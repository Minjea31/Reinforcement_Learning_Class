"""Launch independent algorithm processes, one simulation and model per process."""
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from training.algorithms import ALGORITHMS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',default='runs/studies/algorithm_comparison')
    p.add_argument('--algorithms',nargs='+',choices=ALGORITHMS,default=list(ALGORITHMS))
    p.add_argument('--parallel',type=int,default=3,help='One independent simulation per algorithm, at most three')
    p.add_argument('--web',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--local-viewer',action=argparse.BooleanOptionalAction,default=False)
    p.add_argument('--connection-dir',default='runtime/web_connection')
    args,options=p.parse_known_args()
    if not 1<=args.parallel<=3 or len(args.algorithms)!=len(set(args.algorithms)):
        p.error('Choose 1..3 processes and distinct algorithms')
    if args.local_viewer and not args.web:
        p.error('The local viewer uses the same live video feed as the web; enable --web')
    if importlib.util.find_spec('stable_baselines3') is None:
        p.error('Install requirements-study.txt in your active Conda environment first')
    root=Path(args.run_dir).resolve();root.mkdir(parents=True,exist_ok=True)
    lock=(root/'.study.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:p.error('This comparison is already running')
    from training.algorithm_study import parser as child_parser,validate
    from training.study_budget import restore_stage_start, stage_deadline
    for algorithm in args.algorithms:
        proposed=child_parser().parse_args(['--algorithm',algorithm,'--run-dir',str(root/algorithm),*options])
        try:validate(proposed)
        except ValueError as error:p.error(str(error))
    if proposed.fixed_study_normalization:
        from training.fixed_study_normalization import VERSION, prepare_fixed_stats
        calibration = Path(proposed.fixed_study_normalization).resolve()
        if calibration != root/'fixed_normalization.json':
            p.error('Place fixed_normalization.json in the separate masked study directory')
        for algorithm in args.algorithms:
            folder = root/algorithm
            if folder.exists() and any(folder.iterdir()):
                if not proposed.resume:
                    p.error('The masked study already has results; use --resume or a new study directory')
                config_path = folder/'config.json'
                if (not config_path.exists()
                        or json.loads(config_path.read_text()).get('normalization_version') != VERSION):
                    p.error('A masked study cannot resume or write to an existing unmasked run')
        if proposed.resume and not calibration.exists():
            p.error('The saved calibration file is required to resume this masked study')
        if not calibration.exists():
            print('새 실험의 공통 관측 통계 준비 중…', flush=True)
        prepare_fixed_stats(calibration)
    from web.study_connection import study_identity,connect_study,close_connections,register_study,read_credentials
    study_id=study_identity(root);connections={}
    project=Path(__file__).resolve().parents[1]
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    waiting=list(args.algorithms);running={};results={};stopping=False;signaled=set()
    viewer=None
    old_handlers={}
    def stop(signum,frame):
        nonlocal stopping
        stopping=True
    for sig in (signal.SIGINT,signal.SIGTERM):old_handlers[sig]=signal.signal(sig,stop)
    try:
        stages = dict.fromkeys(args.algorithms, proposed.stage)
        next_registration = 0.
        if '--resume' in options:
            for algorithm in list(waiting):
                path=root/algorithm/'latest/metadata.json'
                if not path.exists():
                    continue
                saved = json.loads(path.read_text())
                saved_config = saved.get('config', {})
                stages[algorithm] = saved_config.get('stage', proposed.stage)
                budget = dict(total_steps=proposed.total_steps, stage_max_steps=proposed.stage_max_steps)
                if proposed.stage_max_steps is not None:
                    try:
                        budget['stage_start_steps'] = restore_stage_start(
                            root/algorithm, saved, proposed.stage, stages[algorithm])
                    except ValueError as error:
                        p.error(str(error))
                if saved['steps']>=stage_deadline(budget) or (
                        proposed.auto_stages and saved_config.get('level_controller', {}).get('completed')):
                    waiting.remove(algorithm);results[algorithm]=0
                    print(algorithm.upper()+' 이미 예산 완료 · 재실행하지 않습니다.',flush=True)
        if args.web:
            connections=connect_study(root,study_id,args.algorithms,args.connection_dir,stages,proposed.total_steps,active=set(waiting))
            print('웹: https://dual-reacher-best-replay.lunarknoll21.chatgpt.site/?study='+study_id,flush=True)
        if args.local_viewer and waiting:
            if not os.environ.get('DISPLAY'):
                print('로컬 창은 그래픽 데스크톱에서 열 수 있습니다. 웹 영상은 계속 전송합니다.', flush=True)
            else:
                with (root/'local_viewer.log').open('ab',buffering=0) as log:
                    viewer=subprocess.Popen([sys.executable,'-B','-m','web.local_viewer','--run-dir',str(root),
                        '--connection-dir',str(Path(args.connection_dir).resolve()),'--wait','60'],
                        cwd=project,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
        (root/'study.json').write_text(json.dumps(dict(id=study_id,algorithms=args.algorithms),indent=2)+'\n')
        while waiting or running:
            while waiting and len(running)<args.parallel and not stopping:
                algorithm=waiting.pop(0);folder=root/algorithm
                log=(root/f'{algorithm}.log').open('ab',buffering=0)
                command=[sys.executable,'-B','-m','training.algorithm_study','--algorithm',algorithm,'--run-dir',str(folder),
                         '--study-id',study_id,'--web-run-name',study_id+'_'+algorithm,*options]
                if args.web:command+=['--web','--connection-dir',str(Path(args.connection_dir).resolve()),'--stream-port',str(connections[algorithm]['port'])]
                process=subprocess.Popen(command,cwd=project,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                log.close();running[algorithm]=process
                print(f'{algorithm.upper()} 시작 · PID {process.pid} · {root/algorithm}',flush=True)
            for algorithm,process in list(running.items()):
                if process.poll() is not None:
                    results[algorithm]=process.returncode;del running[algorithm]
                    print(f'{algorithm.upper()} 종료 · exit {process.returncode}',flush=True)
                    if process.returncode:
                        # A failed algorithm must not terminate the other models.
                        folder=root/algorithm;folder.mkdir(parents=True,exist_ok=True);path=folder/'run_status.json'
                        status=json.loads(path.read_text()) if path.exists() else dict(steps=0,total_steps=proposed.total_steps)
                        status['phase']='error';path.write_text(json.dumps(status)+'\n')
                        if args.web and (folder/'config.json').exists():
                            from web.live_access import ensure_history_watcher
                            ensure_history_watcher(folder,args.connection_dir,process.pid)
            if stopping:
                waiting.clear()
                for process in running.values():
                    if process.poll() is None and process.pid not in signaled:
                        process.send_signal(signal.SIGTERM);signaled.add(process.pid)
            rows={}
            current_stages=stages.copy()
            for algorithm in args.algorithms:
                status=root/algorithm/'comparison_status.json'
                if status.exists():rows[algorithm]=json.loads(status.read_text())
                run_status=root/algorithm/'run_status.json'
                if run_status.exists():
                    current_stages[algorithm] = json.loads(run_status.read_text()).get('stage') or stages[algorithm]
            if args.web and current_stages != stages and time.monotonic() >= next_registration:
                try:
                    register_study(root,study_id,args.algorithms,read_credentials(args.connection_dir),
                                   current_stages,proposed.total_steps)
                except (OSError,ValueError,RuntimeError):
                    next_registration = time.monotonic()+15
                else:
                    stages = current_stages
            temporary=root/'comparison.partial.json';temporary.write_text(json.dumps(rows,indent=2)+'\n');os.replace(temporary,root/'comparison.json')
            if running:time.sleep(.5)
    finally:
        for process in running.values():
            if process.poll() is None:process.send_signal(signal.SIGTERM)
        for process in running.values():process.wait()
        if viewer and viewer.poll() is None:
            viewer.terminate()
            try:viewer.wait(timeout=5)
            except subprocess.TimeoutExpired:viewer.kill();viewer.wait()
        close_connections(connections)
        for sig,handler in old_handlers.items():signal.signal(sig,handler)
        lock.close()
    return 1 if any(results.values()) else 130 if stopping else 0


if __name__=='__main__':
    raise SystemExit(main())
