"""Export and publish the best validated policy separately for each mission."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import time
from urllib import request

from web.live_access import read_credentials


class NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):return None


def checkpoint_source(run_dir,stage):
    run=Path(run_dir);source=run/f'stage{stage}_best.pt'
    if stage==1 and not source.exists():source=run/'pickup_source.pt'
    return source


def snapshot(run_dir,stage):
    import torch
    source=checkpoint_source(run_dir,stage)
    if not source.exists():return None
    raw=source.read_bytes()
    checkpoint=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False)
    if checkpoint.get('env_config',{}).get('stage')!=stage:
        raise ValueError('Mission checkpoint has a different stage')
    digest=hashlib.sha256(raw).hexdigest()
    folder=Path(run_dir)/'mission_replays'/f'stage-{stage}'/digest
    folder.mkdir(parents=True,exist_ok=True)
    frozen=folder/'checkpoint.pt'
    if not frozen.exists():frozen.write_bytes(raw)
    return folder,checkpoint,digest


def prepare(run_dir,stage):
    result=snapshot(run_dir,stage)
    if result is None:return None
    folder,checkpoint,digest=result
    output=folder/'assets'
    if not (output/'replays.json').exists():
        from evaluation.export_replay import export
        level=checkpoint['curriculum']['level']
        export(argparse.Namespace(checkpoint=str(folder/'checkpoint.pt'),output=str(output),
            seed=None,top_k=10,device='cpu',fps=50,evaluate=True,eval_episodes=50,
            run_name=Path(run_dir).resolve().name,
            eval_seed=1_000_000_000+stage*1_000_000+level*10_000))
    return describe(run_dir,stage,folder,checkpoint,digest)


def describe(run_dir,stage,folder,checkpoint,digest):
    output=folder/'assets'
    manifest=json.loads((output/'replays.json').read_text())
    # Cached exports predate evaluation episode numbers. Recover the original
    # order from the complete evaluation, never from the reward-ranked list.
    evaluation=output/'evaluation.json'
    if evaluation.exists():
        records=json.loads(evaluation.read_text())['records']
        episodes={record['seed']:index+1 for index,record in enumerate(records)}
        for entry in manifest['entries']:
            if entry.get('episode') is None and entry.get('seed') in episodes:
                entry['episode']=episodes[entry['seed']]
            metadata=output/entry['metadata']
            if entry.get('episode') is not None and metadata.exists():
                record=json.loads(metadata.read_text())
                if record.get('episode')!=entry['episode']:
                    record['episode']=entry['episode']
                    metadata.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    source=checkpoint_source(run_dir,stage)
    run_config=Path(run_dir)/'config.json'
    experiment=json.loads(run_config.read_text()) if run_config.exists() else checkpoint['config']
    manifest.update(policy_selection='inherited_stage1_checkpoint' if source.name=='pickup_source.pt' else 'best_validation_checkpoint',source_checkpoint=source.name,
        run_name=Path(run_dir).resolve().name,task_version=(checkpoint.get('env_config') or {}).get('task_version',1),
        physics_backend=checkpoint['config'].get('physics_backend','cpu'),num_envs=checkpoint['config']['num_envs'],
        run_profile=dict(task_version=experiment.get('task_version') or 1,physics_backend=experiment.get('physics_backend') or 'cpu',num_envs=experiment['num_envs'],layout_mode=experiment.get('layout_mode') or 'fixed'),
        stage=stage,level=checkpoint['curriculum']['level'],
        asset_base=f'/api/mission-assets/stage-{stage}/{digest}/')
    return output,manifest


def publish(run_dir,output,manifest,connection_dir=None):
    credentials=read_credentials(connection_dir or run_dir)
    origin=credentials['origin'].rstrip('/')
    client=request.build_opener(NoRedirect())
    headers={'OAI-Sites-Authorization':'Bearer '+credentials['site_token']}
    types={'video':'video/mp4','poster':'image/jpeg','metadata':'application/json'}
    for entry in manifest['entries']:
        for kind,mime in types.items():
            relative=entry[kind]
            if relative!=f'rank-{entry["rank"]:02d}/'+{'video':'replay.mp4','poster':'poster.jpg','metadata':'replay.json'}[kind]:
                raise ValueError('Unexpected replay asset path')
            path=Path(output)/relative
            req=request.Request(origin+manifest['asset_base']+relative,data=path.read_bytes(),method='PUT',
                headers=dict(headers,**{'Content-Type':mime}))
            with client.open(req,timeout=60) as response:
                if response.status!=200:raise RuntimeError('Replay upload failed')
    body=json.dumps(manifest,ensure_ascii=False,separators=(',',':')).encode()
    req=request.Request(origin+f'/api/missions/{manifest["stage"]}',data=body,method='POST',
        headers=dict(headers,**{'Content-Type':'application/json'}))
    with client.open(req,timeout=30) as response:
        if response.status!=200:raise RuntimeError('Mission publication failed')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',default='runs/stage1_100m')
    parser.add_argument('--connection-dir',help='Reuse private web credentials from this run directory')
    parser.add_argument('--once',action='store_true')
    parser.add_argument('--export-only',action='store_true')
    parser.add_argument('--interval',type=float,default=30)
    parser.add_argument('--min-refresh-seconds',type=float,default=300)
    args=parser.parse_args()
    if args.interval<1 or args.min_refresh_seconds<0:parser.error('Invalid refresh interval')
    os.environ.setdefault('MUJOCO_GL','egl')
    os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    try:os.nice(10)
    except OSError:pass
    run=Path(args.run_dir).resolve();run.mkdir(parents=True,exist_ok=True)
    state_path=run/'mission_replays'/'published.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else {}
    stopped=False
    def stop(*_):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    if not args.once:(run/'mission_replays.pid').write_text(str(os.getpid())+'\n')
    while not stopped:
        for stage in range(1,5):
            if stopped:break
            try:
                candidate=snapshot(run,stage)
                if candidate is None:continue
                _,_,digest=candidate;previous=state.get(str(stage),{})
                if not args.export_only and (previous.get('hash')==digest or
                    time.time()-previous.get('published_at',0)<args.min_refresh_seconds):continue
                result=prepare(run,stage)
                if stopped:break
                output,manifest=result
                if args.export_only:continue
                publish(run,output,manifest,args.connection_dir)
                state[str(stage)]=dict(hash=manifest['checkpoint_sha256'],steps=manifest['checkpoint_steps'],published_at=time.time())
                temporary=state_path.with_suffix('.tmp');temporary.write_text(json.dumps(state,indent=2)+'\n');os.replace(temporary,state_path)
                print(f'Mission {stage}: published best policy at {manifest["checkpoint_steps"]:,} steps, success {manifest["validation_success_rate"]:.1%}',flush=True)
            except Exception as exc:
                print(f'Mission {stage}: replay update pending ({type(exc).__name__}); training continues.',flush=True)
                if args.once:raise
        if args.once:break
        for _ in range(max(1,int(args.interval*5))):
            if stopped:break
            time.sleep(.2)


if __name__=='__main__':main()
