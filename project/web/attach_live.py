"""Attach live window video to a running trainer without restarting it."""
import argparse
import base64
import csv
import io
import json
import os
from pathlib import Path
import signal
import threading
import time
from urllib import request

from web.live_access import read_credentials,ensure_mission_watcher,ensure_history_watcher
from web.live_stream import FrameHub,start_stream
from web.window_capture import find_viewer,WindowCapture,scene_only


class NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):return None


def verify_trainer(pid,run):
    command=Path(f'/proc/{pid}/cmdline').read_bytes().decode().split('\0')
    if 'training.train' not in command and not any(x.endswith('/train.py') for x in command):
        raise ValueError('PID is not a training process')
    index=command.index('--run-dir')
    path=Path(command[index+1])
    if not path.is_absolute():path=Path(f'/proc/{pid}/cwd').resolve()/path
    if path.resolve()!=run:raise ValueError('Training PID belongs to another experiment')


class Telemetry:
    def __init__(self,run):
        import torch
        from simulation.dual_reacher import DualReacherEnv
        torch.set_num_threads(1)
        self.torch=torch;self.run=run;self.env=DualReacherEnv();self.stamp=None;self.metrics={}
        self.validation_path=None;self.validation={};self.next_eval=float('inf')

    def progress_row(self):
        with (self.run/'metrics.csv').open() as stream:
            header=stream.readline()
            stream.seek(0,2);end=stream.tell();stream.seek(max(0,end-4096))
            lines=stream.read().splitlines()
            if lines and lines[-1]!=header.strip():return next(csv.DictReader([header,lines[-1]]))
        return {}

    def phase(self,row):
        return 'evaluating' if int(row.get('steps',0))>=self.next_eval or self.metrics.get('steps',0)>=self.metrics.get('total_steps',float('inf')) else 'training'

    def refresh(self):
        source=self.run/'latest.pt';stamp=source.stat().st_mtime_ns
        if stamp==self.stamp:
            # Validation begins after the CSV row but before latest.pt is saved.
            self.metrics['phase']=self.phase(self.progress_row())
            return self.metrics.copy()
        checkpoint=self.torch.load(io.BytesIO(source.read_bytes()),map_location='cpu',weights_only=False)
        state=checkpoint['worker_states'][0]
        self.env.set_state(state);info=self.env.info();task=state['task'];config=checkpoint['config']
        curriculum=checkpoint.get('curriculum') or {};stage=curriculum.get('stage',0)
        start=config.get('stage_start_steps',0)
        row=self.progress_row();self.next_eval=checkpoint['next_eval']
        phase='evaluating' if int(row.get('steps',0))>=checkpoint['next_eval'] or checkpoint['steps']>=config['total_steps'] else 'training'
        paths=list(self.run.glob('validation_*.json'))
        latest=max(paths,key=lambda p:int(p.stem.split('_')[1])) if paths else None
        if latest and latest!=self.validation_path:
            self.validation=json.loads(latest.read_text());self.validation_path=latest
        validation=self.validation
        self.metrics=dict(phase=phase,steps=checkpoint['steps'],total_steps=config['total_steps'],
            updates=checkpoint['updates'],fps=int(row['fps']) if row.get('fps') else None,
            stage=stage,level=curriculum.get('level',0),task_version=curriculum.get('task_version',1),
            physics_backend=config.get('physics_backend','cpu'),num_envs=config['num_envs'],
            layout_mode=config.get('layout_mode','fixed'),
            run_name=self.run.name,stage_start_steps=start,stage_steps=checkpoint['steps']-start,
            stage_total_steps=config['total_steps']-start,episode_step=task['steps'],
            episode_return=task['episode_return'],grasped=list(info['grasped']),lifted=bool(task['episode_lifted']),
            success=bool(task['success_steps']*self.env.dt>=(.5 if stage==1 else 1.)-1e-9),
            max_torque=float(abs(info['motor_torque']).max()),max_gripper_force=float(abs(info['gripper_force']).max()),
            goal_distance=info['goal_distance'],payload_mass=info['payload_mass'],
            validation_success_rate=validation.get('success_rate'),validation_grasp_rate=validation.get('grasp_rate'),
            validation_lift_rate=validation.get('lift_rate'),metrics_source='latest_checkpoint',
            metrics_at=source.stat().st_mtime,video_source='actual_training_scene')
        self.stamp=stamp
        return self.metrics.copy()

    def close(self):self.env.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',required=True)
    parser.add_argument('--connection-dir',default='runs/stage1_100m')
    parser.add_argument('--training-pid',required=True,type=int)
    parser.add_argument('--fps',type=int,default=20)
    args=parser.parse_args()
    if not 1<=args.fps<=30:parser.error('fps must be between 1 and 30')
    run=Path(args.run_dir).resolve();verify_trainer(args.training_pid,run)
    window,owner=find_viewer(args.training_pid)
    credentials=read_credentials(args.connection_dir)
    ensure_mission_watcher(run,args.connection_dir)
    ensure_history_watcher(run,args.connection_dir,args.training_pid)
    capture=WindowCapture(window,owner,args.training_pid)
    hub=FrameHub();server=start_stream(credentials.get('stream_port',8767),credentials['stream_token'],hub)
    telemetry=Telemetry(run);guard=threading.Lock();metrics=telemetry.refresh();stopped=threading.Event()
    client=request.build_opener(NoRedirect())
    def local_json(name,data):
        path=run/name;temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(data)+'\n');os.replace(temporary,path)
    def send(final=False):
        with hub.condition:jpeg=hub.jpeg;sequence=hub.sequence
        if jpeg is None:return
        with guard:payload=metrics.copy()
        payload.update(sequence=sequence,sent_at=time.time(),stream_enabled=not final,
                       training_pid=args.training_pid,viewer_pid=owner,video_fps=args.fps,
                       image=base64.b64encode(jpeg).decode('ascii'))
        req=request.Request(credentials['origin']+'/api/live',data=json.dumps(payload,separators=(',',':')).encode(),
            method='POST',headers={'Content-Type':'application/json','OAI-Sites-Authorization':'Bearer '+credentials['site_token']})
        with client.open(req,timeout=4) as response:
            if response.status!=200:raise RuntimeError('Live upload rejected')
        local_json('live_status.json',{k:v for k,v in payload.items() if k!='image'})
        local_json('live_health.json',dict(status='connected',last_upload_at=time.time(),sequence=sequence,
            video_source='actual_training_scene',training_pid=args.training_pid,viewer_pid=owner))
    def upload():
        while not stopped.is_set():
            try:
                current=telemetry.refresh()
                with guard:metrics.update(current)
                send()
            except Exception as ex:
                local_json('live_health.json',dict(status='retrying',error=type(ex).__name__))
            stopped.wait(1.)
    def stop(*_):stopped.set()
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    (run/'window_live.pid').write_text(str(os.getpid())+'\n')
    thread=threading.Thread(target=upload,daemon=True);thread.start()
    print(f'Attached actual viewer {hex(window)} (PID {owner}) to training PID {args.training_pid}; {args.fps}fps.',flush=True)
    try:
        while not stopped.is_set():
            start=time.monotonic()
            if not Path(f'/proc/{args.training_pid}').exists():break
            frame=scene_only(capture.capture());frame.thumbnail((960,600))
            output=io.BytesIO();frame.save(output,format='JPEG',quality=75)
            hub.publish(output.getvalue())
            with guard:metrics.update(frame_at=time.time())
            stopped.wait(max(0.,1./args.fps-(time.monotonic()-start)))
    finally:
        stopped.set();thread.join(timeout=5)
        with guard:
            metrics['phase']='stopped' if not Path(f'/proc/{args.training_pid}').exists() else metrics.get('phase','training')
        try:send(final=True)
        except Exception:pass
        hub.close();server.shutdown();server.server_close();capture.close();telemetry.close()


if __name__=='__main__':main()
