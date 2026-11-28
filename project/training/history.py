"""Durable episode records and reproducible presentation material metadata."""
import csv
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

from training.study_budget import stage_progress

EPISODE_FIELDS=['episode','env_id','steps','stage','level','reward','success','grasped','lifted',
                'used_arms','effort','collisions','goal_distance','episode_steps','termination','time_utc']


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    os.replace(temporary,path)


def report_process(run_dir):
    run=Path(run_dir).resolve();pid_file=run/'reports/generation.pid'
    if pid_file.exists():
        try:
            pid=int(pid_file.read_text());command=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
            if b'evaluation.training_report' in command and str(run).encode() in command:return pid
        except (OSError,ValueError):pass
    return None


def launch_report(run_dir,live=False):
    run=Path(run_dir).resolve();folder=run/'reports';folder.mkdir(exist_ok=True)
    pid_file=folder/'generation.pid'
    active=report_process(run)
    if active:return active
    env=dict(os.environ,MPLBACKEND='Agg',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    with (folder/'generation.log').open('ab',buffering=0) as log:
        command=[sys.executable,'-B','-m','evaluation.training_report','--run-dir',str(run)]
        if live:command.append('--live')
        process=subprocess.Popen(command,
            cwd=Path(__file__).resolve().parents[1],env=env,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    pid_file.write_text(str(process.pid)+'\n')
    return process.pid


class TrainingHistory:
    def __init__(self,run_dir,config,steps):
        import gymnasium,mujoco,numpy,torch
        self.run=Path(run_dir).resolve();self.config=config
        self.run.mkdir(parents=True,exist_ok=True)
        self.episode_count=0
        path=self.run/'episodes.csv'
        if path.exists():
            with path.open() as stream:
                for row in csv.DictReader(stream):self.episode_count=max(self.episode_count,int(row['episode']))
        self.stream=path.open('a',newline='');self.writer=csv.DictWriter(self.stream,fieldnames=EPISODE_FIELDS)
        if self.stream.tell()==0:self.writer.writeheader();self.stream.flush()
        metadata_path=self.run/'artifacts/run_metadata.json'
        if not metadata_path.exists():
            parent=Path(config['resume']).resolve().parent if config.get('resume') else None
            metadata=dict(run_name=self.run.name,started_at=datetime.now(timezone.utc).isoformat(),
                initial_steps=steps,parent_run=str(parent) if parent and parent!=self.run else None,
                parent_cutoff_steps=steps if parent and parent!=self.run else None,
                python=platform.python_version(),versions=dict(torch=torch.__version__,mujoco=mujoco.__version__,numpy=numpy.__version__,gymnasium=gymnasium.__version__),
                observation_dim=config.get('observation_dim',105 if config.get('curriculum') else 89),
                action_dim=14,payload_kg=config.get('payload_mass',1.),config=config.copy())
            atomic_json(metadata_path,metadata)
            root=Path(__file__).resolve().parents[1];manifest=[]
            files=[*root.glob('simulation/*.py'),*root.glob('training/*.py'),*root.glob('evaluation/*.py'),*root.glob('web/*.py'),
                   root/'models/dual_reacher.xml',root/'reward.md',root/'additional_study.md',root/'environment.yml',root/'requirements.txt']
            for source in files:
                if not source.is_file():continue
                relative=source.relative_to(root);target=self.run/'artifacts/source'/relative
                target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
                manifest.append(dict(path=str(relative),sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
            atomic_json(self.run/'artifacts/source_manifest.json',manifest)
        self.status('training',steps)

    def status(self,phase,steps):
        atomic_json(self.run/'run_status.json',dict(phase=phase,steps=steps,total_steps=self.config['total_steps'],
            stage=self.config.get('stage'),level=self.config.get('level'),
            updated_at=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),
            **stage_progress(self.config,steps)))

    def episode(self,info,steps,env_id,stage,level,terminated):
        self.episode_count+=1
        self.writer.writerow(dict(episode=self.episode_count,env_id=env_id,steps=steps,stage=stage,level=level,
            reward=info['episode_return'],success=int(info['is_success']),grasped=int(info['used_arms'].any()),
            lifted=int(info['episode_lifted']),used_arms=int(info['used_arms'].sum()),effort=info['episode_effort'],
            collisions=info.get('collision_total',0),goal_distance=info['goal_distance'],
            episode_steps=info.get('episode_step',0),termination='success' if info['is_success'] else 'fall' if info.get('fallen') else 'terminated' if terminated else 'timeout',
            time_utc=datetime.now(timezone.utc).isoformat()))
        self.stream.flush()

    def update(self,row):
        event=dict(row,event='update',time_utc=datetime.now(timezone.utc).isoformat())
        with (self.run/'training_events.jsonl').open('a') as stream:stream.write(json.dumps(event)+'\n')
        self.status('training',row['steps'])

    def finish(self,phase,steps):
        self.stream.close();self.status(phase,steps)
        if self.config.get('report',True):return launch_report(self.run)
