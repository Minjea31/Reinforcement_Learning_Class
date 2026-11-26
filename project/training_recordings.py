"""Milestone scheduling and nonblocking local video recording."""
import os
from pathlib import Path
import subprocess
import sys


def next_milestone(steps,every):
    return (steps//every+1)*every if every else None


def rollout_length(steps,total_steps,num_envs,rollout_steps,next_record=None):
    boundary=min(total_steps,next_record) if next_record is not None else total_steps
    return min(rollout_steps,max(1,(boundary-steps+num_envs-1)//num_envs))


class RecordingJobs:
    def __init__(self):self.jobs=[]

    def start(self,checkpoint,output,episodes,fps,milestone=None):
        output=Path(output);output.mkdir(parents=True,exist_ok=True)
        if (output/'metadata.json').exists() and (output/'video.mp4').exists():return
        env=os.environ.copy();env.update(MUJOCO_GL='egl',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        command=[sys.executable,'-B',str(Path(__file__).with_name('record_policy.py')),'--checkpoint',str(checkpoint),
                 '--output',str(output),'--episodes',str(episodes),'--fps',str(fps)]
        if milestone is not None:command+=['--milestone',str(milestone)]
        with (output/'recording.log').open('ab') as log:
            process=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        self.jobs=[job for job in self.jobs if job.poll() is None]+[process]
        (output/'recording.pid').write_text(str(process.pid)+'\n')
