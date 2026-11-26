"""Save smooth, real-time-speed evaluation videos of a frozen policy checkpoint."""
import argparse
from datetime import datetime,timezone
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess

os.environ.setdefault('MUJOCO_GL','egl')
import mujoco
import numpy as np
import torch
from PIL import Image,ImageDraw
from curriculum import make_env
from ppo import ActorCritic,RunningNorm


def record(checkpoint_path,output,episodes=5,seed=2_100_000_000,fps=50,milestone=None):
    torch.set_num_threads(1)
    raw=Path(checkpoint_path).read_bytes()
    checkpoint=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False)
    policy=ActorCritic(checkpoint['obs_dim']);policy.load_state_dict(checkpoint['policy']);policy.eval()
    norm=RunningNorm((checkpoint['obs_dim'],));norm.load_state_dict(checkpoint['normalizer'])
    config=checkpoint['config']
    env=make_env(checkpoint.get('env_config'),max_steps=config['max_steps'],reward_gamma=config['gamma'])
    renderer=mujoco.Renderer(env.model,height=600,width=960)
    camera=mujoco.MjvCamera();mujoco.mjv_defaultCamera(camera)
    camera.lookat[:]=[0,0,.78];camera.distance,camera.azimuth,camera.elevation=2.45,90,-40
    stride=round(1/(env.dt*fps));actual_fps=1/(env.dt*stride)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    temporary=output/'video.partial.mp4';video=output/'video.mp4'
    encoder=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pixel_format','rgb24','-video_size','960x600',
        '-framerate',str(actual_fps),'-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','21',
        '-pix_fmt','yuv420p','-threads','2','-movflags','+faststart',str(temporary)],stdin=subprocess.PIPE)
    records=[];frames=0
    try:
        for episode in range(episodes):
            obs,_=env.reset(seed=seed+episode)
            for step in range(env.max_steps):
                with torch.no_grad():action=policy.sample(torch.from_numpy(norm(obs)[None]),deterministic=True)[0][0].numpy()
                obs,_,term,trunc,info=env.step(action)
                if (step+1)%stride==0 or term or trunc:
                    renderer.update_scene(env.data,camera=camera)
                    image=Image.fromarray(renderer.render());draw=ImageDraw.Draw(image)
                    draw.rectangle((10,10,670,56),fill=(12,16,22))
                    stage=checkpoint.get('env_config',{}).get('stage',0) if checkpoint.get('env_config') else 0
                    draw.text((20,18),f'MODEL {checkpoint["steps"]:,} STEPS | STAGE {stage} | EPISODE {episode+1}/{episodes}',fill=(123,212,176))
                    draw.text((20,36),f'SIM TIME {env.data.time:.2f}s | RETURN {info["episode_return"]:.3f}',fill=(233,238,245))
                    encoder.stdin.write(image.tobytes());frames+=1
                if term or trunc:break
            records.append(dict(seed=seed+episode,steps=step+1,sim_seconds=env.data.time,
                                success=bool(info['is_success']),episode_return=info['episode_return']))
        encoder.stdin.close()
        if encoder.wait()!=0:raise RuntimeError('Video encoding failed')
        os.replace(temporary,video)
        metadata=dict(kind='checkpoint_evaluation',recorded_at=datetime.now(timezone.utc).isoformat(),
            milestone_steps=milestone,checkpoint_steps=checkpoint['steps'],checkpoint_sha256=hashlib.sha256(raw).hexdigest(),
            env_config=checkpoint.get('env_config'),fps=actual_fps,frames=frames,
            video_seconds=frames/actual_fps,playback_speed=1.,episodes=records)
        meta_tmp=output/'metadata.partial.json';meta_tmp.write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n');os.replace(meta_tmp,output/'metadata.json')
        print(f'Saved {video}: {actual_fps:g}fps, {frames/actual_fps:.2f}s, model {checkpoint["steps"]:,} steps',flush=True)
        return metadata
    finally:
        if encoder.poll() is None:
            encoder.stdin.close();encoder.terminate();encoder.wait()
        renderer.close();env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--episodes',type=int,default=5);p.add_argument('--seed',type=int,default=2_100_000_000)
    p.add_argument('--fps',type=int,choices=(10,25,50),default=50);p.add_argument('--milestone',type=int)
    a=p.parse_args()
    if a.episodes<1:p.error('--episodes must be positive')
    record(a.checkpoint,a.output,a.episodes,a.seed,a.fps,a.milestone)
