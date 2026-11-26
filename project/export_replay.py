"""Export ranked validation scenarios of one frozen policy for web replay."""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess

os.environ.setdefault('MUJOCO_GL', 'egl')
import mujoco
import numpy as np
import torch
from curriculum import make_env
from evaluate import evaluate_policy, VALIDATION_SEED
from ppo import ActorCritic, RunningNorm, resolve_device


def select_scenarios(validation, seed, top_k):
    if seed is not None:
        if top_k != 1:
            raise ValueError('--seed selects one scenario; use --top-k 1')
        return [dict(seed=seed,episode_return=None)]
    if validation is None or not validation['records']:
        raise ValueError('Matching validation results missing; provide --seed for a specific replay')
    if top_k > len(validation['records']):
        raise ValueError('--top-k exceeds the number of validation scenarios')
    return sorted(validation['records'],key=lambda r:(-r['episode_return'],r['seed']))[:top_k]


def export_scenario(args, checkpoint, checkpoint_hash, validation, selected, rank, output,
                    policy, normalizer, device):
    output.mkdir(parents=True,exist_ok=True)
    config = checkpoint['config']
    env = make_env(checkpoint.get('env_config'),max_steps=config['max_steps'],reward_gamma=config['gamma'])
    obs, info = env.reset(seed=selected['seed'])
    renderer = mujoco.Renderer(env.model,height=600,width=960)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = [0,0,.78]
    camera.distance,camera.azimuth,camera.elevation = 2.45,90,-40
    stride = round(1/(env.dt*args.fps))
    fps = 1/(env.dt*stride)
    video = output/'replay.mp4'
    temporary = output/'replay.partial.mp4'
    command = ['ffmpeg','-v','error','-y','-f','rawvideo','-pixel_format','rgb24',
               '-video_size','960x600','-framerate',str(fps),'-i','pipe:0','-an',
               '-c:v','libx264','-preset','fast','-crf','23','-pix_fmt','yuv420p',
               '-threads','2','-movflags','+faststart',str(temporary)]
    encoder = subprocess.Popen(command,stdin=subprocess.PIPE)
    samples = []
    initial_hands = env.data.site_xpos[env.sites].copy()
    previous_hands = initial_hands.copy()
    hand_path = np.zeros(2)
    late_hand_path = np.zeros(2)
    max_hand_displacement = np.zeros(2)
    max_object_xy_displacement = 0.
    max_object_z_displacement = 0.
    grasp_seconds = 0.
    carried_seconds = 0.
    reward_totals = {}
    def frame(reward, info):
        renderer.update_scene(env.data,camera=camera)
        encoder.stdin.write(renderer.render().tobytes())
        samples.append(dict(time=round(env.data.time,3),step=env.steps,
            reward=round(float(reward),5),episode_return=round(env.episode_return,5),
            goal_distance=round(info['goal_distance'],4),
            max_torque=round(float(np.abs(info['motor_torque']).max()),2),
            max_gripper_force=round(float(np.abs(info['gripper_force']).max()),2),
            object_xy_displacement=round(float(np.linalg.norm(info['payload_position'][:2]-env.start[:2])),6),
            object_z_displacement=round(float(info['payload_position'][2]-env.start[2]),6),
            used_arms=env.used_arms.astype(int).tolist(),grasped=list(info['grasped']),
            carried=env.carried))
    print(f'Rank {rank}: checkpoint {checkpoint["steps"]:,}, seed {selected["seed"]}, device {device}',flush=True)
    try:
        frame(0.,info)
        for step in range(env.max_steps):
            with torch.no_grad():
                action = policy.sample(torch.as_tensor(normalizer(obs)[None],device=device),deterministic=True)[0][0].cpu().numpy()
            obs,reward,terminated,truncated,info = env.step(action)
            hands = env.data.site_xpos[env.sites].copy()
            movement = np.linalg.norm(hands-previous_hands,axis=1)
            hand_path += movement
            if env.data.time > 2.:
                late_hand_path += movement
            previous_hands = hands
            max_hand_displacement = np.maximum(max_hand_displacement,np.linalg.norm(hands-initial_hands,axis=1))
            max_object_xy_displacement = max(max_object_xy_displacement,float(np.linalg.norm(info['payload_position'][:2]-env.start[:2])))
            max_object_z_displacement = max(max_object_z_displacement,float(info['payload_position'][2]-env.start[2]))
            grasp_seconds += env.dt if any(info['grasped']) else 0.
            carried_seconds += env.dt if info['carried'] else 0.
            for key,value in info['reward_terms'].items():
                reward_totals[key] = reward_totals.get(key,0.)+value
            if (step+1)%stride == 0 or terminated or truncated:
                frame(reward,info)
            if terminated or truncated:
                break
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError('Video encoding failed')
        os.replace(temporary,video)
        subprocess.run(['ffmpeg','-v','error','-y','-ss',str(min(.4,float(env.data.time)/2)),'-i',str(video),'-frames:v','1',
                        '-threads','1',str(output/'poster.jpg')],check=True)
        if not (output/'poster.jpg').is_file():
            raise RuntimeError('Video poster was not generated')
        metadata = dict(env_config=checkpoint.get('env_config'),control_mode=checkpoint.get('control_mode','absolute_targets'),title='두 팔 로봇 · 보상 상위 시나리오',rank=rank,
            generated_at=datetime.now(timezone.utc).isoformat(),checkpoint_steps=checkpoint['steps'],
            checkpoint_sha256=checkpoint_hash,seed=selected['seed'],
            selection='ranked_validation_episode_return' if selected.get('episode_return') is not None else 'specified_seed',
            validation_episodes=validation['episodes'] if validation else None,
            validation_success_rate=validation['success_rate'] if validation else None,
            expected_return=selected.get('episode_return'),episode_return=float(info['episode_return']),
            success=bool(info['is_success']),effort=float(info['episode_effort']),used_arms=int(env.used_arms.sum()),
            duration_seconds=float(env.data.time),steps=env.steps,fps=fps,payload_mass=env.mass,
            total_training_steps=config['total_steps'],start=env.start.tolist(),goal=env.goal.tolist(),
            obstacle_position=env.obstacle_position.tolist(),
            obstacle_size=(2*env.model.geom_size[env.obstacle_geom]).tolist(),
            motion=dict(hand_path_m=hand_path.tolist(),hand_path_after_2s_m=late_hand_path.tolist(),
                        max_hand_displacement_m=max_hand_displacement.tolist(),
                        max_object_xy_displacement_m=max_object_xy_displacement,
                        max_object_z_displacement_m=max_object_z_displacement,
                        grasp_seconds=grasp_seconds,carried_seconds=carried_seconds),
            reward_totals=reward_totals,samples=samples)
        temp_json = output/'replay.partial.json'
        temp_json.write_text(json.dumps(metadata,ensure_ascii=False,separators=(',',':'))+'\n')
        os.replace(temp_json,output/'replay.json')
        print(f'Rank {rank} exported: return {info["episode_return"]:.6f}, success {info["is_success"]}, '
              f'object XY movement {max_object_xy_displacement*1000:.3f}mm, '
              f'hand path after 2s {np.round(late_hand_path,3)}m, grasp {grasp_seconds:.2f}s',flush=True)
        return metadata
    finally:
        if encoder.poll() is None:
            encoder.stdin.close()
            encoder.terminate()
            encoder.wait()
        renderer.close()
        env.close()


def export(args):
    torch.set_num_threads(1)
    device = resolve_device(args.device)
    checkpoint_path = Path(args.checkpoint)
    checkpoint_bytes = checkpoint_path.read_bytes()
    checkpoint = torch.load(io.BytesIO(checkpoint_bytes),map_location='cpu',weights_only=False)
    checkpoint_hash = hashlib.sha256(checkpoint_bytes).hexdigest()
    validation_path = checkpoint_path.parent/f'validation_{checkpoint["steps"]}.json'
    validation = json.loads(validation_path.read_text()) if validation_path.exists() else None
    if validation is not None and validation.get('env_config') != checkpoint.get('env_config'):
        validation = None
    policy = ActorCritic(checkpoint['obs_dim']).to(device)
    policy.load_state_dict(checkpoint['policy'])
    policy.eval()
    normalizer = RunningNorm((checkpoint['obs_dim'],))
    normalizer.load_state_dict(checkpoint['normalizer'])
    output = Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    if args.evaluate:
        if args.seed is not None:
            raise ValueError('--evaluate cannot be combined with a single --seed')
        validation=evaluate_policy(policy,normalizer,args.eval_episodes,args.eval_seed,
            checkpoint['config']['max_steps'],checkpoint['config']['gamma'],
            env_config=checkpoint.get('env_config'),progress_label='Replay evaluation')
        validation.update(checkpoint_steps=checkpoint['steps'],checkpoint_sha256=checkpoint_hash)
        (output/'evaluation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n')
    scenarios = select_scenarios(validation,args.seed,args.top_k)
    entries = []
    for rank, selected in enumerate(scenarios,1):
        folder = f'rank-{rank:02d}' if args.top_k > 1 else ''
        metadata = export_scenario(args,checkpoint,checkpoint_hash,validation,selected,rank,output/folder,
                                   policy,normalizer,device)
        entries.append({key:metadata[key] for key in ('rank','seed','episode_return','success','duration_seconds','motion')})
        entries[-1].update(video=f'{folder}/replay.mp4'.lstrip('/'),poster=f'{folder}/poster.jpg'.lstrip('/'),
                           metadata=f'{folder}/replay.json'.lstrip('/'))
    manifest = dict(generated_at=datetime.now(timezone.utc).isoformat(),checkpoint_steps=checkpoint['steps'],
                    checkpoint_sha256=checkpoint_hash,env_config=checkpoint.get('env_config'),
                    selection='ranked_validation_episode_return',fps=args.fps,
                    validation_episodes=validation['episodes'] if validation else None,
                    validation_success_rate=validation['success_rate'] if validation else None,entries=entries)
    partial = output/'replays.partial.json'
    partial.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    os.replace(partial,output/'replays.json')
    print(f'Exported {len(entries)} scenarios to {output}',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',default='runs/ppo_100m/best.pt')
    parser.add_argument('--output',default='replay_site/public/assets')
    parser.add_argument('--seed',type=int)
    parser.add_argument('--top-k',type=int,default=1)
    parser.add_argument('--device',choices=('auto','cpu','cuda'),default='auto')
    parser.add_argument('--fps',type=int,choices=(10,25,50),default=25)
    parser.add_argument('--evaluate',action='store_true',help='Evaluate this frozen checkpoint before selecting the highest-return scenarios')
    parser.add_argument('--eval-episodes',type=int,default=50)
    parser.add_argument('--eval-seed',type=int,default=VALIDATION_SEED)
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error('--top-k must be positive')
    if args.eval_episodes<args.top_k and args.evaluate:
        parser.error('--eval-episodes must be at least --top-k')
    export(args)
