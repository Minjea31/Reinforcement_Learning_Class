"""Deterministic evaluation on seeds never used for training resets."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from curriculum import make_env, STAGE_NAMES
from ppo import ActorCritic, RunningNorm, load_checkpoint, resolve_device
from training_viewer import TrainingViewer

# Training uses seeds below this boundary; validation and final test are disjoint.
VALIDATION_SEED = 1_000_000_000
TEST_SEED = 2_000_000_000


def evaluate_policy(policy, normalizer, episodes, seed, max_steps, gamma=.999, viewer=None, stop_requested=None, progress_label='Evaluation', env_config=None):
    env = make_env(env_config,max_steps=max_steps,reward_gamma=gamma)
    records = []
    device = next(policy.parameters()).device
    was_training = policy.training
    policy.eval()
    started = time.monotonic()
    if progress_label:
        print(f'{progress_label}' + (f' stage {env_config["stage"]} ({STAGE_NAMES[env_config["stage"]]})' if env_config else '') + f': starting {episodes} episodes (up to {env.max_steps:,} steps each)...', flush=True)
    try:
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed+episode)
            for step in range(env.max_steps):
                if stop_requested is not None and stop_requested():
                    return None
                with torch.no_grad():
                    action = policy.sample(torch.as_tensor(normalizer(obs)[None],device=device), deterministic=True)[0][0].cpu().numpy()
                obs, reward, terminated, truncated, info = env.step(action)
                if viewer is not None and viewer.ready():
                    viewer.send(env.get_state(), f'Evaluation seed {seed+episode}\nStep {step+1}  reward {reward:.3f}\nSuccess {info["is_success"]}  used arms {info["used_arms"].astype(int)}\nEffort {info["episode_effort"]:.3f}\nTorques Nm {np.round(info["motor_torque"],1)}\nGripper N {np.round(info["gripper_force"],1)}')
                if terminated or truncated:
                    break
            records.append(dict(seed=seed+episode, success=bool(info['is_success']),
                                used_arms=int(info['used_arms'].sum()), effort=float(info['episode_effort']),
                                episode_return=float(info['episode_return']), steps=step+1,
                                grasped=bool(info['used_arms'].any()),lifted=bool(info['episode_lifted'])))
            completed = episode+1
            if progress_label and (completed % 5 == 0 or completed == episodes):
                successes = sum(r['success'] for r in records)
                print(f'{progress_label}: {completed}/{episodes} episodes  '
                      f'success {successes/completed:.1%}  '
                      f'elapsed {time.monotonic()-started:.0f}s', flush=True)
    finally:
        env.close()
        policy.train(was_training)
    successes = [r for r in records if r['success']]
    result = dict(env_config=env_config,episodes=episodes,
                  grasp_rate=float(np.mean([r['grasped'] for r in records])),
                  lift_rate=float(np.mean([r['lifted'] for r in records])), success_rate=len(successes)/episodes,
                  successful_used_arms=float(np.mean([r['used_arms'] for r in successes])) if successes else None,
                  successful_effort=float(np.mean([r['effort'] for r in successes])) if successes else None,
                  mean_return=float(np.mean([r['episode_return'] for r in records])), records=records)
    return result


def ranking(result):
    return (result['success_rate'], -(result['successful_used_arms'] or 0),
            -(result['successful_effort'] or 0), result['mean_return'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default='runs/ppo/best.pt')
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--seed', type=int, default=TEST_SEED)
    parser.add_argument('--viewer', action='store_true')
    parser.add_argument('--output', default=None)
    parser.add_argument('--stage',type=int,choices=(1,2,3,4),help='Evaluate a curriculum model on this stage at full difficulty')
    parser.add_argument('--device', choices=('auto','cpu','cuda'), default='auto')
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error('--episodes must be positive')
    torch.set_num_threads(1)
    try:
        device = resolve_device(args.device)
    except RuntimeError as error:
        parser.error(str(error))
    print(f'Evaluation device: {device}',flush=True)
    checkpoint = load_checkpoint(args.checkpoint)
    policy = ActorCritic(checkpoint['obs_dim'])
    policy.load_state_dict(checkpoint['policy'])
    policy.to(device)
    norm = RunningNorm((checkpoint['obs_dim'],))
    norm.load_state_dict(checkpoint['normalizer'])
    env_config = checkpoint.get('env_config')
    if args.stage is not None:
        if env_config is None:
            parser.error('--stage requires a curriculum checkpoint')
        env_config = dict(stage=args.stage,difficulty=1.,effort_scale=1. if args.stage==4 else 0.)
    viewer = TrainingViewer(args.viewer, 'PPO: actual evaluation trajectory')
    try:
        result = evaluate_policy(policy, norm, args.episodes, args.seed,
                                 checkpoint['config']['max_steps'], checkpoint['config']['gamma'], viewer,env_config=env_config)
        output = Path(args.output or str(Path(args.checkpoint).parent/'evaluation.json'))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k != 'records'}, indent=2))
        print(f'Results: {output}')
    finally:
        viewer.close()


if __name__ == '__main__':
    main()
