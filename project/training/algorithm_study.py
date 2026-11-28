"""Train one comparison model in its own CPU physics environment.

Each process owns its curriculum controller. The progressive default enables
the reviewed reward and advances only this model's difficulty after validation.
"""
import argparse
import hashlib
import math
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import torch

from evaluation.evaluate import evaluate_policy, curriculum_ranking
from simulation.curriculum import make_env,Curriculum
from simulation.reward_design import PREGRASP_REWARD_SCHEMES, STUDY_REWARD_VERSIONS
from training.algorithms import ALGORITHMS, EpisodeSeeds, EvaluationPolicy, build_model
from training.history import TrainingHistory, atomic_json
from training.study_budget import learn_exact, restore_stage_start, stage_deadline, stage_progress
from training.study_monitor import StudyMonitor
from web.live_training import LiveTrainingFeed


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--algorithm', choices=ALGORITHMS, required=True)
    p.add_argument('--run-dir', required=True)
    p.add_argument('--total-steps', type=int, default=20_000_000)
    p.add_argument('--stage-max-steps', type=int, help='Maximum environment steps in each stage')
    p.add_argument('--stage', type=int, choices=(1, 2, 3, 4), default=1)
    p.add_argument('--level', type=int, choices=range(5), default=0)
    p.add_argument('--layout-mode', choices=('fixed', 'random_separated', 'random_all', 'random_approach','progressive'), default='progressive')
    p.add_argument('--reward-scheme', choices=STUDY_REWARD_VERSIONS, default='reviewed')
    p.add_argument('--payload-mass', type=float, default=1.)
    p.add_argument('--pickup-checkpoint', help='Common frozen native PPO .pt teacher for stage-two initial states')
    p.add_argument('--seed', type=int, default=31)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    p.add_argument('--rollout-steps', type=int, default=256)
    p.add_argument('--fixed-study-normalization', help='Shared calibration file for the stage-one arm-mask study')
    p.add_argument('--eval-every', type=int, default=25_600)
    p.add_argument('--save-every',type=int,default=25_600)
    p.add_argument('--eval-episodes', type=int, default=50)
    p.add_argument('--buffer-size', type=int, default=100_000)
    p.add_argument('--learning-starts', type=int, default=1000)
    p.add_argument('--max-steps',type=int,default=2000)
    p.add_argument('--auto-levels',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--auto-stages',action=argparse.BooleanOptionalAction,default=False)
    p.add_argument('--min-level-steps',type=int,default=100_000)
    p.add_argument('--record-every',type=int,default=5_000_000)
    p.add_argument('--record-episodes',type=int,default=5)
    p.add_argument('--report',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--web',action='store_true')
    p.add_argument('--connection-dir',default='runtime/web_connection')
    p.add_argument('--stream-port',type=int,default=0)
    p.add_argument('--study-id',default='standalone')
    p.add_argument('--web-run-name')
    p.add_argument('--resume', action='store_true')
    return p


def validate(args):
    if args.rollout_steps < 2 or args.total_steps < 1 or args.eval_every < 1 or args.eval_episodes < 1:
        raise ValueError('Steps and episode counts must be positive; rollout must contain at least two steps')
    if args.stage_max_steps is not None and args.stage_max_steps < 1:
        raise ValueError('--stage-max-steps must be positive')
    if args.stage_max_steps is None and args.algorithm in ('ppo', 'a2c') and args.total_steps % args.rollout_steps:
        raise ValueError('On-policy total steps must be a multiple of --rollout-steps')
    if args.buffer_size < 2 or args.learning_starts < 0:
        raise ValueError('Invalid replay-buffer configuration')
    if args.max_steps<1 or args.min_level_steps<0 or args.record_every<0 or args.record_episodes<1 or args.save_every<1:raise ValueError('Invalid curriculum or recording settings')
    if not .05 <= args.payload_mass <= 5.:
        raise ValueError('Payload mass must be between .05 and 5 kg')
    if args.stage == 2 and not args.pickup_checkpoint:
        raise ValueError('Stage two needs a common frozen native --pickup-checkpoint')
    if args.auto_stages and (not args.auto_levels or args.layout_mode != 'progressive'):
        raise ValueError('--auto-stages needs progressive layouts and automatic levels')
    if args.reward_scheme in PREGRASP_REWARD_SCHEMES and (args.layout_mode != 'progressive' or args.fixed_study_normalization):
        raise ValueError('Grasp progress requires the progressive layout with both arms available')
    if args.seed < 0 or args.seed + args.total_steps >= 1_000_000_000:
        raise ValueError('Training reset seeds must remain below the held-out validation range')
    if args.fixed_study_normalization:
        from training.fixed_study_normalization import CALIBRATION_SEED
        if (args.stage != 1 or args.layout_mode != 'progressive'
                or args.payload_mass != 1. or args.pickup_checkpoint):
            raise ValueError('The arm-mask study starts in progressive stage 1 with a 1 kg payload')
        if args.eval_episodes % 2:
            raise ValueError('Use an even evaluation count for balanced left and right scenarios')
        if args.seed + args.total_steps >= CALIBRATION_SEED:
            raise ValueError('Training reset seeds must remain below the calibration range')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable')


def save_bundle(run, model, normalizer, config, reset_seed, name,include_buffer=True):
    temporary = run / (name+'.partial')
    if temporary.exists():shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    model.save(temporary/'model.zip')
    normalizer.save(temporary/'normalization.pkl')
    if config['algorithm']=='sac' and include_buffer:
        model.save_replay_buffer(temporary/'replay_buffer.pkl')
    files={name:hashlib.sha256((temporary/name).read_bytes()).hexdigest() for name in ('model.zip','normalization.pkl')}
    atomic_json(temporary/'metadata.json', dict(format='stable_baselines3',steps=model.num_timesteps,
                next_reset_seed=reset_seed,config=config.copy(),env_config=config['env_config'].copy(),files=files))
    target = run/name
    backup = run/(name+'.previous')
    if backup.exists():shutil.rmtree(backup)
    if target.exists():os.replace(target, backup)
    os.replace(temporary, target)
    if backup.exists():shutil.rmtree(backup)


def freeze_policy(run, target):
    """Keep the best policy available after its stage or difficulty changes."""
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix('.partial')
        if temporary.exists():
            shutil.rmtree(temporary)
        shutil.copytree(run/'best', temporary)
        os.replace(temporary, target)
    return target


def freeze_stage_policy(run, stage):
    return freeze_policy(run, run/'stage_checkpoints'/f'stage-{stage}')


def apply_curriculum_config(config, controller, pickup_checkpoint):
    stage, level = controller.stage, controller.level
    env_config = config['env_config']
    config.update(stage=stage, level=level, level_controller=controller.state_dict())
    env_config.update(stage=stage, difficulty=level/4., effort_scale=level/4. if stage == 4 else 0.,
                      pickup_checkpoint=str(pickup_checkpoint) if stage == 2 and pickup_checkpoint else None)
    if 'arm_mask' in env_config and config['auto_stages']:
        env_config['arm_mask'] = 'both' if stage >= 3 or level == 4 else 'mixed' if level == 3 else 'single'


def next_validation(steps, interval, stage_start):
    return stage_start + ((steps-stage_start)//interval+1)*interval


def main():
    from stable_baselines3 import A2C, PPO, SAC
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    p = parser();args = p.parse_args()
    try:validate(args)
    except ValueError as error:p.error(str(error))
    torch.set_num_threads(1)
    run = Path(args.run_dir).resolve()
    if run.exists() and any(run.iterdir()) and not args.resume:
        p.error('Run directory is not empty; use a new directory or --resume')
    run.mkdir(parents=True,exist_ok=True)
    env_config = dict(stage=args.stage,difficulty=args.level/4.,effort_scale=args.level/4. if args.stage==4 else 0.,
                      task_version=2,pickup_checkpoint=args.pickup_checkpoint,layout_mode=args.layout_mode,payload_mass=args.payload_mass)
    if args.layout_mode=='progressive':env_config['reward_scheme']=args.reward_scheme
    config = dict(vars(args),run_dir=str(run),resume=str(run/'latest') if args.resume else None,curriculum=True,num_envs=1,physics_backend='cpu',
                  task_version=2,reward_version=STUDY_REWARD_VERSIONS[args.reward_scheme] if args.layout_mode=='progressive' else 'existing_v2',
                  policy_version='bounded_v2',normalization_version='bounded_task_state_v1',
                  observation_version='reward_state_v4',action_mode='continuous_14',
                  framework='sb3',web_run_name=args.web_run_name or args.study_id+'_'+args.algorithm,env_config=env_config)
    fixed = None
    if args.fixed_study_normalization:
        from training.fixed_study_normalization import FixedObservation, VERSION, load_fixed_stats
        calibration = Path(args.fixed_study_normalization).resolve()
        fixed = load_fixed_stats(calibration)
        env_config['arm_mask'] = 'single'
        config.update(normalization_version=VERSION, observation_version='reward_state_v4_arm_mask_v1',
                      observation_dim=107, fixed_stats_sha256=hashlib.sha256(calibration.read_bytes()).hexdigest())
    previous = None
    if args.resume:
        if not (run/'latest').exists() and (run/'latest.previous').exists():
            os.replace(run/'latest.previous',run/'latest')
        metadata=run/'latest/metadata.json'
        if not metadata.exists():p.error('No completed study checkpoint found in latest/')
        previous=json.loads(metadata.read_text())
        keys=('algorithm','layout_mode','payload_mass','seed','device','rollout_steps','buffer_size','learning_starts','pickup_checkpoint','reward_version','policy_version','normalization_version','observation_version','max_steps')
        if fixed is not None:
            keys += ('fixed_stats_sha256', 'eval_episodes')
        if any(config[k]!=previous['config'].get(k) for k in keys):
            p.error('Resume requires the same model, environment and optimizer configuration')
        original_stage = previous['config'].get('initial_stage', previous['config']['stage'])
        if args.stage != original_stage:
            p.error('Resume requires the original starting stage')
        if previous['steps']>=args.total_steps:p.error('Increase --total-steps to resume')
    controller=Curriculum(**previous['config']['level_controller']) if previous and previous['config'].get('level_controller') else Curriculum(stage=args.stage,level=args.level,task_version=2)
    config['initial_stage'] = args.stage
    try:
        config['stage_start_steps'] = restore_stage_start(run, previous, args.stage, controller.stage)
    except ValueError as error:
        p.error(str(error))
    pickup_checkpoint = previous['env_config'].get('pickup_checkpoint') if previous else args.pickup_checkpoint
    if previous and previous['config'].get('auto_levels') is False and args.auto_levels:
        controller.passes = 0
    if previous and not previous['config'].get('auto_stages') and args.auto_stages and controller.stage < 4:
        controller.stage_complete = False
    apply_curriculum_config(config, controller, pickup_checkpoint)
    atomic_json(run/'config.json',config)
    raw=EpisodeSeeds(make_env(env_config,max_steps=args.max_steps), previous['next_reset_seed'] if previous else args.seed)
    vec=DummyVecEnv([lambda:FixedObservation(raw, fixed) if fixed is not None else raw])
    env=None;journal=None;model=None;monitor=None;live=LiveTrainingFeed(None,None,run);stopping=False;failed=True;best=(-1,)
    old_handlers={}
    def stop(signum, frame):
        nonlocal stopping
        stopping=True
    try:
        from training.study_normalization import ContextNormalize
        if fixed is not None:
            env, normalizer = vec, fixed
        else:
            env=VecNormalize.load(run/'latest/normalization.pkl',vec) if previous else ContextNormalize(vec,norm_obs=True,norm_reward=False,clip_obs=10.)
            env.norm_reward=False
            normalizer = env
        classes=dict(ppo=PPO,a2c=A2C,sac=SAC)
        if previous:
            model=classes[args.algorithm].load(run/'latest/model.zip',env=env,device=args.device)
            if args.algorithm=='sac':model.load_replay_buffer(run/'latest/replay_buffer.pkl')
        else:model=build_model(args.algorithm,env,args)
        # SB3 seeds VecEnv during model construction/loading. Preserve the
        # journal's next episode seed rather than restarting at the base seed.
        env.seed(raw.next_seed)
        journal=TrainingHistory(run,config,model.num_timesteps)
        if args.web:
            from web.live_access import read_credentials,ensure_history_watcher
            credentials=read_credentials(args.connection_dir)
            live=LiveTrainingFeed(credentials['origin'],credentials['site_token'],run,args.stream_port,credentials['stream_token'])
            ensure_history_watcher(run,args.connection_dir,os.getpid())
            with (run/'mission_replays.log').open('ab',buffering=0) as log:
                subprocess.Popen([sys.executable,'-B','-m','web.algorithm_replays','--run-dir',str(run),
                    '--connection-dir',str(Path(args.connection_dir).resolve()),'--training-pid',str(os.getpid())],
                    cwd=Path(__file__).resolve().parents[1],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True,
                    env=dict(os.environ,MUJOCO_GL='egl',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'))
        live.status(dict(run_name=config['web_run_name'],algorithm=args.algorithm,study_id=args.study_id,phase='training'))
        for sig in (signal.SIGINT,signal.SIGTERM):old_handlers[sig]=signal.signal(sig,stop)
        started=time.monotonic();start_steps=model.num_timesteps
        monitor=StudyMonitor(journal,raw,live,config,lambda:stopping)
        interval=max(args.rollout_steps,(args.eval_every//args.rollout_steps)*args.rollout_steps)
        save_interval=max(args.rollout_steps,(args.save_every//args.rollout_steps)*args.rollout_steps)
        validation_start = config['stage_start_steps'] if args.stage_max_steps is not None else 0
        next_eval=next_validation(model.num_timesteps,interval,validation_start)
        first_eval=validation_start+args.rollout_steps*8
        if model.num_timesteps<first_eval:
            next_eval=min(next_eval,first_eval)
        best_file=run/'best_validation.json'
        if best_file.exists():
            old=json.loads(best_file.read_text())
            old_stage=old['env_config']['stage']
            best=(old_stage, round(old['env_config']['difficulty']*4), *curriculum_ranking(old,old_stage,2))
        if previous is None:save_bundle(run,model,normalizer,config,raw.next_seed,'latest')
        while not stopping and not (args.auto_stages and controller.completed):
            deadline = stage_deadline(config)
            if model.num_timesteps >= deadline:
                break
            chunk=min(save_interval,next_eval-model.num_timesteps,deadline-model.num_timesteps)
            milestone=(model.num_timesteps//args.record_every+1)*args.record_every if args.record_every else None
            if milestone:
                boundary=math.ceil(milestone/args.rollout_steps)*args.rollout_steps
                chunk=min(chunk,boundary-model.num_timesteps)
            if args.stage_max_steps is None:
                model.learn(total_timesteps=chunk,reset_num_timesteps=False,callback=monitor)
            else:
                learn_exact(model, chunk, monitor, lambda:stopping)
            monitor.log(force=True)
            should_evaluate=model.num_timesteps>=next_eval or model.num_timesteps>=deadline
            phase='stopped' if stopping else 'evaluating' if should_evaluate else 'training'
            journal.status(phase,model.num_timesteps)
            live.status(dict(phase=phase,steps=model.num_timesteps,updates=model._n_updates,
                             **stage_progress(config, model.num_timesteps)))
            if not stopping and should_evaluate:
                stage, level = controller.stage, controller.level
                next_eval=next_validation(model.num_timesteps,interval,validation_start)
                result=evaluate_policy(EvaluationPolicy(model,args.algorithm),normalizer.normalize_obs,args.eval_episodes,
                    1_000_000_000+stage*1_000_000+level*10_000,args.max_steps,.999,
                    stop_requested=lambda:stopping,progress_label=None,env_config=env_config)
                if result is not None:
                    progress = stage_progress(config, model.num_timesteps)
                    result.update(algorithm=args.algorithm,steps=model.num_timesteps, **progress)
                    atomic_json(run/f'validation_{model.num_timesteps}.json',result)
                    rank=(stage,level,*curriculum_ranking(result,stage,2))
                    if rank>best:
                        best=rank
                        save_bundle(run,model,normalizer,config,raw.next_seed,'best',include_buffer=False)
                        atomic_json(best_file,result)
                    elapsed=time.monotonic()-started
                    atomic_json(run/'comparison_status.json',dict(algorithm=args.algorithm,steps=model.num_timesteps,
                        success_rate=result['success_rate'],grasp_rate=result['grasp_rate'],lift_rate=result['lift_rate'],
                        mean_return=result['mean_return'],elapsed_seconds=elapsed,fps=(model.num_timesteps-start_steps)/max(elapsed,.001),
                        action_mode=config['action_mode'],validation_level=level,stage=stage,
                        updated_at=datetime.now(timezone.utc).isoformat(), **progress))
                    live.status(dict(phase='training',validation_level=level,validation_success_rate=result['success_rate'],validation_grasp_rate=result['grasp_rate'],validation_lift_rate=result['lift_rate']))
                    print(f'{args.algorithm.upper()}: {model.num_timesteps:,}/{args.total_steps:,} success {result["success_rate"]:.1%}',flush=True)
                    candidate = Curriculum(**controller.state_dict())
                    if args.auto_levels and candidate.observe(
                            result['success_rate'],model.num_timesteps,
                            locked_stage=None if args.auto_stages else stage,min_steps=args.min_level_steps):
                        # Save the validated state first so an interrupted reset can be resumed.
                        save_bundle(run,model,normalizer,config,raw.next_seed,'latest')
                        freeze_policy(run, run/'level_checkpoints'/f'stage-{stage}'/f'level-{level}')
                        if candidate.stage != stage:
                            frozen_stage = freeze_stage_policy(run, stage)
                            config['stage_start_steps'] = model.num_timesteps
                            if args.stage_max_steps is not None:
                                validation_start = model.num_timesteps
                                next_eval = validation_start + min(interval, args.rollout_steps*8)
                            if stage == 1:
                                pickup_checkpoint = frozen_stage
                        apply_curriculum_config(config, candidate, pickup_checkpoint)
                        raw.unwrapped.configure(env_config)
                        model._last_obs = None
                        live.status(dict(stage=candidate.stage,level=candidate.level,validation_level=None,
                                         validation_success_rate=None,validation_grasp_rate=None,validation_lift_rate=None,
                                         **stage_progress(config, model.num_timesteps)))
                        print(f'{args.algorithm.upper()}: stage {stage} level {level} → stage {candidate.stage} level {candidate.level}',flush=True)
                    controller = candidate
                    if args.auto_stages and controller.completed:
                        freeze_stage_policy(run, stage)
                    config['level_controller']=controller.state_dict();atomic_json(run/'config.json',config)
            save_bundle(run,model,normalizer,config,raw.next_seed,'latest')
            if milestone and model.num_timesteps>=milestone:
                folder=run/'recordings'/f'step_{milestone:09d}';folder.mkdir(parents=True,exist_ok=True)
                save_bundle(folder,model,normalizer,config,raw.next_seed,'checkpoint',include_buffer=False)
                with (folder/'recording.log').open('ab',buffering=0) as log:
                    subprocess.Popen([sys.executable,'-B','-m','evaluation.export_algorithm_replay','--bundle',str(folder/'checkpoint'),
                        '--output',str(folder/'videos'),'--record-only','--episodes',str(args.record_episodes)],cwd=Path(__file__).resolve().parents[1],
                        stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True,
                        env=dict(os.environ,MUJOCO_GL='egl',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'))
        if (args.stage_max_steps is not None and not stopping and not controller.completed
                and model.num_timesteps >= config['stage_start_steps']+args.stage_max_steps):
            print(f'{args.algorithm.upper()}: stage {controller.stage} reached its {args.stage_max_steps:,}-step limit',flush=True)
        if fixed is not None and not args.auto_levels and not stopping:
            journal.status('evaluating',model.num_timesteps)
            live.status(dict(phase='evaluating',steps=model.num_timesteps,updates=model._n_updates))
            # Pair both-enabled evaluation with the final masked evaluation.
            unmasked = evaluate_policy(
                EvaluationPolicy(model,args.algorithm), normalizer.normalize_obs, args.eval_episodes,
                1_000_000_000+controller.stage*1_000_000+controller.level*10_000,
                args.max_steps, .999, stop_requested=lambda:stopping,
                progress_label=None, env_config=dict(env_config, arm_mask='both'))
            if unmasked is not None:
                unmasked.update(algorithm=args.algorithm, steps=model.num_timesteps)
                atomic_json(run/'final_unmasked_validation.json', unmasked)
        failed=False
    finally:
        for sig,handler in old_handlers.items():signal.signal(sig,handler)
        try:
            if monitor:monitor.close()
            live.status(dict(phase='error' if failed else 'stopped' if stopping else 'finished',steps=model.num_timesteps if model else 0));live.close()
            if journal:journal.finish('error' if failed else 'stopped' if stopping else 'finished',model.num_timesteps)
        finally:
            if env:env.close()
            else:vec.close()
    return 0


if __name__=='__main__':
    raise SystemExit(main())
