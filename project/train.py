"""Train one centralized PyTorch PPO policy for both physical robot arms."""
import argparse
import csv
import json
import os
from pathlib import Path
import random
import signal
import time
import numpy as np
import torch
from torch import nn
from curriculum import Curriculum, make_env, STAGE_NAMES
from evaluate import evaluate_policy, ranking, VALIDATION_SEED, TEST_SEED
from ppo import ActorCritic, RunningNorm, compute_gae, load_checkpoint, resolve_device
from rollout_workers import RolloutWorkers
from training_viewer import TrainingViewer
from live_training import LiveTrainingFeed
from training_recordings import RecordingJobs,next_milestone,rollout_length
from training_schedule import StageHandoff


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name, default in [('total-steps',None),('num-envs',8),('rollout-steps',256),
                          ('batch-size',256),('epochs',10),('max-steps',2000),('seed',31),
                          ('eval-every',100_000),('eval-episodes',50),('final-eval-episodes',200)]:
        p.add_argument('--'+name, type=int, default=default)
    for name, default in [('lr',3e-4),('gamma',.999),('gae-lambda',.95),('clip',.2),
                          ('entropy-coef',.005),('value-coef',.5),('max-grad-norm',.5),('target-kl',.02)]:
        p.add_argument('--'+name, type=float, default=default)
    p.add_argument('--run-dir',default=None)
    p.add_argument('--resume')
    p.add_argument('--next-stage',type=int,choices=(2,3,4),help='At the current budget boundary, continue the same model in this stage')
    p.add_argument('--next-stage-steps',type=int,help='Additional environment steps for the next fixed stage')
    p.add_argument('--stage',type=int,choices=(1,2,3,4),help='Train only this stage; keep improving it after its levels pass')
    p.add_argument('--record-every',type=int,default=None,help='Save a checkpoint and smooth local evaluation video every N global steps')
    p.add_argument('--record-episodes',type=int,default=5)
    p.add_argument('--record-fps',type=int,choices=(10,25,50),default=50)
    p.add_argument('--record-initial',action='store_true',help='Also record the checkpoint at startup/resume')
    p.add_argument('--live-stream-port',type=int,default=0,help='Authenticated localhost source for continuous live video')
    p.add_argument('--live-url',help='Owner-private Site origin for live training uploads')
    p.add_argument('--curriculum',action='store_true',help='Learn stages 1→2→3→4 with one policy')
    p.add_argument('--gate-success',type=float,default=.8)
    p.add_argument('--gate-passes',type=int,default=2)
    p.add_argument('--min-level-steps',type=int,default=100_000)
    p.add_argument('--viewer',action='store_true')
    p.add_argument('--device',choices=('auto','cpu','cuda'),default='auto')
    return p


def save_checkpoint(path, policy, optimizer, norm, workers, config, steps, updates, best_rank, next_eval, curriculum=None):
    checkpoint = dict(version=2 if curriculum else 1,
        curriculum=curriculum.state_dict() if curriculum else None,
        env_config=curriculum.env_config() if curriculum else None,
        control_mode='incremental_targets' if curriculum else 'absolute_targets', obs_dim=workers.obs.shape[1], policy=policy.state_dict(),
        optimizer=optimizer.state_dict(), normalizer=norm.state_dict(), config=config,
        steps=steps, updates=updates, best_rank=best_rank, next_eval=next_eval,
        worker_states=workers.states(), rng=dict(python=random.getstate(), numpy=np.random.get_state(),
                                               torch=torch.get_rng_state(),
                                               cuda=torch.cuda.get_rng_state_all() if next(policy.parameters()).is_cuda else None))
    temporary = path.with_suffix('.tmp')
    torch.save(checkpoint,temporary)
    os.replace(temporary,path)


def main():
    p = parser()
    args = p.parse_args()
    requested_total_steps=args.total_steps
    if args.total_steps is None:args.total_steps=100_000_000
    requested_handoff=args.next_stage is not None
    requested_handoff_steps=args.next_stage_steps
    if args.stage is not None:
        args.curriculum = True
    if args.run_dir is None:
        args.run_dir = str(Path(args.resume).parent) if args.resume else ('runs/curriculum' if args.curriculum else 'runs/ppo')
    config = vars(args).copy()
    checkpoint = load_checkpoint(args.resume) if args.resume else None
    if checkpoint is not None:
        if args.curriculum and not checkpoint.get('curriculum'):
            p.error('Start a fresh curriculum run; legacy absolute-control checkpoints use a different observation/action schema')
        # Architecture, rollout and task settings stay identical for reproducible resume.
        runtime = {name:config[name] for name in ('run_dir','resume','viewer','device','live_url','record_episodes','record_fps','record_initial','live_stream_port')}
        if requested_total_steps is not None:
            runtime['total_steps']=requested_total_steps
        if args.stage is not None:
            runtime['stage'] = args.stage
        if args.record_every is not None:
            runtime['record_every'] = args.record_every
        if requested_handoff:
            runtime.update(next_stage=args.next_stage,next_stage_steps=requested_handoff_steps)
        config.update(checkpoint['config'])
        config.update(runtime)
        args = argparse.Namespace(**config)
    if requested_handoff:
        if not args.curriculum or args.stage is None or not requested_handoff_steps:
            p.error('--next-stage requires a fixed --stage and positive --next-stage-steps')
        try:
            handoff=StageHandoff(args.stage,args.total_steps,args.next_stage,requested_handoff_steps)
        except ValueError as error:
            p.error(str(error))
        if handoff.at_steps%args.num_envs or handoff.additional_steps%args.num_envs:
            p.error('Stage budgets must be divisible by --num-envs for exact handoff boundaries')
    else:
        handoff=StageHandoff(**config['handoff']) if config.get('handoff') else None
    config['handoff']=handoff.state_dict() if handoff else None
    if args.record_every is None:
        args.record_every=0
    config['record_every']=args.record_every
    for name in ('total_steps','num_envs','rollout_steps','batch_size','epochs','max_steps','eval_every','eval_episodes'):
        if getattr(args,name) < 1:
            p.error(f'{name} must be positive')
    if args.record_every<0 or args.record_episodes<1:
        p.error('Invalid recording interval or episode count')
    if args.final_eval_episodes < 0 or args.num_envs*args.rollout_steps < 2:
        p.error('Need at least two rollout samples and nonnegative final evaluation episodes')
    if not 0 <= args.seed < VALIDATION_SEED-args.num_envs:
        p.error('Training seeds must be below the held-out evaluation range')
    if not 0 <= args.gate_success <= 1 or args.gate_passes < 1 or args.min_level_steps < 0:
        p.error('Invalid curriculum promotion criteria')
    if not 0 < args.gamma <= 1:
        p.error('gamma must be in (0, 1]')
    try:
        device = resolve_device(args.device)
    except RuntimeError as error:
        p.error(str(error))
    print(f'PPO device: {device}' + (f' ({torch.cuda.get_device_name(device)})' if device.type == 'cuda' else ''), flush=True)
    run_dir = Path(args.run_dir)
    if checkpoint is None and (run_dir/'latest.pt').exists():
        p.error('Run directory contains a checkpoint; use --resume or a new --run-dir')
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    curriculum = Curriculum(**checkpoint['curriculum']) if checkpoint and checkpoint.get('curriculum') else (Curriculum(stage=args.stage or 1) if args.curriculum else None)
    stage_switch = bool(curriculum and args.stage is not None and curriculum.stage != args.stage)
    if stage_switch:
        curriculum = Curriculum(stage=args.stage,level_start_steps=checkpoint['steps'])
        config['stage_start_steps']=checkpoint['steps']
    config.setdefault('stage_start_steps',0)
    if handoff and not handoff.done and curriculum and curriculum.stage!=handoff.from_stage:
        p.error('Pending handoff does not match the resumed stage')
    (run_dir/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    env_config = curriculum.env_config() if curriculum else None
    probe = make_env(env_config)
    obs_dim = probe.observation_space.shape[0]
    initial_action = probe.target_action.copy()
    probe.close()
    policy = ActorCritic(obs_dim)
    with torch.no_grad():
        if curriculum:
            policy.log_std.fill_(-.8)
        else:
            policy.actor[-1].bias.copy_(torch.from_numpy(np.arctanh(np.clip(initial_action,-.99,.99))).float())
    policy.to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5)
    norm = RunningNorm((obs_dim,),passthrough=6 if curriculum else 0)
    steps = updates = 0
    best_rank = (-1., 0., 0., -float('inf'))
    next_eval = args.eval_every
    if checkpoint is not None:
        policy.load_state_dict(checkpoint['policy'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        norm.load_state_dict(checkpoint['normalizer'])
        steps, updates = checkpoint['steps'], checkpoint['updates']
        best_rank, next_eval = checkpoint['best_rank'], checkpoint['next_eval']
        if not (run_dir/'best.pt').exists():
            best_rank = (-1., 0., 0., -float('inf'))
    if steps >= args.total_steps and not (handoff and curriculum and handoff.due(curriculum.stage,steps)):
        print(f'Checkpoint already has {steps} steps. Increase --total-steps to continue.')
        return
    live_token = os.environ.pop('REACHER_LIVE_TOKEN',None)
    if args.live_url and not live_token:
        p.error('REACHER_LIVE_TOKEN is required for private live uploads')
    stream_token=os.environ.pop('REACHER_STREAM_TOKEN',None)
    if args.live_stream_port and not stream_token:
        p.error('REACHER_STREAM_TOKEN is required for continuous video')
    workers = RolloutWorkers(args.num_envs,args.seed,args.max_steps,args.gamma,env_config)
    viewer = TrainingViewer(args.viewer)
    live = LiveTrainingFeed(args.live_url,live_token,run_dir,args.live_stream_port,stream_token)
    recordings=RecordingJobs()
    next_record=next_milestone(steps,args.record_every)
    stopping = False
    def request_stop(*_):
        nonlocal stopping
        stopping = True
        print('Stopping after this update; saving latest.pt...',flush=True)
    old_handler = signal.signal(signal.SIGINT,request_stop)
    old_term_handler = signal.signal(signal.SIGTERM,request_stop)
    def persist_config():
        temporary=run_dir/'config.tmp'
        temporary.write_text(json.dumps(config,indent=2)+'\n')
        os.replace(temporary,run_dir/'config.json')

    def perform_handoff():
        nonlocal curriculum,best_rank,next_eval
        if not handoff or not curriculum or not handoff.due(curriculum.stage,steps):return False
        if args.final_eval_episodes and not stopping:
            live.status(dict(phase='evaluating',steps=steps,evaluation_episodes=args.final_eval_episodes))
            evaluation=evaluate_policy(policy,norm,args.final_eval_episodes,TEST_SEED,args.max_steps,args.gamma,
                stop_requested=lambda:stopping,progress_label=f'Stage {curriculum.stage} final evaluation',
                env_config=dict(stage=curriculum.stage,difficulty=1.,effort_scale=1. if curriculum.stage==4 else 0.))
            if evaluation is not None:
                (run_dir/f'stage{curriculum.stage}_final_evaluation.json').write_text(json.dumps(evaluation,indent=2)+'\n')
        if stopping:return False
        previous_stage=curriculum.stage
        save_checkpoint(run_dir/f'stage{previous_stage}_final.pt',policy,optimizer,norm,workers,config,steps,updates,best_rank,next_eval,curriculum)
        curriculum=Curriculum(stage=handoff.to_stage,level_start_steps=steps)
        args.stage=handoff.to_stage
        args.total_steps=handoff.final_steps
        args.next_stage=None;args.next_stage_steps=None
        handoff.done=True
        config.update(stage=args.stage,total_steps=args.total_steps,next_stage=None,next_stage_steps=None,
                      handoff=handoff.state_dict(),stage_start_steps=steps)
        workers.configure(curriculum.env_config())
        best_rank=(-1.,0.,0.,-float('inf'));next_eval=steps+args.eval_every
        persist_config()
        save_checkpoint(run_dir/f'stage{args.stage}_initial.pt',policy,optimizer,norm,workers,config,steps,updates,best_rank,next_eval,curriculum)
        save_checkpoint(run_dir/'best.pt',policy,optimizer,norm,workers,config,steps,updates,best_rank,next_eval,curriculum)
        save_checkpoint(run_dir/'latest.pt',policy,optimizer,norm,workers,config,steps,updates,best_rank,next_eval,curriculum)
        with (run_dir/'stage_handoffs.jsonl').open('a') as history:
            history.write(json.dumps(dict(steps=steps,from_stage=previous_stage,to_stage=args.stage,
                final_steps=args.total_steps,preserved=['policy','optimizer','normalizer','rng']))+'\n')
        live.status(dict(phase='training',steps=steps,total_steps=args.total_steps,stage=args.stage,level=0,
            stage_start_steps=steps,stage_steps=0,stage_total_steps=handoff.additional_steps,
            validation_success_rate=None,validation_grasp_rate=None,validation_lift_rate=None))
        print(f'Stage handoff {previous_stage} → {args.stage}: same model and optimizer, target {args.total_steps:,} global steps',flush=True)
        return True

    failed=True
    try:
        if checkpoint is not None:
            workers.restore(checkpoint['worker_states'])
            if stage_switch:
                workers.configure(curriculum.env_config())
                best_rank = (-1.,0.,0.,-float('inf'))
            random.setstate(checkpoint['rng']['python'])
            np.random.set_state(checkpoint['rng']['numpy'])
            torch.set_rng_state(checkpoint['rng']['torch'])
            if device.type == 'cuda' and checkpoint['rng'].get('cuda') is not None:
                torch.cuda.set_rng_state_all(checkpoint['rng']['cuda'])
        else:
            norm.update(workers.obs)
        start = time.monotonic()
        start_steps = steps
        log_path = run_dir/'metrics.csv'
        fields = ['steps','updates','fps','policy_loss','value_loss','entropy','kl','episodes',
                  'training_success_rate','mean_return','mean_effort','mean_used_arms',
                  'stage','level','difficulty','effort_scale','grasp_rate','lift_rate']
        if log_path.exists() and log_path.stat().st_size:
            with log_path.open(newline='') as previous_log:
                if next(csv.reader(previous_log)) != fields:
                    log_path = run_dir/'metrics_v2.csv'
        if checkpoint is None:
            save_checkpoint(run_dir/'latest.pt',policy,optimizer,norm,workers,config,steps,updates,best_rank,next_eval,curriculum)
        if args.record_initial:
            initial_folder=run_dir/'recordings'/f'initial_{steps:09d}'
            initial_folder.mkdir(parents=True,exist_ok=True)
            save_checkpoint(initial_folder/'checkpoint.pt',policy,optimizer,norm,workers,config,steps,updates,best_rank,next_eval,curriculum)
            recordings.start(initial_folder/'checkpoint.pt',initial_folder,args.record_episodes,args.record_fps)
        if handoff and curriculum and handoff.due(curriculum.stage,steps):
            perform_handoff()
        live.status(dict(phase='training',steps=steps,total_steps=args.total_steps,stage=curriculum.stage if curriculum else 0,level=curriculum.level if curriculum else 0,run_name=run_dir.name,device=str(device),stage_start_steps=config['stage_start_steps'],stage_steps=steps-config['stage_start_steps'],stage_total_steps=args.total_steps-config['stage_start_steps']))
        with log_path.open('a',newline='') as logfile:
            log = csv.DictWriter(logfile,fieldnames=fields)
            if logfile.tell() == 0:
                log.writeheader()
            while steps < args.total_steps and not stopping:
                # Freeze observation statistics throughout each rollout and its PPO update.
                length = rollout_length(steps,args.total_steps,args.num_envs,args.rollout_steps,next_record)
                buffer = {name:[] for name in ('obs','latent','logp','value','reward','next_value','term','trunc')}
                raw_observations, episodes = [], []
                for _ in range(length):
                    observations = norm(workers.obs)
                    raw_observations.append(workers.obs.copy())
                    with torch.no_grad():
                        action, latent, logp, value = policy.sample(torch.as_tensor(observations,device=device))
                    results = workers.step(action.cpu().numpy(),snapshot=viewer.ready() or live.ready())
                    final_obs = np.stack([r[1] for r in results])
                    with torch.no_grad():
                        next_value = policy.critic(torch.as_tensor(norm(final_obs),device=device)).squeeze(-1).cpu().numpy()
                    for name, item in dict(obs=observations,latent=latent.cpu().numpy(),logp=logp.cpu().numpy(),
                        value=value.cpu().numpy(),reward=np.array([r[2] for r in results],dtype=np.float32),
                        next_value=next_value,term=np.array([r[3] for r in results]),
                        trunc=np.array([r[4] for r in results])).items():
                        buffer[name].append(item)
                    steps += args.num_envs
                    for r in results:
                        if r[3] or r[4]:
                            episodes.append(r[5])
                    if results[0][6] is not None:
                        info = results[0][5]
                        live.send(results[0][6],dict(phase='training',steps=steps,total_steps=args.total_steps,stage=curriculum.stage if curriculum else 0,level=curriculum.level if curriculum else 0,run_name=run_dir.name,device=str(device),stage_start_steps=config['stage_start_steps'],stage_steps=steps-config['stage_start_steps'],stage_total_steps=args.total_steps-config['stage_start_steps'],episode_step=info.get('episode_step',results[0][6]['task']['steps']),reward=results[0][2],episode_return=info['episode_return'],grasped=list(info['grasped']),lifted=info['episode_lifted'],success=info['is_success'],max_torque=float(np.abs(info['motor_torque']).max()),max_gripper_force=float(np.abs(info['gripper_force']).max()),goal_distance=info['goal_distance'],payload_mass=info['payload_mass']))
                        viewer.send(results[0][6], f'Stage {curriculum.stage if curriculum else 0} · Steps {steps:,} / {args.total_steps:,}\nWorker 0 reward {results[0][2]:.3f}  success {info["is_success"]}\nUsed arms {info["used_arms"].astype(int)}  carried {info["carried"]}\nEffort {info["episode_effort"]:.3f}\nTorques Nm {np.round(info["motor_torque"],1)}\nGripper N {np.round(info["gripper_force"],1)}')
                data = {k:np.stack(v) for k,v in buffer.items()}
                advantage, returns = compute_gae(data['reward'],data['value'],data['next_value'],
                                                 data['term'],data['trunc'],args.gamma,args.gae_lambda)
                advantage = (advantage-advantage.mean())/(advantage.std()+1e-8)
                tensors = {k:torch.from_numpy(data[k].reshape((-1,)+data[k].shape[2:])) for k in ('obs','latent','logp')}
                tensors.update(advantage=torch.from_numpy(advantage.ravel()), returns=torch.from_numpy(returns.ravel()))
                tensors = {name:value.to(device) for name,value in tensors.items()}
                count = len(tensors['obs'])
                losses = []
                for epoch in range(args.epochs):
                    indices = np.random.permutation(count)
                    stop_epoch = False
                    for offset in range(0,count,args.batch_size):
                        idx = torch.as_tensor(indices[offset:offset+args.batch_size],device=device)
                        new_logp, entropy, value = policy.evaluate(tensors['obs'][idx],tensors['latent'][idx])
                        logratio = new_logp-tensors['logp'][idx]
                        ratio = logratio.exp()
                        adv = tensors['advantage'][idx]
                        policy_loss = -torch.min(ratio*adv,ratio.clamp(1-args.clip,1+args.clip)*adv).mean()
                        value_loss = .5*(value-tensors['returns'][idx]).square().mean()
                        kl = ((ratio-1)-logratio).mean()
                        if kl.item() > args.target_kl:
                            stop_epoch = True
                            break
                        loss = policy_loss+args.value_coef*value_loss-args.entropy_coef*entropy.mean()
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        nn.utils.clip_grad_norm_(policy.parameters(),args.max_grad_norm)
                        optimizer.step()
                        losses.append([policy_loss.item(),value_loss.item(),entropy.mean().item(),kl.item()])
                    if stop_epoch:
                        break
                norm.update(np.concatenate(raw_observations))
                updates += 1
                averages = np.mean(losses,axis=0) if losses else [0.,0.,0.,float(kl)]
                row = dict(zip(['policy_loss','value_loss','entropy','kl'],averages))
                row.update(steps=steps,updates=updates,fps=int((steps-start_steps)/(time.monotonic()-start)),
                    episodes=len(episodes),training_success_rate=np.mean([e['is_success'] for e in episodes]) if episodes else '',
                    mean_return=np.mean([e['episode_return'] for e in episodes]) if episodes else '',
                    mean_effort=np.mean([e['episode_effort'] for e in episodes]) if episodes else '',
                    mean_used_arms=np.mean([e['used_arms'].sum() for e in episodes]) if episodes else '',
                    stage=curriculum.stage if curriculum else 0,level=curriculum.level if curriculum else 0,
                    difficulty=curriculum.env_config()['difficulty'] if curriculum else 1.,
                    effort_scale=curriculum.env_config()['effort_scale'] if curriculum else 1.,
                    grasp_rate=np.mean([e['used_arms'].any() for e in episodes]) if episodes else '',
                    lift_rate=np.mean([e['episode_lifted'] for e in episodes]) if episodes else '')
                log.writerow(row)
                logfile.flush()
                live.status(dict(phase='training',steps=steps,updates=updates,fps=row['fps'],stage=row['stage'],level=row['level'],training_success_rate=row['training_success_rate'],grasp_rate=row['grasp_rate'],lift_rate=row['lift_rate']))
                print((f'Stage {curriculum.stage} {STAGE_NAMES[curriculum.stage]} level {curriculum.level}/4 | ' if curriculum else '') + f'Steps {steps:,}/{args.total_steps:,}  updates {updates}  FPS {row["fps"]}  episodes {len(episodes)}  success {row["training_success_rate"]}  grasp {row["grasp_rate"]}  lift {row["lift_rate"]}',flush=True)
                if next_record is not None and steps>=next_record:
                    folder=run_dir/'recordings'/f'step_{next_record:09d}'
                    folder.mkdir(parents=True,exist_ok=True)
                    save_checkpoint(folder/'checkpoint.pt',policy,optimizer,norm,workers,config,steps,updates,best_rank,next_eval,curriculum)
                    recordings.start(folder/'checkpoint.pt',folder,args.record_episodes,args.record_fps,next_record)
                    next_record=next_milestone(steps,args.record_every)
                if not stopping and (steps >= next_eval or steps >= args.total_steps):
                    env_config = curriculum.env_config() if curriculum else None
                    validation_seed = VALIDATION_SEED+(curriculum.stage*1_000_000+curriculum.level*10_000 if curriculum else 0)
                    live.status(dict(phase='evaluating',steps=steps,evaluation_episodes=args.eval_episodes))
                    result = evaluate_policy(policy,norm,args.eval_episodes,validation_seed,args.max_steps,args.gamma,
                                             stop_requested=lambda: stopping, progress_label='Validation',env_config=env_config)
                    if result is not None:
                        while next_eval <= steps:
                            next_eval += args.eval_every
                        (run_dir/f'validation_{steps}.json').write_text(json.dumps(result,indent=2)+'\n')
                        print(f'Validation success: {result["success_rate"]:.1%}  grasp {result["grasp_rate"]:.1%}  lift {result["lift_rate"]:.1%}',flush=True)
                        live.status(dict(phase='training',steps=steps,validation_success_rate=result['success_rate'],validation_grasp_rate=result['grasp_rate'],validation_lift_rate=result['lift_rate']))
                        candidate_rank = ranking(result) if curriculum is None or curriculum.stage == 4 else (result['success_rate'],0.,0.,result['mean_return'])
                        if candidate_rank > tuple(best_rank):
                            best_rank = candidate_rank
                            save_checkpoint(run_dir/'best.pt',policy,optimizer,norm,workers,config,steps,updates,best_rank,next_eval,curriculum)
                            if curriculum:
                                save_checkpoint(run_dir/f'stage{curriculum.stage}_best.pt',policy,optimizer,norm,workers,config,steps,updates,best_rank,next_eval,curriculum)
                        if curriculum:
                            previous = curriculum.state_dict()
                            promoted = curriculum.observe(result['success_rate'],steps,args.gate_success,args.gate_passes,args.min_level_steps,locked_stage=args.stage)
                            if promoted:
                                with (run_dir/'curriculum_history.jsonl').open('a') as history:
                                    history.write(json.dumps(dict(steps=steps,previous=previous,current=curriculum.state_dict(),success_rate=result['success_rate']))+'\n')
                                workers.configure(curriculum.env_config())
                                best_rank = (-1.,0.,0.,-float('inf'))
                                # The transferred policy is the initial candidate for the new level.
                                save_checkpoint(run_dir/'best.pt',policy,optimizer,norm,workers,config,steps,updates,best_rank,next_eval,curriculum)
                                print(f'Curriculum → stage {curriculum.stage} {STAGE_NAMES[curriculum.stage]} {curriculum.env_config()}',flush=True)
                            elif curriculum.completed and not previous['completed']:
                                print('All four curriculum stages passed; continuing full-task efficiency training.',flush=True)
                # Atomic checkpoint after every complete update: no partial rollout on resume.
                save_checkpoint(run_dir/'latest.pt',policy,optimizer,norm,workers,config,steps,updates,best_rank,next_eval,curriculum)
                if not stopping and handoff and curriculum and handoff.due(curriculum.stage,steps):
                    perform_handoff()
            if not stopping and args.final_eval_episodes:
                best = load_checkpoint(run_dir/'best.pt')
                policy.load_state_dict(best['policy'])
                norm.load_state_dict(best['normalizer'])
                final_stage = args.stage or 4
                final_config = dict(stage=final_stage,difficulty=1.,effort_scale=1. if final_stage==4 else 0.) if curriculum else None
                live.status(dict(phase='evaluating',steps=steps,evaluation_episodes=args.final_eval_episodes))
                result = evaluate_policy(policy,norm,args.final_eval_episodes,TEST_SEED,args.max_steps,args.gamma,
                                         stop_requested=lambda: stopping, progress_label='Final full-task evaluation',env_config=final_config)
                if result is not None:
                    result.update(curriculum_completed=curriculum.completed if curriculum else None,
                                  last_trained_stage=curriculum.stage if curriculum else None)
                    (run_dir/'final_evaluation.json').write_text(json.dumps(result,indent=2)+'\n')
                    print(f'Final held-out success: {result["success_rate"]:.1%}',flush=True)
        failed=False
    finally:
        live.status(dict(phase='error' if failed else 'stopped' if stopping else 'finished',steps=steps))
        live.close()
        signal.signal(signal.SIGINT,old_handler)
        signal.signal(signal.SIGTERM,old_term_handler)
        workers.close()
        viewer.close()
    if curriculum:
        print(f'Curriculum status: stage {curriculum.stage}, level {curriculum.level}/4, all_stages_passed={curriculum.completed}',flush=True)
    print(f'Checkpoint: {run_dir/"latest.pt"}',flush=True)


if __name__ == '__main__':
    main()
