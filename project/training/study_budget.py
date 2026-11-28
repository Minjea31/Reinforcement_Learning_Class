"""Stage budgets and exact environment-step accounting for comparison runs."""
import json


def stage_progress(config, steps):
    start = config.get('stage_start_steps', 0)
    limit = config.get('stage_max_steps') or config['total_steps']-start
    return dict(stage_start_steps=start, stage_steps=steps-start, stage_total_steps=limit)


def stage_deadline(config):
    limit = config.get('stage_max_steps')
    if limit is None:
        return config['total_steps']
    return min(config['total_steps'], config['stage_start_steps']+limit)


def restore_stage_start(run, previous, initial_stage, current_stage):
    if previous is None or current_stage == initial_stage:
        return 0
    if 'stage_start_steps' in previous['config']:
        return previous['config']['stage_start_steps']
    # Runs that already advanced before stage budgets recorded each validation
    # at the transition step, before switching the environment.
    prior_validations = []
    for path in run.glob('validation_*.json'):
        result = json.loads(path.read_text())
        if result['env_config']['stage'] < current_stage and result['steps'] <= previous['steps']:
            prior_validations.append(result['steps'])
    if not prior_validations:
        raise ValueError('Missing previous-stage validation needed to restore the stage step count')
    return max(prior_validations)


def learn_exact(model, steps, callback, stopping):
    """Keep PPO/A2C from rounding a budget boundary up to a full rollout."""
    from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm

    if not isinstance(model, OnPolicyAlgorithm) or steps % model.n_steps == 0:
        model.learn(total_timesteps=steps, reset_num_timesteps=False, callback=callback)
        return

    rollout_steps = model.n_steps
    full_rollouts, tail = divmod(steps, rollout_steps)
    if full_rollouts:
        start = model.num_timesteps
        full_steps = full_rollouts*rollout_steps
        model.learn(total_timesteps=full_steps, reset_num_timesteps=False, callback=callback)
        if stopping() or model.num_timesteps != start+full_steps:
            return

    buffer = model.rollout_buffer
    tail_buffer = model.rollout_buffer_class(
        tail, model.observation_space, model.action_space, device=model.device,
        gamma=model.gamma, gae_lambda=model.gae_lambda, n_envs=model.n_envs,
        **model.rollout_buffer_kwargs,
    )
    model.n_steps = tail
    model.rollout_buffer = tail_buffer
    try:
        model.learn(total_timesteps=tail, reset_num_timesteps=False, callback=callback)
    finally:
        model.n_steps = rollout_steps
        model.rollout_buffer = buffer
