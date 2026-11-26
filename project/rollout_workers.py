"""Spawned MuJoCo workers. Only the policy is centralized; tasks have no roles."""
import multiprocessing as mp
import traceback
import numpy as np
from curriculum import make_env


def worker(connection, seed, max_steps, gamma, env_config=None):
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    env = None
    try:
        env = make_env(env_config,max_steps=max_steps,reward_gamma=gamma)
        obs, _ = env.reset(seed=seed)
        connection.send(('ok', obs))
        while True:
            command, value = connection.recv()
            if command == 'close':
                break
            if command == 'state':
                result = env.get_state()
            elif command == 'configure':
                result = env.configure(value)
            elif command == 'restore':
                result = env.set_state(value)
            elif command == 'step':
                action, snapshot = value
                obs, reward, term, trunc, info = env.step(action)
                final_obs = obs.copy()
                state = env.get_state() if snapshot else None
                if term or trunc:
                    obs, _ = env.reset()
                result = (obs, final_obs, reward, term, trunc, info, state)
            else:
                raise ValueError(command)
            connection.send(('ok', result))
    except (EOFError, KeyboardInterrupt):
        pass
    except Exception:
        connection.send(('error', traceback.format_exc()))
    finally:
        if env is not None:
            env.close()
        connection.close()


class RolloutWorkers:
    def __init__(self, count, seed, max_steps, gamma, env_config=None):
        self.pipes, self.processes = [], []
        context = mp.get_context('spawn')
        try:
            for index in range(count):
                parent, child = context.Pipe()
                process = context.Process(target=worker,args=(child, seed+index,max_steps,gamma,env_config),daemon=True)
                process.start()
                child.close()
                self.pipes.append(parent)
                self.processes.append(process)
            self.obs = np.stack([self.receive(p) for p in self.pipes])
        except BaseException:
            self.close()
            raise

    @staticmethod
    def receive(pipe):
        status, result = pipe.recv()
        if status != 'ok':
            raise RuntimeError(result)
        return result

    def step(self, actions, snapshot=False):
        for i, (pipe, action) in enumerate(zip(self.pipes, actions)):
            pipe.send(('step', (action, snapshot and i == 0)))
        results = [self.receive(p) for p in self.pipes]
        self.obs = np.stack([r[0] for r in results])
        return results

    def configure(self, env_config):
        for pipe in self.pipes:
            pipe.send(('configure',env_config))
        self.obs = np.stack([self.receive(p) for p in self.pipes])

    def states(self):
        for pipe in self.pipes:
            pipe.send(('state', None))
        return [self.receive(p) for p in self.pipes]

    def restore(self, states):
        if len(states) != len(self.pipes):
            raise ValueError('Resume requires the saved number of environments')
        for pipe, state in zip(self.pipes, states):
            pipe.send(('restore', state))
        self.obs = np.stack([self.receive(p) for p in self.pipes])

    def close(self):
        for pipe in self.pipes:
            try:
                pipe.send(('close', None))
            except (BrokenPipeError, EOFError, OSError):
                pass
        for process in self.processes:
            process.join(timeout=3)
            if process.is_alive():
                process.terminate()
                process.join()
        for pipe in self.pipes:
            pipe.close()
