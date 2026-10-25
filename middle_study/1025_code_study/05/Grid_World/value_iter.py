from collections import defaultdict
from common.GridWorld import GridWorld
import common.Grid_World_render as render_helper
from policy_iter import greedy_policy

def value_iter_onestep(V, env, gamma):
    for state in env.states():
        if state == env.goal_state:
            V[state] = 0
            continue

        temp_V = []

        for action in env.actions():
            next_state = env.next_state(state, action)
            reward = env.reward(state, action, next_state)

            value = reward + gamma * V[next_state]
            temp_V.append(value)

        V[state] = max(temp_V)

    return V

def value_iter(V, env, gamma, threshold=0.001, is_render = True):
    while True:
        if is_render:
            env.render_v(V)

        old_V = V.copy()
        V = value_iter_onestep(V, env, gamma)

        delta = 0
        for state in env.states():
            temp = abs(V[state] - old_V[state])
            delta = max(delta, temp)

        if delta < threshold:
            break

    return V

if __name__ == '__main__':
    V = defaultdict(lambda : 0)
    env = GridWorld()
    gamma = 0.9

    V = value_iter(V, env, gamma)
    pi = greedy_policy(V, env, gamma)

    env.render_v(V, pi)