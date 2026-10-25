from collections import defaultdict
from common.GridWorld import GridWorld
import common.Grid_World_render as render_helper
from policy_eval import policy_eval

def argmax(d):
    """d (dict)"""
    max_value = max(d.values())
    max_key = -1
    for key, value in d.items():
        if max_value == value:
            max_key = key

    return max_key

def greedy_policy(V, env, gamma):
    pi = {}
    
    for state in env.states():
        action_value = {}

        for action in env.actions():
            next_state = env.next_state(state, action)
            reward = env.reward(state, action, next_state)
            value = reward + gamma * V[next_state]
            action_value[action] = value

        max_action = argmax(action_value)
        action_probs = {0:0, 1:0, 2:0, 3:0}
        action_probs[max_action] = 1.0
        pi[state] = action_probs

    return pi

def policy_iter(env, gamma, threshold = 0.001, is_render = True):
    pi = defaultdict(lambda : {0:0.25, 1:0.25, 2:0.25, 3:0.25})
    V = defaultdict(lambda : 0)

    while True:
        old_pi = pi.copy()

        V = policy_eval(pi, V, env, gamma, threshold)
        pi = greedy_policy(V, env, gamma)

        if is_render:
            env.render_v(V, pi)

        if pi == old_pi:
            break

    return pi

if __name__ == '__main__':
    env= GridWorld()
    gamma = 0.9
    policy_iter(env, gamma)
        
