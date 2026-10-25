from common.GridWorld import GridWorld
from collections import defaultdict
import common.Grid_World_render as render_helper


def eval_onestep(pi, V, env, gamma=0.9):
    for state in env.states():
        if state == env.goal_state:
            V[state] = 0
            continue

        action_probs = pi[state]
        new_V = 0

        for action, action_prob in action_probs.items():
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)

            new_V += action_prob * (r + gamma * V[next_state])

        V[state] = new_V
    return V

def policy_eval(pi, V, env, gamma, threshold=0.001):
    while True:
        old_V = V.copy()
        V = eval_onestep(pi, V, env, gamma)

        delta = 0
        for state in V.keys():
            t = abs(old_V[state] - V[state])

            delta = max(delta, t)

        if delta < threshold:
            break
    return V

"""
env = GridWorld()
gamma = 0.9

pi_1 = {0:1.0, 1:0.0, 2:0.0, 3:0.0}
pi_2 = {0:0.0, 1:1.0, 2:0.0, 3:0.0}
pi_3 = {0:0.0, 1:0.0, 2:1.0, 3:0.0}
pi_4 = {0:0.0, 1:0.0, 2:0.0, 3:1.0}
pi_5 = {0:0.7, 1:0.1, 2:0.1, 3:0.1}
pi = defaultdict(lambda : pi_1)
V = defaultdict(lambda : 0)

policy_eval(pi, V, env, gamma)
env.render_v(V)

pi = defaultdict(lambda : pi_2)
V = defaultdict(lambda : 0)

policy_eval(pi, V, env, gamma)
env.render_v(V)

pi = defaultdict(lambda : pi_3)
V = defaultdict(lambda : 0)

policy_eval(pi, V, env, gamma)
env.render_v(V)

pi = defaultdict(lambda : pi_4)
V = defaultdict(lambda : 0)

policy_eval(pi, V, env, gamma)
env.render_v(V)

pi = defaultdict(lambda : pi_5)
V = defaultdict(lambda : 0)

policy_eval(pi, V, env, gamma)
env.render_v(V)
"""