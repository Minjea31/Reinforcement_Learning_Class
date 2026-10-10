from GridWorld import GridWorld
import Grid_World_render as render_helper
from collections import defaultdict

def eval_onestep(pi, V, env, gamma=0.9):
    for state in env.states(): # 모든 위치를 다돔
        if state == env.goal_state: # goal 지점의 가치는 0임
            V[state] = 0
            continue

        action_probs = pi[state] # 해당 위치의 policy를 가져옴
        new_V = 0
        # print(action_probs)
        for action, action_prob in action_probs.items(): # action_prob은 행동의 확률임.
            next_state = env.next_state(state, action) # 다음 위치 계산
            r = env.reward(state, action, next_state) # r(s,a,s') 계산

            new_V += action_prob * (r + gamma * V[next_state])

        V[state] = new_V
    return V


def policy_eval(pi, V, env, gamma, threshold=0.001):
    while True:
        old_V = V.copy()
        # env.render_v(V, pi)
        V = eval_onestep(pi, V, env, gamma=0.9)
        #갱신된 양의 최댓값 계산
        delta = 0
        for state in V.keys():
            t = abs(V[state] - old_V[state])
            if delta < t:
                delta = t

        if delta < threshold:
            break
    return V
"""
env = GridWorld()
gamma = 0.9

pi = defaultdict(lambda: {0:0.25, 1:0.25, 2:0.25, 3:0.25})
V = defaultdict(lambda:0)

V = policy_eval(pi, V, env, gamma)
env.render_v(V, pi)
"""

"""
env = GridWorld()
gamma = 0.9

pi = defaultdict(lambda: {0:0.25, 1:0.25, 2:0.25, 3:0.25})
V = defaultdict(lambda:0)

V = policy_eval(pi, V, env, gamma)
env.render_v(V, pi)


pi = defaultdict(lambda: {0:1.0, 1:0, 2:0, 3:0})
V = defaultdict(lambda:0)

V = policy_eval(pi, V, env, gamma)
env.render_v(V, pi)

pi = defaultdict(lambda: {0:0, 1:1.0, 2:0, 3:0})
V = defaultdict(lambda:0)

V = policy_eval(pi, V, env, gamma)
env.render_v(V, pi)
pi = defaultdict(lambda: {0:0, 1:0, 2:1.0, 3:0})
V = defaultdict(lambda:0)

V = policy_eval(pi, V, env, gamma)
env.render_v(V, pi)

pi = defaultdict(lambda: {0:0, 1:0, 2:0, 3:1.0})
V = defaultdict(lambda:0)

V = policy_eval(pi, V, env, gamma)
env.render_v(V, pi)

pi = defaultdict(lambda: {0:0.7, 1:0.1, 2:0.1, 3:0.1})
V = defaultdict(lambda:0)

V = policy_eval(pi, V, env, gamma)
env.render_v(V, pi)
"""