from collections import defaultdict
from pprint import pprint
from GridWorld import GridWorld
from policy_eval import policy_eval

def argmax(d): # policy 중 가장 높은 확률인 action의 index를 반환함.
    """d(dict): key-value 쌍을 가지는 dict"""
    max_value = max(d.values())
    max_key = -1
    for key, value in d.items():
        if value == max_value:
            max_key = key
    return max_key

def greedy_policy(V, env, gamma): # 가장 확률이 높은 action을 선택하도록 함.
    pi = {}
    # print(V)
    for state in env.states(): # 모든 state를 순회
        action_values = {}

        # 해당 state에서의 action에 대한 state_value를 저장.
        for action in env.actions(): # env.actions() = {0, 1, 2, 3}
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)
            value = r + gamma * V[next_state]
            action_values[action] = value

        max_action = argmax(action_values) # action에 대한 state_value 중에서 가장 큰 action을 저장.
        action_probs = {0:0, 1:0, 2:0, 3:0}
        action_probs[max_action] = 1.0
        pi[state] = action_probs
    print(pi)
    return pi

def policy_iter(env, gamma=0.9, threshold=0.001, is_render=True):
    pi = defaultdict(lambda: {0:0.25, 1:0.25, 2:0.25, 3:0.25})
    V = defaultdict(lambda:0)

    while True:
        V = policy_eval(pi, V, env, gamma, threshold) # 평가, policy_eval 은 V를 뱉어냄.
        # print(V) # state value 임 
        
        # env.render_v(V, pi)

        new_pi = greedy_policy(V, env, gamma) # 개선, greedy_policy는 pi를 뱉어내는데 deterministic policy를 뱉어냄.
        # pprint(new_pi, width=60, sort_dicts=False)

        if is_render:
            pprint(new_pi, width=60, sort_dicts=False)
            env.render_v(V, pi)

        # print(pi)
        if new_pi == pi: # 갱신 여부 확인
            break
        pi = new_pi

    return pi



if __name__ == "__main__":
    env = GridWorld()
    gamma = 0.9

    pi = policy_iter(env, gamma)
