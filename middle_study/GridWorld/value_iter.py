from collections import defaultdict
from GridWorld import GridWorld
from policy_iter import greedy_policy

def value_iter_onestep(V, env, gamma=0.9):
    for state in env.states(): # 모든 격자를 순회
        if state == env.goal_state:
            V[state] = 0
            continue

        action_values = []
        for action in env.actions(): # 모든 행동에 차례로 접근
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)
            value = r + gamma * V[next_state] # 새로운 가치 함수
            action_values.append(value)
        # print(action_values)


        V[state] = max(action_values) # 최댓값 추출, 해당 state의 action 중에서 가장 value가 높은걸 저장.
    # print(V)
    return V

def value_iter(V, env, gamma, threshold=0.001, is_render=True): # 얘가 본체임. 초기의 V를 받아서 V를 갱신 후 변화가 없을때 까지의 최적의 value를 return 함.
    while True:
        if is_render:
            env.render_v(V)

        old_V = V.copy()
        V = value_iter_onestep(V, env, gamma)

        # 차이를 계산 해야함.
        delta = 0
        for state in V.keys():
            t = abs(V[state] - old_V[state])
            if delta < t:
                delta = t

        if delta < threshold:
            break
    return V


if __name__ == '__main__':
    V = defaultdict(lambda: 0)
    env = GridWorld()
    gamma = 0.9

    V = value_iter(V, env, gamma) # 가장 좋은 value를 찾는거임.
    
    pi = greedy_policy(V, env, gamma) # 최종 value와 policy 
    env.render_v(V, pi)