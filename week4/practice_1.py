from policy_eval import policy_eval
from gridworld import GridWorld
from collections import defaultdict


if __name__ == '__main__':
    env = GridWorld()
    gamma = 0.9  # 할인율

    pi = defaultdict(lambda: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})  # 정책
    V = defaultdict(lambda: 0)  # 가치 함수

    V = policy_eval(pi, V, env, gamma)  # 정책 평가

    # [그림 4-13] 무작위 정책의 가치 함수
    env.render_v(V, pi)