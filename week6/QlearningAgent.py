from collections import defaultdict, deque
import numpy as np
import Grid_World_render as render_helper
from GridWorld import GridWorld

def greedy_probs(Q, state, action_size=4):
    qs = [Q[(state, action)] for action in range(action_size)]
    max_action = np.argmax(qs)
    action_probs = {action : 0.0 for action in range(action_size)}
    # 이 시점에서 action_probs는 {0:0.0, 1:0.0, 2:0.0, 3:0.0}이 됨
    action_probs[max_action] = 1
    return action_probs

def epsilon_greedy_probs(Q, state, epsilon=0, action_size=4):
    qs = [Q[(state, action)] for action in range(action_size)]
    max_action = np.argmax(qs)

    base_prob = epsilon/action_size
    action_probs = {action : base_prob for action in range(action_size)}
    # 이 시점에서 action_probs는 {0 : epsilon/action_size, 1 : epsilon/action_size, 2 : epsilon/action_size, 3 : epsilon/action_size}이 됨
    action_probs[max_action] += (1- epsilon)
    return action_probs

class QlearningAgent:
    def __init__(self):
        self.gamma = 0.9
        self.alpha = 0.8
        self.epsilon = 0.1
        self.action_size = 4

        random_actions = {0:0.25, 1:0.25, 2:0.25, 3:0.25}
        self.pi = defaultdict(lambda: random_actions)
        self.b = defaultdict(lambda: random_actions) # 행동 정책
        self.Q = defaultdict(lambda : 0)

    def get_action(self, state):
        action_probs = self.b[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)

    def update(self, state, action, reward, next_state, done):
        if done: # 목표에 도달
            next_q_max = 0
        else:  # 그 외에는 다음 상태에서 Q 함수의 최댓값 계산
            next_qs = [self.Q[next_state, a] for a in range(self.action_size)]
            next_q_max = max(next_qs)

        # Q 함수 갱신
        target = reward + self.gamma * next_q_max
        self.Q[state, action] += self.alpha * (target - self.Q[state, action])

        # 행동 정책과 대상 정책 갱신
        self.pi[state] = greedy_probs(self.Q, state)
        self.b[state] = epsilon_greedy_probs(self.Q, state, epsilon=self.epsilon)



env = GridWorld()
agent = QlearningAgent()

episodes = 10000
for episode in range(episodes):
    state = env.reset()

    while True:
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)

        agent.update(state, action, reward, next_state, done)
        if done:
            break
        state = next_state

env.render_q(agent.Q) # Q 함수 시각화
env.render_policy(agent.Q) # 정책 시각화