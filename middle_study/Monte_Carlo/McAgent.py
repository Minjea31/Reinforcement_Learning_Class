from collections import defaultdict
import numpy as np

def greedy_probs(Q, state, action_size=4):
    qs = [Q[(state, action)] for action in range(action_size)]
    max_action = np.argmax(qs)
    action_probs = {action : 0.0 for action in range(action_size)}
    # 이 시점에서 action_probs는 {0:0.0, 1:0.0, 2:0.0, 3:0.0}이 됨
    action_probs[max_action] = 1
    return action_probs

class McAgent:
    def __init__(self):
        self.gamma = 0.9
        self.action_size = 4

        random_actions = {0:0.25, 1:0.25, 2:0.25, 3:0.25}
        self.pi = defaultdict(lambda:random_actions)
        self.Q = defaultdict(lambda : 0) # V 가 아닌 Q를 사용
        self.cnts = defaultdict(lambda:0)
        self.memory = []

    def get_action(self, state):
        actions_probs = self.pi[state]
        actions = list(actions_probs.keys())
        probs = list(actions_probs.values())
        return np.random.choice(actions, p=probs)

    def add(self, state, action, reward):
        data = (state, action, reward)
        self.memory.append(data)

    def reset(self):
        self.memory.clear()

    def update(self):
        G = 0
        for data in reversed(self.memory):
            state, action, reward = data
            G = self.gamma * G + reward
            key = (state, action)
            self.cnts[key] += 1
            self.Q[key] += (G - self.Q[key]) / self.cnts[key]

            # state의 정책 탐욕화
            self.pi[state] = greedy_probs(self.Q, state)