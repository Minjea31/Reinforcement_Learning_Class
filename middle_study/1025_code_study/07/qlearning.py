from collections import defaultdict, deque
import numpy as np
from common.GridWorld import GridWorld
import common.Grid_World_render as render_helper

def greedy_probs(Q, state, epsilon = 0, action_size = 4):
    qs = [Q[state, action] for action in range(action_size)]

    base_prob = epsilon/action_size
    action_probs = {0:base_prob, 1:base_prob, 2:base_prob, 3:base_prob}
    max_action = np.argmax(qs)
    action_probs[max_action] += 1 - epsilon
    return action_probs

class qlearningAgent:
    def __init__(self):
        self.gamma = 0.9
        self.alpha = 0.8
        self.epsilon = 0.1
        self.action_size = 4

        random_actions = {0:0.25, 1:0.25, 2:0.25, 3:0.25}
        self.pi = defaultdict(lambda : random_actions)
        self.b = defaultdict(lambda : random_actions)
        self.Q = defaultdict(lambda : 0)

    def get_action(self, state):
        action_probs = self.b[state]
        action = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(action, p =probs)

    def update(self, state, action, reward, next_state, done):
        if done:
            max_next_qs = 0

        else:
            next_qs = [self.Q[next_state, action] for action in range(self.action_size)]
            max_next_qs = max(next_qs)

        target = reward + self.gamma * max_next_qs
        self.Q[state, action] += self.alpha * (target - self.Q[state, action])

        self.pi[state] = greedy_probs(self.Q, state, epsilon = 0)
        self.b[state] = greedy_probs(self.Q, state, self.epsilon)



env = GridWorld()
agent = qlearningAgent()

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

env.render_q(agent.Q)