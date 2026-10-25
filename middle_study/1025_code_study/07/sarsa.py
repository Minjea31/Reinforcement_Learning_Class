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

class SarsaAgent:
    def __init__(self):
        self.gamma = 0.9
        self.alpha = 0.8
        self.epsilon = 0.1
        self.actions_size = 4

        random_actions = {0:0.25, 1:0.25, 2:0.25, 3:0.25}
        self.pi = defaultdict(lambda : random_actions)
        self.Q = defaultdict(lambda : 0)
        self.memory = deque(maxlen = 2)

    def reset(self):
        self.memory.clear()

    def get_action(self, state):
        action_probs = self.pi[state]
        action = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(action, p =probs)

    def update(self, state, action, reward, done):
        self.memory.append((state, action, reward, done))

        if len(self.memory) < 2:
            return

        state, action, reward, done = self.memory[0]
        next_state, next_action, _, _ = self.memory[1]
        next_q = 0 if done else self.Q[next_state, next_action]

        target = reward + self.gamma * next_q

        self.Q[state, action] += self.alpha * (target - self.Q[state, action])

        self.pi[state] = greedy_probs(self.Q, state, self.epsilon)



env = GridWorld()
agent = SarsaAgent()

episodes = 1000
for episode in range(episodes):
    state = env.reset()
    agent.reset()

    while True:
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)

        agent.update(state, action, reward, done)
        if done:
            agent.update(next_state, None, None, None)
            break
        state = next_state

env.render_q(agent.Q)