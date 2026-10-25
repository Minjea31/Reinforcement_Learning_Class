import numpy as np
from collections import defaultdict
from common.GridWorld import GridWorld
import common.Grid_World_render as render_helper

class RandomAgent:
    def __init__(self):
        self.gamma = 0.9
        self.action_size = 4

        random_actions = {0:0.25, 1:0.25, 2:0.25, 3:0.25}
        self.pi = defaultdict(lambda : random_actions)
        self.V = defaultdict(lambda : 0)
        self.cnts = defaultdict(lambda : 0)
        self.memory = []

    def get_action(self, state):
        action_probs = self.pi[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p = probs)

    def add(self, state, action, reward):
        data = (state, action, reward)
        self.memory.append(data)

    def reset(self):
        self.memory.clear()

    def eval(self):
        G = 0
        for data in reversed(self.memory):
            state, action, reward = data
            G = self.gamma * G + reward
            self.cnts[state] += 1
            self.V[state] += (G - self.V[state]) / self.cnts[state]

def greedy_probs(Q, state, epsilon = 0, action_size = 4):
    qs = [Q[state, action] for action in range(action_size)]

    base_prob = epsilon/action_size
    action_probs = {0:base_prob, 1:base_prob, 2:base_prob, 3:base_prob}
    max_action = np.argmax(qs)
    action_probs[max_action] += 1 - epsilon
    return action_probs


class McAgent:
    def __init__(self):
        self.gamma = 0.9
        self.action_size = 4
        self.epsilon = 0.1
        self.alpha = 0.1

        random_actions = {0:0.25, 1:0.25, 2:0.25, 3:0.25}
        self.pi = defaultdict(lambda : random_actions)
        self.Q = defaultdict(lambda : 0)
        self.cnts = defaultdict(lambda : 0)
        self.memory = []

    def get_action(self, state):
        action_probs = self.pi[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p = probs)

    def add(self, state, action, reward):
        data = (state, action, reward)
        self.memory.append(data)

    def reset(self):
        self.memory.clear() 

    def update(self):
        G = 0
        for data in reversed(self.memory):
            state, action, reward = data
            G = reward + self.gamma * G
            key = (state, action)
            """
            self.cnts[key] += 1
            self.Q[key] += (G - self.Q[key]) / self.cnts[key]
            """
            
            # exponential 일 경우 
            self.Q[key] += (G - self.Q[key]) * self.alpha
        

            self.pi[state] = greedy_probs(self.Q, state, self.epsilon)


env= GridWorld()
agent = McAgent()

episodes = 10000
for episode in range(episodes):
    state = env.reset()
    agent.reset()

    while True:
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)

        agent.add(state, action, reward)
        if done:
            agent.update()
            break

        state = next_state

env.render_q(agent.Q) # Q 함수 시각화



"""
env = GridWorld()
agent = RandomAgent()

episodes = 1000
for episode in range(episodes):
    state = env.reset()
    agent.reset()

    while True:
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)

        agent.add(state, action, reward)
        if done:
            agent.eval()
            break

        state = next_state

env.render_v(agent.V)
"""