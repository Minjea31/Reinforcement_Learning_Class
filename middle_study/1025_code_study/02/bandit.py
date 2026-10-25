import numpy as np
import matplotlib.pyplot as plt

class Bandit:
    def __init__(self, arms=10):
        self.rates = np.random.rand(arms)

    def play(self, arm):
        rate = self.rates[arm]
        if np.random.rand() < rate:
            return 1
        else:
            return 0


class Agent:
    def __init__(self, epsilon, action_size=10):
        self.epsilon = epsilon
        self.Qs = np.zeros(action_size) # 각 슬롯머신의 가치 추정치
        self.ns = np.zeros(action_size) # 각 슬롯머신의 플레이 횟수

    def update(self, action, reward):
        self.ns[action] += 1
        self.Qs[action] += (reward - self.Qs[action]) / self.ns[action]

    def get_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, len(self.Qs)) # 무작위 행동 선택
        else:
            return np.argmax(self.Qs) # 탐욕 행동 선택

if __name__ == '__main__':
    steps = 1000
    epsilon = 0.1

    bandit = Bandit()
    agent = Agent(epsilon)
    total_rewrad = 0
    total_rewrads = []
    rates = []

    for step in range(steps):
        action = agent.get_action()
        reward = bandit.play(action)
        agent.update(action, reward)
        total_rewrad += reward

        total_rewrads.append(total_rewrad)
        rates.append(total_rewrad/(step + 1))

    print(total_rewrad)

    plt.ylabel('Total reward')
    plt.xlabel('Steps')
    plt.plot(total_rewrads)
    plt.show()

    plt.ylabel('Rates')
    plt.xlabel('Steps')
    plt.plot(rates)
    plt.show()

"""
bandit = Bandit()

for i in range(3):
    print(bandit.play(0))
    """