import numpy as np
import matplotlib.pyplot as plt


class Bandit:
    def __init__(self, arms=10):
        self.rates = np.random.rand(arms) # arms개 만큼 0~1 사이의 난수를 생성해 저장.

    def play(self, arm):
        rate = self.rates[arm]

        if rate > np.random.rand():
            return 1
        else:
            return 0

class Agent:
    def __init__(self, epsilon, action_size=10):
        self.epsilon = epsilon
        self.Qs = np.zeros(action_size) # Qs : 각 슬롯의 상태 가치를 저장하기 위한 배열 0으로 초기화
        self.ns = np.zeros(action_size) # Qs(n) = Qs(n-1) + (Rs - Qs(n-1)) / n 으로 가치를 갱신하므로 횟수인 n을 저장할 메모리

    def update(self, action, reward):
        self.ns[action] += 1
        self.Qs[action] += (reward - self.Qs[action]) / self.ns[action] # iterative하게 업데이트

    def get_action(self):
        if self.epsilon > np.random.rand():
            return np.random.randint(0, len(self.Qs)) # 무작위 행동 선택
        else:
            return np.argmax(self.Qs)

"""
bandit = Bandit()
for i in range(3):
    print(bandit.play(0))
"""

"""
if __name__ == '__main__':
    steps = 1000
    epsilon = 0.1

    bandit = Bandit()
    agent = Agent(epsilon)
    total_reward = 0
    total_rewards = []
    rates = []

    for step in range(steps):
        action = agent.get_action()
        reward = bandit.play(action)
        agent.update(action, reward)
        total_reward += reward

        total_rewards.append(total_reward)
        rates.append(total_reward/(step+1))
    print(total_reward)

plt.ylabel('Total reward')
plt.xlabel('Steps')
plt.plot(total_rewards)
plt.show()

plt.ylabel('Rates')
plt.xlabel('Steps')
plt.plot(rates)
plt.show()
"""