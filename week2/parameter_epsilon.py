import numpy as np
import matplotlib.pyplot as plt
from bandit import Bandit, Agent

runs = 200
steps = 1000
epsilons = [0.1, 0.3, 0.01]

for eps in epsilons:
    all_rates = np.zeros((runs, steps))

    for run in range(runs):
        bandit = Bandit()
        agent = Agent(eps)
        total_reward = 0
        rates = []

        for step in range(steps):
            action = agent.get_action()
            reward = bandit.play(action)
            agent.update(action, reward)
            total_reward += reward
            rates.append(total_reward / (step + 1))

        all_rates[run] = rates

    avg_rates = np.average(all_rates, axis=0)
    plt.plot(avg_rates, label=str(eps))

plt.ylabel('Rates')
plt.xlabel('Steps')
plt.legend()
plt.show()