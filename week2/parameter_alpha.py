import numpy as np
import matplotlib.pyplot as plt
from bandit import Agent
from non_stationary import NonStatBandit, AlphaAgent

runs = 200
steps = 1000
epsilon = 0.1
alphas = [0.2, 0.5, 0.8]

agent_types = ['sample average'] + [f'alpha={a}' for a in alphas]
results = {}

for agent_type in agent_types:
    all_rates = np.zeros((runs, steps))  # (200, 1000)

    for run in range(runs):
        if agent_type == 'sample average':
            agent = Agent(epsilon)
        else:
            alpha = float(agent_type.split('=')[1])
            agent = AlphaAgent(epsilon, alpha)

        bandit = NonStatBandit()
        total_reward = 0
        rates = []

        for step in range(steps):
            action = agent.get_action()
            reward = bandit.play(action)
            agent.update(action, reward)
            total_reward += reward
            rates.append(total_reward / (step + 1))

        all_rates[run] = rates

    results[agent_type] = np.average(all_rates, axis=0)

plt.figure()
plt.ylabel('Average Rates')
plt.xlabel('Steps')
for key, avg_rates in results.items():
    plt.plot(avg_rates, label=key)
plt.legend()
plt.show()

# 보고서용 수치
for key, avg_rates in results.items():
    print(f'{key:16s} final={avg_rates[-1]:.4f}  step500={avg_rates[499]:.4f}')