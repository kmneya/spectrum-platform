# train.py
from simulation.engine import run_real_simulation, init_maddpg
import numpy as np
import matplotlib.pyplot as plt

NUM_EPISODES = 2000
lte_agents = 3
wifi_aps = 5

init_maddpg(lte_agents)
fairness_history = []
reward_history = []

for ep in range(NUM_EPISODES):
    # Randomise scenario parameters
    traffic = np.random.randint(1, 6)
    result = run_real_simulation(lte_agents, wifi_aps, traffic, algorithm="madrl")
    fairness_history.append(result['fairness'])
    reward_history.append(result['rl']['reward'])
    if ep % 100 == 0:
        print(f"Episode {ep}: Fairness = {result['fairness']:.3f}, Reward = {result['rl']['reward']:.2f}")

# Plot learning curve
plt.plot(fairness_history)
plt.xlabel('Episode')
plt.ylabel('Jain Fairness Index')
plt.title('MADDPG Learning Progress')
plt.show()