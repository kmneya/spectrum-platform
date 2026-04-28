# train.py
from simulation.engine import run_real_simulation, init_maddpg
import numpy as np
import matplotlib.pyplot as plt

NUM_EPISODES = 2000
lte_agents = 3
wifi_aps = 5

# Sample traffic values covering all three deployment categories:
#   Residential (1–10), Office (11–30), Urban (31–1000)
TRAFFIC_POOL = (
    list(range(1, 11))        # residential
    + list(range(11, 31))     # office
    + list(range(31, 201))    # urban (capped at 200 for training speed)
)

init_maddpg(lte_agents)
fairness_history = []
reward_history = []

for ep in range(NUM_EPISODES):
    traffic = int(np.random.choice(TRAFFIC_POOL))
    result = run_real_simulation(lte_agents, wifi_aps, traffic, algorithm="madrl")
    fairness_history.append(result['fairness'])
    reward_history.append(result['rl']['reward'])
    if ep % 100 == 0:
        cat = result.get('category', '')
        print(f"Episode {ep} [{cat:12s} t={traffic:4d}]: Fairness = {result['fairness']:.3f}, Reward = {result['rl']['reward']:.2f}")

# Plot learning curve
plt.plot(fairness_history)
plt.xlabel('Episode')
plt.ylabel('Jain Fairness Index')
plt.title('MADDPG Learning Progress')
plt.show()