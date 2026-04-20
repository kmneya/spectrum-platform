# simulation/engine.py
import random
import numpy as np

# Import the MADDPG implementation
try:
    from .maddpg_agent import MADDPG
except ImportError:
    # Fallback for testing if file not yet created
    MADDPG = None

# Global MADDPG instance
maddpg = None

def init_maddpg(num_agents, state_dim=3, action_dim=1):
    global maddpg
    if MADDPG is not None:
        maddpg = MADDPG(num_agents, state_dim, action_dim)

def run_real_simulation(lte_agents, wifi_aps, traffic, algorithm):
    global maddpg

    # ---------------------------
    # Input validation
    # ---------------------------
    lte_agents = max(0, int(lte_agents))
    wifi_aps = max(0, int(wifi_aps))
    traffic = max(1, min(5, int(traffic)))

    # ---------------------------
    # 1. Initialize or load MADDPG if needed
    # ---------------------------
    if algorithm == "madrl":
        if maddpg is None or maddpg.num_agents != lte_agents:
            if lte_agents > 0:
                init_maddpg(lte_agents, state_dim=3, action_dim=1)
            else:
                maddpg = None

    # ---------------------------
    # 2. Prepare observations for each LTE agent
    # ---------------------------
    observations = []
    for i in range(lte_agents):
        obs = [
            traffic / 10.0,
            0.5,
            wifi_aps / max(1, lte_agents + wifi_aps)
        ]
        observations.append(obs)

    # ---------------------------
    # 3. Get actions (duty cycles)
    # ---------------------------
    if algorithm == "madrl" and lte_agents > 0 and maddpg is not None:
        duty_cycles = maddpg.act(observations, noise=0.05).flatten().tolist()
    elif algorithm == "duty_cycle":
        duty_cycles = [0.5] * lte_agents
    else:  # no_algorithm
        duty_cycles = [0.9] * lte_agents

   # ---------------------------
    # 4. Simulate throughput & fairness
    # ---------------------------
    lte_rate_per_agent = 20.0   # Mbps max
    wifi_rate_per_ap = 15.0     # Mbps max

    avg_lte_duty = np.mean(duty_cycles) if duty_cycles else 0.0
    wifi_airtime = max(0.0, 1.0 - avg_lte_duty * lte_agents * 0.15)

    throughput_lte = []
    for dc in duty_cycles:
        t = lte_rate_per_agent * dc * (1 + 0.1 * traffic)
        throughput_lte.append(t + np.random.uniform(-1, 2))

    throughput_wifi = wifi_rate_per_ap * wifi_aps * wifi_airtime * (1 + 0.05 * traffic)
    throughput_wifi = max(0, throughput_wifi + np.random.uniform(-2, 3))

    total_lte = sum(throughput_lte)
    total = total_lte + throughput_wifi

    # Jain's Fairness Index between LTE and WiFi
    # Special case: if only one technology is present, fairness = 1.0
    if lte_agents == 0:
        fairness = 1.0
    elif wifi_aps == 0:
        fairness = 1.0
    else:
        sum_x = total_lte + throughput_wifi
        sum_sq = total_lte**2 + throughput_wifi**2
        if sum_sq > 0:
            fairness = (sum_x ** 2) / (2 * sum_sq)
        else:
            fairness = 0.0

    # Reward: fairness-focused
    reward = fairness * 100 + total * 0.1

    # ---------------------------
    # 5. Store transition and update MADDPG
    # ---------------------------
    if algorithm == "madrl" and lte_agents > 0 and maddpg is not None:
        next_obs = observations  # static scenario
        dones = [False] * lte_agents
        rewards_per_agent = [reward] * lte_agents

        maddpg.store_transition(
            np.array(observations),
            np.array(duty_cycles).reshape(lte_agents, 1),
            np.array(rewards_per_agent),
            np.array(next_obs),
            np.array(dones)
        )
        maddpg.update()

    # ---------------------------
    # 6. Packet loss
    # ---------------------------
    packet_loss_lte = []
    for dc in duty_cycles:
        pl = np.random.uniform(1, 8) * (1 - dc)
        packet_loss_lte.append(round(pl, 2))

    packet_loss_wifi = np.random.uniform(3, 12) * (1 - wifi_airtime)
    packet_loss_wifi = round(packet_loss_wifi, 2)

    # ---------------------------
    # 7. Build response
    # ---------------------------
    return {
        "throughput": {
            "lte": [round(t, 2) for t in throughput_lte],
            "wifi": round(throughput_wifi, 2),
            "total": round(total, 2)
        },
        "fairness": round(fairness, 3),
        "packet_loss": {
            "lte": packet_loss_lte,
            "wifi": packet_loss_wifi
        },
        "rl": {
            "duty_cycles": [round(dc, 2) for dc in duty_cycles],
            "reward": round(reward, 2)
        }
    }