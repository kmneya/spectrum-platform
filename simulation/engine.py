# simulation/engine.py
import numpy as np

try:
    from .maddpg_agent import MADDPG
except ImportError:
    MADDPG = None

maddpg = None
STATE_DIM = 4  # [traffic_norm, channel_quality, wifi_ratio, lte_load_norm]


def init_maddpg(num_agents):
    global maddpg
    if MADDPG is not None and num_agents > 0:
        maddpg = MADDPG(num_agents, state_dim=STATE_DIM, action_dim=1)


def get_traffic_profile(traffic):
    """
    Map traffic value (1-1000) to a deployment category and physics parameters.
      Residential :  1 – 10   (low device density, WiFi-friendly)
      Office      : 11 – 30   (moderate contention, bursty patterns)
      Urban       : 31 – 1000 (high contention, dense IoT + mobile devices)
    """
    if traffic <= 10:
        category = "residential"
        t = (traffic - 1) / 9.0                   # 0→1 within range
        background_contention = 0.02 + t * 0.04   # 0.02→0.06
        lte_rate_boost        = 1.00 + t * 0.05   # 1.00→1.05
        wifi_rate_boost       = 1.00               # full rate; little competition
        pl_base               = 0.5  + t * 1.0    # 0.5%→1.5%
    elif traffic <= 30:
        category = "office"
        t = (traffic - 11) / 19.0
        background_contention = 0.06 + t * 0.09   # 0.06→0.15
        lte_rate_boost        = 1.05 + t * 0.10   # 1.05→1.15
        wifi_rate_boost       = 0.95 - t * 0.10   # 0.95→0.85
        pl_base               = 1.5  + t * 2.0    # 1.5%→3.5%
    else:  # urban 31-1000
        category = "urban"
        t = min((traffic - 31) / 969.0, 1.0)
        background_contention = 0.15 + t * 0.35   # 0.15→0.50
        lte_rate_boost        = 1.15 + t * 0.20   # 1.15→1.35
        wifi_rate_boost       = 0.85 - t * 0.30   # 0.85→0.55
        pl_base               = 3.5  + t * 8.5    # 3.5%→12.0%

    # Log-scaled demand factor prevents linear blowup at high traffic values
    traffic_factor = 1.0 + 0.4 * np.log1p(traffic / 20.0)

    return {
        "category":              category,
        "traffic_norm":          traffic / 1000.0,
        "background_contention": background_contention,
        "lte_rate_boost":        lte_rate_boost,
        "wifi_rate_boost":       wifi_rate_boost,
        "pl_base":               pl_base,
        "traffic_factor":        traffic_factor,
    }


def run_real_simulation(lte_agents, wifi_aps, traffic, algorithm):
    global maddpg

    lte_agents = max(0, int(lte_agents))
    wifi_aps   = max(0, int(wifi_aps))
    traffic    = max(1, min(1000, int(traffic)))

    profile = get_traffic_profile(traffic)
    bc = profile["background_contention"]
    tf = profile["traffic_factor"]

    # Re-init MADDPG when agent count changes
    if algorithm == "madrl":
        if maddpg is None or maddpg.num_agents != lte_agents:
            if lte_agents > 0:
                init_maddpg(lte_agents)
            else:
                maddpg = None

    # Observations: [traffic_norm, channel_quality, wifi_ratio, lte_load_norm]
    channel_quality = 1.0 - bc
    wifi_ratio      = wifi_aps / max(1, lte_agents + wifi_aps)
    lte_load_norm   = min(lte_agents / 10.0, 1.0)
    obs_vec         = [profile["traffic_norm"], channel_quality, wifi_ratio, lte_load_norm]
    observations    = [obs_vec[:] for _ in range(lte_agents)]

    # Duty cycles
    if algorithm == "madrl" and lte_agents > 0 and maddpg is not None:
        duty_cycles = maddpg.act(observations, noise=0.05).flatten().tolist()
    elif algorithm == "duty_cycle":
        duty_cycles = [0.5] * lte_agents
    else:  # no_algorithm — LTE dominant
        duty_cycles = [0.9] * lte_agents

    # Throughput
    lte_rate  = 20.0 * profile["lte_rate_boost"]
    wifi_rate = 15.0 * profile["wifi_rate_boost"]

    avg_lte_duty  = np.mean(duty_cycles) if duty_cycles else 0.0
    # LTE occupancy grows with duty cycle, agent count, and a log-scaled interference term
    lte_occupancy = min(1.0, avg_lte_duty * lte_agents * (0.10 + 0.03 * np.log1p(traffic / 10.0)))
    # WiFi airtime = remaining channel minus background device contention
    wifi_airtime  = max(0.0, (1.0 - lte_occupancy) * (1.0 - bc))

    throughput_lte = [
        max(0.0, lte_rate * dc * tf + np.random.uniform(-1, 2))
        for dc in duty_cycles
    ]
    throughput_wifi = max(0.0, wifi_rate * wifi_aps * wifi_airtime * tf + np.random.uniform(-2, 3))

    total_lte = sum(throughput_lte)
    total     = total_lte + throughput_wifi

    # Jain's Fairness Index between LTE aggregate and WiFi aggregate
    if lte_agents == 0 or wifi_aps == 0:
        fairness = 1.0
    else:
        s  = total_lte + throughput_wifi
        sq = total_lte ** 2 + throughput_wifi ** 2
        fairness = (s ** 2) / (2 * sq) if sq > 0 else 0.0

    reward = fairness * 100 + total * 0.1

    # Store transition and update MADDPG
    if algorithm == "madrl" and lte_agents > 0 and maddpg is not None:
        maddpg.store_transition(
            np.array(observations),
            np.array(duty_cycles).reshape(lte_agents, 1),
            np.array([reward] * lte_agents),
            np.array(observations),  # static next_obs for single-step scenario
            np.array([False] * lte_agents),
        )
        maddpg.update()

    # Packet loss — scales with category base rate and unused duty cycle / airtime
    pl_base = profile["pl_base"]
    packet_loss_lte  = [round(max(0.0, pl_base * (1 - dc) + np.random.uniform(0, 1)), 2) for dc in duty_cycles]
    packet_loss_wifi = round(max(0.0, pl_base * 1.5 * (1 - wifi_airtime) + np.random.uniform(0, 2)), 2)

    # Latency (ms) — grows with load and contention
    latency_lte  = round(5.0 + avg_lte_duty * tf * 15.0 + np.random.uniform(0, 3), 2) if lte_agents > 0 else 0.0
    latency_wifi = round(10.0 / max(0.01, wifi_airtime) * (1.0 + bc) + np.random.uniform(0, 5), 2) if wifi_aps > 0 else 0.0

    return {
        "category": profile["category"],
        "throughput": {
            "lte":   [round(t, 2) for t in throughput_lte],
            "wifi":  round(throughput_wifi, 2),
            "total": round(total, 2),
        },
        "fairness": round(fairness, 3),
        "packet_loss": {
            "lte":  packet_loss_lte,
            "wifi": packet_loss_wifi,
        },
        "latency": {
            "lte":  latency_lte,
            "wifi": latency_wifi,
        },
        "rl": {
            "duty_cycles": [round(dc, 2) for dc in duty_cycles],
            "reward":      round(reward, 2),
        },
    }
