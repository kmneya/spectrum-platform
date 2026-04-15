import random
from .rl_agent import (
    load_q_table,
    save_q_table,
    get_state,
    choose_action,
    update_q_table
)

reward_history = []

def run_real_simulation(lte_agents, wifi_aps, traffic, algorithm):

    lte_capacity = 10
    wifi_capacity = 8

    base_lte = lte_agents * lte_capacity
    base_wifi = wifi_aps * wifi_capacity

    traffic_factor = 1 + (traffic * 0.2)

    # -----------------------------
    # RL LOGIC
    # -----------------------------
    duty_cycle = 0.5

    if algorithm == "madrl":
        q_table = load_q_table()

        state = get_state(lte_agents, wifi_aps, traffic)

        duty_cycle = choose_action(state, q_table)

    elif algorithm == "duty_cycle":
        duty_cycle = 0.5

    elif algorithm == "no_algorithm":
        duty_cycle = 0.9

    # -----------------------------
    # THROUGHPUT
    # -----------------------------

    throughput_lte = base_lte * duty_cycle * traffic_factor
    throughput_wifi = base_wifi * (1 - duty_cycle) * traffic_factor

    # Add noise
    throughput_lte += random.uniform(0, 5)
    throughput_wifi += random.uniform(0, 5)

    total = throughput_lte + throughput_wifi

    # -----------------------------
    # FAIRNESS (JAIN)
    # -----------------------------

    denominator = 2 * (throughput_lte**2 + throughput_wifi**2)
    fairness = (total ** 2) / denominator if denominator != 0 else 0

    # -----------------------------
    # REWARD FUNCTION (CRITICAL)
    # -----------------------------

    reward = total * 0.6 + fairness * 100

    reward_history.append(float(reward))

    # Keep last 30 points
    if len(reward_history) > 30:
        reward_history.pop(0)

    # -----------------------------
    # UPDATE RL
    # -----------------------------

    if algorithm == "madrl":
        update_q_table(q_table, state, duty_cycle, reward)
        save_q_table(q_table)

    # -----------------------------
    # PACKET LOSS
    # -----------------------------

    packet_loss_lte = random.uniform(1, 10)
    packet_loss_wifi = random.uniform(5, 15) * (1 - duty_cycle)

    return {
        "throughput": [
            round(throughput_lte, 2),
            round(throughput_wifi, 2),
            round(total, 2)
        ],
        "fairness": [round(fairness, 3)],
        "packet_loss": [
            round(packet_loss_lte, 2),
            round(packet_loss_wifi, 2)
        ],
        "rl": {
            "duty_cycle": round(duty_cycle, 2),
            "reward": round(reward, 2),
            "history": reward_history
        }
    }