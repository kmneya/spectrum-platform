import random

def run_baseline(lte_agents, wifi_aps, traffic, algorithm):
    """Run baseline algorithms (no RL)"""
    
    # Base throughput per agent
    base_lte = lte_agents * 10
    base_wifi = wifi_aps * 8
    
    # Apply algorithm effects
    if algorithm == "duty_cycle":
        lte = base_lte * 0.5
        wifi = base_wifi * 0.5
        packet_loss_lte = 5 + random.uniform(0, 3)
        packet_loss_wifi = 10 + random.uniform(0, 5)
    elif algorithm == "no_algorithm":
        lte = base_lte * 1.2
        wifi = base_wifi * 0.2
        packet_loss_lte = 3 + random.uniform(0, 2)
        packet_loss_wifi = 30 + random.uniform(0, 10)
    else:
        # Default
        lte = base_lte
        wifi = base_wifi
        packet_loss_lte = 5
        packet_loss_wifi = 15
    
    total = lte + wifi
    
    # Jain's Fairness Index (proper formula)
    if lte == 0 and wifi == 0:
        fairness = 0
    else:
        fairness = (lte + wifi) ** 2 / (2 * (lte**2 + wifi**2))
    
    return {
        "throughput": [round(lte, 2), round(wifi, 2), round(total, 2)],
        "fairness": [round(fairness, 3)],
        "packet_loss": [round(packet_loss_lte, 2), round(packet_loss_wifi, 2)],
        "rl": None  # No RL data for baseline
    }


# Global variable to store learning data (simple version)
reward_history = []

def run_madrl(lte_agents, wifi_aps, traffic):
    """Run MADRL simulation"""
    
    # In a real system, this would call your trained model
    # For now, we simulate learning behavior
    
    global reward_history
    
    # Simulate agent decision based on traffic
    # More traffic → LTE becomes more aggressive
    if traffic <= 2:
        duty_cycle = 0.3  # Polite (more WiFi friendly)
    elif traffic <= 4:
        duty_cycle = 0.6  # Moderate
    else:
        duty_cycle = 0.8  # Aggressive
    
    # Calculate throughput based on duty cycle
    lte_capacity = 10 * duty_cycle
    wifi_capacity = 8 * (1 - duty_cycle * 0.8)  # WiFi suffers when LTE is aggressive
    
    lte = lte_agents * lte_capacity
    wifi = wifi_aps * wifi_capacity
    
    total = lte + wifi
    
    # Jain's Fairness
    if lte == 0 and wifi == 0:
        fairness = 0
    else:
        fairness = (lte + wifi) ** 2 / (2 * (lte**2 + wifi**2))
    
    # Reward = throughput + fairness weight
    reward = total + 50 * fairness
    
    # Store for history (for graph)
    reward_history.append(reward)
    if len(reward_history) > 20:
        reward_history.pop(0)
    
    # Packet loss (lower is better)
    packet_loss_lte = 5 - (duty_cycle * 2) + random.uniform(-1, 2)
    packet_loss_wifi = 8 + (duty_cycle * 10) + random.uniform(-2, 3)
    
    return {
        "throughput": [round(lte, 2), round(wifi, 2), round(total, 2)],
        "fairness": [round(fairness, 3)],
        "packet_loss": [round(packet_loss_lte, 2), round(packet_loss_wifi, 2)],
        "rl": {
            "duty_cycle": round(duty_cycle, 2),
            "reward": round(reward, 2),
            "history": reward_history.copy()
        }
    }


def run_real_simulation(lte_agents, wifi_aps, traffic, algorithm):
    """Main entry point for simulation"""
    
    # Handle zero agents case
    if lte_agents == 0 and wifi_aps == 0:
        return {
            "throughput": [0, 0, 0],
            "fairness": [0],
            "packet_loss": [0, 0],
            "rl": None
        }
    
    if algorithm == "madrl":
        return run_madrl(lte_agents, wifi_aps, traffic)
    else:
        return run_baseline(lte_agents, wifi_aps, traffic, algorithm)