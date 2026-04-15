import random
import json
import os

Q_TABLE_FILE = "q_table.json"

# Possible duty cycle actions
ACTIONS = [0.3, 0.5, 0.7, 0.9]

# Learning parameters
ALPHA = 0.1   # learning rate
GAMMA = 0.9   # discount factor
EPSILON = 0.2 # exploration rate


def load_q_table():
    if os.path.exists(Q_TABLE_FILE):
        with open(Q_TABLE_FILE, "r") as f:
            return json.load(f)
    return {}


def save_q_table(q_table):
    with open(Q_TABLE_FILE, "w") as f:
        json.dump(q_table, f)


def get_state(lte_agents, wifi_aps, traffic):
    return f"{lte_agents}_{wifi_aps}_{traffic}"


def choose_action(state, q_table):
    if random.uniform(0, 1) < EPSILON:
        return random.choice(ACTIONS)

    if state not in q_table:
        q_table[state] = {str(a): 0 for a in ACTIONS}

    return float(max(q_table[state], key=q_table[state].get))


def update_q_table(q_table, state, action, reward):
    action = str(action)

    if state not in q_table:
        q_table[state] = {str(a): 0 for a in ACTIONS}

    current_q = q_table[state][action]
    new_q = current_q + ALPHA * (reward - current_q)

    q_table[state][action] = new_q