import random

class SimpleRLAgent:
    def __init__(self):
        self.actions = [0.2, 0.4, 0.6, 0.8, 1.0]
        self.q_table = {}

    def get_state_key(self, state):
        return tuple(state)

    def select_action(self, state):
        key = self.get_state_key(state)

        if key not in self.q_table:
            self.q_table[key] = [0] * len(self.actions)

        # ε-greedy
        if random.random() < 0.2:
            return random.choice(self.actions)

        return self.actions[self.q_table[key].index(max(self.q_table[key]))]

    def update(self, state, action, reward):
        key = self.get_state_key(state)
        idx = self.actions.index(action)

        self.q_table[key][idx] += 0.1 * (reward - self.q_table[key][idx])