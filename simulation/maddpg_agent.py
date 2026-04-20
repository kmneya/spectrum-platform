# maddpg_agent.py
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# ---------------------------
# Actor and Critic Networks
# ---------------------------
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Sigmoid()   # duty cycle in [0,1]
        )

    def forward(self, state):
        return self.net(state)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, num_agents, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim * num_agents + action_dim * num_agents, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, states, actions):
        # states: (batch, num_agents, state_dim)
        # actions: (batch, num_agents, action_dim)
        batch_size = states.size(0)
        x = torch.cat([states.view(batch_size, -1), actions.view(batch_size, -1)], dim=-1)
        return self.net(x)

# ---------------------------
# MADDPG Agent Wrapper
# ---------------------------
class MADDPG:
    def __init__(self, num_agents, state_dim, action_dim,
                 lr_actor=1e-3, lr_critic=1e-3, gamma=0.95, tau=0.01):
        self.num_agents = num_agents
        self.gamma = gamma
        self.tau = tau

        self.actors = [Actor(state_dim, action_dim) for _ in range(num_agents)]
        self.target_actors = [Actor(state_dim, action_dim) for _ in range(num_agents)]
        self.critic = Critic(state_dim, action_dim, num_agents)
        self.target_critic = Critic(state_dim, action_dim, num_agents)

        # Copy parameters to target networks
        for i in range(num_agents):
            self.target_actors[i].load_state_dict(self.actors[i].state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizers = [optim.Adam(actor.parameters(), lr=lr_actor) for actor in self.actors]
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.memory = deque(maxlen=100000)
        self.batch_size = 64

    def act(self, observations, noise=0.1):
        """Return continuous actions for all agents."""
        actions = []
        for i, obs in enumerate(observations):
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            action = self.actors[i](obs_t).detach().numpy().flatten()
            # Add exploration noise
            action += noise * np.random.randn(*action.shape)
            action = np.clip(action, 0.0, 1.0)
            actions.append(action)
        return np.array(actions)

    def store_transition(self, obs, actions, rewards, next_obs, dones):
        self.memory.append((obs, actions, rewards, next_obs, dones))

    def update(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)

        obs = torch.FloatTensor(np.array(obs))
        actions = torch.FloatTensor(np.array(actions))
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(-1)
        next_obs = torch.FloatTensor(np.array(next_obs))
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(-1)

        # ---------------------
        # Update Critic
        # ---------------------
        with torch.no_grad():
            next_actions = torch.stack([self.target_actors[i](next_obs[:, i]) for i in range(self.num_agents)], dim=1)
            target_q = self.target_critic(next_obs, next_actions)
            target_q = rewards + self.gamma * (1 - dones) * target_q

        current_q = self.critic(obs, actions)
        critic_loss = nn.MSELoss()(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # ---------------------
        # Update Actors
        # ---------------------
        for i in range(self.num_agents):
            # Freeze critic parameters for actor update
            for p in self.critic.parameters():
                p.requires_grad = False

            # Actor loss = -Q(s, a1..aN) with a_i from current actor
            new_actions = actions.clone()
            new_actions[:, i] = self.actors[i](obs[:, i])
            actor_loss = -self.critic(obs, new_actions).mean()

            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(), 1.0)
            self.actor_optimizers[i].step()

            # Unfreeze critic
            for p in self.critic.parameters():
                p.requires_grad = True

        # ---------------------
        # Soft Update Targets
        # ---------------------
        for target, source in zip(self.target_critic.parameters(), self.critic.parameters()):
            target.data.copy_(self.tau * source.data + (1 - self.tau) * target.data)
        for i in range(self.num_agents):
            for target, source in zip(self.target_actors[i].parameters(), self.actors[i].parameters()):
                target.data.copy_(self.tau * source.data + (1 - self.tau) * target.data)

    def save(self, path):
        torch.save({
            'actors': [actor.state_dict() for actor in self.actors],
            'critic': self.critic.state_dict()
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        for i, state_dict in enumerate(checkpoint['actors']):
            self.actors[i].load_state_dict(state_dict)
            self.target_actors[i].load_state_dict(state_dict)
        self.critic.load_state_dict(checkpoint['critic'])
        self.target_critic.load_state_dict(checkpoint['critic'])