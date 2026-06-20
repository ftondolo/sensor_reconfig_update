"""DQN Agent for adaptive resolution/framerate control.

Standard DQN with:
  - Experience replay buffer
  - Target network (soft update)
  - Epsilon-greedy exploration with decay
  - Dueling architecture (optional)
"""

import random, os, json
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from .environment import STATE_DIM, NUM_ACTIONS, RES_TIERS, FPS_TIERS, action_to_tiers
except ImportError:
    from environment import STATE_DIM, NUM_ACTIONS, RES_TIERS, FPS_TIERS, action_to_tiers


class QNetwork(nn.Module):
    """Dueling DQN: shared feature layers → value stream + advantage stream."""

    def __init__(self, state_dim=STATE_DIM, n_actions=NUM_ACTIONS, hidden=128):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        # Value stream
        self.value = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        # Advantage stream
        self.advantage = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x):
        feat = self.feature(x)
        val = self.value(feat)            # [B, 1]
        adv = self.advantage(feat)        # [B, n_actions]
        # Dueling: Q = V + (A - mean(A))
        q = val + adv - adv.mean(dim=1, keepdim=True)
        return q


class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards, dtype=np.float32),
                np.array(next_states), np.array(dones, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """DQN agent with epsilon-greedy exploration and target network."""

    def __init__(self, state_dim=STATE_DIM, n_actions=NUM_ACTIONS,
                 lr=1e-3, gamma=0.99, tau=0.005,
                 eps_start=1.0, eps_end=0.05, eps_decay=5000,
                 buffer_size=50000, batch_size=64,
                 device=None):
        self.n_actions = n_actions
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.steps_done = 0

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.q_net = QNetwork(state_dim, n_actions).to(self.device)
        self.target_net = QNetwork(state_dim, n_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)

    @property
    def epsilon(self):
        return self.eps_end + (self.eps_start - self.eps_end) * \
               np.exp(-self.steps_done / self.eps_decay)

    def select_action(self, state, greedy=False):
        """Epsilon-greedy action selection."""
        if not greedy and random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)

        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            q_values = self.q_net(state_t)
            return q_values.argmax(dim=1).item()

    def store(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)
        self.steps_done += 1

    def train_step(self):
        """One gradient step on a batch from replay buffer."""
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states_t = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions_t = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states_t = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32).to(self.device)

        # Current Q values
        q_values = self.q_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Double DQN: use q_net to select action, target_net to evaluate
        with torch.no_grad():
            next_actions = self.q_net(next_states_t).argmax(dim=1)
            next_q = self.target_net(next_states_t).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards_t + self.gamma * next_q * (1 - dones_t)

        loss = nn.functional.smooth_l1_loss(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        # Soft update target network
        for p, tp in zip(self.q_net.parameters(), self.target_net.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        return loss.item()

    def save(self, path):
        """Save agent weights and config."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "steps_done": self.steps_done,
            "config": {
                "gamma": self.gamma, "tau": self.tau,
                "eps_start": self.eps_start, "eps_end": self.eps_end,
                "eps_decay": self.eps_decay,
            }
        }, path)
        print(f"[Agent] Saved to {path}")

    def load(self, path):
        """Load agent weights."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.steps_done = ckpt.get("steps_done", 0)
        print(f"[Agent] Loaded from {path}, steps={self.steps_done}")

    def get_policy_table(self):
        """Generate a human-readable policy table for common states."""
        table = []
        for scenario in ["day", "night", "adverse"]:
            for rgb_c in [0.6, 0.3, 0.1]:
                therm_c = 1.0 - rgb_c - 0.1
                radar_c = 0.1

                state = np.array([
                    rgb_c, therm_c, radar_c,
                    0.5,   # avg conf
                    0.1,   # n_dets normalized
                    0.0, 0.0,  # current tiers (full)
                    1.0, 1.0,  # current fps (full)
                ], dtype=np.float32)

                action = self.select_action(state, greedy=True)
                rt, tt = action_to_tiers(action)
                table.append({
                    "scenario": scenario,
                    "rgb_contrib": rgb_c, "therm_contrib": therm_c,
                    "rgb_res": RES_TIERS[rt][1], "rgb_fps": FPS_TIERS[rt],
                    "therm_res": RES_TIERS[tt][1], "therm_fps": FPS_TIERS[tt],
                })
        return table
