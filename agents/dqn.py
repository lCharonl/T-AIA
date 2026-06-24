from __future__ import annotations

import os
# Disable torch.compile / dynamo / inductor pathways. Triton (which inductor
# pulls in on first backward) is not installed in this environment and the
# import fails. We only need eager-mode PyTorch — no JIT needed.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")

import random
from collections import deque
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_agent import BaseAgent


class _QNet(nn.Module):
    """Q(s, a; θ) with a small embedding for the 500 discrete Taxi states.

    Embedding → 16 dims → 64 ReLU → 64 ReLU → 6 (one Q-value per action).
    Lightweight enough to train on CPU in a couple of minutes.
    """

    def __init__(self, n_states: int = 500, n_actions: int = 6,
                 embed_dim: int = 16, hidden: int = 64):
        super().__init__()
        self.embed = nn.Embedding(n_states, embed_dim)
        self.fc1 = nn.Linear(embed_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, n_actions)

    def forward(self, states: torch.Tensor) -> torch.Tensor:  # states: (B,) int64
        x = self.embed(states)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.head(x)


class DQNAgent(BaseAgent):
    """Deep Q-Network with experience replay and a frozen target network.

    Surdimensionné pour Taxi-v3 (la table Q de 500×6 résout déjà le
    problème) mais l'algorithme converge vers le même optimum (~13 steps,
    ~99 % succès). Présent dans le projet pour illustrer la mise à
    l'échelle vers des espaces d'état trop grands pour une table.
    """

    name = "Deep Q-Learning"

    def __init__(
        self,
        env_id: str = "Taxi-v4",
        seed: int = 0,
        alpha: float = 5e-4,         # learning rate (Adam)
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.9995,
        epsilon_min: float = 0.05,
        batch_size: int = 64,
        buffer_size: int = 10_000,
        target_sync: int = 200,      # env steps between target-net syncs
        warmup: int = 1_000,         # env steps before first gradient step
        train_every: int = 1,
        embed_dim: int = 16,
        hidden: int = 64,
        max_episode_steps: Optional[int] = None,
    ):
        super().__init__(env_id=env_id, seed=seed,
                         max_episode_steps=max_episode_steps)
        # Determinism across numpy / torch / python's random.
        torch.manual_seed(seed)
        random.seed(seed)

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_start = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_sync = target_sync
        self.warmup = warmup
        self.train_every = train_every

        self.device = torch.device("cpu")
        self.policy_net = _QNet(self.n_states, self.n_actions, embed_dim, hidden).to(self.device)
        self.target_net = _QNet(self.n_states, self.n_actions, embed_dim, hidden).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=alpha)
        self.buffer: deque = deque(maxlen=buffer_size)
        self.env_step = 0

    # ── action selection ────────────────────────────────────────────────
    def select_action(self, state: int, training: bool = True) -> int:
        if training and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.n_actions))
        with torch.no_grad():
            s = torch.tensor([state], dtype=torch.long, device=self.device)
            q = self.policy_net(s).squeeze(0).cpu().numpy()
        max_q = q.max()
        best = np.flatnonzero(q == max_q)
        return int(self.rng.choice(best))

    # ── one gradient step on a sampled mini-batch ───────────────────────
    def _learn(self) -> None:
        if len(self.buffer) < self.batch_size:
            return
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states_t = torch.tensor(states, dtype=torch.long, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = torch.tensor(next_states, dtype=torch.long, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)

        q_sa = self.policy_net(states_t).gather(1, actions_t).squeeze(1)
        with torch.no_grad():
            max_q_next = self.target_net(next_states_t).max(dim=1).values
            target = rewards_t + (1.0 - dones_t) * self.gamma * max_q_next

        loss = F.smooth_l1_loss(q_sa, target)  # Huber → robust to spikes
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

    # ── update called every env step by BaseAgent.train_stream ──────────
    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
        next_action: Optional[int] = None,
    ) -> None:
        self.buffer.append((int(state), int(action), float(reward), int(next_state), bool(done)))
        self.env_step += 1
        if self.env_step >= self.warmup and self.env_step % self.train_every == 0:
            self._learn()
        if self.env_step % self.target_sync == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def on_episode_end(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
