from __future__ import annotations

from typing import Optional

import numpy as np

from .base_agent import BaseAgent


class QLearningAgent(BaseAgent):
    """Tabular Q-Learning (off-policy TD control).

    Update rule:
        Q(s, a) ← Q(s, a) + α [r + γ · max_a' Q(s', a') − Q(s, a)]

    Exploration: ε-greedy with exponential decay clipped at `epsilon_min`.
    """

    name = "Q-Learning"

    def __init__(
        self,
        env_id: str = "Taxi-v3",
        seed: int = 0,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.9995,
        epsilon_min: float = 0.01,
    ):
        super().__init__(env_id=env_id, seed=seed)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_start = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = np.zeros((self.n_states, self.n_actions), dtype=np.float64)

    def select_action(self, state: int, training: bool = True) -> int:
        if training and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.n_actions))
        # Ties broken randomly to avoid getting stuck on action 0 early on.
        q_row = self.q_table[state]
        max_q = q_row.max()
        best = np.flatnonzero(q_row == max_q)
        return int(self.rng.choice(best))

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
        next_action: Optional[int] = None,
    ) -> None:
        target = reward
        if not done:
            target += self.gamma * self.q_table[next_state].max()
        td_error = target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error

    def on_episode_end(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
