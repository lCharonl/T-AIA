from __future__ import annotations

from .base_agent import BaseAgent


class BruteForceAgent(BaseAgent):
    """Naive baseline: picks a uniformly random action every step.

    No learning. By default we lift Gymnasium's TimeLimit (200) to 5000 so
    that we measure the *real* cost of a random policy on Taxi-v4 — the
    expected ~1800 mean steps and ~66 % eventual success rate. With the
    default TimeLimit, brute force would falsely appear to "fail in 198
    steps", which is just the truncation, not a real failure.
    """

    name = "Brute Force"

    def __init__(self, env_id: str = "Taxi-v4", seed: int = 0,
                 max_episode_steps: int = 3000):
        super().__init__(env_id=env_id, seed=seed,
                         max_episode_steps=max_episode_steps)
        self.epsilon = 1.0  # always exploring, never exploiting

    def select_action(self, state: int, training: bool = True) -> int:
        return int(self.rng.integers(0, self.n_actions))
