from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, Optional

import gymnasium as gym
import numpy as np


@dataclass
class StepEvent:
    """One step emitted by a training/evaluation generator.

    `frame` is an RGB array (H, W, 3) of the rendered environment, or None
    when rendering is disabled. `episode_done` is True on the final step of
    an episode (terminated or truncated). `episode_reward` / `episode_steps`
    carry the running totals for the current episode.
    """

    episode: int
    step: int
    state: int
    action: int
    reward: float
    next_state: int
    done: bool
    episode_done: bool
    episode_reward: float
    episode_steps: int
    epsilon: float
    frame: Optional[np.ndarray] = None


class BaseAgent:
    """Common interface for every agent in the project.

    Subclasses must implement `select_action`, `update` and `name`.
    The class provides shared training/evaluation loops + a streaming
    generator (`train_stream`) so the dashboard can show the taxi moving
    in real time.
    """

    name: str = "BaseAgent"

    def __init__(self, env_id: str = "Taxi-v4", seed: int = 0,
                 max_episode_steps: Optional[int] = None):
        self.env_id = env_id
        self.seed = seed
        # max_episode_steps overrides Gymnasium's default TimeLimit wrapper.
        # None ⇒ keep the env default (200 for Taxi-v3). Pass e.g. 5000 for
        # brute force to measure its real cost without truncation.
        self.max_episode_steps = max_episode_steps
        self.n_states = 500
        self.n_actions = 6
        self.rng = np.random.default_rng(seed)
        self.epsilon = 0.0
        self.history: dict = {
            "rewards": [],
            "steps": [],
            "epsilon": [],
        }

    # ── methods every agent must override ─────────────────────────────────
    def select_action(self, state: int, training: bool = True) -> int:
        raise NotImplementedError

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
        next_action: Optional[int] = None,
    ) -> None:
        # Default: no learning (used by brute force).
        pass

    def on_episode_end(self) -> None:
        pass

    # ── helpers ───────────────────────────────────────────────────────────
    def _make_env(self, render: bool):
        kwargs = dict(render_mode="rgb_array" if render else None)
        if self.max_episode_steps is not None:
            kwargs["max_episode_steps"] = self.max_episode_steps
        return gym.make(self.env_id, **kwargs)

    def _maybe_render(self, env, render: bool) -> Optional[np.ndarray]:
        if not render:
            return None
        try:
            return env.render()
        except Exception:
            return None

    # ── streaming training (yields every step) ────────────────────────────
    def train_stream(
        self,
        episodes: int,
        render_every: int = 10,
        max_steps: Optional[int] = None,
        seed_offset: int = 0,
    ) -> Generator[StepEvent, None, None]:
        """Train for `episodes` and yield a StepEvent for every step.

        Frames are only rendered for one episode out of `render_every`
        (and the very first + very last episode). When `render_every <= 0`,
        no frames are ever produced — useful for headless CLI training.

        `max_steps` is the Python-side safety cap on the inner loop. Defaults
        to `self.max_episode_steps` if set (so brute force without TimeLimit
        can run long episodes), else 200 (matches Gymnasium's TimeLimit).
        """
        if max_steps is None:
            max_steps = self.max_episode_steps if self.max_episode_steps is not None else 200
        env = self._make_env(render=render_every > 0)
        try:
            for ep in range(episodes):
                render_this = render_every > 0 and (
                    ep == 0 or ep == episodes - 1 or ep % render_every == 0
                )
                obs, _ = env.reset(seed=self.seed + seed_offset + ep)
                state = int(obs)
                action = self.select_action(state, training=True)
                ep_reward = 0.0
                ep_steps = 0

                for step in range(max_steps):
                    next_obs, reward, terminated, truncated, _ = env.step(action)
                    next_state = int(next_obs)
                    done = bool(terminated or truncated)
                    ep_reward += float(reward)
                    ep_steps += 1

                    next_action = (
                        self.select_action(next_state, training=True)
                        if not done
                        else 0
                    )
                    self.update(state, action, float(reward), next_state, done, next_action)

                    frame = self._maybe_render(env, render_this) if render_this else None

                    yield StepEvent(
                        episode=ep,
                        step=step,
                        state=state,
                        action=int(action),
                        reward=float(reward),
                        next_state=next_state,
                        done=done,
                        episode_done=done,
                        episode_reward=ep_reward,
                        episode_steps=ep_steps,
                        epsilon=float(self.epsilon),
                        frame=frame,
                    )

                    state = next_state
                    action = next_action
                    if done:
                        break

                self.history["rewards"].append(ep_reward)
                self.history["steps"].append(ep_steps)
                self.history["epsilon"].append(float(self.epsilon))
                self.on_episode_end()
        finally:
            env.close()

    # ── headless training (no yields, fast) ───────────────────────────────
    def train(self, episodes: int, max_steps: Optional[int] = None) -> dict:
        for _ in self.train_stream(episodes, render_every=0, max_steps=max_steps):
            pass
        return self.history

    # ── evaluation ────────────────────────────────────────────────────────
    def evaluate(
        self,
        episodes: int,
        max_steps: Optional[int] = None,
        render: bool = False,
        seed_offset: int = 10_000,
    ) -> dict:
        if max_steps is None:
            max_steps = self.max_episode_steps if self.max_episode_steps is not None else 200
        env = self._make_env(render=render)
        rewards, steps_list, successes, penalties_list = [], [], [], []
        frames_per_episode: list[list[np.ndarray]] = []
        try:
            for ep in range(episodes):
                obs, _ = env.reset(seed=self.seed + seed_offset + ep)
                state = int(obs)
                ep_reward, ep_steps, success = 0.0, 0, False
                ep_penalties = 0  # count of illegal pickup/dropoff (-10) actions
                frames: list[np.ndarray] = []
                if render:
                    f = self._maybe_render(env, True)
                    if f is not None:
                        frames.append(f)
                for _ in range(max_steps):
                    action = self.select_action(state, training=False)
                    next_obs, reward, terminated, truncated, _ = env.step(action)
                    state = int(next_obs)
                    ep_reward += float(reward)
                    ep_steps += 1
                    if reward <= -10:
                        ep_penalties += 1
                    if render:
                        f = self._maybe_render(env, True)
                        if f is not None:
                            frames.append(f)
                    if terminated:
                        success = True
                        break
                    if truncated:
                        break
                rewards.append(ep_reward)
                steps_list.append(ep_steps)
                successes.append(success)
                penalties_list.append(ep_penalties)
                if render:
                    frames_per_episode.append(frames)
        finally:
            env.close()

        return {
            "rewards": rewards,
            "steps": steps_list,
            "successes": successes,
            "penalties": penalties_list,
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "std_reward": float(np.std(rewards)) if rewards else 0.0,
            "mean_steps": float(np.mean(steps_list)) if steps_list else 0.0,
            "std_steps": float(np.std(steps_list)) if steps_list else 0.0,
            "success_rate": float(np.mean(successes)) if successes else 0.0,
            "total_penalties": int(sum(penalties_list)),
            "mean_penalties": float(np.mean(penalties_list)) if penalties_list else 0.0,
            "frames_per_episode": frames_per_episode,
        }

    # ── stream a single evaluation episode (for live replay) ──────────────
    def play_stream(
        self,
        max_steps: int = 200,
        seed: Optional[int] = None,
    ) -> Generator[StepEvent, None, None]:
        env = self._make_env(render=True)
        try:
            obs, _ = env.reset(seed=seed if seed is not None else self.seed + 99_999)
            state = int(obs)
            ep_reward, ep_steps = 0.0, 0
            frame = self._maybe_render(env, True)
            yield StepEvent(
                episode=0, step=0, state=state, action=-1, reward=0.0,
                next_state=state, done=False, episode_done=False,
                episode_reward=0.0, episode_steps=0,
                epsilon=float(self.epsilon), frame=frame,
            )
            for step in range(max_steps):
                action = self.select_action(state, training=False)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                next_state = int(next_obs)
                done = bool(terminated or truncated)
                ep_reward += float(reward)
                ep_steps += 1
                frame = self._maybe_render(env, True)
                yield StepEvent(
                    episode=0, step=step + 1, state=state, action=int(action),
                    reward=float(reward), next_state=next_state,
                    done=done, episode_done=done,
                    episode_reward=ep_reward, episode_steps=ep_steps,
                    epsilon=float(self.epsilon), frame=frame,
                )
                state = next_state
                if done:
                    break
        finally:
            env.close()
