from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Callable, Optional

from agents import BaseAgent, StepEvent


@contextmanager
def time_it():
    """Context manager returning a single-element list whose [0] becomes the elapsed seconds."""
    holder = [0.0]
    start = time.perf_counter()
    try:
        yield holder
    finally:
        holder[0] = time.perf_counter() - start


def train_headless(agent: BaseAgent, episodes: int, max_steps: int = 200) -> tuple[dict, float]:
    """Train without rendering. Returns (history, elapsed_seconds)."""
    with time_it() as t:
        history = agent.train(episodes=episodes, max_steps=max_steps)
    return history, t[0]


def train_with_callback(
    agent: BaseAgent,
    episodes: int,
    on_step: Optional[Callable[[StepEvent], None]] = None,
    on_episode_end: Optional[Callable[[int, dict], None]] = None,
    render_every: int = 10,
    max_steps: int = 200,
    should_stop: Optional[Callable[[], bool]] = None,
) -> tuple[dict, float]:
    """Train and invoke callbacks per step / per episode (used by the dashboard).

    Returns (history, elapsed_seconds). `should_stop` is polled between
    steps so the UI can request an early stop.
    """
    start = time.perf_counter()
    last_ep = -1
    for evt in agent.train_stream(
        episodes=episodes, render_every=render_every, max_steps=max_steps
    ):
        if on_step is not None:
            on_step(evt)
        if evt.episode_done and on_episode_end is not None:
            on_episode_end(evt.episode, agent.history)
            last_ep = evt.episode
        if should_stop is not None and should_stop():
            break
    elapsed = time.perf_counter() - start
    # Guarantee a final episode callback even if the last episode was truncated.
    if on_episode_end is not None and last_ep != episodes - 1 and agent.history.get("rewards"):
        on_episode_end(len(agent.history["rewards"]) - 1, agent.history)
    return agent.history, elapsed
