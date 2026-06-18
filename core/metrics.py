from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable

import numpy as np


def moving_average(values: Iterable[float], window: int = 100) -> list[float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return []
    window = max(1, min(window, arr.size))
    cumsum = np.cumsum(np.insert(arr, 0, 0.0))
    out = np.empty(arr.size, dtype=np.float64)
    for i in range(arr.size):
        start = max(0, i + 1 - window)
        out[i] = (cumsum[i + 1] - cumsum[start]) / (i + 1 - start)
    return out.tolist()


def summarize_history(history: dict) -> dict:
    rewards = history.get("rewards", [])
    steps = history.get("steps", [])
    if not rewards:
        return {
            "episodes": 0,
            "mean_reward": 0.0,
            "mean_steps": 0.0,
            "last_reward": 0.0,
            "last_steps": 0,
            "best_reward": 0.0,
        }
    return {
        "episodes": len(rewards),
        "mean_reward": float(np.mean(rewards)),
        "mean_steps": float(np.mean(steps)),
        "last_reward": float(rewards[-1]),
        "last_steps": int(steps[-1]),
        "best_reward": float(np.max(rewards)),
    }


def compare_algorithms(results: dict[str, dict]) -> list[dict]:
    """Build comparison rows from a `{algo_name: result_dict}` mapping.

    Each value should contain at least:
        - "eval": dict from `BaseAgent.evaluate`
        - "train_time": float (seconds)
        - "infer_time": float (seconds for the evaluation pass)
    """
    rows = []
    for algo, payload in results.items():
        ev = payload.get("eval", {})
        mean_steps = ev.get("mean_steps", 0.0)
        std_steps = ev.get("std_steps", 0.0)
        mean_r = ev.get("mean_reward", 0.0)
        std_r = ev.get("std_reward", 0.0)
        rows.append({
            "Algorithm": algo,
            "Mean steps": round(mean_steps, 2),
            "Steps ±std": f"{mean_steps:.1f} ± {std_steps:.1f}",
            "Mean reward": round(mean_r, 2),
            "Reward ±std": f"{mean_r:.1f} ± {std_r:.1f}",
            "Success rate": round(ev.get("success_rate", 0.0) * 100, 1),
            "Train time (s)": round(payload.get("train_time", 0.0), 2),
            "Infer time (s)": round(payload.get("infer_time", 0.0), 3),
        })
    return rows


@contextmanager
def stopwatch():
    """Yields a list that, on exit, contains the elapsed seconds at index 0."""
    import time
    holder = [0.0]
    start = time.perf_counter()
    try:
        yield holder
    finally:
        holder[0] = time.perf_counter() - start
