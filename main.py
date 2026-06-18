"""CLI entry point — train and evaluate agents without the dashboard.

Examples:
    python main.py --algo q-learning --train 2000 --test 100
    python main.py --algo brute-force --train 0 --test 100
    python main.py --algo sarsa --train 2000 --test 100 --mode time-limited
    python main.py --compare --train 2000 --test 100
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from agents import ALGO_REGISTRY
from core import (
    DEFAULT_USER_PARAMS,
    OPTIMIZED_PARAMS,
    build_agent,
    compare_algorithms,
    summarize_history,
    train_headless,
    time_it,
)

ALGO_ALIASES = {
    "brute-force": "Brute Force",
    "bruteforce": "Brute Force",
    "q-learning": "Q-Learning",
    "qlearning": "Q-Learning",
    "ql": "Q-Learning",
    "sarsa": "SARSA",
    "deep-q-learning": "Deep Q-Learning",
    "deepqlearning": "Deep Q-Learning",
    "dqn": "Deep Q-Learning",
}


def resolve_algo(name: str) -> str:
    key = name.strip().lower()
    if key in ALGO_ALIASES:
        return ALGO_ALIASES[key]
    for algo in ALGO_REGISTRY:
        if algo.lower() == key:
            return algo
    raise SystemExit(f"Unknown algorithm '{name}'. Choose from {list(ALGO_REGISTRY)}.")


def run_one(algo_name: str, train_episodes: int, test_episodes: int, mode: str, seed: int) -> dict:
    params_table = OPTIMIZED_PARAMS if mode == "time-limited" else DEFAULT_USER_PARAMS
    params = params_table.get(algo_name, {})

    print(f"\n──────── {algo_name} ({mode}) ────────")
    print(f"params = {params}")
    agent = build_agent(algo_name, params=params, seed=seed)

    if train_episodes > 0:
        print(f"Training for {train_episodes} episodes…")
        history, train_time = train_headless(agent, episodes=train_episodes)
        s = summarize_history(history)
        print(
            f"  → trained in {train_time:.2f}s | mean reward {s['mean_reward']:.2f} "
            f"| mean steps {s['mean_steps']:.1f} | best reward {s['best_reward']:.1f}"
        )
        first_100 = float(np.mean(history["rewards"][:100])) if len(history["rewards"]) >= 1 else 0
        last_100 = float(np.mean(history["rewards"][-100:])) if len(history["rewards"]) >= 1 else 0
        print(f"  → reward first 100 = {first_100:.2f} | last 100 = {last_100:.2f}")
    else:
        train_time = 0.0
        print("(no training — brute force baseline)")

    print(f"Evaluating on {test_episodes} episodes…")
    with time_it() as t_infer:
        ev = agent.evaluate(episodes=test_episodes)
    print(
        f"  → reward {ev['mean_reward']:.2f} ± {ev.get('std_reward', 0):.2f} | "
        f"steps {ev['mean_steps']:.1f} ± {ev.get('std_steps', 0):.1f} | "
        f"success {ev['success_rate']*100:.1f}% | infer {t_infer[0]:.3f}s"
    )

    return {
        "algo": algo_name,
        "eval": ev,
        "train_time": train_time,
        "infer_time": t_infer[0],
    }


def print_comparison_table(rows: list[dict]) -> None:
    if not rows:
        return
    cols = list(rows[0].keys())
    widths = {c: max(len(c), max(len(str(r[c])) for r in rows)) for c in cols}
    line = " | ".join(c.ljust(widths[c]) for c in cols)
    print("\n" + line)
    print("-" * len(line))
    for r in rows:
        print(" | ".join(str(r[c]).ljust(widths[c]) for c in cols))


def main() -> int:
    parser = argparse.ArgumentParser(description="Taxi-v3 RL training & evaluation CLI")
    parser.add_argument("--algo", default="q-learning",
                        help="brute-force | q-learning | sarsa (ignored if --compare)")
    parser.add_argument("--train", type=int, default=2000, help="number of training episodes")
    parser.add_argument("--test", type=int, default=100, help="number of test (eval) episodes")
    parser.add_argument("--mode", choices=["user", "time-limited"], default="time-limited",
                        help="user mode = default params, time-limited = pre-optimized")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compare", action="store_true",
                        help="run all algorithms and print the comparison table")
    args = parser.parse_args()

    if args.compare:
        results: dict[str, dict] = {}
        # Brute force never trains.
        results["Brute Force"] = run_one("Brute Force", 0, args.test, args.mode, args.seed)
        for name in ("Q-Learning", "SARSA", "Deep Q-Learning"):
            r = run_one(name, args.train, args.test, args.mode, args.seed)
            results[name] = r
        print_comparison_table(compare_algorithms({k: v for k, v in results.items()}))
    else:
        algo_name = resolve_algo(args.algo)
        train = 0 if algo_name == "Brute Force" else args.train
        run_one(algo_name, train, args.test, args.mode, args.seed)

    return 0


if __name__ == "__main__":
    sys.exit(main())
