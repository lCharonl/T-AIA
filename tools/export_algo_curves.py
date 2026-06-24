"""Entraîne réellement chaque algo une fois et exporte ses vraies courbes
de convergence vers web/algo_curves.json, consommé statiquement par le front
(pages « Algorithmes »).

Usage :
    .venv/Scripts/python.exe tools/export_algo_curves.py

Les courbes sont lissées (moyenne glissante 100) puis sous-échantillonnées à
60 points pour rester légères, exactement comme les charts de l'écran
Entraînement. À relancer si l'on veut rafraîchir les données mesurées.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

# Console Windows (cp1252) → forcer UTF-8 pour les caractères accentués/flèches.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gymnasium as gym  # noqa: E402

from core import OPTIMIZED_PARAMS, build_agent  # noqa: E402

OUT = ROOT / "web" / "algo_curves.json"
N_POINTS = 60
MA_WINDOW = 100


def _resolve_env() -> str:
    """Use Taxi-v3 (le défaut du projet) si la version installée de
    gymnasium l'expose encore, sinon bascule sur Taxi-v4 — tâche identique,
    même optimum (~13 pas / 100 %)."""
    for env_id in ("Taxi-v3", "Taxi-v4"):
        try:
            gym.make(env_id).close()
            return env_id
        except Exception:
            continue
    raise RuntimeError("Aucun environnement Taxi disponible")


ENV_ID = _resolve_env()


def smooth_bin(arr, n: int = N_POINTS, ma_window: int = MA_WINDOW) -> list[float]:
    """Moyenne glissante puis binning à n points (réplique le front)."""
    a = np.asarray(arr, dtype=float)
    if a.size == 0:
        return []
    cs = np.cumsum(np.insert(a, 0, 0.0))
    ma = np.empty(a.size)
    for i in range(a.size):
        lo = max(0, i - ma_window + 1)
        ma[i] = (cs[i + 1] - cs[lo]) / (i - lo + 1)
    if ma.size <= n:
        binned = ma
    else:
        step = ma.size / n
        binned = np.array([
            ma[int(i * step):max(int(i * step) + 1, int((i + 1) * step))].mean()
            for i in range(n)
        ])
    return [round(float(x), 2) for x in binned]


def train_steps(algo: str, episodes: int, params: dict, want_eps: bool = False):
    """Entraîne `algo` et renvoie (steps_binned, [epsilon_binned])."""
    t0 = time.perf_counter()
    agent = build_agent(algo, params={**params, "env_id": ENV_ID}, seed=0)
    agent.train(episodes=episodes)
    dt = time.perf_counter() - t0
    ev = agent.evaluate(episodes=100)
    print(f"  · {algo:16s} {episodes:5d} ép. → {ev['mean_steps']:6.1f} pas / "
          f"{ev['success_rate'] * 100:5.1f}% ({dt:.0f}s)")
    steps = smooth_bin(agent.history["steps"])
    if want_eps:
        eps = smooth_bin(agent.history["epsilon"], ma_window=1)
        return steps, eps
    return steps, None


def brute_histogram(n_episodes: int = 100):
    """Évalue la politique aléatoire (sans truncation) → histogramme des pas."""
    print("  · Brute Force      eval 100 ép. (sans truncation)…")
    t0 = time.perf_counter()
    agent = build_agent("Brute Force", params={"max_episode_steps": 3000, "env_id": ENV_ID}, seed=0)
    ev = agent.evaluate(episodes=n_episodes, max_steps=3000)
    dt = time.perf_counter() - t0
    steps = np.asarray(ev["steps"], dtype=float)
    edges = [0, 200, 500, 1000, 2000, 3000, np.inf]
    labels = ["0–200", "200–500", "500–1k", "1k–2k", "2k–3k", "3k+"]
    counts = [int(((steps >= edges[i]) & (steps < edges[i + 1])).sum())
              for i in range(len(labels))]
    print(f"    → moyenne {ev['mean_steps']:.0f} pas / "
          f"{ev['success_rate'] * 100:.0f}% réussite ({dt:.0f}s)")
    return {
        "labels": labels,
        "counts": counts,
        "mean_steps": round(ev["mean_steps"], 1),
        "success_rate": round(ev["success_rate"], 3),
        "n_episodes": n_episodes,
    }


def main() -> None:
    print(f"Export des courbes réelles (entraînements live · {ENV_ID})…")

    # ── Q-Learning : convergence + ε sur 2000 épisodes ──
    q_steps, q_eps = train_steps("Q-Learning", 2000, OPTIMIZED_PARAMS["Q-Learning"], want_eps=True)

    # ── SARSA vs Q-Learning sur le même axe (8000 ép.) ──
    sarsa_steps, _ = train_steps("SARSA", 8000, OPTIMIZED_PARAMS["SARSA"])
    q8000_steps, _ = train_steps("Q-Learning", 8000, OPTIMIZED_PARAMS["Q-Learning"])

    # ── DQN vs Q-Learning sur le même axe (4000 ép.) ──
    dqn_steps, _ = train_steps("Deep Q-Learning", 4000, OPTIMIZED_PARAMS["Deep Q-Learning"])
    q4000_steps, _ = train_steps("Q-Learning", 4000, OPTIMIZED_PARAMS["Q-Learning"])

    # ── Force brute : histogramme ──
    brute = brute_histogram(100)

    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M"),
        "env_id": ENV_ID,
        "n_points": N_POINTS,
        "brute": brute,
        "qlearning": {"episodes": 2000, "steps": q_steps, "epsilon": q_eps},
        "sarsa": {"episodes": 8000, "sarsa": sarsa_steps, "qlearning": q8000_steps},
        "dqn": {"episodes": 4000, "dqn": dqn_steps, "qlearning": q4000_steps},
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✓ Écrit : {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1024:.1f} Ko)")


if __name__ == "__main__":
    main()
