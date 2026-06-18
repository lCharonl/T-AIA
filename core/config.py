from __future__ import annotations

from agents import ALGO_REGISTRY, BaseAgent

# ── Time-limited (pre-optimized) parameters ───────────────────────────────
# These were found by manual sweeping and are the values the dashboard's
# "Time-Limited" mode locks in to converge as fast as possible.
OPTIMIZED_PARAMS = {
    "Q-Learning": {
        "alpha": 0.4,
        "gamma": 0.99,
        "epsilon": 1.0,
        "epsilon_decay": 0.999,
        "epsilon_min": 0.01,
    },
    "SARSA": {
        # Tuned to match the presentation's slide 9: 13.2 steps / 100 %.
        # Needs ~8000 episodes — on-policy with ε_min=0.005 is required so
        # the policy actually fully exploits in the late training phase.
        "alpha": 0.4,
        "gamma": 0.99,
        "epsilon": 1.0,
        "epsilon_decay": 0.9995,
        "epsilon_min": 0.005,
    },
    "Brute Force": {},
    "Deep Q-Learning": {
        # Tuned: 13.72 steps / 99.5 % success / +7.17 reward on 4000 episodes,
        # 287 s on CPU. Matches the presentation's slide 9 target (13.4 / 99.8 %).
        # Insight from sweep: target_sync=200 with eps_decay=0.9997 collapses
        # (~60 % success) — the target net updates slower than the policy can
        # exploit. target_sync=100 + eps_decay=0.9995 is the sweet spot.
        "alpha": 1e-3,
        "gamma": 0.99,
        "epsilon": 1.0,
        "epsilon_decay": 0.9995,
        "epsilon_min": 0.02,
        "batch_size": 64,
        "buffer_size": 10_000,
        "target_sync": 100,
        "warmup": 1_000,
    },
}

# ── User mode defaults (the sliders' starting positions) ──────────────────
DEFAULT_USER_PARAMS = {
    "Q-Learning": {
        "alpha": 0.1,
        "gamma": 0.99,
        "epsilon": 1.0,
        "epsilon_decay": 0.9995,
        "epsilon_min": 0.01,
    },
    "SARSA": {
        "alpha": 0.1,
        "gamma": 0.99,
        "epsilon": 1.0,
        "epsilon_decay": 0.9995,
        "epsilon_min": 0.05,
    },
    "Brute Force": {},
    "Deep Q-Learning": {
        "alpha": 1e-3,
        "gamma": 0.99,
        "epsilon": 1.0,
        "epsilon_decay": 0.999,
        "epsilon_min": 0.05,
        "batch_size": 64,
        "buffer_size": 10_000,
        "target_sync": 200,
        "warmup": 500,
    },
}


# ── Named presets (slide-reproducible benchmarks) ────────────────────────
# Each preset bundles {params, train_episodes, label, description}. They are
# the "preset" rows of the dashboard and feed the slide-9 benchmark table.
PRESETS = {
    "Brute Force (no truncation)": {
        "algo": "Brute Force",
        "params": {"max_episode_steps": 3000},
        "train_episodes": 0,
        "description": "Random policy, TimeLimit lifted to 3000 → ~1700 steps / ~70 % success "
                       "(slide benchmark: 1835 / 66 %).",
    },
    "Q-Learning — non optimisé": {
        "algo": "Q-Learning",
        "params": {
            "alpha": 0.1, "gamma": 0.99,
            "epsilon": 1.0, "epsilon_decay": 0.98, "epsilon_min": 0.05,
        },
        "train_episodes": 800,
        "description": "Early-stage Q-Learning (too-fast ε decay, too-few episodes) → "
                       "~125 steps / ~40 % success (slide benchmark: 126 / 39 %).",
    },
    "Q-Learning — optimisé": {
        "algo": "Q-Learning",
        "params": OPTIMIZED_PARAMS["Q-Learning"],
        "train_episodes": 2000,
        "description": "Tuned Q-Learning → ~13 steps / 100 % success "
                       "(slide benchmark: 13.1 / 100 %).",
    },
    "SARSA — optimisé": {
        "algo": "SARSA",
        "params": OPTIMIZED_PARAMS["SARSA"],
        "train_episodes": 8000,
        "description": "Tuned SARSA → ~13 steps / 100 % success "
                       "(slide benchmark: 13.2 / 100 %).",
    },
    "Deep Q-Learning — optimisé": {
        "algo": "Deep Q-Learning",
        "params": OPTIMIZED_PARAMS["Deep Q-Learning"],
        "train_episodes": 4000,
        "description": "Tuned DQN → ~13.7 steps / 99.5 % success "
                       "(slide benchmark: 13.4 / 99.8 %).",
    },
}


def list_algorithms() -> list[str]:
    return list(ALGO_REGISTRY.keys())


def build_agent(name: str, params: dict | None = None, seed: int = 0) -> BaseAgent:
    if name not in ALGO_REGISTRY:
        raise ValueError(f"Unknown algorithm: {name}")
    cls = ALGO_REGISTRY[name]
    return cls(seed=seed, **(params or {}))
