"""Helpers around the agent layer: state decoding, episode rollout, shaping."""
from __future__ import annotations

import base64
import io
import time
from typing import Any, Generator, Optional

import gymnasium as gym
import numpy as np
from PIL import Image

from agents import ALGO_REGISTRY, BaseAgent
from core import build_agent

ACTION_NAMES = {
    0: "Sud ↓",
    1: "Nord ↑",
    2: "Est →",
    3: "Ouest ←",
    4: "Prise en charge",
    5: "Dépose ✓",
}

# Default landmarks (row, col) for the 4 destinations on Taxi-v4.
LANDMARKS = {0: (0, 0), 1: (0, 4), 2: (4, 0), 3: (4, 3)}


def decode_state(obs: int) -> dict[str, int]:
    """Decode a Taxi-v4 observation int into its 4 components."""
    dest_idx = obs % 4
    obs //= 4
    pass_loc = obs % 5
    obs //= 5
    taxi_col = obs % 5
    taxi_row = obs // 5
    return {
        "taxi_row": int(taxi_row),
        "taxi_col": int(taxi_col),
        "pass_loc": int(pass_loc),
        "dest_idx": int(dest_idx),
    }


def manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def potential(state: int, lam: float = 1.0) -> float:
    """Φ(s) = -λ · Manhattan(taxi, current target).

    Current target = passenger location if not yet picked up, else destination.
    Used for Ng et al. 1999 potential-based shaping:
        r' = r + γ·Φ(s') − Φ(s)
    """
    s = decode_state(state)
    if s["pass_loc"] < 4:  # passenger not in taxi
        target = LANDMARKS[s["pass_loc"]]
    else:
        target = LANDMARKS[s["dest_idx"]]
    taxi = (s["taxi_row"], s["taxi_col"])
    return -lam * manhattan(taxi, target)


def _frame_to_b64(
    arr: np.ndarray,
    *,
    jpeg: bool = False,
    quality: int = 70,
    half_res: bool = False,
) -> str:
    """Encode an (H,W,3) RGB numpy array as a base64 data URI.

    Defaults to PNG full-res (~65 KB for Taxi-v4) for the Environnement
    rollout where pixel-perfectness matters and the trace is short.

    For WebSocket-streamed training samples (~80 frames × 12 samples),
    pass `jpeg=True, half_res=True` to get ~10 KB per frame → ~800 KB
    per sample, safely under the WS 1 MB limit.
    """
    img = Image.fromarray(arr.astype(np.uint8), mode="RGB")
    if half_res:
        img = img.resize((img.width // 2, img.height // 2), Image.NEAREST)
    buf = io.BytesIO()
    if jpeg:
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    img.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# ── In-memory cache of trained agents ────────────────────────────────────
# Key = (algo, sorted-params, train_episodes, seed). Value = trained agent.
# Lets the Environnement screen replay an episode instantly when the same
# (algo, params) was already trained — either by /api/episode itself or
# during the SQLite-persisted training history.
import hashlib
import json
from threading import Lock

_AGENT_CACHE: dict[str, BaseAgent] = {}
_AGENT_LOCK = Lock()

# Cache des épisodes déjà rendus. Clef = (algo, params, train_ep, seed,
# episode_seed_offset). Valeur = dict {trace, frames, solved}. Premier
# clic = calcul (~1.5s pour rollout + render + base64), clics suivants
# avec MÊMES paramètres = retour immédiat (~5ms). Le cache est borné
# pour éviter une fuite mémoire si l'utilisateur joue avec beaucoup de
# seeds différents.
_EPISODE_CACHE: dict[str, dict] = {}
_EPISODE_LOCK = Lock()
_EPISODE_CACHE_MAX = 32


def _norm_params(params: dict) -> dict:
    # Le navigateur sérialise 1.0 en "1" (JS confond int/float). Côté serveur
    # json.dumps(1.0) = "1.0". On normalise tout en float pour que la clé
    # MD5 soit identique quelle que soit la source (curl vs browser).
    out: dict = {}
    for k, v in (params or {}).items():
        if isinstance(v, bool):
            out[k] = v
        elif isinstance(v, (int, float)):
            out[k] = float(v)
        else:
            out[k] = v
    return out


def _agent_key(algo: str, params: dict, train_episodes: int, seed: int) -> str:
    payload = json.dumps(
        [algo, _norm_params(params), train_episodes, seed],
        sort_keys=True, default=str,
    )
    return hashlib.md5(payload.encode()).hexdigest()


def _episode_key(algo: str, params: dict, train_episodes: int, seed: int,
                 episode_seed_offset: int) -> str:
    payload = json.dumps(
        [algo, _norm_params(params), train_episodes, seed, episode_seed_offset],
        sort_keys=True, default=str,
    )
    return hashlib.md5(payload.encode()).hexdigest()


class AgentNotReady(Exception):
    """Levée quand on demande un agent qui n'est pas en cache et que
    `allow_train=False` interdit de l'entraîner à la volée."""


def _get_or_train_agent(
    algo: str,
    params: dict,
    train_episodes: int,
    seed: int,
    allow_train: bool = True,
) -> BaseAgent:
    key = _agent_key(algo, params or {}, train_episodes, seed)
    with _AGENT_LOCK:
        agent = _AGENT_CACHE.get(key)
    if agent is not None:
        return agent
    if not allow_train:
        raise AgentNotReady(
            f"L'agent {algo} ({train_episodes} ép.) n'est pas encore prêt. "
            f"Attendez la fin du warmup (peut prendre ~5 min pour DQN au premier lancement)."
        )
    agent = build_agent(algo, params=params or {}, seed=seed)
    if train_episodes > 0:
        agent.train(episodes=train_episodes)
    with _AGENT_LOCK:
        _AGENT_CACHE[key] = agent
        # Bound the cache: keep at most 8 trained agents.
        if len(_AGENT_CACHE) > 8:
            oldest_key = next(iter(_AGENT_CACHE))
            del _AGENT_CACHE[oldest_key]
    return agent


# ── Episode rollout (for the Environnement screen) ────────────────────────
def rollout_episode(
    algo: str,
    params: dict[str, Any] | None,
    *,
    seed: int = 0,
    train_episodes: int = 0,
    max_steps: Optional[int] = None,
    include_frames: bool = True,
    episode_seed_offset: int = 99_999,
    allow_train: bool = True,
) -> dict[str, Any]:
    """Train (or fetch from cache), then play one greedy episode. Returns
    the full step trace + (optionally) the Gymnasium-rendered RGB frames
    as base64 PNG strings — usable directly as <img src=…>.

    Episode-level cache : si la même requête (algo, params, train_ep,
    seed, episode_seed_offset) a déjà été calculée, retour immédiat
    sans re-rendre les frames.

    allow_train=False : si l'agent n'est pas en cache, lève AgentNotReady
    au lieu de l'entraîner. Utilisé par /api/episode pour empêcher des
    trainings parallèles déclenchés par des clics utilisateur."""
    if include_frames:
        ep_key = _episode_key(algo, params or {}, train_episodes, seed, episode_seed_offset)
        with _EPISODE_LOCK:
            cached = _EPISODE_CACHE.get(ep_key)
        if cached is not None:
            return cached

    agent: BaseAgent = _get_or_train_agent(
        algo, params or {}, train_episodes, seed, allow_train=allow_train
    )

    # Brute Force: lift the 200-step TimeLimit so a truly random agent
    # has a chance to deliver, otherwise we just stop on truncation.
    if algo == "Brute Force":
        env = gym.make("Taxi-v4", render_mode="rgb_array", max_episode_steps=300)
        if max_steps is None:
            max_steps = 300
    else:
        env = gym.make("Taxi-v4", render_mode="rgb_array")
        if max_steps is None:
            max_steps = 200

    # Brute Force = 300 steps → 300 frames. PNG full-res serait ~45 s à
    # encoder et ~25 MB de payload. On garde JPEG mais qualité haute (90)
    # SANS resize : visuellement quasi identique au PNG, ~4x plus rapide,
    # ~6x plus léger. Les algos appris (~15 frames) restent en PNG full-res.
    use_jpeg = (algo == "Brute Force")

    def _encode(f):
        if use_jpeg:
            return _frame_to_b64(f, jpeg=True, half_res=False, quality=90)
        return _frame_to_b64(f)

    try:
        obs, _ = env.reset(seed=seed + episode_seed_offset)
        state = int(obs)
        frame0 = env.render() if include_frames else None
        trace = [{
            "step": 0, "action": -1, "action_name": "—", "reward": 0.0,
            **decode_state(state), "done": False, "carrying": False,
        }]
        frames: list[str] = [_encode(frame0)] if frame0 is not None else []
        cum = 0.0
        carrying = False
        for step in range(max_steps):
            action = agent.select_action(state, training=False)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            state = int(next_obs)
            cum += float(reward)
            if action == 4 and reward != -10:  # legal pickup
                carrying = True
            if action == 5 and reward != -10:  # legal dropoff
                carrying = False
            done = bool(terminated or truncated)
            trace.append({
                "step": step + 1,
                "action": int(action),
                "action_name": ACTION_NAMES.get(action, str(action)),
                "reward": float(reward),
                "cum_reward": float(cum),
                **decode_state(state),
                "done": done,
                "carrying": carrying,
            })
            if include_frames:
                f = env.render()
                if f is not None:
                    frames.append(_encode(f))
            if done:
                break
    finally:
        env.close()

    result = {
        "algo": algo,
        "trace": trace,
        "frames": frames,
        "solved": trace[-1]["done"],
    }
    # Mettre en cache pour éviter de re-rendre les frames au prochain clic
    # avec les mêmes paramètres. Borné à _EPISODE_CACHE_MAX entrées (FIFO).
    if include_frames:
        with _EPISODE_LOCK:
            _EPISODE_CACHE[ep_key] = result
            if len(_EPISODE_CACHE) > _EPISODE_CACHE_MAX:
                oldest = next(iter(_EPISODE_CACHE))
                del _EPISODE_CACHE[oldest]
    return result


# ── Streaming training generator (for /ws/train) ──────────────────────────
def stream_training(
    algo: str,
    params: dict[str, Any] | None,
    episodes: int,
    *,
    seed: int = 0,
    shaped: bool = False,
    shaping_lambda: float = 0.5,
) -> Generator[dict[str, Any], None, dict[str, Any]]:
    """Yield {type, ...} dicts as the agent trains.

    When `shaped=True`, applies Ng et al. potential-based shaping by
    intercepting `agent.update` and rewriting `reward` to
    `r + γ · Φ(s') − Φ(s)`. The policy this produces is theoretically
    identical to the un-shaped optimal policy (Ng et al. 1999).

    Yield protocol:
        {"type": "start", "algo", "episodes", "params"}
        {"type": "episode", "i", "reward", "steps", "epsilon"}  ← every episode
        {"type": "done", "history": {"rewards", "steps", "epsilon"}}
    """
    agent: BaseAgent = build_agent(algo, params=params or {}, seed=seed)

    # ── shaping: monkey-patch agent.update to rewrite the reward ──────────
    if shaped and hasattr(agent, "gamma"):
        original_update = agent.update
        gamma = float(getattr(agent, "gamma", 0.99))

        def shaped_update(state, action, reward, next_state, done, next_action=None):  # type: ignore[override]
            phi_s = potential(state, shaping_lambda)
            phi_sp = 0.0 if done else potential(next_state, shaping_lambda)
            shaped_r = reward + gamma * phi_sp - phi_s
            return original_update(state, action, shaped_r, next_state, done, next_action)

        agent.update = shaped_update  # type: ignore[assignment]

    yield {
        "type": "start",
        "algo": algo,
        "episodes": episodes,
        "params": params or {},
        "shaped": shaped,
    }

    # Sample one greedy rollout every `sample_every` training episodes for
    # the live mini-grid on the Entraînement screen. Cap the number of
    # samples to ~12 across the whole training so the WS doesn't drown
    # the front with frames (frames are JPEG ~12 KB each).
    sample_every = max(1, episodes // 12)
    import gymnasium as _gym
    sample_env = _gym.make("Taxi-v4", render_mode="rgb_array")

    def _sample_frames(ep_idx: int) -> dict[str, Any]:
        obs, _ = sample_env.reset(seed=42_000 + ep_idx)
        s = int(obs)
        frames = [_frame_to_b64(sample_env.render(), jpeg=True, half_res=True, quality=60)]
        cum = 0.0
        term = False
        # Cap at 60: a trained policy never needs more, an untrained one
        # gets truncated. Keeps every sample under ~800 KB (1 MB WS limit).
        for _ in range(60):
            a = agent.select_action(s, training=False)
            no, r, term, trunc, _ = sample_env.step(a)
            s = int(no)
            cum += float(r)
            frames.append(_frame_to_b64(sample_env.render(), jpeg=True, half_res=True, quality=60))
            if term or trunc:
                break
        return {
            "type": "sample",
            "i": ep_idx,
            "frames": frames,
            "cum_reward": cum,
            "solved": bool(term),
        }

    t0 = time.perf_counter()
    try:
        for evt in agent.train_stream(episodes=episodes, render_every=0):
            if evt.episode_done:
                yield {
                    "type": "episode",
                    "i": evt.episode,
                    "reward": float(evt.episode_reward),
                    "steps": int(evt.episode_steps),
                    "epsilon": float(evt.epsilon),
                }
                # Sample BEFORE the very first episode and every N afterwards.
                if evt.episode == 0 or (evt.episode + 1) % sample_every == 0 \
                   or evt.episode == episodes - 1:
                    yield _sample_frames(evt.episode)
    finally:
        sample_env.close()
    elapsed = time.perf_counter() - t0
    return {
        "type": "done",
        "history": dict(agent.history),
        "train_time_s": elapsed,
        "agent": agent,
    }


def evaluate_agent(agent: BaseAgent, episodes: int = 100) -> dict[str, Any]:
    ev = agent.evaluate(episodes=episodes)
    return {
        "mean_steps": ev["mean_steps"],
        "mean_reward": ev["mean_reward"],
        "success_rate": ev["success_rate"],
        "std_steps": ev.get("std_steps", 0.0),
        "std_reward": ev.get("std_reward", 0.0),
    }


def list_algos() -> list[str]:
    return list(ALGO_REGISTRY.keys())
