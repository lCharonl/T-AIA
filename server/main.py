"""FastAPI app : serves the static maquette and exposes the RL API."""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

# Ensure project root is on sys.path so `from agents import …` resolves
# whether the server is launched via `python -m server.main` or `uvicorn server.main:app`.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from core import DEFAULT_USER_PARAMS, OPTIMIZED_PARAMS, PRESETS, build_agent
from server import history
from server.runner import (
    evaluate_agent,
    list_algos,
    rollout_episode,
    stream_training,
)

WEB_DIR = ROOT / "web"

app = FastAPI(title="Taxi Driver — RL Platform")

# Tracks an in-flight warmup so the front (and concurrent /api/benchmark calls)
# know to wait instead of triggering a duplicate run.
_warmup_state: dict[str, Any] = {"running": False, "started_at": None}


def _seed_cache_now() -> None:
    """Populate benchmark cache by running every preset. CPU-bound."""
    _warmup_state["running"] = True
    _warmup_state["started_at"] = int(time.time())
    try:
        rows = []
        for name, preset in PRESETS.items():
            agent = build_agent(preset["algo"], params=preset["params"], seed=0)
            t0 = time.perf_counter()
            if preset["train_episodes"] > 0:
                agent.train(episodes=preset["train_episodes"])
            train_t = time.perf_counter() - t0
            ev = agent.evaluate(episodes=100)
            rows.append({
                "label": name,
                "algo": preset["algo"],
                "train_episodes": preset["train_episodes"],
                "mean_steps": ev["mean_steps"],
                "mean_reward": ev["mean_reward"],
                "success_rate": ev["success_rate"],
                "std_steps": ev.get("std_steps", 0.0),
                "std_reward": ev.get("std_reward", 0.0),
                "total_penalties": ev.get("total_penalties", 0),
                "mean_penalties": ev.get("mean_penalties", 0.0),
                "train_time_s": train_t,
            })
        history.save_benchmark_cache(rows, 100, 0)
        print("[warmup] benchmark cache populated:", [r["label"] for r in rows])
    finally:
        _warmup_state["running"] = False


@app.on_event("startup")
async def _warmup_benchmark() -> None:
    """At boot, if the benchmark cache is empty, populate it in the
    background so the first visit to /benchmark is instant. Takes ~5 min."""
    if history.get_benchmark_cache() is not None:
        return
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _seed_cache_now)


# ── Static front ─────────────────────────────────────────────────────────
@app.get("/")
async def root() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


# Also expose proto.css / proto.js / api.js at the root so the maquette's
# original <link rel="stylesheet" href="proto.css"> resolves.
@app.get("/{filename}")
async def serve_asset(filename: str) -> FileResponse:
    p = WEB_DIR / filename
    if not p.is_file():
        raise HTTPException(status_code=404, detail=f"not found: {filename}")
    return FileResponse(p)


# ── Schemas ──────────────────────────────────────────────────────────────
class EpisodeRequest(BaseModel):
    algo: str = "Q-Learning"
    params: Optional[dict[str, Any]] = None
    train_episodes: int = 2000
    seed: int = 0
    episode_seed_offset: int = 99_999  # change to vary the initial state


class TrainRequest(BaseModel):
    algo: str
    params: dict[str, Any] = Field(default_factory=dict)
    episodes: int = 2000
    seed: int = 0
    shaped: bool = False
    shaping_lambda: float = 0.5
    eval_episodes: int = 200


class BenchmarkRequest(BaseModel):
    presets: Optional[list[str]] = None  # None = all
    eval_episodes: int = 100
    seed: int = 0
    force: bool = False  # ignore SQLite cache and re-run everything


class ShapingRequest(BaseModel):
    algo: str = "Q-Learning"
    params: Optional[dict[str, Any]] = None
    episodes: int = 2000
    shaping_lambda: float = 0.5
    seed: int = 0


# ── Meta endpoints ───────────────────────────────────────────────────────
@app.get("/api/algos")
async def api_algos() -> dict[str, Any]:
    return {
        "algos": list_algos(),
        "optimized": OPTIMIZED_PARAMS,
        "defaults": DEFAULT_USER_PARAMS,
        "presets": {
            name: {k: v for k, v in p.items() if k != "params"} | {"params": p["params"]}
            for name, p in PRESETS.items()
        },
    }


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


# ── Environnement screen: play one greedy episode ────────────────────────
@app.post("/api/episode")
async def api_episode(req: EpisodeRequest) -> dict[str, Any]:
    # Heavy on the first call (training), cached afterwards by runner.
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: rollout_episode(
            req.algo, req.params,
            seed=req.seed,
            train_episodes=0 if req.algo == "Brute Force" else req.train_episodes,
            episode_seed_offset=req.episode_seed_offset,
        ),
    )
    return result


# Replay an episode from a saved run's hyperparameters (Historique → Env).
@app.post("/api/runs/{run_id}/episode")
async def api_run_episode(run_id: int, episode_seed_offset: int = 99_999) -> dict[str, Any]:
    runs = history.list_runs(limit=10_000)
    run = next((r for r in runs if r["id"] == run_id), None)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: rollout_episode(
            run["algo"], run["params"],
            seed=0,
            train_episodes=run["episodes"],
            episode_seed_offset=episode_seed_offset,
        ),
    )
    result["run"] = {
        "id": run["id"], "algo": run["algo"],
        "eval_steps": run["eval_steps"], "eval_success": run["eval_success"],
        "episodes": run["episodes"],
    }
    return result


# ── Benchmark screen: run every preset (or serve cached) ─────────────────
@app.post("/api/benchmark")
async def api_benchmark(req: BenchmarkRequest) -> dict[str, Any]:
    loop = asyncio.get_event_loop()

    # Serve cache when allowed: only if it covers all the presets the caller
    # asked for (or the full default set) and was produced with the same
    # eval_episodes/seed. Otherwise fall through to a fresh run.
    if not req.force:
        cached = history.get_benchmark_cache()
        if cached and cached["eval_episodes"] == req.eval_episodes and cached["seed"] == req.seed:
            wanted = set(req.presets or PRESETS.keys())
            have = {r["label"] for r in cached["rows"]}
            if wanted.issubset(have):
                rows = [r for r in cached["rows"] if r["label"] in wanted]
                return {
                    "rows": rows,
                    "eval_episodes": req.eval_episodes,
                    "from_cache": True,
                    "cached_at": cached["ts"],
                }
        # If a warmup is currently running, do NOT kick off a parallel run —
        # let the front poll /api/benchmark/cache instead. Returning 202
        # with a clear message keeps the client side simple.
        if _warmup_state["running"]:
            raise HTTPException(
                status_code=202,
                detail="warmup in progress — poll /api/benchmark/cache",
            )

    def _run() -> list[dict[str, Any]]:
        rows = []
        targets = req.presets or list(PRESETS.keys())
        for name in targets:
            preset = PRESETS[name]
            agent = build_agent(preset["algo"], params=preset["params"], seed=req.seed)
            t0 = time.perf_counter()
            if preset["train_episodes"] > 0:
                agent.train(episodes=preset["train_episodes"])
            train_t = time.perf_counter() - t0
            ev = agent.evaluate(episodes=req.eval_episodes)
            rows.append({
                "label": name,
                "algo": preset["algo"],
                "train_episodes": preset["train_episodes"],
                "mean_steps": ev["mean_steps"],
                "mean_reward": ev["mean_reward"],
                "success_rate": ev["success_rate"],
                "std_steps": ev.get("std_steps", 0.0),
                "std_reward": ev.get("std_reward", 0.0),
                "total_penalties": ev.get("total_penalties", 0),
                "mean_penalties": ev.get("mean_penalties", 0.0),
                "train_time_s": train_t,
            })
        return rows

    rows = await loop.run_in_executor(None, _run)
    # Persist only when we ran the full default preset set — partial snapshots
    # would pollute the cache and trigger stale hits later.
    if req.presets is None or set(req.presets) == set(PRESETS.keys()):
        await loop.run_in_executor(
            None, lambda: history.save_benchmark_cache(rows, req.eval_episodes, req.seed)
        )
    return {
        "rows": rows,
        "eval_episodes": req.eval_episodes,
        "from_cache": False,
        "cached_at": int(time.time()),
    }


# ── Read-only access to the latest cached benchmark (for the front) ──────
@app.get("/api/benchmark/cache")
async def api_benchmark_cache() -> dict[str, Any]:
    cached = history.get_benchmark_cache()
    payload: dict[str, Any] = {
        "cached": cached is not None,
        "warmup_running": bool(_warmup_state["running"]),
        "warmup_started_at": _warmup_state["started_at"],
    }
    if cached is not None:
        payload.update(cached)
    return payload


# ── Reward shaping screen: compare with/without ──────────────────────────
@app.post("/api/shaping_compare")
async def api_shaping_compare(req: ShapingRequest) -> dict[str, Any]:
    """Two trainings of `req.algo` (one base, one shaped) — same seed,
    same params. Returns the steps curve (downsampled) for each."""
    loop = asyncio.get_event_loop()

    def _train_once(shaped: bool) -> dict[str, Any]:
        agent = build_agent(req.algo, params=req.params or {}, seed=req.seed)
        # Reuse the streaming generator's monkey-patch for shaping.
        if shaped and hasattr(agent, "gamma"):
            from server.runner import potential
            gamma = float(getattr(agent, "gamma", 0.99))
            orig = agent.update

            def shaped_update(s, a, r, sp, done, na=None):
                phi_s = potential(s, req.shaping_lambda)
                phi_sp = 0.0 if done else potential(sp, req.shaping_lambda)
                return orig(s, a, r + gamma * phi_sp - phi_s, sp, done, na)
            agent.update = shaped_update  # type: ignore[assignment]

        t0 = time.perf_counter()
        agent.train(episodes=req.episodes)
        elapsed = time.perf_counter() - t0
        ev = agent.evaluate(episodes=100)
        return {
            "shaped": shaped,
            "history": {
                "steps": [int(x) for x in agent.history["steps"]],
                "rewards": [float(x) for x in agent.history["rewards"]],
            },
            "conv_episode": history.find_conv_episode(agent.history["steps"]),
            "eval": {
                "mean_steps": ev["mean_steps"],
                "mean_reward": ev["mean_reward"],
                "success_rate": ev["success_rate"],
            },
            "train_time_s": elapsed,
        }

    base = await loop.run_in_executor(None, _train_once, False)
    shaped = await loop.run_in_executor(None, _train_once, True)
    return {"base": base, "shaped": shaped, "algo": req.algo,
            "shaping_lambda": req.shaping_lambda, "episodes": req.episodes}


# ── History endpoints ────────────────────────────────────────────────────
@app.get("/api/runs")
async def api_runs_list() -> dict[str, Any]:
    return {"runs": history.list_runs()}


@app.delete("/api/runs")
async def api_runs_clear() -> dict[str, int]:
    n = history.clear_all()
    return {"deleted": n}


@app.delete("/api/runs/{run_id}")
async def api_runs_delete(run_id: int) -> dict[str, bool]:
    ok = history.delete_run(run_id)
    if not ok:
        raise HTTPException(status_code=404, detail="run not found")
    return {"deleted": True}


# ── WebSocket: live training stream ──────────────────────────────────────
@app.websocket("/ws/train")
async def ws_train(ws: WebSocket) -> None:
    await ws.accept()
    try:
        msg = await ws.receive_text()
        req = TrainRequest.model_validate_json(msg)
    except Exception as e:
        await ws.send_json({"type": "error", "message": f"bad request: {e}"})
        await ws.close()
        return

    # We need to (a) stream training events to the WS, (b) afterwards
    # evaluate the agent and (c) persist the run. The generator runs in
    # a thread (it's CPU-bound), the main asyncio loop drains it via a queue.
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    final: dict[str, Any] = {}

    def _producer() -> None:
        try:
            gen = stream_training(
                req.algo, req.params, req.episodes,
                seed=req.seed, shaped=req.shaped,
                shaping_lambda=req.shaping_lambda,
            )
            while True:
                try:
                    evt = next(gen)
                    asyncio.run_coroutine_threadsafe(queue.put(evt), loop)
                except StopIteration as stop:
                    # Generator returned the final dict (history + agent)
                    final.update(stop.value or {})
                    break
        except Exception as e:
            asyncio.run_coroutine_threadsafe(
                queue.put({"type": "error", "message": str(e)}), loop
            )
        finally:
            asyncio.run_coroutine_threadsafe(queue.put({"type": "_eof"}), loop)

    producer = loop.run_in_executor(None, _producer)
    try:
        while True:
            evt = await queue.get()
            if evt.get("type") == "_eof":
                break
            await ws.send_json(evt)
        await producer

        if "agent" in final:
            agent = final["agent"]
            ev = await loop.run_in_executor(
                None, lambda: agent.evaluate(episodes=req.eval_episodes)
            )
            run_id = await loop.run_in_executor(None, lambda: history.insert_run(
                algo=req.algo,
                params=req.params,
                episodes=req.episodes,
                curve_steps=[int(x) for x in agent.history["steps"]],
                curve_reward=[float(x) for x in agent.history["rewards"]],
                eval_steps=ev["mean_steps"],
                eval_reward=ev["mean_reward"],
                eval_success=ev["success_rate"],
                train_time_s=final.get("train_time_s", 0.0),
            ))
            await ws.send_json({
                "type": "done",
                "run_id": run_id,
                "train_time_s": final.get("train_time_s", 0.0),
                "eval": {
                    "mean_steps": ev["mean_steps"],
                    "mean_reward": ev["mean_reward"],
                    "success_rate": ev["success_rate"],
                    "std_steps": ev.get("std_steps", 0.0),
                    "std_reward": ev.get("std_reward", 0.0),
                },
            })
    except WebSocketDisconnect:
        return
    finally:
        await ws.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.main:app", host="0.0.0.0", port=8000, reload=False)
