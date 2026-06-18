"""SQLite-backed history of training runs.

Schema is deliberately minimal: one row per run, hyperparameters and
sampled learning curves stored as JSON strings.
"""
from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

DB_PATH = Path(__file__).resolve().parent / "history.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              INTEGER NOT NULL,
    algo            TEXT NOT NULL,
    params_json     TEXT NOT NULL,
    episodes        INTEGER NOT NULL,
    curve_steps     TEXT NOT NULL,   -- downsampled to ~60 points
    curve_reward    TEXT NOT NULL,   -- downsampled to ~60 points
    eval_steps      REAL,
    eval_reward     REAL,
    eval_success    REAL,
    conv_episode    INTEGER,         -- episode where steps MA(100) first <20
    train_time_s    REAL
);
CREATE INDEX IF NOT EXISTS idx_runs_ts ON runs (ts DESC);

CREATE TABLE IF NOT EXISTS benchmark_cache (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              INTEGER NOT NULL,
    eval_episodes   INTEGER NOT NULL,
    seed            INTEGER NOT NULL,
    rows_json       TEXT NOT NULL    -- full list of preset rows
);
CREATE INDEX IF NOT EXISTS idx_bench_ts ON benchmark_cache (ts DESC);
"""


def _init() -> None:
    with sqlite3.connect(DB_PATH) as cx:
        cx.executescript(SCHEMA)


_init()


@contextmanager
def _conn() -> Iterator[sqlite3.Connection]:
    cx = sqlite3.connect(DB_PATH)
    cx.row_factory = sqlite3.Row
    try:
        yield cx
        cx.commit()
    finally:
        cx.close()


def _downsample(values: list[float], target: int = 60) -> list[float]:
    if not values:
        return []
    if len(values) <= target:
        return [float(v) for v in values]
    step = len(values) / target
    return [float(values[min(int(i * step), len(values) - 1)]) for i in range(target)]


def find_conv_episode(steps: list[int], window: int = 100, threshold: float = 20.0) -> Optional[int]:
    """First episode at which the rolling mean of `steps` drops below `threshold`."""
    if not steps:
        return None
    cumsum = 0.0
    buf: list[int] = []
    for i, s in enumerate(steps):
        buf.append(s)
        cumsum += s
        if len(buf) > window:
            cumsum -= buf.pop(0)
        ma = cumsum / len(buf)
        if len(buf) >= min(window, len(steps)) and ma < threshold:
            return i + 1
    return None


def insert_run(
    *,
    algo: str,
    params: dict[str, Any],
    episodes: int,
    curve_steps: list[int],
    curve_reward: list[float],
    eval_steps: float,
    eval_reward: float,
    eval_success: float,
    train_time_s: float,
) -> int:
    conv = find_conv_episode(curve_steps)
    with _conn() as cx:
        cur = cx.execute(
            """INSERT INTO runs
               (ts, algo, params_json, episodes,
                curve_steps, curve_reward,
                eval_steps, eval_reward, eval_success,
                conv_episode, train_time_s)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                int(time.time()),
                algo,
                json.dumps(params),
                int(episodes),
                json.dumps(_downsample(curve_steps)),
                json.dumps(_downsample(curve_reward)),
                float(eval_steps),
                float(eval_reward),
                float(eval_success),
                conv,
                float(train_time_s),
            ),
        )
        return int(cur.lastrowid)


def list_runs(limit: int = 100) -> list[dict[str, Any]]:
    with _conn() as cx:
        rows = cx.execute(
            "SELECT * FROM runs ORDER BY ts DESC LIMIT ?", (limit,)
        ).fetchall()
    out = []
    for r in rows:
        out.append({
            "id": r["id"],
            "ts": r["ts"],
            "algo": r["algo"],
            "params": json.loads(r["params_json"]),
            "episodes": r["episodes"],
            "curve_steps": json.loads(r["curve_steps"]),
            "curve_reward": json.loads(r["curve_reward"]),
            "eval_steps": r["eval_steps"],
            "eval_reward": r["eval_reward"],
            "eval_success": r["eval_success"],
            "conv_episode": r["conv_episode"],
            "train_time_s": r["train_time_s"],
        })
    return out


def delete_run(run_id: int) -> bool:
    with _conn() as cx:
        cur = cx.execute("DELETE FROM runs WHERE id = ?", (run_id,))
        return cur.rowcount > 0


def clear_all() -> int:
    with _conn() as cx:
        cur = cx.execute("DELETE FROM runs")
        return cur.rowcount


# ── Benchmark cache ──────────────────────────────────────────────────────
def get_benchmark_cache() -> Optional[dict[str, Any]]:
    """Return the most recent benchmark snapshot, or None if cache is empty."""
    with _conn() as cx:
        row = cx.execute(
            "SELECT * FROM benchmark_cache ORDER BY ts DESC LIMIT 1"
        ).fetchone()
    if not row:
        return None
    return {
        "ts": row["ts"],
        "eval_episodes": row["eval_episodes"],
        "seed": row["seed"],
        "rows": json.loads(row["rows_json"]),
    }


def save_benchmark_cache(rows: list[dict[str, Any]], eval_episodes: int, seed: int) -> int:
    """Insert a fresh snapshot; keep only the last 5 to bound disk."""
    with _conn() as cx:
        cur = cx.execute(
            "INSERT INTO benchmark_cache (ts, eval_episodes, seed, rows_json) VALUES (?, ?, ?, ?)",
            (int(time.time()), int(eval_episodes), int(seed), json.dumps(rows)),
        )
        cx.execute(
            "DELETE FROM benchmark_cache WHERE id NOT IN "
            "(SELECT id FROM benchmark_cache ORDER BY ts DESC LIMIT 5)"
        )
        return int(cur.lastrowid)
