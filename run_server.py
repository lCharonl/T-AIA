"""Convenience launcher for the Taxi Driver web platform.

Run:
    python run_server.py            # boots on http://localhost:8000
    python run_server.py --port 8765
"""
from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Taxi Driver web platform")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="dev mode")
    args = parser.parse_args()

    import uvicorn
    uvicorn.run("server.main:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
