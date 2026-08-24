from __future__ import annotations

import argparse
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local stock research dashboard.")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--stock-root")
    parser.add_argument("--host", default="127.0.0.1", help="Use 0.0.0.0 to allow access from other devices (LAN/Tailscale).")
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    from stock_research.dashboard.server import run

    run(port=args.port, stock_root=args.stock_root, host=args.host)


if __name__ == "__main__":
    main()
