from __future__ import annotations

"""Local dashboard server: stdlib http.server only (no new dependency).

Single-user local research tool -- runs on localhost, serves the frontend
HTML/JS/CSS from the repo's dashboard/ directory, and exposes a small JSON
API backed by stock_research.dashboard.{state,data_collection,engine}.
"""

import json
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from stock_research.dashboard import data_collection, engine, today
from stock_research.dashboard import state as dashboard_state
from stock_research.paths import ProjectPaths, load_paths

REPO_ROOT = Path(__file__).resolve().parents[3]
FRONTEND_DIR = REPO_ROOT / "dashboard"
DEFAULT_PORT = 8765


class DashboardHandler(BaseHTTPRequestHandler):
    paths: ProjectPaths

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        print(f"[dashboard] {self.address_string()} - {format % args}")

    def _send_json(self, payload: object, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, message: str, status: int = 400) -> None:
        self._send_json({"error": message}, status=status)

    def _read_json_body(self) -> dict[str, object]:
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8")) if raw else {}

    def _serve_static(self, path: str) -> None:
        relative = path.lstrip("/") or "app.html"
        if relative in {"app", "today"}:
            relative = "app.html"
        file_path = (FRONTEND_DIR / relative).resolve()
        if FRONTEND_DIR not in file_path.parents and file_path != FRONTEND_DIR:
            self._send_error_json("not found", 404)
            return
        if not file_path.exists() or not file_path.is_file():
            self._send_error_json("not found", 404)
            return
        content_type = {
            ".html": "text/html; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".json": "application/json; charset=utf-8",
            ".webmanifest": "application/manifest+json; charset=utf-8",
            ".png": "image/png",
        }.get(file_path.suffix, "application/octet-stream")
        body = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        try:
            if parsed.path == "/api/universe":
                state = dashboard_state.load_state(self.paths)
                self._send_json(state.as_dict())
                return
            if parsed.path == "/api/today":
                payload = today.load_latest(self.paths)
                if payload is None:
                    payload = today.run_today(self.paths)
                self._send_json(payload)
                return
            if parsed.path.startswith("/api/ticker/") and parsed.path.endswith("/history"):
                ticker = parsed.path.split("/")[3]
                start = query.get("start", [None])[0]
                end = query.get("end", [None])[0]
                if not start or not end:
                    self._send_error_json("start and end query params are required")
                    return
                state = dashboard_state.load_state(self.paths)
                result = engine.get_ticker_history(self.paths, state, ticker, start=start, end=end)
                self._send_json(result)
                return
            if not parsed.path.startswith("/api/"):
                self._serve_static(parsed.path)
                return
            self._send_error_json("unknown endpoint", 404)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self._send_error_json(str(exc), 500)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            body = self._read_json_body()
            if parsed.path == "/api/universe/add":
                ticker = str(body["ticker"]).upper().strip()
                company = str(body.get("company") or ticker)
                result = data_collection.collect_ticker(self.paths, ticker, company)
                self._send_json(result)
                return
            if parsed.path == "/api/universe/remove":
                ticker = str(body["ticker"]).upper().strip()
                state = dashboard_state.remove_ticker(self.paths, ticker)
                self._send_json(state.as_dict())
                return
            if parsed.path == "/api/universe/collect":
                # Re-sync SEC filings for the whole current universe (cheap
                # for already-cached tickers) without adding a new one.
                state = dashboard_state.load_state(self.paths)
                artifacts = data_collection.sync_filings_for_dashboard(
                    self.paths, state, refresh_metadata=True
                )
                self._send_json({"status": "ok", "point_in_time_features_csv": str(artifacts.point_in_time_features_csv)})
                return
            if parsed.path == "/api/universe/top_k":
                top_k = int(body["top_k"])
                state = dashboard_state.set_top_k(self.paths, top_k)
                self._send_json(state.as_dict())
                return
            if parsed.path == "/api/backtest":
                state = dashboard_state.load_state(self.paths)
                result = engine.run_backtest(
                    self.paths,
                    state,
                    start=str(body["start"]),
                    end=str(body["end"]),
                    top_k=int(body["top_k"]) if body.get("top_k") else None,
                )
                self._send_json(result)
                return
            if parsed.path == "/api/today":
                result = today.run_today(
                    self.paths,
                    top_k=int(body["top_k"]) if body.get("top_k") else None,
                    refresh_prices=bool(body.get("refresh_prices", False)),
                    refresh_filings=bool(body.get("refresh_filings", False)),
                )
                self._send_json(result)
                return
            self._send_error_json("unknown endpoint", 404)
        except KeyError as exc:
            self._send_error_json(f"missing field: {exc}", 400)
        except ValueError as exc:
            self._send_error_json(str(exc), 400)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self._send_error_json(str(exc), 500)


def run(port: int = DEFAULT_PORT, stock_root: str | None = None, host: str = "127.0.0.1") -> None:
    DashboardHandler.paths = load_paths(stock_root)
    server = ThreadingHTTPServer((host, port), DashboardHandler)
    print(f"Dashboard running at http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
