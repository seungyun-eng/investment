from __future__ import annotations

"""Refresh, validate, test, and publish Alpha Desk's weekly model signal.

This is the single scheduled publication path.  It intentionally calls the
same Python dashboard engine used by simulation instead of reimplementing the
financial formulas in a Worker.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from stock_research.dashboard import data_collection, today
from stock_research.dashboard import state as dashboard_state
from stock_research.dashboard import tsla_signal_ledger
from stock_research.dashboard.forward_ledger import LEDGER_FILENAME, LEDGER_RESULTS_FOLDER
from stock_research.dashboard.ticker_request_review import (
    WranglerD1TickerRequests,
    WranglerD1UniverseRemovals,
    apply_approved_state,
    record_universe_change,
    review_requests,
)
from stock_research.paths import load_paths


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _run(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    subprocess.run(command, cwd=cwd, env=env, check=True)


def _local_node_tool(app_root: Path, name: str) -> Path:
    """Resolve a package binary on both Windows and Linux CI runners."""

    suffix = ".CMD" if os.name == "nt" else ""
    executable = app_root / "node_modules" / ".bin" / f"{name}{suffix}"
    if not executable.exists():
        raise FileNotFoundError(f"Node tool not found: {executable}")
    return executable


def _run_optional(command: list[str], *, cwd: Path, label: str) -> None:
    """Run a step whose failure must not abort the publish.

    Used for research-only monitoring panels: a stale panel is a much smaller
    problem than a weekly signal that never ships.
    """

    try:
        subprocess.run(command, cwd=cwd, check=True)
    except (subprocess.CalledProcessError, OSError) as error:
        print(json.dumps({"step": label, "status": "skipped", "error": str(error)}, ensure_ascii=False))


def _most_recent_friday(day: date) -> date:
    return day - timedelta(days=(day.weekday() - 4) % 7)


def _expected_completed_friday(now: datetime) -> date:
    """Return the latest Friday whose New York market session has completed."""

    local = now.astimezone(ZoneInfo("America/New_York"))
    day = local.date()
    if local.weekday() == 4 and local.hour < 16:
        day -= timedelta(days=1)
    return _most_recent_friday(day)


def _write_status(path: Path, **payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_monitor_universe(paths, state, destination: Path, *, signal_as_of: str) -> None:
    """Publish the live universe and validated SEC identifiers for the cloud monitor."""

    entries: list[dict[str, object]] = []
    for member in state.tickers:
        ticker = member.ticker
        cik: str | None = None
        submissions_path = paths.stock_root / "SEC Filings" / ticker / "submissions.json"
        if submissions_path.exists():
            try:
                submissions = json.loads(submissions_path.read_text(encoding="utf-8-sig"))
                submission_tickers = {
                    str(value).strip().upper().replace(".", "-")
                    for value in submissions.get("tickers", [])
                }
                raw_cik = str(submissions.get("cik", "")).strip()
                if ticker in submission_tickers and raw_cik.isdigit():
                    cik = raw_cik.zfill(10)
            except (OSError, json.JSONDecodeError):
                pass
        entries.append({"ticker": ticker, "company": member.company, "cik": cik})

    _write_status(
        destination,
        updatedAt=datetime.now(UTC).isoformat(),
        signalAsOf=signal_as_of,
        count=len(entries),
        entries=entries,
    )


def _prepare_forward_ledger_history(paths) -> None:
    """Collect the union of every point-in-time forward universe.

    A clean cloud runner starts with only the current universe's prices and
    filings. The forward ledger must still replay U001/U002 without silently
    substituting U003, so removed historical members need their own data too.
    """

    snapshots_dir = (
        paths.repo_root
        / "config"
        / "dashboard_model_registry"
        / "universe_snapshots"
    )
    entries: dict[str, dashboard_state.TickerEntry] = {}
    for snapshot_path in sorted(snapshots_dir.glob("*.json")):
        snapshot = dashboard_state.state_from_snapshot(snapshot_path)
        for entry in snapshot.tickers:
            entries.setdefault(entry.ticker, entry)
    if not entries:
        raise RuntimeError("No forward-ledger universe snapshots found")

    for entry in entries.values():
        destination = paths.stock_root / "Dashboard Data" / "Prices" / f"{entry.ticker}.csv"
        if not destination.exists():
            data_collection.download_price_history(entry.ticker, destination)
        entry.price_path = f"Dashboard Data/Prices/{entry.ticker}.csv"

    live = dashboard_state.load_state(paths)
    historical = dashboard_state.DashboardState(
        top_k=live.top_k,
        sec_user_agent=live.sec_user_agent,
        tickers=list(entries.values()),
    )
    data_collection.sync_filings_for_dashboard(
        paths,
        historical,
        refresh_metadata=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish the latest Alpha Desk weekly signal.")
    parser.add_argument("--stock-root")
    parser.add_argument("--skip-deploy", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_now = datetime.now(UTC)
    repo_root = Path(__file__).resolve().parents[2]
    app_root = repo_root / "alpha-desk-cloud"
    public_data = app_root / "public" / "data"
    paths = load_paths(args.stock_root)
    result_dir = paths.results / LEDGER_RESULTS_FOLDER
    status_path = result_dir / "weekly_publish_status.json"
    checked_at = date.today().isoformat()
    request_store = WranglerD1TickerRequests(repo_root)
    requests = request_store.fetch_actionable()
    removal_store = WranglerD1UniverseRemovals(repo_root)
    requested_removals = removal_store.fetch_requested()
    prior_status = (
        json.loads(status_path.read_text(encoding="utf-8-sig"))
        if status_path.exists()
        else None
    )

    if prior_status and not args.force:
        expected = _expected_completed_friday(run_now).isoformat()
        # A current weekly signal is no longer sufficient to skip.  Closes move
        # every session and the app bundles price history as a static file, so
        # gating on the signal alone froze the dashboard on Friday's prices for
        # the rest of the week.  Only a run that already completed today is
        # genuinely redundant; whether prices actually moved is decided after
        # the refresh, which is the only thing that knows the real last close.
        if (
            not requests
            and not requested_removals
            and prior_status.get("status") == "ok"
            and str(prior_status.get("signalAsOf", "")) >= expected
            and str(prior_status.get("checkedAt", "")) == checked_at
        ):
            print(
                json.dumps(
                    {
                        "status": "already_ran_today",
                        "signalAsOf": prior_status["signalAsOf"],
                        "pricesAsOf": prior_status.get("pricesAsOf"),
                    }
                )
            )
            return

    _write_status(status_path, status="running", checkedAt=checked_at)
    original_state = dashboard_state.load_state(paths)
    approved_decisions = []
    added_tickers: list[str] = []
    state_applied = False
    signal_built = False
    try:
        if requests:
            request_store.mark_reviewing(requests)
            try:
                decisions, reviewed_state = review_requests(
                    paths, requests, as_of=date.today().isoformat()
                )
                request_store.save_decisions(decisions)
            except Exception as error:
                request_store.mark_errors(requests, error)
                raise
            approved_decisions = [decision for decision in decisions if decision.approved]
            original_tickers = {entry.ticker for entry in original_state.tickers}
            added_tickers = [
                entry.ticker
                for entry in reviewed_state.tickers
                if entry.ticker not in original_tickers
            ]
            if added_tickers:
                apply_approved_state(paths, reviewed_state)
                state_applied = True

        removed_tickers: list[str] = []
        if requested_removals:
            # Unlike additions, a removal needs no price/SEC/model review --
            # the ticker is already a validated, currently-held universe
            # member, so it is applied directly.
            live_tickers = {entry.ticker for entry in dashboard_state.load_state(paths).tickers}
            for ticker in requested_removals:
                if ticker in live_tickers:
                    dashboard_state.remove_ticker(paths, ticker)
                    removed_tickers.append(ticker)
            if removed_tickers:
                state_applied = True

        if requests:
            # A current signal does not need rebuilding when every request was
            # deferred or was already present. D1 is live, so those review
            # results are visible in the app immediately.
            if prior_status and not args.force and not added_tickers and not removed_tickers:
                expected = _expected_completed_friday(run_now).isoformat()
                if prior_status.get("status") == "ok" and str(prior_status.get("signalAsOf", "")) >= expected:
                    request_store.save_decisions(
                        approved_decisions,
                        effective_signal_date=str(prior_status["signalAsOf"]),
                    )
                    _write_status(status_path, **prior_status)
                    print(
                        json.dumps(
                            {
                                "status": "requests_reviewed_signal_already_current",
                                "signalAsOf": prior_status["signalAsOf"],
                                "reviewed": len(decisions),
                                "approved": len(approved_decisions),
                            },
                            ensure_ascii=False,
                        )
                    )
                    return

        payload = today.run_today(
            paths,
            refresh_prices=True,
            refresh_filings=True,
        )
        signal_built = True
        price_refresh = (payload.get("refresh") or {}).get("prices") or {}
        if price_refresh.get("failed"):
            raise RuntimeError(f"price refresh failed: {price_refresh.get('failures')}")

        signal_as_of = str(payload["market_as_of"])
        # Two different clocks: selection is weekly (weekly_signal_dates only
        # returns completed W-FRI weeks), while marks are daily.  Price history
        # must follow the daily one or the app cannot value a mid-week session.
        prices_as_of = str(price_refresh.get("latest_date") or signal_as_of)
        if added_tickers or removed_tickers:
            current_state = dashboard_state.load_state(paths)
            record_universe_change(
                paths,
                current_state,
                added=added_tickers,
                removed=removed_tickers,
                # A weekend run still changes the just-completed Friday signal;
                # the forward ledger segment must therefore start on that
                # signal date, not on Saturday's wall-clock date.
                change_date=signal_as_of,
            )
        if removed_tickers:
            removal_store.mark_applied(removed_tickers)
        if requested_removals:
            # Any requested ticker no longer in the universe (already removed,
            # or never was a member) has nothing left to apply -- clear it
            # from the queue too so it doesn't retry forever.
            removal_store.mark_applied(
                ticker for ticker in requested_removals if ticker not in removed_tickers
            )
        if approved_decisions:
            request_store.save_decisions(
                approved_decisions,
                effective_signal_date=signal_as_of,
            )
        expected = _expected_completed_friday(run_now).isoformat()
        if signal_as_of < expected:
            raise RuntimeError(
                f"weekly signal is stale: expected at least {expected}, got {signal_as_of}"
            )

        # A weekend or repeat run that moved neither the weekly signal nor the
        # last close has nothing new to ship, so stop before the export, build
        # and deploy instead of redeploying identical bytes.
        if (
            prior_status
            and not args.force
            and not added_tickers
            and prior_status.get("status") == "ok"
            and str(prior_status.get("signalAsOf", "")) >= signal_as_of
            and str(prior_status.get("pricesAsOf", "")) >= prices_as_of
        ):
            _write_status(status_path, **{**prior_status, "checkedAt": checked_at})
            print(
                json.dumps(
                    {
                        "status": "already_current",
                        "signalAsOf": signal_as_of,
                        "pricesAsOf": prices_as_of,
                    }
                )
            )
            return

        _prepare_forward_ledger_history(paths)

        score_output = result_dir / "score_history.json"
        reports_output = result_dir / "company_reports.json"
        _run(
            [
                sys.executable,
                str(repo_root / "scripts" / "dashboard" / "export_score_history.py"),
                "--output",
                str(score_output),
                "--report-output",
                str(reports_output),
                "--end",
                prices_as_of,
                "--stock-root",
                str(paths.stock_root),
            ],
            cwd=repo_root,
        )
        _run(
            [
                sys.executable,
                str(repo_root / "scripts" / "dashboard" / "update_forward_ledger.py"),
                "--end",
                prices_as_of,
                "--stock-root",
                str(paths.stock_root),
            ],
            cwd=repo_root,
        )
        _run(
            [
                sys.executable,
                "-m",
                "scripts.dashboard.update_forward_shadow",
                "--stock-root",
                str(paths.stock_root),
                "--cutoff",
                prices_as_of,
                "--public-json",
                str(public_data / "forward_shadow.json"),
            ],
            cwd=repo_root,
        )

        _run_optional(
            [
                sys.executable,
                str(repo_root / "scripts" / "dashboard" / "refresh_remaining_upside.py"),
            ],
            cwd=repo_root,
            label="refresh_remaining_upside",
        )

        _run_optional(
            [
                sys.executable,
                str(repo_root / "scripts" / "dashboard" / "export_tesla_validation.py"),
            ],
            cwd=repo_root,
            label="export_tesla_validation",
        )

        # After the price refresh above, so the technical panel always reflects
        # the same closes the rest of this publication was valued on.
        _run_optional(
            [
                sys.executable,
                str(repo_root / "scripts" / "dashboard" / "export_tesla_technicals.py"),
                "--stock-root",
                str(paths.stock_root),
            ],
            cwd=repo_root,
            label="export_tesla_technicals",
        )

        _atomic_copy(today.latest_path(paths), public_data / "latest_today.json")
        tsla_signal_path = tsla_signal_ledger.ledger_path(paths)
        if tsla_signal_path.exists():
            _atomic_copy(tsla_signal_path, public_data / tsla_signal_ledger.LEDGER_FILENAME)
        _atomic_copy(score_output, public_data / "score_history.json")
        _atomic_copy(reports_output, public_data / "company_reports.json")
        _atomic_copy(result_dir / LEDGER_FILENAME, public_data / LEDGER_FILENAME)
        _atomic_copy(
            repo_root / "config" / "dashboard_model_registry" / "registry.json",
            public_data / "registry.json",
        )

        publish_env = os.environ.copy()
        node = shutil.which("node")
        if not node:
            # Codex Desktop does not put its bundled Node runtime on PATH.
            bundled_node = (
                Path.home()
                / ".cache"
                / "codex-runtimes"
                / "codex-primary-runtime"
                / "dependencies"
                / "node"
                / "bin"
                / "node.exe"
            )
            if not bundled_node.exists():
                raise FileNotFoundError("Node.js executable not found")
            node = str(bundled_node)
            publish_env["PATH"] = os.pathsep.join(
                [str(bundled_node.parent), publish_env.get("PATH", "")]
            )
        vinext = _local_node_tool(app_root, "vinext")
        deployer = _local_node_tool(app_root, "vinext-cloudflare")
        wrangler = _local_node_tool(app_root, "wrangler")
        monitor_root = app_root / "cloudflare-monitor"
        monitor_universe_path = result_dir / "monitor_universe.json"
        _write_monitor_universe(
            paths,
            dashboard_state.load_state(paths),
            monitor_universe_path,
            signal_as_of=signal_as_of,
        )
        _run(
            [
                str(wrangler),
                "kv",
                "key",
                "put",
                "active-universe",
                "--binding",
                "MONITOR_STATUS",
                "--remote",
                "--path",
                str(monitor_universe_path),
                "--config",
                str(monitor_root / "wrangler.jsonc"),
            ],
            cwd=monitor_root,
            env=publish_env,
        )
        _run(
            [
                str(wrangler),
                "d1",
                "migrations",
                "apply",
                "alpha-desk-db",
                "--remote",
                "--config",
                str(app_root / "wrangler.jsonc"),
            ],
            cwd=app_root,
            env=publish_env,
        )
        _run([str(vinext), "build"], cwd=app_root, env=publish_env)
        _run(
            [str(node), "--test", "tests/rendered-html.test.mjs"],
            cwd=app_root,
            env=publish_env,
        )
        if not args.skip_deploy:
            _run(
                [str(deployer), "deploy", "--config", "dist/server/wrangler.json"],
                cwd=app_root,
                env=publish_env,
            )

        picks = [str(row.get("ticker")) for row in payload.get("top_picks", [])]
        _write_status(
            status_path,
            status="ok",
            checkedAt=checked_at,
            signalAsOf=signal_as_of,
            pricesAsOf=prices_as_of,
            topPicks=picks,
        )
        print(
            json.dumps(
                {"signalAsOf": signal_as_of, "pricesAsOf": prices_as_of, "topPicks": picks},
                ensure_ascii=False,
            )
        )
    except Exception as error:
        if state_applied and not signal_built:
            dashboard_state.save_state(paths, original_state)
        if approved_decisions and not signal_built:
            request_store.mark_errors(
                [decision.request for decision in approved_decisions], error
            )
        _write_status(status_path, status="error", checkedAt=checked_at, error=str(error))
        raise


if __name__ == "__main__":
    main()
