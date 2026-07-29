from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from stock_research.cross_sectional.overfitting_diagnostics import (
    build_overfitting_diagnostic,
    input_schema,
    write_overfitting_outputs,
)
from stock_research.paths import load_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate V6-B DSR and coarse four-block CSCV PBO."
    )
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--selected-strategy", required=True)
    parser.add_argument("--selected-equity", required=True)
    parser.add_argument("--selected-candidate", type=int, default=1931)
    parser.add_argument("--nominal-trials", type=int, default=2000)
    parser.add_argument(
        "--equity-scenario",
        default="BASELINE_CURRENT_SNAPSHOT_NONCAUSAL",
    )
    parser.add_argument("--equity-period", default="TRAIN_2020_2024")
    parser.add_argument("--stock-root")
    parser.add_argument("--output-dir")
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    paths = load_paths(args.stock_root)
    candidates_path = _resolve(paths.repo_root, args.candidates)
    strategy_path = _resolve(paths.repo_root, args.selected_strategy)
    equity_path = _resolve(paths.repo_root, args.selected_equity)
    strategy = json.loads(strategy_path.read_text(encoding="utf-8"))
    _validate_frozen_strategy(
        strategy,
        selected_candidate=args.selected_candidate,
        nominal_trials=args.nominal_trials,
    )

    print("INPUT SCHEMA")
    print(input_schema().to_string(index=False))
    candidates = pd.read_csv(candidates_path)
    equity = pd.read_csv(equity_path)
    diagnostic = build_overfitting_diagnostic(
        candidates,
        equity,
        selected_candidate=args.selected_candidate,
        nominal_trials=args.nominal_trials,
        scenario=args.equity_scenario,
        period=args.equity_period,
    )
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else paths.results
        / "Cross_Sectional"
        / "overfitting_diagnostic"
        / _run_name()
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    outputs = write_overfitting_outputs(
        diagnostic,
        candidates,
        output_dir,
        selected_candidate=args.selected_candidate,
    )
    _atomic_json(
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model_status": "POST_HOC_DIAGNOSTIC_PASS",
            "validation_is_fresh": False,
            "selected_candidate": args.selected_candidate,
            "nominal_trials": args.nominal_trials,
            "search_dimensions": 9,
            "candidates": str(candidates_path),
            "selected_strategy": str(strategy_path),
            "selected_equity": str(equity_path),
            "equity_scenario": args.equity_scenario,
            "equity_period": args.equity_period,
            "exact_daily_candidate_return_matrix_available": False,
            "pbo_status": "COARSE_S4_AGGREGATE_BLOCK_PROXY",
            "contamination_warning": (
                "Survivorship bias, multiple testing, and post-hoc "
                "observation of 2025/2026 remain."
            ),
            "outputs": {key: str(value) for key, value in outputs.items()},
        },
        output_dir / "manifest.json",
    )
    print("\nDSR")
    print(diagnostic.dsr.to_string(index=False))
    print("\nEFFECTIVE TRIALS")
    print(diagnostic.effective_trials.to_string(index=False))
    print("\nPBO")
    print(diagnostic.pbo_summary.to_string(index=False))
    print("\nIS-2025 RANK CORRELATION")
    print(diagnostic.rank_correlation.to_string(index=False))
    print(f"\nOutput: {output_dir}")


def _validate_frozen_strategy(
    strategy: dict[str, object],
    *,
    selected_candidate: int,
    nominal_trials: int,
) -> None:
    if strategy.get("selected_candidate") != selected_candidate:
        raise ValueError("Selected candidate does not match frozen strategy")
    settings = strategy.get("settings", {})
    if not isinstance(settings, dict):
        raise ValueError("Frozen strategy settings are missing")
    if settings.get("candidate_count") != nominal_trials:
        raise ValueError("Nominal trial count does not match frozen strategy")
    if strategy.get("model_status") != "POST_HOC_DIAGNOSTIC_PASS":
        raise ValueError("Unexpected model status")
    methodology = strategy.get("methodology", {})
    if not isinstance(methodology, dict):
        raise ValueError("Frozen strategy methodology is missing")
    if methodology.get("validation_is_fresh") is not False:
        raise ValueError("validation_is_fresh must remain false")


def _resolve(repo_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _run_name() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    return f"{stamp}_v6_b_dsr_pbo"


def _atomic_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        suffix=".tmp",
        prefix=f"{path.stem}_",
        dir=path.parent,
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
