from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from datetime import UTC, datetime

from stock_research.io_utils import atomic_to_csv
from stock_research.macro_sp500.config import (
    iter_parameter_candidates,
    load_search_space,
    load_settings,
)
from stock_research.macro_sp500.data import load_macro_sp500_data
from stock_research.macro_sp500.optimization import optimize_walk_forward
from stock_research.macro_sp500.reporting import generate_optimization_report
from stock_research.parameters import append_parameter_record, load_parameters
from stock_research.paths import load_paths


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--price-file")
    parser.add_argument("--vix-file")
    parser.add_argument("--first-test-year", type=int)
    parser.add_argument("--training-years", type=int)
    parser.add_argument("--max-candidates", type=int)
    args = parser.parse_args()

    paths = load_paths()
    settings = load_settings()
    if args.first_test_year is not None:
        settings = replace(settings, first_test_year=args.first_test_year)
    if args.training_years is not None:
        settings = replace(settings, training_years=args.training_years)
    candidates = iter_parameter_candidates(load_search_space())
    if args.max_candidates is not None:
        candidates = candidates[: args.max_candidates]
    data = load_macro_sp500_data(
        paths.macro,
        price_file=args.price_file,
        vix_file=args.vix_file,
    )
    result = optimize_walk_forward(data, candidates, settings)
    params = result.latest_params
    source_note = (
        f"Price source: {data.attrs['price_source']}; "
        f"close column: {data.attrs['close_column']}; "
        f"dividend adjusted: {data.attrs['dividend_adjusted']}; "
        f"VIX source: {data.attrs['vix_source']}. "
        "Walk-forward folds reset portfolio state at each test-year boundary."
    )
    record = {
        "Asset": "S&P500_PROXY",
        "DataStart": data["Date"].min(),
        "DataEnd": data["Date"].max(),
        "CandidateCount": result.candidate_count,
        "FoldCount": len(result.folds),
        "OOS_ROI(%)": result.oos_summary.roi_percent,
        "OOS_CAGR(%)": result.oos_summary.cagr_percent,
        "OOS_MDD(%)": result.oos_summary.max_drawdown_percent,
        "OOS_Calmar": result.oos_summary.calmar_ratio,
        "Benchmark_ROI(%)": result.benchmark_summary.roi_percent,
        "Benchmark_CAGR(%)": result.benchmark_summary.cagr_percent,
        "Benchmark_MDD(%)": result.benchmark_summary.max_drawdown_percent,
        "PriceSource": data.attrs["price_source"],
        "VixSource": data.attrs["vix_source"],
        "DividendAdjusted": data.attrs["dividend_adjusted"],
        **params.as_dict(),
    }
    parameter_file = append_parameter_record(
        paths.parameters,
        "macro_sp500",
        record,
    )
    parameter_index = int(load_parameters(paths.parameters, "macro_sp500")["Index"].iloc[-1])
    output_folder = paths.results / "SP500" / "macro_sp500"
    timestamp = datetime.now(UTC).astimezone().strftime("%Y%m%d_%H%M%S_%f")
    folds_file = atomic_to_csv(
        result.folds,
        output_folder / f"{parameter_index}_walk_forward_folds_{timestamp}.csv",
        index=False,
    )
    daily_file = atomic_to_csv(
        result.oos_daily,
        output_folder / f"{parameter_index}_oos_daily_{timestamp}.csv",
        index=False,
    )
    trades_file = atomic_to_csv(
        result.oos_trades,
        output_folder / f"{parameter_index}_oos_rebalances_{timestamp}.csv",
        index=False,
    )
    report = generate_optimization_report(
        result,
        output_folder,
        source_note=source_note,
    )
    print(f"ParameterIndex={parameter_index}")
    print(f"OOS ROI={result.oos_summary.roi_percent:.2f}%")
    print(f"OOS CAGR={result.oos_summary.cagr_percent:.2f}%")
    print(f"OOS MDD={result.oos_summary.max_drawdown_percent:.2f}%")
    print(f"Benchmark ROI={result.benchmark_summary.roi_percent:.2f}%")
    print(f"Parameters={parameter_file}")
    print(f"Folds={folds_file}")
    print(f"Daily={daily_file}")
    print(f"Trades={trades_file}")
    print(f"Report={report}")


if __name__ == "__main__":
    main()
