from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from datetime import UTC, datetime

from stock_research.io_utils import atomic_to_csv
from stock_research.macro_sp500.config_v2 import (
    iter_v2_candidates,
    load_v2_search_space,
    load_v2_settings,
)
from stock_research.macro_sp500.data import load_macro_sp500_data
from stock_research.macro_sp500.optimization_v2 import optimize_v2_walk_forward
from stock_research.macro_sp500.reporting_v2 import (
    generate_v2_optimization_report,
)
from stock_research.parameters import append_parameter_record, load_parameters
from stock_research.paths import load_paths


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--price-file")
    parser.add_argument("--vix-file")
    parser.add_argument("--cash-rate-file")
    parser.add_argument("--first-test-year", type=int)
    parser.add_argument("--training-years", type=int)
    parser.add_argument("--max-candidates", type=int)
    args = parser.parse_args()

    paths = load_paths()
    settings = load_v2_settings()
    if args.first_test_year is not None:
        settings = replace(settings, first_test_year=args.first_test_year)
    if args.training_years is not None:
        settings = replace(settings, training_years=args.training_years)
    candidates = iter_v2_candidates(load_v2_search_space())
    if args.max_candidates is not None:
        candidates = candidates[: args.max_candidates]
    data = load_macro_sp500_data(
        paths.macro,
        price_file=args.price_file,
        vix_file=args.vix_file,
        cash_rate_file=args.cash_rate_file,
        require_cash_rate=True,
    )
    if not data.attrs["dividend_adjusted"]:
        raise SystemExit(
            "V2 requires an adjusted-close price file. "
            "Run update_adjusted_spy.py first."
        )
    result = optimize_v2_walk_forward(data, candidates, settings)
    summary = result.oos_strategy.summary
    benchmark = result.buy_hold.summary
    record = {
        "Asset": "SPY_ADJUSTED",
        "DataStart": data["Date"].min(),
        "DataEnd": data["Date"].max(),
        "CandidateCount": result.candidate_count,
        "FoldCount": len(result.folds),
        "OOS_ROI(%)": summary.roi_percent,
        "OOS_CAGR(%)": summary.cagr_percent,
        "OOS_MDD(%)": summary.max_drawdown_percent,
        "OOS_Calmar": summary.calmar_ratio,
        "Benchmark_ROI(%)": benchmark.roi_percent,
        "Benchmark_CAGR(%)": benchmark.cagr_percent,
        "Benchmark_MDD(%)": benchmark.max_drawdown_percent,
        "Static70_ROI(%)": result.static_70_30.summary.roi_percent,
        "Static70_MDD(%)": result.static_70_30.summary.max_drawdown_percent,
        "PriceSource": data.attrs["price_source"],
        "VixSource": data.attrs["vix_source"],
        "CashRateSource": data.attrs["cash_rate_source"],
        "DividendAdjusted": data.attrs["dividend_adjusted"],
        **result.latest_params.as_dict(),
    }
    parameter_file = append_parameter_record(
        paths.parameters,
        "macro_sp500_v2",
        record,
    )
    parameter_index = int(
        load_parameters(paths.parameters, "macro_sp500_v2")["Index"].iloc[-1]
    )
    output_folder = paths.results / "SP500" / "macro_sp500_v2"
    stamp = datetime.now(UTC).astimezone().strftime("%Y%m%d_%H%M%S_%f")
    folds_file = atomic_to_csv(
        result.folds,
        output_folder / f"{parameter_index}_folds_{stamp}.csv",
        index=False,
    )
    daily_file = atomic_to_csv(
        result.oos_strategy.daily,
        output_folder / f"{parameter_index}_oos_daily_{stamp}.csv",
        index=False,
    )
    trades_file = atomic_to_csv(
        result.oos_strategy.trades,
        output_folder / f"{parameter_index}_oos_rebalances_{stamp}.csv",
        index=False,
    )
    source_note = (
        f"Adjusted price: {data.attrs['price_source']}; "
        f"VIX: {data.attrs['vix_source']}; "
        f"cash rate: {data.attrs['cash_rate_source']}. "
        "Monthly Fed Funds observations are carried backward-to-forward "
        "within their known effective month. OOS portfolio and crisis state "
        "continue across test-year boundaries."
    )
    report = generate_v2_optimization_report(
        result,
        output_folder,
        source_note=source_note,
    )
    print(f"ParameterIndex={parameter_index}")
    print(f"OOS ROI={summary.roi_percent:.2f}%")
    print(f"OOS CAGR={summary.cagr_percent:.2f}%")
    print(f"OOS MDD={summary.max_drawdown_percent:.2f}%")
    print(f"BuyHold ROI={benchmark.roi_percent:.2f}%")
    print(f"Static70 ROI={result.static_70_30.summary.roi_percent:.2f}%")
    print(f"Parameters={parameter_file}")
    print(f"Folds={folds_file}")
    print(f"Daily={daily_file}")
    print(f"Trades={trades_file}")
    print(f"Report={report}")


if __name__ == "__main__":
    main()
