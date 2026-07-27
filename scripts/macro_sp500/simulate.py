from __future__ import annotations

import argparse
import sys

from stock_research.io_utils import atomic_to_csv
from stock_research.macro_sp500.config import (
    MacroSp500Params,
    load_settings,
)
from stock_research.macro_sp500.data import load_macro_sp500_data
from stock_research.macro_sp500.features import add_macro_features
from stock_research.macro_sp500.portfolio import (
    run_buy_and_hold,
    run_target_weight_backtest,
)
from stock_research.macro_sp500.reporting import generate_simulation_report
from stock_research.parameters import load_parameters
from stock_research.paths import load_paths


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("index", type=int)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--price-file")
    parser.add_argument("--vix-file")
    args = parser.parse_args()

    paths = load_paths()
    settings = load_settings()
    records = load_parameters(paths.parameters, "macro_sp500")
    selected = records[records["Index"] == args.index]
    if selected.empty:
        raise SystemExit(f"Macro SP500 parameter index not found: {args.index}")
    row = selected.iloc[0]
    params = MacroSp500Params(
        vix_lookback_years=int(row["vix_lookback_years"]),
        core_weight=float(row["core_weight"]),
        warning_score_min=int(row["warning_score_min"]),
        warning_addition=float(row["warning_addition"]),
        vix_entry_quantile=float(row["vix_entry_quantile"]),
        exit_vix_quantile=float(row["exit_vix_quantile"]),
        minimum_hold_days=int(row["minimum_hold_days"]),
    )
    data = load_macro_sp500_data(
        paths.macro,
        price_file=args.price_file,
        vix_file=args.vix_file,
    )
    features = add_macro_features(
        data,
        vix_lookback_years=params.vix_lookback_years,
        warning_lookback_days=settings.warning_lookback_days,
        drawdown_lookback_days=settings.drawdown_lookback_days,
        minimum_vix_observations=settings.minimum_vix_observations,
    )
    strategy = run_target_weight_backtest(
        features,
        params,
        settings,
        start=args.start,
        end=args.end,
    )
    benchmark = run_buy_and_hold(
        features,
        settings,
        start=args.start,
        end=args.end,
    )
    output_folder = paths.results / "SP500" / "macro_sp500"
    daily = atomic_to_csv(
        strategy.daily,
        output_folder / f"{args.index}_simulation_daily_{args.start}_{args.end}.csv",
        index=False,
    )
    trades = atomic_to_csv(
        strategy.trades,
        output_folder / f"{args.index}_simulation_rebalances_{args.start}_{args.end}.csv",
        index=False,
    )
    source_note = (
        f"Price source: {data.attrs['price_source']}; "
        f"dividend adjusted: {data.attrs['dividend_adjusted']}; "
        f"VIX source: {data.attrs['vix_source']}."
    )
    report = generate_simulation_report(
        strategy,
        benchmark,
        params,
        output_folder,
        source_note=source_note,
        start=args.start,
        end=args.end,
    )
    print(f"Strategy ROI={strategy.summary.roi_percent:.2f}%")
    print(f"Benchmark ROI={benchmark.summary.roi_percent:.2f}%")
    print(f"Daily={daily}")
    print(f"Trades={trades}")
    print(f"Report={report}")


if __name__ == "__main__":
    main()
