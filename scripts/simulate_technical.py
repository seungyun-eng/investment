from __future__ import annotations

import argparse

from stock_research.data_loading import load_processed
from stock_research.parameters import load_parameters
from stock_research.paths import load_paths
from stock_research.simulation import (
    buy_and_hold,
    daily_dca,
    perfect_foresight,
    run_strategy,
    save_simulation_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("index", type=int)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--extra-on-buy", action="store_true")
    parser.add_argument("--daily-dca-amount", type=float, default=10_000.0)
    args = parser.parse_args()

    paths = load_paths()
    parameters = load_parameters(paths.parameters, "technical")
    rows = parameters[parameters["Index"] == args.index]
    if rows.empty:
        raise SystemExit(f"Parameter index not found: {args.index}")
    row = rows.iloc[0]
    company = str(row["종목"])
    data = load_processed(paths.processed, company, args.start, args.end)

    result = run_strategy(
        data, "technical", row.to_dict(), extra_on_buy=args.extra_on_buy
    )
    outputs = {
        "once" if not args.extra_on_buy else "extra": result.trades,
        "buy_and_hold": buy_and_hold(data),
        "perfect_foresight_DIAGNOSTIC": perfect_foresight(data),
        "daily_dca": daily_dca(data, args.daily_dca_amount),
    }
    saved = save_simulation_outputs(
        paths.results,
        company,
        "technical",
        args.index,
        args.start,
        args.end,
        outputs,
    )
    print(f"Strategy ROI={result.summary.roi_percent:.2f}%")
    for path in saved:
        print(path)


if __name__ == "__main__":
    main()
