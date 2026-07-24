from __future__ import annotations

import argparse
from dataclasses import asdict

from stock_research.data_loading import load_processed
from stock_research.optimization import optimize_technical
from stock_research.parameters import append_parameter_record
from stock_research.paths import load_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("company")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--tpe-trials", type=int, default=500)
    parser.add_argument("--cma-trials", type=int, default=200)
    args = parser.parse_args()

    paths = load_paths()
    data = load_processed(paths.processed, args.company, args.start, args.end)
    params, roi, importance = optimize_technical(
        data, tpe_trials=args.tpe_trials, cma_trials=args.cma_trials
    )
    record = {
        "종목": args.company,
        "Start": args.start,
        "End": args.end,
        "ROI(%)": roi,
        **asdict(params),
        **{f"importance_{k}": v for k, v in importance.items()},
    }
    output = append_parameter_record(paths.parameters, "technical", record)
    print(f"ROI={roi:.2f}%")
    print(output)


if __name__ == "__main__":
    main()
