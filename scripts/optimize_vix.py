from __future__ import annotations

import argparse

from stock_research.data_loading import load_processed_with_vix
from stock_research.optimization import optimize_vix, run_vix_backtest
from stock_research.parameters import append_parameter_record
from stock_research.paths import load_paths
from stock_research.strategies.vix import load_vix_rule_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("company")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--tpe-trials", type=int, default=1500)
    parser.add_argument("--cma-trials", type=int, default=500)
    args = parser.parse_args()

    paths = load_paths()
    data = load_processed_with_vix(
        paths.processed, paths.macro, args.company, args.start, args.end
    )
    rules = load_vix_rule_config()
    params, roi, importance = optimize_vix(
        data, rules=rules, tpe_trials=args.tpe_trials, cma_trials=args.cma_trials
    )
    result = run_vix_backtest(data, params)
    record = {
        "종목": args.company,
        "Start": args.start,
        "End": args.end,
        "ROI(%)": roi,
        "rsi_buy_th": params.rsi_buy_th,
        "rsi_sell_th": params.rsi_sell_th,
        "boll_buffer": params.boll_buffer,
        "ActualVixBuyLevel": rules.vix_buy_level,
        "ActualVixSellLevel": rules.vix_sell_level,
        "VixRuleSource": rules.source,
        "BuyCount": result.summary.buys,
        "SignalSellCount": result.summary.sells,
        "LiquidationCount": result.summary.liquidations,
        "CompletedTrades": result.summary.completed_trades,
        **{f"importance_{k}": v for k, v in importance.items()},
    }
    output = append_parameter_record(paths.parameters, "vix", record)
    print(f"ROI={roi:.2f}%")
    print(output)


if __name__ == "__main__":
    main()
