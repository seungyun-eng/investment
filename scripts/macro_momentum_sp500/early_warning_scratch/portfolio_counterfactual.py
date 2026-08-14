from __future__ import annotations
import json
import pandas as pd
from stock_research.macro_momentum_sp500.config import load_research_config
from stock_research.macro_momentum_sp500.data import load_research_data
from stock_research.macro_momentum_sp500.event_warning import detect_drawdown_events
from stock_research.macro_momentum_sp500.features import build_features
from stock_research.macro_momentum_sp500.portfolio import run_weight_backtest
from stock_research.paths import load_paths

config = load_research_config("config/macro_momentum_sp500/research.json")
data = load_research_data(load_paths().macro, config)
features = build_features(data, config).reset_index(drop=True)
events = detect_drawdown_events(features, -0.15)
dates = pd.to_datetime(features["Date"])

def summary_row(result, label):
    s = result.summary
    return {"label": label, "CAGR_pct": round(s.cagr_percent, 2),
            "MDD_pct": round(s.max_drawdown_percent, 2), "Sharpe": round(s.sharpe_ratio, 3)}

base = features[["Date", "Open", "Close", "CashRate"]].copy()
base["TargetWeight"] = 1.0
base["SignalState"] = "BUY_AND_HOLD"
baseline_result = run_weight_backtest(base, config, name="buy_and_hold")
rows = [summary_row(baseline_result, "Buy & Hold SPY (frozen baseline, 1993-2026)")]

alert_date_2007 = pd.Timestamp("2007-08-02")
alert_date_2021 = pd.Timestamp("2021-12-02")
event_2007 = next(e for e in events if e.peak_date.year == 2007)
event_2022 = next(e for e in events if e.peak_date.year == 2022)

scenarios = {
    "A_exit_at_alert_reenter_at_trough_HINDSIGHT": {
        2007: (alert_date_2007, event_2007.trough_date),
        2022: (alert_date_2021, event_2022.trough_date),
    },
    "B_exit_at_alert_reenter_at_recovery_HINDSIGHT": {
        2007: (alert_date_2007, event_2007.recovery_date),
        2022: (alert_date_2021, event_2022.recovery_date),
    },
    "C_exit_at_alert_reenter_63d_after_trough_REALISTIC": {
        2007: (alert_date_2007, None),
        2022: (alert_date_2021, None),
    },
}

for name, windows in scenarios.items():
    weights = features[["Date", "Open", "Close", "CashRate"]].copy()
    weights["TargetWeight"] = 1.0
    for year, (start, end) in windows.items():
        if end is None:
            trough = event_2007.trough_date if year == 2007 else event_2022.trough_date
            trough_idx = dates[dates.eq(trough)].index[0]
            end_idx = min(trough_idx + 63, len(dates) - 1)
            end = dates.iloc[end_idx]
        mask = dates.ge(start) & dates.le(end)
        weights.loc[mask, "TargetWeight"] = 0.0
    weights["SignalState"] = "EVENT_WARNING_OVERLAY"
    result = run_weight_backtest(weights, config, name=name)
    days_in_cash = int((weights["TargetWeight"] == 0.0).sum())
    row = summary_row(result, name)
    row["days_in_cash"] = days_in_cash
    rows.append(row)

print(json.dumps(rows, indent=2, ensure_ascii=False))

print()
print("Event windows used:")
print(f"2007: alert={alert_date_2007.date()}, peak={event_2007.peak_date.date()}, "
      f"trough={event_2007.trough_date.date()}, recovery={event_2007.recovery_date.date() if event_2007.recovery_date else None}")
print(f"2022: alert={alert_date_2021.date()}, peak={event_2022.peak_date.date()}, "
      f"trough={event_2022.trough_date.date()}, recovery={event_2022.recovery_date.date() if event_2022.recovery_date else None}")
