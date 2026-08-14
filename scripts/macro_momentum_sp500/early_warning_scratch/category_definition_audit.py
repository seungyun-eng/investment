from __future__ import annotations
import pandas as pd
from stock_research.macro_momentum_sp500.config import load_research_config
from stock_research.macro_momentum_sp500.data import load_research_data
from stock_research.macro_momentum_sp500.features import build_features
from stock_research.paths import load_paths

paths = load_paths()
config = load_research_config("config/macro_momentum_sp500/research.json")
features = build_features(load_research_data(paths.macro, config), config)

cols = ["Date", "Close", "Treasury2Y", "RealYield5Y", "Treasury2Y_Change21_Z252",
        "RealYield5Y_Change21_Z252", "HYYield_Change21_Z252", "NFCI_Change21_Z252",
        "VIX_Change21_Z252", "InitialJoblessClaims_Change21_Z252"]

def window(start, end, label):
    print(f"\n=== {label}: {start} to {end} ===")
    f = features.loc[(features["Date"] >= start) & (features["Date"] <= end), cols]
    f = f.iloc[::10]  # every 10th row to keep it short
    with pd.option_context('display.width', 200, 'display.max_columns', 20):
        print(f.to_string(index=False))

window("2008-06-01", "2008-11-15", "GFC run-up + crash")
window("2011-06-01", "2011-10-15", "2011 European debt crisis")
