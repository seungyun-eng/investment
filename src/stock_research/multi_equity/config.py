from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EquitySpec:
    ticker: str
    price_file: str


def load_equity_specs(path: Path) -> list[EquitySpec]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    specs = [EquitySpec(**item) for item in payload.get("universe", [])]
    if not specs:
        raise ValueError(f"No equities configured in {path}")
    tickers = [spec.ticker for spec in specs]
    if len(tickers) != len(set(tickers)):
        raise ValueError(f"Duplicate tickers configured in {path}")
    return specs
