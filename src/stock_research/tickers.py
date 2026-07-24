from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TickerConfig:
    ticker: str
    company_slug: str
    fiscal_year_end_month: int

    @property
    def display_name(self) -> str:
        return self.company_slug.replace("-", " ").title()


def load_tickers(config_path: str | Path) -> dict[str, TickerConfig]:
    path = Path(config_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, TickerConfig] = {}
    for ticker, details in raw.items():
        result[ticker.upper()] = TickerConfig(
            ticker=ticker.upper(),
            company_slug=details["company_slug"],
            fiscal_year_end_month=int(details.get("fiscal_year_end_month", 12)),
        )
    return result
