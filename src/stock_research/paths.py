from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    repo_root: Path
    stock_root: Path
    raw_prices: Path
    processed: Path
    macro: Path
    financial_raw: Path
    financial_legacy: Path
    results: Path
    parameters: Path
    transformer_results: Path

    def ensure_output_dirs(self) -> None:
        for path in (
            self.processed,
            self.macro,
            self.financial_raw,
            self.results,
            self.parameters,
            self.transformer_results,
        ):
            path.mkdir(parents=True, exist_ok=True)


def find_repo_root() -> Path:
    # .../stock-project/src/stock_research/paths.py -> .../stock-project
    return Path(__file__).resolve().parents[2]


def load_paths(stock_root: str | Path | None = None) -> ProjectPaths:
    repo_root = find_repo_root()

    # Preferred layout:
    # OneDrive/주식/
    # ├── stock-project/   <- repository
    # ├── Back Test/
    # ├── Processed Data/
    # └── Results/
    configured = stock_root or os.getenv("STOCK_DATA_ROOT")
    resolved_stock_root = (
        Path(configured).expanduser().resolve()
        if configured
        else repo_root.parent.resolve()
    )

    paths = ProjectPaths(
        repo_root=repo_root,
        stock_root=resolved_stock_root,
        raw_prices=resolved_stock_root / "Back Test",
        processed=resolved_stock_root / "Processed Data",
        macro=resolved_stock_root / "Macro Data",
        financial_raw=resolved_stock_root / "Financial_Data_real",
        financial_legacy=resolved_stock_root / "Financial Data",
        results=resolved_stock_root / "Results",
        parameters=resolved_stock_root / "Results" / "Parameters",
        transformer_results=resolved_stock_root / "Results" / "Transformer",
    )
    paths.ensure_output_dirs()
    return paths
