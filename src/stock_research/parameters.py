from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from .io_utils import atomic_to_excel


def parameter_path(folder: Path, strategy: str) -> Path:
    safe = strategy.strip().lower().replace(" ", "_")
    return folder / f"{safe}_parameters.xlsx"


def append_parameter_record(
    folder: Path,
    strategy: str,
    record: dict,
) -> Path:
    path = parameter_path(folder, strategy)
    existing = pd.read_excel(path) if path.exists() else pd.DataFrame()
    row = dict(record)
    row.setdefault("Strategy", strategy)
    row.setdefault("OptimizedAt", datetime.now().isoformat(timespec="seconds"))
    updated = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    if "Index" in updated:
        updated = updated.drop(columns=["Index"])
    updated.insert(0, "Index", range(1, len(updated) + 1))
    atomic_to_excel({"Parameters": updated}, path, index=False)
    return path


def load_parameters(folder: Path, strategy: str) -> pd.DataFrame:
    path = parameter_path(folder, strategy)
    if not path.exists():
        raise FileNotFoundError(f"Parameter file does not exist: {path}")
    return pd.read_excel(path, sheet_name="Parameters")
