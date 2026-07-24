from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Iterable

import pandas as pd


def read_csv_fallback(path: str | Path, **kwargs) -> pd.DataFrame:
    path = Path(path)
    errors: list[str] = []
    for encoding in ("utf-8-sig", "utf-8", "cp949"):
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
    raise UnicodeError(f"Unable to decode {path}. Attempts: {errors}")


def atomic_to_csv(df: pd.DataFrame, path: str | Path, **kwargs) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".tmp",
        prefix=path.stem + "_",
        dir=path.parent,
        delete=False,
        encoding=kwargs.pop("encoding", "utf-8-sig"),
        newline="",
    ) as handle:
        tmp = Path(handle.name)
    try:
        df.to_csv(tmp, **kwargs)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def atomic_to_excel(
    sheets: dict[str, pd.DataFrame],
    path: str | Path,
    *,
    index: bool | dict[str, bool] = True,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        suffix=".xlsx", prefix=path.stem + "_", dir=path.parent
    )
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        with pd.ExcelWriter(tmp, engine="xlsxwriter") as writer:
            for sheet, df in sheets.items():
                use_index = index.get(sheet, True) if isinstance(index, dict) else index
                df.to_excel(writer, sheet_name=sheet[:31], index=use_index)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def newest_file(paths: Iterable[Path]) -> Path | None:
    existing = [p for p in paths if p.exists()]
    return max(existing, key=lambda p: p.stat().st_mtime) if existing else None
