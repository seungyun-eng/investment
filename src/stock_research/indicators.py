from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .io_utils import atomic_to_csv, read_csv_fallback

COLUMN_ALIASES = {
    "date": "날짜",
    "price": "종가",
    "close": "종가",
    "adjclose": "종가",
    "adj_close": "종가",
    "open": "시가",
    "high": "고가",
    "low": "저가",
    "vol": "거래량(raw)",
    "volume": "거래량(raw)",
    "change%": "등락률(%)",
    "changepct": "등락률(%)",
    "change": "등락률(%)",
}


def _clean_key(value: str) -> str:
    return value.strip().lower().replace(" ", "").replace(".", "")


def parse_number(value) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip().replace(",", "").replace("$", "")
    if not text or text.lower() in {"nan", "none", "-"}:
        return np.nan
    multiplier = 1.0
    if text[-1:].lower() == "k":
        multiplier, text = 1_000.0, text[:-1]
    elif text[-1:].lower() == "m":
        multiplier, text = 1_000_000.0, text[:-1]
    elif text[-1:].lower() == "b":
        multiplier, text = 1_000_000_000.0, text[:-1]
    text = text.replace("%", "")
    try:
        return float(text) * multiplier
    except ValueError:
        return np.nan


def normalize_price_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename_map: dict[str, str] = {}
    for col in out.columns:
        key = _clean_key(str(col))
        if key in COLUMN_ALIASES:
            rename_map[col] = COLUMN_ALIASES[key]
    out = out.rename(columns=rename_map)

    required = {"날짜", "종가"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"Missing required price columns: {sorted(missing)}")

    out["날짜"] = pd.to_datetime(out["날짜"], errors="coerce")
    for col in ("종가", "시가", "고가", "저가"):
        if col in out:
            out[col] = out[col].map(parse_number)

    if "거래량(raw)" in out:
        out["거래량"] = out["거래량(raw)"].map(parse_number)
        out = out.drop(columns=["거래량(raw)"])
    elif "거래량" in out:
        out["거래량"] = out["거래량"].map(parse_number)

    if "등락률(%)" in out:
        out["등락률(%)"] = out["등락률(%)"].map(parse_number)

    keep = [
        c for c in ("날짜", "종가", "시가", "고가", "저가", "거래량", "등락률(%)")
        if c in out.columns
    ]
    return (
        out[keep]
        .dropna(subset=["날짜", "종가"])
        .sort_values("날짜")
        .drop_duplicates("날짜", keep="last")
        .reset_index(drop=True)
    )


def add_indicators(df: pd.DataFrame, *, drop_warmup: bool = False) -> pd.DataFrame:
    if not {"날짜", "종가"}.issubset(df.columns):
        raise ValueError("Data requires 날짜 and 종가 columns.")

    out = df.copy().sort_values("날짜").reset_index(drop=True)
    close = pd.to_numeric(out["종가"], errors="coerce")
    volume = pd.to_numeric(out.get("거래량"), errors="coerce")

    # RSI 14: preserves the simple rolling-average method used in the notebooks.
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out["RSI (14일)"] = 100 - (100 / (1 + rs))
    out["RSI_14"] = out["RSI (14일)"]
    out["RSI_SIG9"] = out["RSI_14"].rolling(9).mean()

    # Bollinger 20; pandas sample std (ddof=1) matches the original backtest notebooks.
    mid = close.rolling(20).mean()
    std = close.rolling(20).std()
    out["볼린저밴드 상단"] = mid + 2 * std
    out["볼린저밴드 하단"] = mid - 2 * std
    out["BB_MID"] = mid
    out["BB_STD"] = std
    out["BB_UPPER"] = out["볼린저밴드 상단"]
    out["BB_LOWER"] = out["볼린저밴드 하단"]

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    out["MACD"] = ema12 - ema26
    out["MACD 시그널"] = out["MACD"].ewm(span=9, adjust=False).mean()
    out["MACD_SIG"] = out["MACD 시그널"]

    for period in (5, 10, 20, 60, 120, 200):
        out[f"SMA {period}일"] = close.rolling(period).mean()

    for label, periods in {"2주": 10, "3개월": 63, "6개월": 126, "1년": 252}.items():
        out[f"가격 상승률 ({label})"] = close.pct_change(periods) * 100
        if volume is not None:
            out[f"거래량 상승률 ({label})"] = volume.pct_change(periods) * 100

    if volume is not None and not volume.isna().all():
        out["OBV"] = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        out["OBV_SIG9"] = out["OBV"].rolling(9).mean()

    if drop_warmup:
        essential = [
            c for c in (
                "RSI_14", "RSI_SIG9", "BB_UPPER", "BB_LOWER",
                "MACD", "MACD_SIG", "OBV", "OBV_SIG9"
            ) if c in out.columns
        ]
        out = out.dropna(subset=essential).reset_index(drop=True)
    return out


def preprocess_company_dir(
    company_dir: Path,
    output_root: Path,
) -> Path | None:
    frames: list[pd.DataFrame] = []
    for csv_path in sorted(company_dir.glob("*.csv")):
        try:
            frames.append(
                normalize_price_columns(read_csv_fallback(csv_path))
            )
        except Exception as exc:  # noqa: BLE001
            print(f"SKIP {csv_path.name}: {exc}")
    if not frames:
        return None
    combined = (
        pd.concat(frames, ignore_index=True)
        .sort_values("날짜")
        .drop_duplicates("날짜", keep="last")
        .reset_index(drop=True)
    )
    processed = add_indicators(combined, drop_warmup=False)
    output = output_root / f"{company_dir.name}_지표포함.csv"
    atomic_to_csv(processed, output, index=False)
    return output


def preprocess_folder(raw_root: Path, output_root: Path) -> list[Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for company_dir in sorted(p for p in raw_root.iterdir() if p.is_dir()):
        output = preprocess_company_dir(company_dir, output_root)
        if output is not None:
            outputs.append(output)
    return outputs
