"""SPY tactical-defense rule (V2), promoted from research scratch to the package.

The rule watches SPY against its 150-day moving average and produces a single
daily *exposure* number in {0.0, 0.30, 0.80, 1.0}. It is used as a de-risking
overlay on top of a portfolio model: on a defensive day the portfolio is held at
`exposure` of its normal weight and the remainder sits in cash. It is NOT a
signal to rotate into SPY -- that variant was tested and rejected.

State machine, evaluated once per trading day `i`. Every trigger reads day
`i - 1` values, so the decision is knowable at the previous close and executes
on day `i` (1-day lag, no look-ahead):

    invested (exposure 1.0)
      -> close below MA150 for `sell_confirm` consecutive days: exposure 0.0

    defensive (exposure < 1.0)
      -> quick recovery: if no RSI buy has fired yet this episode, the sell was
         shallow (within `quick_recovery_depth` of the MA) and price has closed
         back above the sell price within `quick_recovery_window` days,
         restore 1.0. Catches the whipsaw sells that dominate the trade log.
      -> otherwise RSI(14) tiers: below `rsi_high` -> `rsi_high_exposure`,
         below `rsi_low` -> `rsi_low_exposure`. Targets ratchet up only; they
         are never reduced, and the low tier can be reached without touching
         the high tier.
      -> restore: close above MA150 for `restore_confirm` consecutive days
         -> 1.0. The RSI tiers cap out below 1.0, so the last slice always
         needs either a quick recovery or this restore.

Parameters are frozen as `TUNED_DEFENSE` -- selected on the combined V7.3
overlay objective, see docs/macro/SPY_TACTICAL_DEFENSE.md. Do not change them
without rerunning that comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np
import pandas as pd

Phase = Literal["INVESTED", "DEFENSIVE"]
EventKind = Literal["LIQUIDATE", "QUICK_RESTORE", "SCALE_IN", "RESTORE"]

MOVING_AVERAGE_MINIMUM_OBSERVATIONS = 150
RSI_PERIOD = 14


@dataclass(frozen=True)
class DefenseParams:
    """Tuned SPY tactical-defense parameters."""

    moving_average_window: int = 150
    sell_confirm: int = 3
    restore_confirm: int = 2
    rsi_high: float = 35.0
    rsi_high_exposure: float = 0.30
    rsi_low: float = 30.0
    rsi_low_exposure: float = 0.80
    quick_recovery_depth: float | None = -0.05
    quick_recovery_window: int = 5

    def __post_init__(self) -> None:
        if self.moving_average_window < 2:
            raise ValueError("moving_average_window must be at least 2")
        if self.sell_confirm < 1 or self.restore_confirm < 1:
            raise ValueError("confirmation windows must be at least 1 day")
        if not 0.0 <= self.rsi_high_exposure <= self.rsi_low_exposure <= 1.0:
            raise ValueError("RSI tier exposures must satisfy 0 <= high <= low <= 1")
        if self.rsi_low > self.rsi_high:
            raise ValueError("rsi_low must be at or below rsi_high")


TUNED_DEFENSE = DefenseParams()


@dataclass(frozen=True)
class DefenseEvent:
    """One exposure change produced by the rule."""

    date: str
    kind: EventKind
    close: float
    exposure_from: float
    exposure_to: float
    rsi: float | None
    moving_average_deviation: float | None


@dataclass(frozen=True)
class DefenseTail:
    """State-machine internals as of the last evaluated day.

    These are what the next trigger depends on, so the dashboard can say what
    it is waiting for instead of only what already happened.
    """

    below_streak: int
    restore_streak: int
    rsi_bought: bool
    sell_date: str | None
    sell_price: float | None
    sell_depth: float | None
    days_since_sell: int | None


@dataclass(frozen=True)
class DefenseState:
    """Full point-in-time evaluation of the rule over a price history."""

    dates: list[str]
    close: np.ndarray
    moving_average: np.ndarray
    rsi: np.ndarray
    exposure: np.ndarray
    tail: DefenseTail
    events: list[DefenseEvent] = field(default_factory=list)

    @property
    def as_of(self) -> str:
        return self.dates[-1]

    @property
    def current_exposure(self) -> float:
        return float(self.exposure[-1])

    @property
    def phase(self) -> Phase:
        return "INVESTED" if self.current_exposure >= 1.0 else "DEFENSIVE"

    def event_on(self, day: str) -> DefenseEvent | None:
        for event in reversed(self.events):
            if event.date == day:
                return event
        return None


def compute_indicators(
    close: Sequence[float], window: int
) -> tuple[np.ndarray, np.ndarray]:
    """Simple moving average and Wilder RSI(14), both NaN until warm."""

    series = pd.Series(np.asarray(close, dtype=float))
    moving_average = series.rolling(window).mean().to_numpy()
    delta = series.diff()
    average_gain = delta.clip(lower=0).ewm(alpha=1 / RSI_PERIOD, adjust=False).mean()
    average_loss = (-delta.clip(upper=0)).ewm(alpha=1 / RSI_PERIOD, adjust=False).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        # pandas 3 may expose a read-only zero-copy NumPy view.  We intentionally
        # mask the warm-up period below, so request an owned, writable array.
        rsi = (100 - 100 / (1 + average_gain / average_loss)).to_numpy(copy=True)
    rsi[:RSI_PERIOD] = np.nan
    return moving_average, rsi


def evaluate_defense(
    frame: pd.DataFrame,
    params: DefenseParams = TUNED_DEFENSE,
    *,
    date_column: str = "Date",
    close_column: str = "Close",
) -> DefenseState:
    """Run the state machine over a daily price frame sorted by date."""

    if frame.empty:
        raise ValueError("price frame is empty")
    ordered = frame.sort_values(date_column).reset_index(drop=True)
    dates = [str(pd.Timestamp(value).date()) for value in ordered[date_column]]
    close = ordered[close_column].to_numpy(dtype=float)
    moving_average, rsi = compute_indicators(close, params.moving_average_window)

    total = len(close)
    exposure = np.ones(total)
    current = 1.0
    phase: Phase = "INVESTED"
    sell_price: float | None = None
    sell_index: int | None = None
    sell_depth: float | None = None
    below_streak = 0
    restore_streak = 0
    rsi_bought = False
    events: list[DefenseEvent] = []
    quick_recovery_on = params.quick_recovery_depth is not None

    def record(index: int, kind: EventKind, previous: float) -> None:
        prior_average = moving_average[index - 1]
        deviation = (
            float(close[index - 1] / prior_average - 1)
            if not np.isnan(prior_average)
            else None
        )
        prior_rsi = rsi[index - 1]
        events.append(
            DefenseEvent(
                date=dates[index],
                kind=kind,
                close=float(close[index]),
                exposure_from=previous,
                exposure_to=current,
                rsi=None if np.isnan(prior_rsi) else float(prior_rsi),
                moving_average_deviation=deviation,
            )
        )

    for index in range(1, total):
        price = close[index]
        if np.isnan(moving_average[index]):
            exposure[index] = current
            continue

        prior_average = moving_average[index - 1]
        prior_below = (
            bool(close[index - 1] < prior_average) if not np.isnan(prior_average) else False
        )
        below_streak = below_streak + 1 if prior_below else 0

        if phase == "INVESTED":
            if below_streak >= params.sell_confirm and current > 0.0:
                previous, current = current, 0.0
                phase = "DEFENSIVE"
                sell_price, sell_index = float(price), index
                sell_depth = (
                    float(close[index - 1] / prior_average - 1)
                    if not np.isnan(prior_average)
                    else 0.0
                )
                restore_streak = 0
                rsi_bought = False
                record(index, "LIQUIDATE", previous)
        else:
            restored = False
            if (
                quick_recovery_on
                and not rsi_bought
                and sell_index is not None
                and index - sell_index <= params.quick_recovery_window
                and current < 1.0
                and sell_depth is not None
                and sell_depth > params.quick_recovery_depth
                and sell_price is not None
                and price > sell_price
            ):
                previous, current = current, 1.0
                phase, restored = "INVESTED", True
                record(index, "QUICK_RESTORE", previous)

            if not restored:
                prior_rsi = rsi[index - 1]
                target = 0.0
                if not np.isnan(prior_rsi):
                    if prior_rsi < params.rsi_low:
                        target = params.rsi_low_exposure
                    elif prior_rsi < params.rsi_high:
                        target = params.rsi_high_exposure
                if target > current:
                    previous, current = current, target
                    rsi_bought = True
                    if current >= 1.0:
                        phase = "INVESTED"
                    record(index, "SCALE_IN", previous)

                if phase == "DEFENSIVE" and current < 1.0:
                    prior_above = (
                        bool(close[index - 1] > prior_average)
                        if not np.isnan(prior_average)
                        else False
                    )
                    restore_streak = restore_streak + 1 if prior_above else 0
                    if restore_streak >= params.restore_confirm:
                        previous, current = current, 1.0
                        phase = "INVESTED"
                        record(index, "RESTORE", previous)

        exposure[index] = current

    return DefenseState(
        dates=dates,
        close=close,
        moving_average=moving_average,
        rsi=rsi,
        exposure=exposure,
        tail=DefenseTail(
            below_streak=below_streak,
            restore_streak=restore_streak,
            rsi_bought=rsi_bought,
            sell_date=dates[sell_index] if sell_index is not None else None,
            sell_price=sell_price,
            sell_depth=sell_depth,
            days_since_sell=(total - 1 - sell_index) if sell_index is not None else None,
        ),
        events=events,
    )
