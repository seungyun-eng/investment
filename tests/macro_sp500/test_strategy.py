from __future__ import annotations

import pandas as pd
import pytest

from stock_research.macro_sp500.config import (
    MacroSp500Params,
    MacroSp500Settings,
)
from stock_research.macro_sp500.strategy import generate_target_weights


def test_warning_panic_and_recovery_state_machine(
    macro_settings: MacroSp500Settings,
    macro_params: MacroSp500Params,
) -> None:
    features = pd.DataFrame(
        {
            "Date": pd.bdate_range("2024-01-02", periods=4),
            "VixPercentile": [0.50, 0.91, 0.30, 0.30],
            "Drawdown": [0.0, 0.0, 0.0, 0.0],
            "WarningScore": [2, 2, 0, 0],
            "FeaturesReady": [True, True, True, True],
        }
    )

    signals = generate_target_weights(features, macro_params, macro_settings)

    assert signals["State"].tolist() == ["ARMED", "PANIC", "PANIC", "RECOVERY"]
    assert signals["TargetWeight"].tolist() == pytest.approx([0.80, 0.80, 0.80, 0.70])
    assert signals.loc[1, "Reason"] == "PANIC_CONFIRMED_LEVEL_1"
    assert signals.loc[3, "Reason"].startswith("RECOVERY_EXIT")
