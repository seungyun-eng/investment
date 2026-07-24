import pandas as pd

from stock_research.strategies.technical import (
    TechnicalParams,
    technical_buy_signal,
    technical_sell_signal,
)
from stock_research.strategies.vix import VixParams, vix_buy_signal, vix_sell_signal


def test_vix_signals():
    params = VixParams(25, 15, 35, 65, 0.02)
    buy_row = pd.Series({
        "VIX": 30, "RSI (14일)": 30, "종가": 90,
        "볼린저밴드 하단": 90, "볼린저밴드 상단": 110,
        "MACD": 2, "MACD 시그널": 1,
    })
    sell_row = pd.Series({
        "VIX": 10, "RSI (14일)": 70, "종가": 120,
        "볼린저밴드 하단": 90, "볼린저밴드 상단": 110,
        "MACD": 0, "MACD 시그널": 1,
    })
    assert vix_buy_signal(buy_row, params)
    assert vix_sell_signal(sell_row, params)


def test_technical_signals():
    params = TechnicalParams(rsi_sell_th=70, boll_buffer=0.02)
    buy_row = pd.Series({
        "OBV": 11, "OBV_SIG9": 10, "RSI_14": 40, "RSI_SIG9": 45,
        "종가": 90, "BB_LOWER": 90, "BB_UPPER": 110,
        "MACD": 2, "MACD_SIG": 1,
    })
    sell_row = pd.Series({
        "OBV": 9, "OBV_SIG9": 10, "RSI_14": 75, "RSI_SIG9": 60,
        "종가": 120, "BB_LOWER": 90, "BB_UPPER": 110,
        "MACD": 0, "MACD_SIG": 1,
    })
    assert technical_buy_signal(buy_row, params)
    assert technical_sell_signal(sell_row, params)
