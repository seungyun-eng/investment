import pandas as pd

from stock_research.strategies.technical import (
    TechnicalParams,
    technical_buy_signal,
    technical_sell_signal,
)
from stock_research.strategies.vix import VixParams, vix_buy_signal, vix_sell_signal


def test_vix_signals():
    params = VixParams(35, 65, 0.02, 25, 20)
    buy_row = pd.Series({
        "VIX": 30, "RSI (14일)": 30, "종가": 90,
        "볼린저밴드 하단": 90, "볼린저밴드 상단": 110,
        "MACD": 2, "MACD 시그널": 1,
    })
    sell_row = pd.Series({
        "VIX": 20, "RSI (14일)": 70, "종가": 120,
        "볼린저밴드 하단": 90, "볼린저밴드 상단": 110,
        "MACD": 0, "MACD 시그널": 1,
    })
    assert vix_buy_signal(buy_row, params)
    assert vix_sell_signal(sell_row, params)


def test_vix_signals_accept_either_bollinger_or_macd_but_keep_vix_gate():
    params = VixParams(35, 65, 0.0, 25, 20)
    bollinger_only_buy = pd.Series({
        "VIX": 30.0, "RSI (14일)": 30, "종가": 89,
        "볼린저밴드 하단": 90, "볼린저밴드 상단": 110,
        "MACD": 0, "MACD 시그널": 1,
    })
    macd_only_buy = bollinger_only_buy.copy()
    macd_only_buy["종가"] = 95
    macd_only_buy["MACD"] = 2
    blocked_buy = macd_only_buy.copy()
    blocked_buy["VIX"] = 24.99

    bollinger_only_sell = pd.Series({
        "VIX": 19.0, "RSI (14일)": 70, "종가": 111,
        "볼린저밴드 하단": 90, "볼린저밴드 상단": 110,
        "MACD": 2, "MACD 시그널": 1,
    })
    macd_only_sell = bollinger_only_sell.copy()
    macd_only_sell["종가"] = 105
    macd_only_sell["MACD"] = 0
    blocked_sell = macd_only_sell.copy()
    blocked_sell["VIX"] = 20.01

    assert vix_buy_signal(bollinger_only_buy, params)
    assert vix_buy_signal(macd_only_buy, params)
    assert not vix_buy_signal(blocked_buy, params)
    assert vix_sell_signal(bollinger_only_sell, params)
    assert vix_sell_signal(macd_only_sell, params)
    assert not vix_sell_signal(blocked_sell, params)


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
