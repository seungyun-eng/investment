from __future__ import annotations

import argparse

from stock_research.data_loading import load_processed
from stock_research.paths import load_paths
from stock_research.transformer import TransformerConfig, train_transformer


DEFAULT_FEATURES = [
    "종가", "거래량", "RSI_14", "MACD", "MACD_SIG",
    "BB_UPPER", "BB_LOWER", "OBV", "OBV_SIG9",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("company")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=20)
    args = parser.parse_args()

    paths = load_paths()
    data = load_processed(paths.processed, args.company, args.start, args.end)
    features = [column for column in DEFAULT_FEATURES if column in data.columns]
    config = TransformerConfig(epochs=args.epochs, horizon_days=args.horizon)
    output = paths.transformer_results / f"{args.company}_transformer.keras"
    train_transformer(data, features, output, config)
    print(output)


if __name__ == "__main__":
    main()
