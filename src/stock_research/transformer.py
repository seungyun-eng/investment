from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TransformerConfig:
    sequence_length: int = 60
    horizon_days: int = 20
    d_model: int = 64
    heads: int = 4
    ff_dim: int = 128
    layers: int = 2
    epochs: int = 50
    batch_size: int = 32


def _require_ml():
    try:
        import tensorflow as tf
        from tensorflow import keras
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise RuntimeError(
            "Transformer support is optional. Install requirements-ml.txt."
        ) from exc
    return tf, keras, StandardScaler


def prepare_regression_sequences(
    frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    target_column: str = "종가",
    sequence_length: int = 60,
    horizon_days: int = 20,
):
    clean = frame[feature_columns + [target_column]].apply(
        pd.to_numeric, errors="coerce"
    ).dropna()
    values = clean[feature_columns].to_numpy(dtype=np.float32)
    target = clean[target_column].pct_change(horizon_days).shift(-horizon_days)
    target_values = target.to_numpy(dtype=np.float32)

    x, y = [], []
    for end in range(sequence_length, len(clean) - horizon_days):
        start = end - sequence_length
        if np.isnan(target_values[end]):
            continue
        x.append(values[start:end])
        y.append(target_values[end])
    return np.asarray(x), np.asarray(y)


def build_transformer_regressor(
    sequence_length: int,
    feature_count: int,
    *,
    d_model: int = 64,
    heads: int = 4,
    ff_dim: int = 128,
    layers: int = 2,
):
    tf, keras, _ = _require_ml()
    inputs = keras.Input(shape=(sequence_length, feature_count))
    x = keras.layers.Dense(d_model)(inputs)
    for _ in range(layers):
        attention = keras.layers.MultiHeadAttention(
            num_heads=heads, key_dim=d_model // heads
        )(x, x)
        x = keras.layers.LayerNormalization()(x + attention)
        feed_forward = keras.Sequential([
            keras.layers.Dense(ff_dim, activation="relu"),
            keras.layers.Dense(d_model),
        ])(x)
        x = keras.layers.LayerNormalization()(x + feed_forward)
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dense(64, activation="relu")(x)
    output = keras.layers.Dense(1)(x)
    model = keras.Model(inputs, output)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def train_transformer(
    frame: pd.DataFrame,
    feature_columns: list[str],
    output_path: Path,
    config: TransformerConfig = TransformerConfig(),
):
    _, _, StandardScaler = _require_ml()
    data = frame.copy()
    scaler = StandardScaler()
    data[feature_columns] = scaler.fit_transform(
        data[feature_columns].apply(pd.to_numeric, errors="coerce")
    )
    x, y = prepare_regression_sequences(
        data,
        feature_columns,
        sequence_length=config.sequence_length,
        horizon_days=config.horizon_days,
    )
    if len(x) < 20:
        raise ValueError("Not enough valid rows to train the Transformer.")
    split = int(len(x) * 0.8)
    model = build_transformer_regressor(
        config.sequence_length,
        len(feature_columns),
        d_model=config.d_model,
        heads=config.heads,
        ff_dim=config.ff_dim,
        layers=config.layers,
    )
    model.fit(
        x[:split],
        y[:split],
        validation_data=(x[split:], y[split:]),
        epochs=config.epochs,
        batch_size=config.batch_size,
        verbose=1,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)
    return model
