"""Feature engineering and LSTM windowing.

All features are computed from PAST data only (rolling windows, EWMs, lagged
returns). Targets are explicitly shifted into the future and dropped from the
feature matrix at train time, so there is no look-ahead leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET_COLUMNS = ["Target_Close_1d", "Target_Return_1d"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators + next-day targets, then drop warm-up NaN rows."""
    d = df.copy()

    # Lagged returns (past only)
    d["Return_1d"] = d["Close"].pct_change(1)
    d["Return_7d"] = d["Close"].pct_change(7)
    d["Return_30d"] = d["Close"].pct_change(30)

    # Moving averages / MACD
    d["MA7"] = d["Close"].rolling(7).mean()
    d["MA30"] = d["Close"].rolling(30).mean()
    d["MA90"] = d["Close"].rolling(90).mean()
    d["EMA12"] = d["Close"].ewm(span=12, adjust=False).mean()
    d["EMA26"] = d["Close"].ewm(span=26, adjust=False).mean()
    d["MACD"] = d["EMA12"] - d["EMA26"]

    # Bollinger bands
    d["BB_mid"] = d["Close"].rolling(20).mean()
    bb_std = d["Close"].rolling(20).std()
    d["BB_upper"] = d["BB_mid"] + 2 * bb_std
    d["BB_lower"] = d["BB_mid"] - 2 * bb_std
    d["BB_width"] = d["BB_upper"] - d["BB_lower"]

    # Volatility (30d std of daily returns)
    d["Volatility"] = d["Return_1d"].rolling(30).std()

    # RSI(14)
    delta = d["Close"].diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    d["RSI"] = 100 - 100 / (1 + rs)

    # Volume trend
    d["Volume_MA7"] = d["Volume"].rolling(7).mean()

    # Next-day targets (future — never used as features)
    d["Target_Close_1d"] = d["Close"].shift(-1)
    d["Target_Return_1d"] = d["Close"].shift(-1) / d["Close"] - 1.0

    d = d.dropna()
    return d


def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric feature matrix with all Target_* columns removed."""
    x = df.drop(columns=[c for c in df.columns if c.startswith("Target_")], errors="ignore")
    return x.select_dtypes(include=[np.number])


def build_windows(values: np.ndarray, look_back: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """Build (X, y) sliding windows for the LSTM.

    X[i] = values[i : i+look_back], y[i] = values[i+look_back]
    (each window strictly precedes its target — no leakage).
    """
    values = np.asarray(values, dtype=float).ravel()
    if len(values) <= look_back:
        raise ValueError(f"Need > {look_back} points, got {len(values)}")
    X, y = [], []
    for i in range(len(values) - look_back):
        X.append(values[i:i + look_back])
        y.append(values[i + look_back])
    return np.array(X)[..., np.newaxis], np.array(y)
