"""Feature engineering and LSTM windowing.

All features are computed from PAST data only (rolling windows, EWMs, lagged
returns). Targets are explicitly shifted into the future and dropped from the
feature matrix at train time, so there is no look-ahead leakage.

Multi-horizon targets (1d / 7d / 30d by default) support production price prediction.
Optional external columns (FearGreed, Funding, FRED) are left-joined by date.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET_COLUMNS = ["Target_Close_1d", "Target_Return_1d"]
DEFAULT_HORIZONS = (1, 7, 30)


def add_features(
    df: pd.DataFrame,
    *,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    external: pd.DataFrame | None = None,
    cross_asset: pd.DataFrame | None = None,
    for_prediction: bool = False,
) -> pd.DataFrame:
    """Add technical indicators + multi-horizon targets, then drop warm-up NaN rows.

    for_prediction=True keeps the latest rows (no Target_* dropna) so we can
    score a live next-day/week/month forecast from the most recent close.
    """
    d = df.copy()

    # Lagged returns (past only)
    d["Return_1d"] = d["Close"].pct_change(1)
    d["Return_7d"] = d["Close"].pct_change(7)
    d["Return_30d"] = d["Close"].pct_change(30)

    # Moving averages / MACD (MA50/200 + golden/death cross)
    d["MA7"] = d["Close"].rolling(7).mean()
    d["MA30"] = d["Close"].rolling(30).mean()
    d["MA50"] = d["Close"].rolling(50).mean()
    d["MA90"] = d["Close"].rolling(90).mean()
    d["MA200"] = d["Close"].rolling(200).mean()
    d["EMA12"] = d["Close"].ewm(span=12, adjust=False).mean()
    d["EMA26"] = d["Close"].ewm(span=26, adjust=False).mean()
    d["MACD"] = d["EMA12"] - d["EMA26"]
    d["Price_vs_MA50"] = d["Close"] / d["MA50"] - 1.0
    d["Price_vs_MA200"] = d["Close"] / d["MA200"] - 1.0
    d["MA50_vs_MA200"] = d["MA50"] / d["MA200"] - 1.0
    # 1 = golden cross day, -1 = death cross day, 0 otherwise
    cross = np.sign(d["MA50"] - d["MA200"])
    d["GoldenDeathCross"] = cross.diff().fillna(0.0).clip(-1, 1)

    # Bollinger bands
    d["BB_mid"] = d["Close"].rolling(20).mean()
    bb_std = d["Close"].rolling(20).std()
    d["BB_upper"] = d["BB_mid"] + 2 * bb_std
    d["BB_lower"] = d["BB_mid"] - 2 * bb_std
    d["BB_width"] = d["BB_upper"] - d["BB_lower"]

    # Volatility (30d std of daily returns)
    d["Volatility"] = d["Return_1d"].rolling(30).std()

    # ATR(14) + range / drawdown (regime-friendly)
    prev_close = d["Close"].shift(1)
    tr = pd.concat([
        d["High"] - d["Low"],
        (d["High"] - prev_close).abs(),
        (d["Low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    d["ATR14"] = tr.rolling(14).mean()
    d["ATR14_pct"] = d["ATR14"] / d["Close"]
    d["HL_range"] = (d["High"] - d["Low"]) / d["Close"]
    d["Drawdown_30d"] = d["Close"] / d["Close"].rolling(30).max() - 1.0

    # RSI(14)
    delta = d["Close"].diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    d["RSI"] = 100 - 100 / (1 + rs)

    # Volume trend
    d["Volume_MA7"] = d["Volume"].rolling(7).mean()
    d["Volume_z20"] = (d["Volume"] - d["Volume"].rolling(20).mean()) / d["Volume"].rolling(20).std()

    # Free alpha stack (Kaggle / QuantForge-style): lags, ROC, OBV, vol regimes
    for lag in (1, 2, 3, 5, 10):
        d[f"Return_lag{lag}"] = d["Return_1d"].shift(lag)
    d["ROC7"] = d["Close"].pct_change(7)
    d["ROC14"] = d["Close"].pct_change(14)
    d["ROC30"] = d["Close"].pct_change(30)
    # On-Balance Volume (signed volume accumulation)
    signed_vol = np.sign(d["Close"].diff().fillna(0.0)) * d["Volume"]
    d["OBV"] = signed_vol.cumsum()
    d["OBV_chg7"] = d["OBV"].pct_change(7).replace([np.inf, -np.inf], np.nan)
    d["OBV_z20"] = (d["OBV"] - d["OBV"].rolling(20).mean()) / d["OBV"].rolling(20).std()
    # Realized vol regimes + vol-of-vol
    d["Vol7"] = d["Return_1d"].rolling(7).std()
    d["Vol14"] = d["Return_1d"].rolling(14).std()
    d["VolOfVol30"] = d["Volatility"].rolling(30).std()
    d["Vol_regime"] = (d["Volatility"] / d["Volatility"].rolling(90).median() - 1.0)
    # Range position (price-level agnostic)
    roll_hi = d["High"].rolling(20).max()
    roll_lo = d["Low"].rolling(20).min()
    d["RangePos20"] = (d["Close"] - roll_lo) / (roll_hi - roll_lo).replace(0, np.nan)
    d["Close_vs_MA7"] = d["Close"] / d["MA7"] - 1.0
    d["Close_vs_MA30"] = d["Close"] / d["MA30"] - 1.0
    # Day-of-week cyclical (crypto trades 24/7 but still mild weekly seasonality)
    dow = d.index.dayofweek.astype(float)
    d["Dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    d["Dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

    if external is not None and len(external):
        ext = external.copy()
        ext.index = pd.to_datetime(ext.index).tz_localize(None)
        # Drop near-empty snapshot columns (e.g. one-day CMC dominance) so
        # they do not wipe the whole frame via dropna().
        keep = [c for c in ext.columns if ext[c].notna().mean() >= 0.4]
        ext = ext[keep]
        if len(ext.columns):
            d = d.join(ext, how="left")
            d[list(ext.columns)] = d[list(ext.columns)].ffill(limit=5)

    if cross_asset is not None and len(cross_asset):
        ca = cross_asset.copy()
        ca.index = pd.to_datetime(ca.index).tz_localize(None)
        d = d.join(ca, how="left")
        d[list(ca.columns)] = d[list(ca.columns)].ffill(limit=3)

    # Multi-horizon targets (future — never used as features)
    for h in horizons:
        d[f"Target_Close_{h}d"] = d["Close"].shift(-h)
        d[f"Target_Return_{h}d"] = d["Close"].shift(-h) / d["Close"] - 1.0

    d = d.replace([np.inf, -np.inf], np.nan)
    if for_prediction:
        # Keep latest bars; require warm-up through MA200 when present
        need = ["Close", "Return_1d", "MA90", "MA200", "RSI", "Volatility", "ATR14"]
        d = d.dropna(subset=[c for c in need if c in d.columns])
        return d

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
