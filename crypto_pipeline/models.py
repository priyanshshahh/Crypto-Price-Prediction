"""Forecasting models with honest, time-ordered evaluation.

Evaluation policy (applies to every model here):
  * Chronological 80/20 split — never shuffled.
  * Scalers are fit on the TRAIN portion only.
  * Primary metrics are RETURNS-based (next-day return RMSE/MAE/R^2 and
    directional accuracy), because price-level R^2 on next-day close is
    dominated by autocorrelation and is trivially high even for a naive
    "predict yesterday's price" model.
  * Every model is compared against the persistence baseline
    (predict tomorrow's close = today's close, i.e. zero return).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import build_windows, feature_matrix

RANDOM_STATE = 42


# ── Shared evaluation helpers ─────────────────────────────────────────────────

def time_split(n: int, test_frac: float = 0.2) -> int:
    """Index that splits [0, n) chronologically into train/test."""
    return int(n * (1 - test_frac))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def directional_accuracy(actual_returns: np.ndarray, predicted_returns: np.ndarray) -> float:
    """Share of test days where the predicted sign of the move matches reality."""
    actual = np.sign(np.asarray(actual_returns))
    pred = np.sign(np.asarray(predicted_returns))
    mask = actual != 0  # ignore flat days
    if mask.sum() == 0:
        return 0.5
    return float((actual[mask] == pred[mask]).mean())


def price_metrics_from_returns(prev_close: np.ndarray, actual_ret: np.ndarray,
                               pred_ret: np.ndarray) -> dict:
    """Convert return predictions to price predictions and score them."""
    actual_price = prev_close * (1 + actual_ret)
    pred_price = prev_close * (1 + pred_ret)
    return regression_metrics(actual_price, pred_price)


def persistence_baseline(prev_close: np.ndarray, actual_ret: np.ndarray) -> dict:
    """Naive baseline: tomorrow's close = today's close (zero predicted return)."""
    zeros = np.zeros_like(actual_ret)
    out = {
        "returns": regression_metrics(actual_ret, zeros),
        "price": price_metrics_from_returns(prev_close, actual_ret, zeros),
        # persistence has no directional view; report the always-up strawman too
        "up_day_share": float((actual_ret > 0).mean()),
    }
    return out


# ── Regularized regressions & tree ensembles ─────────────────────────────────

def run_regressions(df_feat: pd.DataFrame, test_frac: float = 0.2) -> dict:
    """Train 6 regressors to predict NEXT-DAY RETURN. Returns metrics + baseline."""
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import ElasticNet, Lasso, Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR

    X = feature_matrix(df_feat)
    y = df_feat["Target_Return_1d"].to_numpy()
    close = df_feat["Close"].to_numpy()

    split = time_split(len(X), test_frac)
    X_tr, X_te = X.iloc[:split], X.iloc[split:]
    y_tr, y_te = y[:split], y[split:]
    close_te_prev = close[split:]  # Close at day t (the day the prediction is made)

    scaler = StandardScaler().fit(X_tr)          # train-only fit: no leakage
    X_tr_sc = scaler.transform(X_tr)
    X_te_sc = scaler.transform(X_te)

    models = {
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.0005, max_iter=50000),
        "ElasticNet": ElasticNet(alpha=0.0005, l1_ratio=0.5, max_iter=50000),
        "SVR": SVR(C=1.0, epsilon=0.001),
        "RandomForest": RandomForestRegressor(n_estimators=200, max_depth=6,
                                              random_state=RANDOM_STATE, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                                      random_state=RANDOM_STATE),
    }

    results = []
    for name, model in models.items():
        model.fit(X_tr_sc, y_tr)
        pred = model.predict(X_te_sc)
        row = {
            "name": name,
            "returns": regression_metrics(y_te, pred),
            "price": price_metrics_from_returns(close_te_prev, y_te, pred),
            "directional_accuracy": directional_accuracy(y_te, pred),
        }
        results.append(row)

    baseline = persistence_baseline(close_te_prev, y_te)
    return {
        "models": results,
        "baseline_persistence": baseline,
        "n_train": int(split),
        "n_test": int(len(X) - split),
        "test_start": str(X_te.index[0].date()),
        "test_end": str(X_te.index[-1].date()),
    }


# ── ARIMA ─────────────────────────────────────────────────────────────────────

def run_arima(close: pd.Series, test_frac: float = 0.2, steps: int = 30,
              orders: tuple | None = None) -> dict:
    """ARIMA/ARMA on log-returns (not raw prices).

    Why not ARIMA(p,1,q) on price levels: AIC often picks (0,1,1)/(0,1,0), whose
    multi-step forecast is a *horizontal line* (random walk, no drift). That looks
    broken in the UI even though statsmodels is "working."

    Approach (standard for near-RW crypto):
      1. Fit ARMA(p,q) on daily log-returns with AIC order search
      2. Holdout: one-step-ahead return forecasts → price = prev * exp(r̂)
      3. Future path: multi-step return forecast → compound from last close
      4. Attach 80% prediction intervals on the return path → price bands

    Callers: pipeline.py, tests/test_models.py.
    Forecast schema: list of (date, price) or (date, price, lo, hi).
    """
    import warnings

    from statsmodels.tsa.arima.model import ARIMA

    if orders is None:
        orders = tuple(
            (p, 0, q)
            for p in range(0, 4)
            for q in range(0, 4)
            if not (p == 0 and q == 0)
        )

    close = close.astype(float).dropna()
    log_px = np.log(close.to_numpy())
    log_ret = pd.Series(np.diff(log_px), index=close.index[1:], name="log_ret")

    n = len(log_ret)
    split = time_split(n, test_frac)
    train_ret = log_ret.iloc[:split]

    best_aic, best_order, best_res = np.inf, None, None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for order in orders:
            try:
                # trend='c' = nonzero mean log-return (drift) — without this,
                # MA-only models mean-revert to 0 and the compounded path goes flat.
                res = ARIMA(train_ret, order=order, trend="c").fit()
                if np.isfinite(res.aic) and res.aic < best_aic:
                    best_aic, best_order, best_res = float(res.aic), order, res
            except Exception:
                continue
    if best_res is None:
        best_order = (1, 0, 0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            best_res = ARIMA(train_ret, order=best_order, trend="c").fit()
            best_aic = float(best_res.aic)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res_full = best_res.apply(log_ret)
        pred_ret_log = res_full.get_prediction(
            start=split, dynamic=False
        ).predicted_mean.to_numpy()

    prev_close = close.iloc[split:n].to_numpy()
    actual_price = close.iloc[split + 1:].to_numpy()
    m = min(len(pred_ret_log), len(prev_close), len(actual_price))
    pred_ret_log = pred_ret_log[:m]
    prev_close = prev_close[:m]
    actual_price = actual_price[:m]
    pred_price = prev_close * np.exp(pred_ret_log)
    actual_ret = actual_price / prev_close - 1.0
    pred_ret = pred_price / prev_close - 1.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final_res = ARIMA(log_ret, order=best_order, trend="c").fit()
        fc = final_res.get_forecast(steps=steps)
    fc_ret = np.asarray(fc.predicted_mean, dtype=float)
    try:
        ci = fc.conf_int(alpha=0.2)
        if isinstance(ci, pd.DataFrame):
            lo_ret = ci.iloc[:, 0].to_numpy(dtype=float)
            hi_ret = ci.iloc[:, 1].to_numpy(dtype=float)
        else:
            lo_ret = np.asarray(ci)[:, 0]
            hi_ret = np.asarray(ci)[:, 1]
    except Exception:
        sigma = float(np.std(log_ret))
        lo_ret = fc_ret - 1.28 * sigma
        hi_ret = fc_ret + 1.28 * sigma

    last = float(close.iloc[-1])
    fc_vals = last * np.cumprod(np.exp(fc_ret))
    fc_lo = last * np.cumprod(np.exp(lo_ret))
    fc_hi = last * np.cumprod(np.exp(hi_ret))
    fc_lo, fc_hi = np.minimum(fc_lo, fc_hi), np.maximum(fc_lo, fc_hi)
    fc_dates = pd.date_range(close.index[-1] + pd.Timedelta(days=1), periods=steps, freq="D")

    forecasts = [
        (d.strftime("%Y-%m-%d"), float(v), float(lo), float(hi))
        for d, v, lo, hi in zip(fc_dates, fc_vals, fc_lo, fc_hi)
    ]

    return {
        "order": list(best_order),
        "aic": float(best_aic),
        "on": "log_returns",
        "price": regression_metrics(actual_price, pred_price),
        "returns": regression_metrics(actual_ret, pred_ret),
        "directional_accuracy": directional_accuracy(actual_ret, pred_ret),
        "baseline_persistence": persistence_baseline(prev_close, actual_ret),
        "n_test": int(m),
        "forecasts": forecasts,
        "pred_price": pred_price,
        "actual_price": actual_price,
        "forecast_path_std": float(np.std(fc_vals)),
    }


# ── LSTM / GRU (deep sequence models) ─────────────────────────────────────────

def run_lstm(close: pd.Series, test_frac: float = 0.2, steps: int = 30,
             look_back: int = 60, epochs: int = 40, batch_size: int = 32) -> dict:
    """Two-layer LSTM predicting next-day close from a 60-day window."""
    return _run_rnn(
        close, cell="lstm", test_frac=test_frac, steps=steps,
        look_back=look_back, epochs=epochs, batch_size=batch_size,
    )


def run_gru(close: pd.Series, test_frac: float = 0.2, steps: int = 30,
            look_back: int = 60, epochs: int = 40, batch_size: int = 32) -> dict:
    """Two-layer GRU — often similar accuracy to LSTM, fewer parameters (research pillar)."""
    return _run_rnn(
        close, cell="gru", test_frac=test_frac, steps=steps,
        look_back=look_back, epochs=epochs, batch_size=batch_size,
    )


def _run_rnn(close: pd.Series, *, cell: str, test_frac: float, steps: int,
             look_back: int, epochs: int, batch_size: int) -> dict:
    """Shared LSTM/GRU trainer. MinMaxScaler fit on TRAIN closes only (no leakage).

    Callers: run_lstm / run_gru ← pipeline.py (classic + production).
    Extends existing LSTM path; no separate file. Forecasts: (YYYY-MM-DD, price).
    """
    import random

    import tensorflow as tf
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import GRU, LSTM, Dense, Dropout, Input
    from tensorflow.keras.models import Sequential

    tf.get_logger().setLevel("ERROR")
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)

    Cell = LSTM if cell == "lstm" else GRU
    n = len(close)
    split = time_split(n, test_frac)

    scaler = MinMaxScaler()
    scaler.fit(close.iloc[:split].to_numpy().reshape(-1, 1))
    scaled = scaler.transform(close.to_numpy().reshape(-1, 1)).ravel()

    X_all, y_all = build_windows(scaled, look_back)
    first_test_window = split - look_back
    X_tr, X_te = X_all[:first_test_window], X_all[first_test_window:]
    y_tr, y_te = y_all[:first_test_window], y_all[first_test_window:]

    model = Sequential([
        Input(shape=(look_back, 1)),
        Cell(50, return_sequences=True),
        Dropout(0.2),
        Cell(50),
        Dropout(0.2),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    history = model.fit(
        X_tr, y_tr,
        validation_split=0.1,
        epochs=epochs, batch_size=batch_size, verbose=0,
        callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)],
    )

    pred_scaled = model.predict(X_te, verbose=0).ravel()
    pred_price = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
    actual_price = close.iloc[split:].to_numpy()
    prev_close = close.iloc[split - 1:n - 1].to_numpy()
    actual_ret = actual_price / prev_close - 1
    pred_ret = pred_price / prev_close - 1

    last_seq = scaled[-look_back:].copy()
    preds_sc = []
    for _ in range(steps):
        p = float(model.predict(last_seq.reshape(1, look_back, 1), verbose=0)[0, 0])
        preds_sc.append(p)
        last_seq = np.roll(last_seq, -1)
        last_seq[-1] = p
    fc_vals = scaler.inverse_transform(np.array(preds_sc).reshape(-1, 1)).ravel()
    fc_dates = pd.date_range(close.index[-1] + pd.Timedelta(days=1), periods=steps, freq="D")

    return {
        "cell": cell.upper(),
        "look_back": look_back,
        "epochs_configured": epochs,
        "epochs_ran": len(history.history["loss"]),
        "price": regression_metrics(actual_price, pred_price),
        "returns": regression_metrics(actual_ret, pred_ret),
        "directional_accuracy": directional_accuracy(actual_ret, pred_ret),
        "baseline_persistence": persistence_baseline(prev_close, actual_ret),
        "n_test": int(len(y_te)),
        "forecasts": [(d.strftime("%Y-%m-%d"), float(v)) for d, v in zip(fc_dates, fc_vals)],
        "pred_price": pred_price,
        "actual_price": actual_price,
    }
