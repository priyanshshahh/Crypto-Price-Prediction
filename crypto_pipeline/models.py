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
              orders: tuple = ((1, 1, 1), (2, 1, 1), (1, 1, 2), (0, 1, 1), (2, 1, 2))) -> dict:
    """ARIMA with AIC order selection on train, walk-forward ONE-STEP-AHEAD
    evaluation on the holdout (fixed parameters, updated history), and a
    30-day future forecast refit on the full series."""
    import warnings

    from statsmodels.tsa.arima.model import ARIMA

    n = len(close)
    split = time_split(n, test_frac)
    train, test = close.iloc[:split], close.iloc[split:]

    best_aic, best_order, best_res = np.inf, None, None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for order in orders:
            try:
                res = ARIMA(train, order=order).fit()
                if res.aic < best_aic:
                    best_aic, best_order, best_res = res.aic, order, res
            except Exception:
                continue
    if best_res is None:
        raise RuntimeError("All ARIMA orders failed to fit")

    # Walk-forward 1-step-ahead predictions over the test window:
    # apply train-fitted parameters to the full series, then take the
    # one-step-ahead (dynamic=False) predictions for the test index.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res_full = best_res.apply(close)
        pred_price = res_full.get_prediction(start=split, dynamic=False).predicted_mean.to_numpy()

    actual_price = test.to_numpy()
    prev_close = close.iloc[split - 1:n - 1].to_numpy()
    actual_ret = actual_price / prev_close - 1
    pred_ret = pred_price / prev_close - 1

    # Future forecast: refit best order on ALL data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final_res = ARIMA(close, order=best_order).fit()
        fc_vals = np.asarray(final_res.forecast(steps=steps), dtype=float)
    fc_dates = pd.date_range(close.index[-1] + pd.Timedelta(days=1), periods=steps, freq="D")

    return {
        "order": list(best_order),
        "aic": float(best_aic),
        "price": regression_metrics(actual_price, pred_price),
        "returns": regression_metrics(actual_ret, pred_ret),
        "directional_accuracy": directional_accuracy(actual_ret, pred_ret),
        "baseline_persistence": persistence_baseline(prev_close, actual_ret),
        "n_test": int(n - split),
        "forecasts": [(d.strftime("%Y-%m-%d"), float(v)) for d, v in zip(fc_dates, fc_vals)],
        "pred_price": pred_price,
        "actual_price": actual_price,
    }


# ── LSTM ──────────────────────────────────────────────────────────────────────

def run_lstm(close: pd.Series, test_frac: float = 0.2, steps: int = 30,
             look_back: int = 60, epochs: int = 40, batch_size: int = 32) -> dict:
    """Two-layer LSTM predicting next-day close from a 60-day window.

    MinMaxScaler is fit on the TRAIN closes only (the original implementation
    fit it on the full series — that was look-ahead leakage and was fixed).
    Epochs are capped (default 40) with EarlyStopping(patience=5) to keep the
    one-command run fast on CPU.
    """
    import random

    import tensorflow as tf
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.models import Sequential

    tf.get_logger().setLevel("ERROR")
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)

    n = len(close)
    split = time_split(n, test_frac)

    scaler = MinMaxScaler()
    scaler.fit(close.iloc[:split].to_numpy().reshape(-1, 1))   # train-only fit
    scaled = scaler.transform(close.to_numpy().reshape(-1, 1)).ravel()

    X_all, y_all = build_windows(scaled, look_back)
    # target of window i is scaled[i + look_back]; test targets are indices >= split
    first_test_window = split - look_back
    X_tr, X_te = X_all[:first_test_window], X_all[first_test_window:]
    y_tr, y_te = y_all[:first_test_window], y_all[first_test_window:]

    model = Sequential([
        Input(shape=(look_back, 1)),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    history = model.fit(
        X_tr, y_tr,
        validation_split=0.1,        # last 10% of train (Keras takes the tail — chronological)
        epochs=epochs, batch_size=batch_size, verbose=0,
        callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)],
    )

    pred_scaled = model.predict(X_te, verbose=0).ravel()
    pred_price = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
    actual_price = close.iloc[split:].to_numpy()
    prev_close = close.iloc[split - 1:n - 1].to_numpy()
    actual_ret = actual_price / prev_close - 1
    pred_ret = pred_price / prev_close - 1

    # Autoregressive 30-day future forecast from the last observed window
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
