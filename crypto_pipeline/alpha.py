"""Research-grade free alpha stack — LightGBM/XGBoost ensemble, direction
calibration, conformal bands, recency weights.

Callers: crypto_pipeline.horizons (production forecasting).
Schema: adds model names LightGBM/XGBoost/Ensemble; live forecast fields
  direction_prob_up, confidence, conformal_lo/hi, ensemble_models.
User instruction: build best free-scope crypto prediction system, not a demo;
  compare to best predictors and implement rigorously.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RANDOM_STATE = 42


def recency_weights(n: int, half_life: float = 180.0) -> np.ndarray:
    """Exponential recency weights (more weight on recent train rows)."""
    ages = np.arange(n)[::-1].astype(float)
    w = np.exp(-np.log(2) * ages / half_life)
    return w * (n / w.sum())


def _tree_models():
    """Best free OSS predictors: LightGBM + XGBoost + HistGB fallback."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import ElasticNet, Lasso, Ridge

    models = {
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.0005, max_iter=50000),
        "ElasticNet": ElasticNet(alpha=0.0005, l1_ratio=0.5, max_iter=50000),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_depth=4, learning_rate=0.05, max_iter=250, random_state=RANDOM_STATE
        ),
    }
    try:
        import lightgbm as lgb

        models["LightGBM"] = lgb.LGBMRegressor(
            n_estimators=400,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            verbose=-1,
        )
    except ImportError:
        pass
    try:
        import xgboost as xgb

        models["XGBoost"] = xgb.XGBRegressor(
            n_estimators=400,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            verbosity=0,
        )
    except ImportError:
        pass
    return models


def fit_predict_ensemble(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
    fast: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Fit models; Ensemble = mean of tree models. fast=True → LightGBM/HistGB only."""
    preds: dict[str, np.ndarray] = {}
    fitted: dict[str, object] = {}
    models = _tree_models()
    if fast:
        # Walk-forward must stay leak-free but cheap
        try:
            import lightgbm as lgb
            models = {
                "LightGBM": lgb.LGBMRegressor(
                    n_estimators=120,
                    learning_rate=0.05,
                    num_leaves=23,
                    max_depth=4,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=RANDOM_STATE,
                    verbose=-1,
                )
            }
        except ImportError:
            from sklearn.ensemble import HistGradientBoostingRegressor
            models = {
                "HistGradientBoosting": HistGradientBoostingRegressor(
                    max_depth=4, learning_rate=0.05, max_iter=120, random_state=RANDOM_STATE
                )
            }
    for name, model in models.items():
        m = model
        try:
            if sample_weight is not None:
                m.fit(X_tr, y_tr, sample_weight=sample_weight)
            else:
                m.fit(X_tr, y_tr)
            preds[name] = np.asarray(m.predict(X_te), dtype=float)
            fitted[name] = m
        except Exception:  # noqa: BLE001
            continue
    tree_names = [n for n in ("LightGBM", "XGBoost", "HistGradientBoosting") if n in preds]
    if len(tree_names) >= 2:
        preds["Ensemble"] = np.mean([preds[n] for n in tree_names], axis=0)
    elif tree_names:
        preds["Ensemble"] = preds[tree_names[0]].copy()
    return preds, fitted


def calibrated_direction(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """P(up) via LightGBM/sklearn classifier + isotonic calibration on train tail."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import HistGradientBoostingClassifier

    y_cls = (y_tr > 0).astype(int)
    if y_cls.sum() < 10 or (len(y_cls) - y_cls.sum()) < 10:
        return np.full(len(X_te), 0.5), 0.5

    base = HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=150, random_state=RANDOM_STATE
    )
    try:
        import lightgbm as lgb

        base = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=15,
            max_depth=4,
            random_state=RANDOM_STATE,
            verbose=-1,
        )
    except ImportError:
        pass

    n = len(X_tr)
    cut = max(int(n * 0.8), n - 60)
    if cut < 50 or n - cut < 20:
        base.fit(X_tr, y_cls, sample_weight=sample_weight)
        proba = base.predict_proba(X_te)[:, 1]
        return np.asarray(proba, dtype=float), float((y_cls == 1).mean())

    X_fit, y_fit = X_tr[:cut], y_cls[:cut]
    X_cal, y_cal = X_tr[cut:], y_cls[cut:]
    w_fit = sample_weight[:cut] if sample_weight is not None else None
    base.fit(X_fit, y_fit, sample_weight=w_fit)
    try:
        cal = CalibratedClassifierCV(base, method="isotonic", cv="prefit")
        cal.fit(X_cal, y_cal)
        proba = cal.predict_proba(X_te)[:, 1]
    except Exception:  # noqa: BLE001
        proba = base.predict_proba(X_te)[:, 1]
    return np.asarray(proba, dtype=float), float((y_cls == 1).mean())


def conformal_bands(
    y_cal: np.ndarray,
    pred_cal: np.ndarray,
    pred_te: np.ndarray,
    *,
    alpha: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Split-conformal residual bands."""
    resid = np.abs(y_cal - pred_cal)
    q = float(np.quantile(resid, min(1.0, 1.0 - alpha)))
    return pred_te - q, pred_te + q


def vol_normalize_target(y: np.ndarray, vol: np.ndarray, floor: float = 1e-4) -> np.ndarray:
    v = np.maximum(np.asarray(vol, dtype=float), floor)
    return np.asarray(y, dtype=float) / v


def vol_denormalize(y_hat: np.ndarray, vol: np.ndarray, floor: float = 1e-4) -> np.ndarray:
    v = np.maximum(np.asarray(vol, dtype=float), floor)
    return np.asarray(y_hat, dtype=float) * v
