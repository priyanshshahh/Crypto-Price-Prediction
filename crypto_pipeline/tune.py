"""Prediction-skill upgrades: Optuna-tuned LightGBM, regime split, OOF stack.

Goal: beat persistence RMSE on holdout / walk-forward — not UI.
Callers: crypto_pipeline.horizons.run_horizon
Schema: tune metadata on horizon result (params, val_margin).
  why are you limiting yourself?
"""

from __future__ import annotations

import numpy as np

RANDOM_STATE = 42


def select_top_features(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    *,
    k: int = 40,
    sample_weight: np.ndarray | None = None,
) -> list[int]:
    """Keep top-k features by LightGBM gain (or variance fallback)."""
    k = min(k, X_tr.shape[1])
    try:
        import lightgbm as lgb

        m = lgb.LGBMRegressor(
            n_estimators=200, learning_rate=0.05, num_leaves=23, max_depth=4,
            random_state=RANDOM_STATE, verbose=-1,
        )
        m.fit(X_tr, y_tr, sample_weight=sample_weight)
        gain = np.asarray(m.feature_importances_, dtype=float)
        idx = np.argsort(gain)[::-1][:k]
        return sorted(int(i) for i in idx)
    except Exception:
        var = np.nanvar(X_tr, axis=0)
        idx = np.argsort(var)[::-1][:k]
        return sorted(int(i) for i in idx)


def tune_lightgbm(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
    n_trials: int = 40,
    val_frac: float = 0.2,
) -> dict:
    """Optuna search maximizing margin vs persistence (zero-return) on chrono val."""
    try:
        import lightgbm as lgb
        import optuna
        from optuna.samplers import TPESampler
    except ImportError:
        return {}

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    n = len(X_tr)
    cut = max(int(n * (1.0 - val_frac)), n - 90)
    if cut < 80 or n - cut < 20:
        return {}

    X_fit, X_val = X_tr[:cut], X_tr[cut:]
    y_fit, y_val = y_tr[:cut], y_tr[cut:]
    w_fit = sample_weight[:cut] if sample_weight is not None else None
    pers = float(np.sqrt(np.mean(y_val ** 2)))

    def objective(trial: "optuna.Trial") -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, 64),
            "max_depth": trial.suggest_int("max_depth", 2, 7),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 80),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 2.0, log=True),
            "random_state": RANDOM_STATE,
            "verbose": -1,
        }
        m = lgb.LGBMRegressor(**params)
        m.fit(X_fit, y_fit, sample_weight=w_fit)
        pred = m.predict(X_val)
        rmse = float(np.sqrt(np.mean((y_val - pred) ** 2)))
        return pers - rmse

    study = optuna.create_study(
        direction="maximize", sampler=TPESampler(seed=RANDOM_STATE)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = dict(study.best_params)
    best.update({"random_state": RANDOM_STATE, "verbose": -1})
    return {
        "params": best,
        "val_margin_vs_persistence": float(study.best_value),
        "persistence_rmse": pers,
        "n_trials": n_trials,
    }


def fit_tuned_lgbm(params: dict, X: np.ndarray, y: np.ndarray,
                   sample_weight: np.ndarray | None = None):
    import lightgbm as lgb

    m = lgb.LGBMRegressor(**params)
    m.fit(X, y, sample_weight=sample_weight)
    return m


def stack_predict(
    base_preds_tr: np.ndarray,
    y_tr: np.ndarray,
    base_preds_te: np.ndarray,
) -> np.ndarray:
    """Ridge stacker on columns of base model predictions."""
    from sklearn.linear_model import Ridge

    if base_preds_tr.ndim == 1:
        base_preds_tr = base_preds_tr.reshape(-1, 1)
        base_preds_te = base_preds_te.reshape(-1, 1)
    stack = Ridge(alpha=1.0)
    stack.fit(base_preds_tr, y_tr)
    return np.asarray(stack.predict(base_preds_te), dtype=float)


def oof_tree_preds(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_splits: int = 5,
    sample_weight: np.ndarray | None = None,
    lgb_params: dict | None = None,
) -> np.ndarray:
    """Expanding OOF predictions for stacking (chronological folds)."""
    try:
        import lightgbm as lgb
    except ImportError:
        return np.zeros(len(y))

    n = len(y)
    oof = np.zeros(n)
    params = lgb_params or {
        "n_estimators": 200, "learning_rate": 0.05, "num_leaves": 23,
        "max_depth": 4, "random_state": RANDOM_STATE, "verbose": -1,
    }
    edges = np.linspace(int(n * 0.4), n, n_splits + 1, dtype=int)
    for i in range(len(edges) - 1):
        tr_end, te_end = int(edges[i]), int(edges[i + 1])
        if te_end <= tr_end:
            continue
        w = sample_weight[:tr_end] if sample_weight is not None else None
        m = lgb.LGBMRegressor(**params)
        m.fit(X[:tr_end], y[:tr_end], sample_weight=w)
        oof[tr_end:te_end] = m.predict(X[tr_end:te_end])
    head = int(edges[0]) if len(edges) else 0
    if head > 0:
        oof[:head] = float(np.mean(y[:head]))
    return oof
