"""Multi-horizon price/return forecasting with LightGBM/XGBoost ensemble,
calibrated direction, conformal bands, persistence baselines + paper eval.

Callers: pipeline.py --mode production.
Schema: per-horizon models list + live_forecasts with confidence / conformal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .alpha import (calibrated_direction, conformal_bands, fit_predict_ensemble,
                    recency_weights)
from .cv import chronological_holdout, expanding_walk_forward
from .features import feature_matrix
from .models import (directional_accuracy, persistence_baseline, price_metrics_from_returns,
                     regression_metrics)
from .paper import select_best_mode
from .tune import (fit_tuned_lgbm, oof_tree_preds, select_top_features, stack_predict,
                   tune_lightgbm)

RANDOM_STATE = 42
DEFAULT_HORIZONS = (1, 7, 30)


def run_horizon(
    df_feat: pd.DataFrame,
    horizon: int,
    *,
    test_frac: float = 0.2,
    symbol: str = "BTC",
    fee_bps: float = 50.0,
    slippage_bps: float = 10.0,
    min_abs_pred: float | None = None,
    adaptive_plan: dict | None = None,
) -> dict:
    """Train ensemble for Target_Return_{horizon}d on chronological holdout."""
    from sklearn.preprocessing import StandardScaler

    target_col = f"Target_Return_{horizon}d"
    if target_col not in df_feat.columns:
        raise KeyError(f"Missing {target_col}; call add_features(..., horizons=...)")

    X = feature_matrix(df_feat)
    y_raw = df_feat[target_col].to_numpy()
    close = df_feat["Close"].to_numpy()
    vol = (
        df_feat["Volatility"].to_numpy()
        if "Volatility" in df_feat.columns
        else np.full(len(X), np.nanstd(y_raw) or 0.02)
    )
    vol = np.where(np.isfinite(vol), vol, np.nanmedian(vol[np.isfinite(vol)]))

    # Train on raw forward returns (vol-norm hurt holdout RMSE vs Lasso baseline)
    y = y_raw

    fold = chronological_holdout(len(X), test_frac)
    X_tr, X_te = X.iloc[:fold.train_end], X.iloc[fold.test_start:fold.test_end]
    y_tr = y[:fold.train_end]
    y_raw_tr = y_raw[:fold.train_end]
    y_raw_te = y_raw[fold.test_start:fold.test_end]
    close_te = close[fold.test_start:fold.test_end]
    vol_te = vol[fold.test_start:fold.test_end]

    scaler = StandardScaler().fit(X_tr)
    X_tr_sc, X_te_sc = scaler.transform(X_tr), scaler.transform(X_te)
    weights = recency_weights(len(X_tr_sc), half_life=180.0)
    adaptive_meta = {"applied": False, "reason": "no_plan"}

    # Gated adaptive weighting from WF journal wrongs (micro-val must not worsen RMSE)
    cell_key = f"{symbol}-{horizon}d"
    if adaptive_plan and cell_key in (adaptive_plan.get("eligible_cells") or []):
        try:
            from .walkforward_ledger import error_aware_weights
            err_dates = (adaptive_plan.get("error_as_of") or {}).get(cell_key) or []
            w_adapt = error_aware_weights(
                X_tr.index,
                error_as_of=err_dates,
                boost=float(adaptive_plan.get("boost", 2.25)),
                neighbor_days=int(adaptive_plan.get("neighbor_days", 3)),
            )
            # micro-val: last 15% of train
            cut = max(int(len(X_tr_sc) * 0.85), len(X_tr_sc) - 40)
            if cut >= 80 and len(X_tr_sc) - cut >= 15:
                from sklearn.metrics import mean_squared_error
                import lightgbm as lgb
                def _rmse(w):
                    m = lgb.LGBMRegressor(
                        n_estimators=120, learning_rate=0.05, num_leaves=23,
                        max_depth=4, random_state=RANDOM_STATE, verbose=-1,
                    )
                    m.fit(X_tr_sc[:cut], y_tr[:cut], sample_weight=w[:cut])
                    p = m.predict(X_tr_sc[cut:])
                    return float(np.sqrt(mean_squared_error(y_tr[cut:], p)))
                base_rmse = _rmse(weights)
                adapt_rmse = _rmse(w_adapt)
                if adapt_rmse <= base_rmse * 1.01:
                    weights = w_adapt
                    adaptive_meta = {
                        "applied": True,
                        "base_val_rmse": base_rmse,
                        "adapt_val_rmse": adapt_rmse,
                        "n_error_dates": len(err_dates),
                        "reason": "micro_val_pass",
                    }
                else:
                    adaptive_meta = {
                        "applied": False,
                        "base_val_rmse": base_rmse,
                        "adapt_val_rmse": adapt_rmse,
                        "reason": "micro_val_reject_overfit_risk",
                    }
            else:
                adaptive_meta = {"applied": False, "reason": "train_too_short_for_micro_val"}
        except Exception as err:  # noqa: BLE001
            adaptive_meta = {"applied": False, "reason": f"adaptive_error:{err}"}

    # Feature selection → Optuna LightGBM → OOF stack (prediction skill, not UI)
    feat_idx = select_top_features(X_tr_sc, y_tr, k=min(45, X_tr_sc.shape[1]),
                                   sample_weight=weights)
    X_tr_fs, X_te_fs = X_tr_sc[:, feat_idx], X_te_sc[:, feat_idx]

    baseline = persistence_baseline(close_te, y_raw_te)

    preds_n, _fitted = fit_predict_ensemble(X_tr_fs, y_tr, X_te_fs, sample_weight=weights)
    tune_meta: dict = {}
    try:
        tune_meta = tune_lightgbm(
            X_tr_fs, y_tr, sample_weight=weights, n_trials=25, val_frac=0.2
        )
        if tune_meta.get("params"):
            tuned = fit_tuned_lgbm(tune_meta["params"], X_tr_fs, y_tr, sample_weight=weights)
            preds_n["TunedLightGBM"] = np.asarray(tuned.predict(X_te_fs), dtype=float)
            oof = oof_tree_preds(
                X_tr_fs, y_tr, sample_weight=weights, lgb_params=tune_meta["params"]
            )
            base_te = preds_n["TunedLightGBM"].reshape(-1, 1)
            stacked = stack_predict(oof.reshape(-1, 1), y_tr, base_te)
            shrink = 0.85
            preds_n["StackedTuned"] = shrink * stacked
    except Exception as err:  # noqa: BLE001
        tune_meta = {"error": str(err)}

    # Regime blend: high-vol vs low-vol specialist (if enough samples each)
    try:
        import lightgbm as lgb
        thr = float(np.nanmedian(vol[:fold.train_end]))
        hi_tr = vol[:fold.train_end] >= thr
        lo_tr = ~hi_tr
        hi_te = vol_te >= thr
        if hi_tr.sum() >= 80 and lo_tr.sum() >= 80:
            params = (tune_meta.get("params") or {
                "n_estimators": 250, "learning_rate": 0.04, "num_leaves": 23,
                "max_depth": 4, "random_state": RANDOM_STATE, "verbose": -1,
            })
            m_hi = lgb.LGBMRegressor(**params)
            m_lo = lgb.LGBMRegressor(**params)
            w_hi = weights[hi_tr] if weights is not None else None
            w_lo = weights[lo_tr] if weights is not None else None
            m_hi.fit(X_tr_fs[hi_tr], y_tr[hi_tr], sample_weight=w_hi)
            m_lo.fit(X_tr_fs[lo_tr], y_tr[lo_tr], sample_weight=w_lo)
            regime_pred = np.empty(len(X_te_fs))
            if hi_te.any():
                regime_pred[hi_te] = m_hi.predict(X_te_fs[hi_te])
            if (~hi_te).any():
                regime_pred[~hi_te] = m_lo.predict(X_te_fs[~hi_te])
            preds_n["RegimeLGBM"] = regime_pred
            tune_meta["regime_vol_threshold"] = thr
    except Exception:
        pass

    results = []
    best_pred = None
    best_name = "Ensemble"
    best_rmse = float("inf")
    for name, pred in preds_n.items():
        pred = np.asarray(pred, dtype=float)
        row = {
            "name": name,
            "horizon": horizon,
            "returns": regression_metrics(y_raw_te, pred),
            "price": price_metrics_from_returns(close_te, y_raw_te, pred),
            "directional_accuracy": directional_accuracy(y_raw_te, pred),
            "beats_persistence_rmse": bool(
                regression_metrics(y_raw_te, pred)["rmse"] < baseline["returns"]["rmse"]
            ),
        }
        results.append(row)
        if row["returns"]["rmse"] < best_rmse:
            best_rmse = row["returns"]["rmse"]
            best_pred = pred
            best_name = name

    dir_proba, base_rate = calibrated_direction(
        X_tr_sc, y_raw_tr, X_te_sc, sample_weight=weights
    )
    dir_pred = np.where(dir_proba >= 0.5, 1.0, -1.0)
    dir_acc = float((np.sign(y_raw_te) == dir_pred).mean()) if len(y_raw_te) else 0.5
    conf_mask = np.abs(dir_proba - 0.5) >= 0.15
    conf_dir_acc = (
        float((np.sign(y_raw_te[conf_mask]) == dir_pred[conf_mask]).mean())
        if conf_mask.any() else None
    )

    n_tr = len(X_tr_sc)
    cal_cut = max(int(n_tr * 0.8), n_tr - 80)
    if best_pred is not None and cal_cut < n_tr:
        from sklearn.ensemble import HistGradientBoostingRegressor
        cal_model = HistGradientBoostingRegressor(
            max_depth=4, learning_rate=0.05, max_iter=200, random_state=RANDOM_STATE
        )
        try:
            import lightgbm as lgb
            cal_model = lgb.LGBMRegressor(
                n_estimators=300, learning_rate=0.03, max_depth=5,
                random_state=RANDOM_STATE, verbose=-1,
            )
        except ImportError:
            pass
        cal_model.fit(X_tr_sc[:cal_cut], y_tr[:cal_cut], sample_weight=weights[:cal_cut])
        pred_cal = np.asarray(cal_model.predict(X_tr_sc[cal_cut:]), dtype=float)
        _lo, hi_n = conformal_bands(y_raw_tr[cal_cut:], pred_cal, best_pred, alpha=0.2)
        conformal = {
            "alpha": 0.2,
            "coverage_target": 0.8,
            "mean_width": float(np.mean(hi_n - _lo)),
        }
    else:
        conformal = {"alpha": 0.2, "coverage_target": 0.8, "mean_width": None}

    min_abs = min_abs_pred if min_abs_pred is not None else {
        1: 0.005, 7: 0.02, 30: 0.05, 5: 0.015, 10: 0.025
    }.get(horizon, 0.015)
    paper = select_best_mode(
        y_raw_te, best_pred if best_pred is not None else np.zeros_like(y_raw_te),
        fee_bps_round_trip=fee_bps,
        slippage_bps=slippage_bps,
        min_abs_pred=min_abs,
        realized_vol=vol_te,
    )

    wf = _walk_forward_lgbm(X, y_raw, horizon=horizon, purge=horizon)

    best_row = next(r for r in results if r["name"] == best_name)
    wf_ok = wf["pass_frac"] >= 0.55
    dir_ok = (best_row["directional_accuracy"] >= 0.55) or (
        conf_dir_acc is not None and conf_dir_acc >= 0.58
    )
    trustworthy = bool(best_row["beats_persistence_rmse"] and wf_ok and dir_ok)

    return {
        "horizon": horizon,
        "symbol": symbol,
        "models": results,
        "best_model": best_name,
        "baseline_persistence": baseline,
        "direction": {
            "accuracy": dir_acc,
            "base_rate_up": base_rate,
            "high_confidence_accuracy": conf_dir_acc,
            "high_confidence_frac": float(conf_mask.mean()) if len(conf_mask) else 0.0,
            "mean_prob_up": float(dir_proba.mean()) if len(dir_proba) else 0.5,
        },
        "conformal": conformal,
        "paper": paper,
        "walk_forward": wf,
        "trustworthy": trustworthy,
        "tune": tune_meta,
        "adaptive_weights": adaptive_meta,
        "n_features_selected": len(feat_idx),
        "n_train": int(fold.train_end),
        "n_test": int(fold.test_end - fold.test_start),
        "best_pred": best_pred.tolist() if best_pred is not None else [],
        "actual": y_raw_te.tolist(),
        "test_index": [str(i.date()) for i in X_te.index],
    }


def _walk_forward_lgbm(X: pd.DataFrame, y_raw: np.ndarray,
                       *, horizon: int, purge: int) -> dict:
    from sklearn.preprocessing import StandardScaler

    folds = expanding_walk_forward(
        len(X), min_train=180, test_size=30, embargo=5, purge=max(purge, 1)
    )
    if not folds:
        return {"n_folds": 0, "pass_frac": 0.0, "fold_rmse": []}

    fold_rmse = []
    beats = 0
    for f in folds:
        X_tr, X_te = X.iloc[:f.train_end], X.iloc[f.test_start:f.test_end]
        y_tr_raw = y_raw[:f.train_end]
        y_te_raw = y_raw[f.test_start:f.test_end]
        scaler = StandardScaler().fit(X_tr)
        w = recency_weights(len(X_tr), half_life=180.0)
        preds, _ = fit_predict_ensemble(
            scaler.transform(X_tr), y_tr_raw, scaler.transform(X_te),
            sample_weight=w, fast=True,
        )
        pred_n = preds.get("Ensemble")
        if pred_n is None:
            pred_n = preds.get("LightGBM")
        if pred_n is None:
            pred_n = next(iter(preds.values()))
        pred = np.asarray(pred_n, dtype=float)
        rmse = float(np.sqrt(np.mean((y_te_raw - pred) ** 2)))
        base = float(np.sqrt(np.mean(y_te_raw ** 2)))
        fold_rmse.append({"rmse": rmse, "persistence_rmse": base, "beats": rmse < base})
        if rmse < base:
            beats += 1
    return {
        "n_folds": len(folds),
        "pass_frac": beats / len(folds),
        "fold_rmse": fold_rmse,
    }


def run_all_horizons(
    df_feat: pd.DataFrame,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    **kwargs,
) -> dict:
    out = {}
    for h in horizons:
        out[h] = run_horizon(df_feat, h, **kwargs)
    summary = []
    for h, res in out.items():
        best = min(res["models"], key=lambda m: m["returns"]["rmse"])
        summary.append({
            "horizon": h,
            "label": {1: "next_day", 7: "next_week", 30: "next_month"}.get(h, f"{h}d"),
            "best_model": best["name"],
            "ret_rmse": best["returns"]["rmse"],
            "persistence_rmse": res["baseline_persistence"]["returns"]["rmse"],
            "beats_persistence": best["beats_persistence_rmse"],
            "directional_accuracy": best["directional_accuracy"],
            "direction_classifier_acc": res["direction"]["accuracy"],
            "direction_high_conf_acc": res["direction"]["high_confidence_accuracy"],
            "paper_best_mode": res["paper"]["best"]["mode"],
            "paper_net_sharpe": res["paper"]["best"]["net_sharpe"],
            "wf_pass_frac": res["walk_forward"]["pass_frac"],
            "trustworthy": res["trustworthy"],
            "adaptive_weights_applied": bool(
                (res.get("adaptive_weights") or {}).get("applied")
            ),
        })
    return {"by_horizon": out, "summary": summary}


def live_price_forecasts(
    df_ohlcv: pd.DataFrame,
    df_train: pd.DataFrame,
    horizon_results: dict,
    *,
    symbol: str = "BTC",
    external: pd.DataFrame | None = None,
    cross_asset: pd.DataFrame | None = None,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
) -> list[dict]:
    """Build live multi-horizon price rows with P(up), bands, and enrichments.

    Callers: pipeline.py production. Enriches each row via market_overview.enrich_live_row
    (technical bias, risk, scorecard, reasoning, quant, fundamentals).
    """
    from sklearn.preprocessing import StandardScaler
    from .features import add_features, feature_matrix
    from .market_overview import enrich_live_row

    live = add_features(
        df_ohlcv, horizons=horizons, external=external, cross_asset=cross_asset,
        for_prediction=True,
    )
    close_now = float(live["Close"].iloc[-1])
    x_live = feature_matrix(live).iloc[[-1]]

    rows = []
    for h, res in horizon_results.items():
        h = int(h)
        best = min(res["models"], key=lambda m: m["returns"]["rmse"])
        target = f"Target_Return_{h}d"
        X = feature_matrix(df_train)
        for c in X.columns:
            if c not in x_live.columns:
                x_live[c] = 0.0
        x_live_aligned = x_live.reindex(columns=X.columns, fill_value=0.0)
        x_live_aligned = x_live_aligned.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        X_clean = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        y_raw = df_train[target].to_numpy()
        scaler = StandardScaler().fit(X_clean)
        X_sc = scaler.transform(X_clean)
        x_sc = scaler.transform(x_live_aligned)
        w = recency_weights(len(X_sc), half_life=180.0)
        preds, fitted = fit_predict_ensemble(X_sc, y_raw, x_sc, sample_weight=w)
        name = best["name"] if best["name"] in preds else None
        if name is None:
            name = "Ensemble" if "Ensemble" in preds else next(iter(preds))
        pred_ret = float(preds[name][0])
        lo, hi = np.quantile(y_raw, [0.01, 0.99])
        pred_ret = float(np.clip(pred_ret, lo, hi))

        cut = max(int(len(X_sc) * 0.8), len(X_sc) - 80)
        model = fitted.get(name)
        if model is None:
            model = fitted.get("LightGBM")
        if model is None:
            model = next(iter(fitted.values()))
        pred_cal = np.asarray(model.predict(X_sc[cut:]), dtype=float)
        c_lo, c_hi = conformal_bands(y_raw[cut:], pred_cal, np.array([pred_ret]), alpha=0.2)
        ret_lo = float(np.clip(c_lo[0], lo, hi))
        ret_hi = float(np.clip(c_hi[0], lo, hi))

        dir_p, _ = calibrated_direction(X_sc, y_raw, x_sc, sample_weight=w)
        p_up = float(dir_p[0])
        confidence = float(abs(p_up - 0.5) * 2.0)

        pred_price = max(close_now * (1.0 + pred_ret), 0.0)
        price_lo = max(close_now * (1.0 + ret_lo), 0.0)
        price_hi = max(close_now * (1.0 + ret_hi), 0.0)
        label = {1: "1D", 7: "1W", 30: "1M"}.get(h, f"{h}D")
        n_test = int(best.get("n_test") or res.get("n_test") or 0) or None
        row = {
            "horizon_days": h,
            "horizon_label": label,
            "as_of": str(live.index[-1].date()),
            "current_price": close_now,
            "predicted_price": pred_price,
            "predicted_price_p10": price_lo,
            "predicted_price_p90": price_hi,
            "predicted_return_pct": pred_ret * 100.0,
            "predicted_return_p10_pct": ret_lo * 100.0,
            "predicted_return_p90_pct": ret_hi * 100.0,
            "direction_prob_up": p_up,
            "confidence": confidence,
            "model": name,
            "holdout_beats_persistence": best["beats_persistence_rmse"],
            "holdout_directional_accuracy": best["directional_accuracy"],
            "holdout_ret_rmse": best["returns"]["rmse"],
            "persistence_ret_rmse": res["baseline_persistence"]["returns"]["rmse"],
            "wf_pass_frac": res["walk_forward"]["pass_frac"],
            "trustworthy": bool(res.get("trustworthy", False)),
        }
        enriched = enrich_live_row(row, live, n_test=n_test)
        # Quant Monte Carlo overlay (video strategy #7)
        try:
            from .quant import quant_overlay
            mc = quant_overlay(
                df_ohlcv["Close"], h,
                pred_return=pred_ret,
            )
            enriched["monte_carlo"] = mc
            enriched["mc_price_p10"] = mc["price_p10"]
            enriched["mc_price_p50"] = mc["price_p50"]
            enriched["mc_price_p90"] = mc["price_p90"]
            enriched["mc_prob_up"] = mc["prob_up"]
        except Exception as err:  # noqa: BLE001
            enriched["monte_carlo_error"] = str(err)
        # Polymarket discrepancy vs model P(up)
        if "Polymarket_Yes_Prob" in live.columns:
            pm = live["Polymarket_Yes_Prob"].dropna()
            if len(pm):
                crowd = float(pm.iloc[-1])
                enriched["polymarket_yes"] = crowd
                enriched["polymarket_model_gap"] = float(p_up - crowd)
        rows.append(enriched)
    # Attach fundamentals snapshot once on first row
    if rows:
        try:
            from .fundamentals import fundamental_snapshot
            fund = fundamental_snapshot(symbol, live)
            for r in rows:
                r["fundamentals"] = fund
        except Exception:
            pass
    return rows
