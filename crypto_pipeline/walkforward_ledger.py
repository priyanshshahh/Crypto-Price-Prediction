"""Dense walk-forward skill ledger for the production ensemble.

Replaces MA50/200 journal seeds with leak-free LightGBM (fast) predictions:
  train only on rows whose horizon labels are fully resolved (purge=horizon).

Also: wrong-cluster analysis + gated adaptive sample-weight plan
(upweight recent errors / high-vol only when a micro-val gate passes).

Callers: pipeline.py --mode production; journal.merge_walkforward_ledger.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from .alpha import fit_predict_ensemble, recency_weights
from .features import feature_matrix
from .journal import _direction

RANDOM_STATE = 42
HORIZON_LABEL = {1: "1D", 7: "1W", 30: "1M"}


def _grade_dirs(pred_ret: float, actual_ret: float) -> str:
    pred_dir = _direction(pred_ret * 100.0)
    act_dir = _direction(actual_ret * 100.0)
    if pred_dir == "flat" or act_dir == "flat":
        return "Incomplete"
    return "Correct" if pred_dir == act_dir else "Incorrect"


def walkforward_ledger_for_horizon(
    df_feat: pd.DataFrame,
    horizon: int,
    *,
    symbol: str,
    min_train: int = 252,
    dense_days: int = 60,
    dense_step: int = 1,
    archive_days: int = 120,
    archive_step: int = 5,
) -> list[dict[str, Any]]:
    """Expanding-window predictions → resolved journal rows (no look-ahead).

    At index i (as_of), Target_Return_{h} uses closes through i+h, so training
    may only use rows j with j < i - h (purge=horizon).
    """
    from sklearn.preprocessing import StandardScaler

    target = f"Target_Return_{horizon}d"
    if target not in df_feat.columns:
        return []

    X = feature_matrix(df_feat)
    y = df_feat[target].to_numpy(dtype=float)
    close = df_feat["Close"].to_numpy(dtype=float)
    dates = pd.to_datetime(X.index).tz_localize(None)
    n = len(X)
    h = int(horizon)
    last_i = n - h - 1  # need actual y[i] known → need i+h within frame
    if last_i < min_train + h:
        return []

    # Build evaluation indices: archive (coarser) then dense (day-by-day)
    dense_start = max(min_train + h, last_i - dense_days + 1)
    archive_start = max(min_train + h, dense_start - archive_days)
    indices: list[int] = []
    i = archive_start
    while i < dense_start:
        indices.append(i)
        i += max(archive_step, 1)
    i = dense_start
    while i <= last_i:
        indices.append(i)
        i += max(dense_step, 1)
    # unique sorted
    indices = sorted(set(indices))

    rows: list[dict[str, Any]] = []
    label = HORIZON_LABEL.get(h, f"{h}D")
    vol = (
        df_feat["Volatility"].to_numpy(dtype=float)
        if "Volatility" in df_feat.columns
        else np.full(n, np.nan)
    )

    for i in indices:
        train_end = i - h  # exclusive; all train labels resolved before as_of
        if train_end < min_train:
            continue
        X_tr = X.iloc[:train_end]
        y_tr = y[:train_end]
        if not np.isfinite(y[i]):
            continue
        # drop non-finite train rows
        mask = np.isfinite(y_tr)
        if mask.sum() < min_train // 2:
            continue
        X_tr = X_tr.iloc[mask]
        y_tr = y_tr[mask]

        scaler = StandardScaler().fit(X_tr)
        X_tr_sc = scaler.transform(X_tr)
        X_te_sc = scaler.transform(X.iloc[[i]])
        w = recency_weights(len(X_tr_sc), half_life=180.0)
        preds, _ = fit_predict_ensemble(
            X_tr_sc, y_tr, X_te_sc, sample_weight=w, fast=True
        )
        name = "Ensemble" if "Ensemble" in preds else (
            "LightGBM" if "LightGBM" in preds else next(iter(preds))
        )
        pred_ret = float(preds[name][0])
        actual_ret = float(y[i])
        verdict = _grade_dirs(pred_ret, actual_ret)
        as_of = dates[i]
        resolve = dates[min(i + h, n - 1)]
        px0 = float(close[i])
        rows.append({
            "symbol": symbol,
            "as_of": str(as_of.date()),
            "resolve_date": str(resolve.date()),
            "horizon_days": h,
            "horizon_label": label,
            "predicted_return_pct": pred_ret * 100.0,
            "actual_return_pct": actual_ret * 100.0,
            "predicted_direction": _direction(pred_ret * 100.0),
            "actual_direction": _direction(actual_ret * 100.0),
            "verdict": verdict,
            "predicted_price": px0 * (1.0 + pred_ret),
            "model": f"WalkForward-{name}",
            "source": "walkforward_ml",
            "trustworthy": False,
            "train_end_rows": int(train_end),
            "volatility": None if not np.isfinite(vol[i]) else float(vol[i]),
        })
    return rows


def build_walkforward_ledger(
    featured: dict[str, pd.DataFrame],
    *,
    horizons: tuple[int, ...] = (1, 7, 30),
    dense_days: int = 60,
    dense_step: int = 1,
    archive_days: int = 120,
    archive_step: int = 5,
) -> list[dict[str, Any]]:
    """Run WF ledger for all symbols × horizons."""
    all_rows: list[dict[str, Any]] = []
    for sym, feat in featured.items():
        for h in horizons:
            print(f"      WF ledger {sym} H={h}d ...", flush=True)
            part = walkforward_ledger_for_horizon(
                feat, h, symbol=sym,
                dense_days=dense_days, dense_step=dense_step,
                archive_days=archive_days, archive_step=archive_step,
            )
            graded = [r for r in part if r["verdict"] in ("Correct", "Incorrect")]
            hit = (
                sum(1 for r in graded if r["verdict"] == "Correct") / len(graded)
                if graded else None
            )
            if hit is not None:
                print(f"         {len(part)} events · hit={hit:.0%}", flush=True)
            else:
                print(f"         {len(part)} events · hit=n/a", flush=True)
            all_rows.extend(part)
    return all_rows


def analyze_wrong_clusters(resolved: list[dict[str, Any]]) -> dict[str, Any]:
    """Since-last-wrong-cluster analysis (scheduled retrain signal — not instant overfit)."""
    ml = [r for r in resolved if r.get("source") == "walkforward_ml"
          and r.get("verdict") in ("Correct", "Incorrect")]
    by_key: dict[tuple, list] = defaultdict(list)
    for r in ml:
        by_key[(r.get("symbol"), int(r.get("horizon_days") or 0))].append(r)

    per_cell = []
    recommend = []
    for (sym, h), rows in sorted(by_key.items()):
        rows = sorted(rows, key=lambda x: x.get("as_of") or "")
        streak = max_streak = 0
        for r in rows:
            if r["verdict"] == "Incorrect":
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 0
        recent = rows[-20:]
        recent_err = (
            sum(1 for r in recent if r["verdict"] == "Incorrect") / len(recent)
            if recent else 0.0
        )
        hit = sum(1 for r in rows if r["verdict"] == "Correct") / len(rows)
        cell = {
            "symbol": sym,
            "horizon_days": h,
            "n": len(rows),
            "hit_rate": round(hit, 4),
            "max_incorrect_streak": max_streak,
            "recent_20_error_rate": round(recent_err, 4),
            "recommend_retrain_focus": bool(recent_err >= 0.55 and max_streak >= 3),
        }
        per_cell.append(cell)
        if cell["recommend_retrain_focus"]:
            recommend.append(f"{sym}-{h}d")

    return {
        "cells": per_cell,
        "retrain_focus": recommend,
        "note": (
            "Retrain focus = elevated recent error + streak. "
            "Does not auto-fit on single wrongs (avoids memorizing crashes)."
        ),
    }


def adaptive_weight_plan(
    resolved: list[dict[str, Any]],
    wrong_clusters: dict[str, Any],
    *,
    boost: float = 2.25,
    neighbor: int = 3,
) -> dict[str, Any]:
    """Build per-(symbol,horizon) error date boosts; gated by cluster recommend OR low hit.

    Actual application still needs micro-val gate inside run_horizon.
    """
    focus = set(wrong_clusters.get("retrain_focus") or [])
    # also allow mild adaptive if cell hit_rate < 0.48 with n>=30
    low_hit = {
        f"{c['symbol']}-{c['horizon_days']}d"
        for c in wrong_clusters.get("cells") or []
        if c.get("n", 0) >= 30 and c.get("hit_rate", 1) < 0.48
    }
    eligible = focus | low_hit

    errors_by_cell: dict[str, list[str]] = defaultdict(list)
    for r in resolved:
        if r.get("source") != "walkforward_ml" or r.get("verdict") != "Incorrect":
            continue
        key = f"{r.get('symbol')}-{int(r.get('horizon_days') or 0)}d"
        if key not in eligible:
            continue
        errors_by_cell[key].append(str(r.get("as_of")))

    return {
        "eligible_cells": sorted(eligible),
        "error_as_of": {k: v[-40:] for k, v in errors_by_cell.items()},
        "boost": boost,
        "neighbor_days": neighbor,
        "gate": "micro_val_rmse — apply only if adaptive val RMSE <= baseline val RMSE * 1.01",
    }


def error_aware_weights(
    train_dates: pd.DatetimeIndex,
    *,
    error_as_of: list[str],
    half_life: float = 180.0,
    boost: float = 2.25,
    neighbor_days: int = 3,
) -> np.ndarray:
    """Recency weights × boost near Incorrect as_of dates (purged training only)."""
    n = len(train_dates)
    w = recency_weights(n, half_life=half_life)
    if not error_as_of:
        return w
    err = set(error_as_of)
    dates = pd.to_datetime(train_dates).tz_localize(None).normalize()
    for i, d in enumerate(dates):
        for e in err:
            ed = pd.Timestamp(e).normalize()
            if abs((d - ed).days) <= neighbor_days:
                w[i] *= boost
                break
    return w * (n / w.sum())
