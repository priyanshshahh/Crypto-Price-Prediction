"""Prediction journal — Correct/Incorrect after each horizon resolves.

Primary skill ledger = walk-forward ML (source=walkforward_ml), not MA seeds.
Callers: pipeline.py production via walkforward_ledger + update_journal.
"""

from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd


def _direction(ret_pct: float) -> str:
    if ret_pct > 0.05:
        return "up"
    if ret_pct < -0.05:
        return "down"
    return "flat"


def grade_forecast(
    as_of: str,
    horizon_days: int,
    predicted_return_pct: float,
    closes: pd.Series,
) -> dict[str, Any] | None:
    """Return Correct/Incorrect if enough calendar days of closes exist."""
    as_of_ts = pd.Timestamp(as_of).normalize()
    target_day = as_of_ts + pd.Timedelta(days=int(horizon_days))
    closes = closes.copy()
    closes.index = pd.to_datetime(closes.index).tz_localize(None).normalize()
    if as_of_ts not in closes.index:
        prior = closes.loc[:as_of_ts]
        if prior.empty:
            return None
        px0 = float(prior.iloc[-1])
        as_used = prior.index[-1]
    else:
        px0 = float(closes.loc[as_of_ts])
        as_used = as_of_ts
    future = closes.loc[as_used + pd.Timedelta(days=1): target_day]
    if future.empty:
        return None
    px1 = float(future.iloc[-1])
    actual_ret_pct = (px1 / px0 - 1.0) * 100.0
    pred_dir = _direction(float(predicted_return_pct))
    act_dir = _direction(actual_ret_pct)
    if pred_dir == "flat" or act_dir == "flat":
        verdict = "Incomplete"
    elif pred_dir == act_dir:
        verdict = "Correct"
    else:
        verdict = "Incorrect"
    return {
        "as_of": str(pd.Timestamp(as_used).date()),
        "resolve_date": str(pd.Timestamp(future.index[-1]).date()),
        "horizon_days": int(horizon_days),
        "predicted_return_pct": float(predicted_return_pct),
        "actual_return_pct": float(actual_ret_pct),
        "predicted_direction": pred_dir,
        "actual_direction": act_dir,
        "verdict": verdict,
    }


def _is_ma_seed(row: dict) -> bool:
    model = str(row.get("model") or "")
    return "retrospective seed" in model.lower() or model.startswith("MA50/200")


def _summary(pending: list, resolved: list) -> dict[str, Any]:
    ml = [r for r in resolved if r.get("source") == "walkforward_ml"
          and r.get("verdict") in ("Correct", "Incorrect")]
    graded_ok = ml if ml else [
        r for r in resolved
        if r.get("verdict") in ("Correct", "Incorrect") and not _is_ma_seed(r)
    ]
    n_correct = sum(1 for r in graded_ok if r["verdict"] == "Correct")
    return {
        "n_resolved": len(graded_ok),
        "n_correct": n_correct,
        "hit_rate": (n_correct / len(graded_ok)) if graded_ok else None,
        "n_pending": len(pending),
        "n_walkforward_ml": len(ml),
        "disclaimer": (
            "Primary hit rate = leak-free walk-forward LightGBM ensemble "
            "(source=walkforward_ml). Live pending grade after each horizon resolves. "
            "MA50/200 seeds are stripped."
        ),
    }


def replace_with_walkforward_ledger(
    wf_rows: list[dict[str, Any]],
    *,
    path: str = "results/prediction_journal.json",
    keep_live_pending: bool = True,
) -> dict[str, Any]:
    """Overwrite resolved ledger with WF-ML rows; drop MA seeds."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    pending: list[dict] = []
    if keep_live_pending and os.path.exists(path):
        try:
            with open(path) as f:
                old = json.load(f)
            pending = [
                p for p in (old.get("pending") or [])
                if not _is_ma_seed(p)
            ]
        except json.JSONDecodeError:
            pass

    resolved = [r for r in wf_rows if not _is_ma_seed(r)]
    # dedupe by key keeping last
    by_key = {}
    for r in resolved:
        key = (r.get("symbol"), r.get("as_of"), r.get("horizon_days"))
        by_key[key] = r
    resolved = list(by_key.values())
    resolved.sort(key=lambda r: (r.get("as_of") or "", r.get("symbol") or "",
                                 int(r.get("horizon_days") or 0)))

    out = {
        "pending": pending,
        "resolved": resolved[-800:],
        "summary": _summary(pending, resolved),
        "wrong_clusters": {},
        "adaptive_plan": {},
    }
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    return out


def update_journal(
    frames: dict[str, pd.DataFrame],
    live_by_symbol: dict[str, list[dict]],
    path: str = "results/prediction_journal.json",
) -> dict[str, Any]:
    """Merge live forecasts into pending; grade any that have resolved."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    doc: dict[str, Any] = {"pending": [], "resolved": [], "summary": {}}
    if os.path.exists(path):
        try:
            with open(path) as f:
                doc = json.load(f)
        except json.JSONDecodeError:
            pass

    pending = [p for p in (doc.get("pending") or []) if not _is_ma_seed(p)]
    resolved = [r for r in (doc.get("resolved") or []) if not _is_ma_seed(r)]

    keys = {(p.get("symbol"), p.get("as_of"), p.get("horizon_days")) for p in pending}
    keys |= {(r.get("symbol"), r.get("as_of"), r.get("horizon_days")) for r in resolved}

    for sym, rows in live_by_symbol.items():
        for row in rows:
            key = (sym, row.get("as_of"), row.get("horizon_days"))
            if key in keys:
                continue
            pending.append({
                "symbol": sym,
                "as_of": row.get("as_of"),
                "horizon_days": row.get("horizon_days"),
                "horizon_label": row.get("horizon_label"),
                "predicted_return_pct": row.get("predicted_return_pct"),
                "predicted_price": row.get("predicted_price"),
                "model": row.get("model"),
                "trustworthy": row.get("trustworthy"),
                "source": "live_production",
            })
            keys.add(key)

    still_pending = []
    for p in pending:
        sym = p.get("symbol")
        if sym not in frames:
            still_pending.append(p)
            continue
        graded = grade_forecast(
            str(p.get("as_of")),
            int(p.get("horizon_days") or 1),
            float(p.get("predicted_return_pct") or 0),
            frames[sym]["Close"],
        )
        if graded is None:
            still_pending.append(p)
            continue
        resolved.append({**p, **graded, "source": p.get("source") or "live_production"})

    out = {
        "pending": still_pending,
        "resolved": resolved[-800:],
        "summary": _summary(still_pending, resolved),
        "wrong_clusters": doc.get("wrong_clusters") or {},
        "adaptive_plan": doc.get("adaptive_plan") or {},
    }
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    return out
