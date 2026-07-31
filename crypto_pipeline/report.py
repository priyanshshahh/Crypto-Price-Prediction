"""Assemble results/metrics.json (with provenance) and the dashboard JSON
consumed by the React app (public/data/crypto_data.json), plus schema checks.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pandas as pd

DASHBOARD_TABLES = {
    "cryptocurrencies": ["id", "symbol", "name", "created_at"],
    "price_history": ["id", "crypto_id", "date", "open", "high", "low", "close", "volume"],
    "regression_models": ["id", "crypto_id", "model_name", "r2_score", "rmse", "mae",
                          "directional_accuracy", "is_best"],
    "forecasts": ["id", "crypto_id", "model_type", "forecast_date", "predicted_price"],
    "clustering_results": ["id", "crypto_id", "algorithm", "optimal_clusters",
                           "silhouette_score", "is_best"],
}


def _uid() -> str:
    return str(uuid.uuid4())


def build_metrics_json(symbols: dict, data_meta: dict, regression_all: dict,
                       ts_all: dict, clustering_all: dict, seed: int,
                       lstm_config: dict | None = None) -> dict:
    """Full honest metrics file with provenance. Every number is measured."""
    out = {
        "provenance": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "random_seed": seed,
            "evaluation": "chronological 80/20 holdout; scalers fit on train only; "
                          "one-step-ahead walk-forward for ARIMA; "
                          "returns-based metrics are primary",
            "baseline": "persistence (tomorrow's close = today's close)",
            "lstm_config": lstm_config or {},
            "data_sources": {sym: data_meta[sym] for sym in symbols},
        },
        "symbols": {},
    }
    for sym in symbols:
        entry = {"regression": regression_all[sym], "clustering": clustering_all[sym]["results"]}
        ts = ts_all.get(sym, {})
        for model_name in ("ARIMA", "LSTM", "GRU"):
            if model_name in ts:
                m = {k: v for k, v in ts[model_name].items()
                     if k not in ("pred_price", "actual_price", "forecasts")}
                entry[model_name.lower()] = m
        out["symbols"][sym] = entry
    return out


def build_dashboard_json(symbols: dict, frames: dict, regression_all: dict,
                         ts_all: dict, clustering_all: dict,
                         history_days: int = 90,
                         horizon_forecasts: list | None = None) -> dict:
    """JSON in the shape the React dashboard (and its mock Supabase client) expects.

    Callers: pipeline.py (classic + production). Optional horizon_forecasts adds
    Optional 1D/1W/1M live forecast rows (extra table; not in DASHBOARD_TABLES required set).
    """
    now = datetime.now(timezone.utc).isoformat()
    ids = {sym: _uid() for sym in symbols}

    crypto_list = [{"id": ids[s], "symbol": s, "name": symbols[s], "created_at": now}
                   for s in symbols]

    price_history = []
    for sym, df in frames.items():
        for date, row in df.tail(history_days).iterrows():
            price_history.append({
                "id": _uid(), "crypto_id": ids[sym],
                "date": pd.Timestamp(date).strftime("%Y-%m-%d"),
                "open": round(float(row["Open"]), 6), "high": round(float(row["High"]), 6),
                "low": round(float(row["Low"]), 6), "close": round(float(row["Close"]), 6),
                "volume": round(float(row["Volume"]), 2),
            })

    regression_models = []
    for sym, reg in regression_all.items():
        rows = reg["models"]
        best_da = max(m["directional_accuracy"] for m in rows)
        for m in rows:
            regression_models.append({
                "id": _uid(), "crypto_id": ids[sym], "model_name": m["name"],
                "r2_score": round(m["returns"]["r2"], 6),
                "rmse": round(m["returns"]["rmse"], 6),
                "mae": round(m["returns"]["mae"], 6),
                "directional_accuracy": round(m["directional_accuracy"], 6),
                "is_best": m["directional_accuracy"] == best_da,
            })
        # persistence baseline as an explicit comparison row
        base = reg["baseline_persistence"]
        regression_models.append({
            "id": _uid(), "crypto_id": ids[sym], "model_name": "Persistence (baseline)",
            "r2_score": round(base["returns"]["r2"], 6),
            "rmse": round(base["returns"]["rmse"], 6),
            "mae": round(base["returns"]["mae"], 6),
            "directional_accuracy": round(base["up_day_share"], 6),
            "is_best": False,
        })

    forecasts = []
    for sym, ts in ts_all.items():
        for model_type in ("ARIMA", "LSTM", "GRU"):
            if model_type not in ts:
                continue
            for item in ts[model_type]["forecasts"]:
                # Support (date, price) or (date, price, lo, hi) from ARIMA log-return path
                if len(item) >= 4:
                    date_str, price, lo, hi = item[0], item[1], item[2], item[3]
                else:
                    date_str, price = item[0], item[1]
                    lo = hi = None
                row = {
                    "id": _uid(), "crypto_id": ids[sym], "model_type": model_type,
                    "forecast_date": date_str, "predicted_price": round(float(price), 6),
                }
                if lo is not None and hi is not None:
                    row["predicted_price_lo"] = round(float(lo), 6)
                    row["predicted_price_hi"] = round(float(hi), 6)
                forecasts.append(row)

    clustering_results = []
    for sym, clus in clustering_all.items():
        algs = clus["results"]
        best_sil = max(a["silhouette_score"] for a in algs)
        for a in algs:
            clustering_results.append({
                "id": _uid(), "crypto_id": ids[sym], "algorithm": a["algorithm"],
                "optimal_clusters": a["optimal_clusters"],
                "silhouette_score": round(a["silhouette_score"], 6),
                "is_best": a["silhouette_score"] == best_sil,
            })

    doc = {
        "cryptocurrencies": crypto_list,
        "price_history": price_history,
        "regression_models": regression_models,
        "forecasts": forecasts,
        "clustering_results": clustering_results,
    }
    if horizon_forecasts:
        # Attach crypto_id if callers passed symbol
        rows = []
        for row in horizon_forecasts:
            r = dict(row)
            sym = r.pop("symbol", None)
            if sym and "crypto_id" not in r:
                r["crypto_id"] = ids[sym]
            if "id" not in r:
                r["id"] = _uid()
            rows.append(r)
        doc["horizon_forecasts"] = rows
    return doc


def validate_dashboard_json(doc: dict) -> None:
    """Raise ValueError if the dashboard JSON doesn't match the expected schema."""
    for table, fields in DASHBOARD_TABLES.items():
        if table not in doc:
            raise ValueError(f"Missing table: {table}")
        if not isinstance(doc[table], list) or not doc[table]:
            raise ValueError(f"Table {table} must be a non-empty list")
        for row in doc[table]:
            missing = [f for f in fields if f not in row]
            if missing:
                raise ValueError(f"{table} row missing fields: {missing}")
    crypto_ids = {c["id"] for c in doc["cryptocurrencies"]}
    for table in ("price_history", "regression_models", "forecasts", "clustering_results"):
        bad = [r for r in doc[table] if r["crypto_id"] not in crypto_ids]
        if bad:
            raise ValueError(f"{table} contains rows with unknown crypto_id")
