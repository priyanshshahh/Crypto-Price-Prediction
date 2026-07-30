"""Results-schema tests: dashboard JSON builder + committed artifacts."""

import json
import os

import pytest

from crypto_pipeline.report import (build_dashboard_json, build_metrics_json,
                                    validate_dashboard_json)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _fake_run_outputs(btc_fixture):
    symbols = {"BTC": "Bitcoin"}
    frames = {"BTC": btc_fixture}
    regression_all = {"BTC": {
        "models": [{"name": "Ridge",
                    "returns": {"r2": -0.01, "rmse": 0.02, "mae": 0.015},
                    "price": {"r2": 0.9, "rmse": 900.0, "mae": 700.0},
                    "directional_accuracy": 0.52}],
        "baseline_persistence": {"returns": {"r2": 0.0, "rmse": 0.021, "mae": 0.016},
                                 "price": {"r2": 0.9, "rmse": 910.0, "mae": 710.0},
                                 "up_day_share": 0.51},
        "n_train": 160, "n_test": 40,
        "test_start": "2025-01-01", "test_end": "2025-02-10",
    }}
    ts_all = {"BTC": {"ARIMA": {
        "order": [1, 1, 1], "aic": 1000.0,
        "price": {"r2": 0.9, "rmse": 900.0, "mae": 700.0},
        "returns": {"r2": -0.02, "rmse": 0.02, "mae": 0.015},
        "directional_accuracy": 0.5,
        "baseline_persistence": {"returns": {"r2": 0.0, "rmse": 0.02, "mae": 0.015},
                                 "price": {"r2": 0.9, "rmse": 905.0, "mae": 705.0},
                                 "up_day_share": 0.5},
        "n_test": 40,
        "forecasts": [("2025-02-11", 65000.0)],
    }}}
    clustering_all = {"BTC": {"results": [
        {"algorithm": "KMeans", "optimal_clusters": 3, "silhouette_score": 0.25}],
        "n_points": 200}}
    return symbols, frames, regression_all, ts_all, clustering_all


def test_build_and_validate_dashboard_json(btc_fixture):
    symbols, frames, reg, ts, clus = _fake_run_outputs(btc_fixture)
    doc = build_dashboard_json(symbols, frames, reg, ts, clus)
    validate_dashboard_json(doc)  # must not raise
    assert len(doc["price_history"]) == 90
    # baseline row is included alongside the model rows
    names = [m["model_name"] for m in doc["regression_models"]]
    assert "Persistence (baseline)" in names


def test_validate_rejects_missing_table(btc_fixture):
    symbols, frames, reg, ts, clus = _fake_run_outputs(btc_fixture)
    doc = build_dashboard_json(symbols, frames, reg, ts, clus)
    del doc["forecasts"]
    with pytest.raises(ValueError):
        validate_dashboard_json(doc)


def test_metrics_json_has_provenance(btc_fixture):
    symbols, _, reg, ts, clus = _fake_run_outputs(btc_fixture)
    meta = {"BTC": {"symbol": "BTC", "source": "coinbase", "fetched_at": "x",
                    "start": "2024-01-01", "end": "2025-02-10", "rows": 200}}
    doc = build_metrics_json(symbols, meta, reg, ts, clus, seed=42)
    prov = doc["provenance"]
    for key in ("generated_at", "random_seed", "evaluation", "baseline", "data_sources"):
        assert key in prov
    assert doc["symbols"]["BTC"]["arima"]["directional_accuracy"] == 0.5
    # raw arrays must not leak into the JSON
    assert "pred_price" not in doc["symbols"]["BTC"]["arima"]


def test_committed_artifacts_valid_if_present():
    """After a real pipeline run, the committed artifacts must satisfy the schema."""
    dash_path = os.path.join(ROOT, "public", "data", "crypto_data.json")
    metrics_path = os.path.join(ROOT, "results", "metrics.json")
    if not (os.path.exists(dash_path) and os.path.exists(metrics_path)):
        pytest.skip("run `python pipeline.py` first")
    with open(dash_path) as f:
        validate_dashboard_json(json.load(f))
    with open(metrics_path) as f:
        metrics = json.load(f)
    assert set(metrics["symbols"]) == {"BTC", "ETH", "DOGE"}
    for sym, entry in metrics["symbols"].items():
        assert "regression" in entry and "arima" in entry
