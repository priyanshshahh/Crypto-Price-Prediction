"""Model smoke tests on the real-data fixture (fast: small data, tiny configs)."""

import numpy as np
import pytest

from crypto_pipeline.features import add_features
from crypto_pipeline.models import (directional_accuracy, persistence_baseline,
                                    regression_metrics, run_arima,
                                    run_regressions, time_split)


@pytest.fixture(scope="module")
def feat(btc_fixture):
    return add_features(btc_fixture)


def test_time_split_chronological():
    assert time_split(100, 0.2) == 80
    assert time_split(250, 0.2) == 200


def test_directional_accuracy():
    actual = np.array([0.01, -0.02, 0.03, -0.01])
    assert directional_accuracy(actual, actual) == 1.0
    assert directional_accuracy(actual, -actual) == 0.0
    assert directional_accuracy(actual, np.array([0.01, 0.02, 0.03, 0.01])) == 0.5


def test_persistence_baseline_math():
    prev = np.array([100.0, 100.0])
    ret = np.array([0.10, -0.10])
    base = persistence_baseline(prev, ret)
    # predicting zero return on +/-10% moves -> price RMSE = 10
    assert base["price"]["rmse"] == pytest.approx(10.0)
    assert base["returns"]["rmse"] == pytest.approx(0.10)
    assert base["up_day_share"] == 0.5


def test_run_regressions_smoke(feat):
    out = run_regressions(feat)
    names = {m["name"] for m in out["models"]}
    assert names == {"Ridge", "Lasso", "ElasticNet", "SVR", "RandomForest", "GradientBoosting"}
    for m in out["models"]:
        for key in ("r2", "rmse", "mae"):
            assert np.isfinite(m["returns"][key])
        assert 0.0 <= m["directional_accuracy"] <= 1.0
    assert out["n_train"] > out["n_test"] > 0
    assert np.isfinite(out["baseline_persistence"]["price"]["rmse"])


def test_run_arima_smoke(feat, btc_fixture):
    out = run_arima(btc_fixture["Close"], steps=5, orders=((1, 0, 1), (1, 0, 0)))
    assert len(out["forecasts"]) == 5
    assert len(out["pred_price"]) == out["n_test"] == len(out["actual_price"])
    assert np.isfinite(out["price"]["rmse"])
    assert out["on"] == "log_returns"
    prices = [item[1] for item in out["forecasts"]]
    # Must not be a flat random-walk line (old ARIMA(0,1,1)-on-price bug)
    assert float(np.std(prices)) > 0 or abs(prices[-1] - prices[0]) > 0
    for item in out["forecasts"]:
        date_str, price = item[0], item[1]
        assert len(date_str) == 10 and price > 0
        if len(item) >= 4:
            assert item[3] >= item[2]


def test_run_lstm_smoke(btc_fixture):
    tf = pytest.importorskip("tensorflow")  # noqa: F841
    from crypto_pipeline.models import run_lstm
    out = run_lstm(btc_fixture["Close"], steps=3, look_back=20, epochs=2)
    assert out["epochs_ran"] <= 2
    assert len(out["forecasts"]) == 3
    assert np.isfinite(out["price"]["rmse"])
    assert len(out["pred_price"]) == out["n_test"]


def test_clustering_smoke(feat):
    from crypto_pipeline.clustering import run_clustering
    out = run_clustering(feat)
    algos = {r["algorithm"] for r in out["results"]}
    assert algos == {"KMeans", "Agglomerative", "GMM", "DBSCAN"}
    km = next(r for r in out["results"] if r["algorithm"] == "KMeans")
    assert 2 <= km["optimal_clusters"] <= 6
    assert -1.0 <= km["silhouette_score"] <= 1.0
    assert out["pca_coords"].shape == (out["n_points"], 2)
