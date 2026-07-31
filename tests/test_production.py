"""Tests for multi-horizon features, CV, and paper trading."""

import numpy as np
import pandas as pd
import pytest

from crypto_pipeline.cv import chronological_holdout, expanding_walk_forward
from crypto_pipeline.features import add_features, feature_matrix
from crypto_pipeline.paper import PaperConfig, run_paper, select_best_mode


def test_multi_horizon_targets(btc_fixture):
    feat = add_features(btc_fixture, horizons=(1, 7, 30))
    for h in (1, 7, 30):
        assert f"Target_Return_{h}d" in feat.columns
        assert f"Target_Close_{h}d" in feat.columns
    X = feature_matrix(feat)
    assert not any(c.startswith("Target_") for c in X.columns)


def test_horizon_alignment(btc_fixture):
    feat = add_features(btc_fixture, horizons=(1, 5))
    close = btc_fixture["Close"]
    for t in list(feat.index)[10:15]:
        pos = close.index.get_loc(t)
        if pos + 5 < len(close):
            expected = close.iloc[pos + 5] / close.iloc[pos] - 1
            assert feat.loc[t, "Target_Return_5d"] == pytest.approx(expected)


def test_external_join_no_leak_columns(btc_fixture):
    ext = pd.DataFrame(
        {"FearGreed": np.linspace(20, 80, len(btc_fixture))},
        index=btc_fixture.index,
    )
    feat = add_features(btc_fixture, horizons=(1,), external=ext)
    assert "FearGreed" in feat.columns
    assert "FearGreed" in feature_matrix(feat).columns


def test_walk_forward_folds_ordered():
    folds = expanding_walk_forward(400, min_train=180, test_size=30, embargo=5, purge=5)
    assert len(folds) >= 3
    for f in folds:
        assert f.train_end <= f.test_start
        assert f.test_start < f.test_end


def test_chronological_holdout():
    f = chronological_holdout(100, 0.2)
    assert f.train_end == 80 and f.test_end == 100


def test_paper_both_modes_and_select():
    rng = np.random.default_rng(42)
    actual = rng.normal(0.001, 0.02, 120)
    pred = actual * 0.3 + rng.normal(0, 0.01, 120)
    out = select_best_mode(actual, pred, min_abs_pred=0.01)
    assert out["best"]["mode"] in ("long_flat", "long_short")
    assert "long_flat" in out["modes"] and "long_short" in out["modes"]
    assert np.isfinite(out["best"]["net_sharpe"])


def test_paper_costs_reduce_sharpe():
    actual = np.array([0.02, -0.01, 0.03, -0.02, 0.01] * 20)
    pred = actual.copy()
    free = run_paper(actual, pred, PaperConfig(fee_bps_round_trip=0, slippage_bps=0,
                                               min_abs_pred=0.005, mode="long_short"))
    costly = run_paper(actual, pred, PaperConfig(fee_bps_round_trip=50, slippage_bps=10,
                                                 min_abs_pred=0.005, mode="long_short"))
    assert costly["net_sharpe"] <= free["gross_sharpe"] + 1e-9


def test_ma50_atr_features(btc_fixture):
    feat = add_features(btc_fixture, horizons=(1,))
    for col in ("MA50", "MA200", "ATR14", "ATR14_pct", "GoldenDeathCross", "Price_vs_MA200"):
        assert col in feat.columns


def test_for_prediction_keeps_latest(btc_fixture):
    live = add_features(btc_fixture, horizons=(1, 7, 30), for_prediction=True)
    assert len(live) > 0
    assert live["Close"].iloc[-1] == pytest.approx(btc_fixture["Close"].iloc[-1])
