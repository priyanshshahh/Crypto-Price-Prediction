"""Tests for Polymarket/fundamentals/quant/journal modules (8-strategy stack)."""

import numpy as np
import pandas as pd

from crypto_pipeline.fundamentals import fundamental_snapshot
from crypto_pipeline.journal import grade_forecast, update_journal
from crypto_pipeline.quant import ewma_vol, monte_carlo_paths, quant_overlay
from crypto_pipeline.prediction_markets import _parse_yes_price


def test_parse_yes_price():
    assert _parse_yes_price('["0.6", "0.4"]') == 0.6
    assert _parse_yes_price([0.55, 0.45]) == 0.55


def test_ewma_and_monte_carlo():
    rng = np.random.default_rng(42)
    rets = pd.Series(rng.normal(0, 0.02, 200))
    v = ewma_vol(rets)
    assert v > 0
    mc = monte_carlo_paths(100.0, 0.0, v, 7, n_paths=500)
    assert mc["price_p10"] <= mc["price_p50"] <= mc["price_p90"]


def test_quant_overlay(btc_fixture):
    out = quant_overlay(btc_fixture["Close"], 7, pred_return=0.02)
    assert "price_p50" in out
    assert out["n_paths"] == 1000


def test_fundamental_snapshot_empty():
    snap = fundamental_snapshot("BTC", pd.DataFrame())
    assert snap["available"] is False


def test_journal_grade_and_update(btc_fixture, tmp_path):
    closes = btc_fixture["Close"]
    as_of = str(closes.index[-40].date())
    graded = grade_forecast(as_of, 7, 5.0, closes)
    assert graded is not None
    assert graded["verdict"] in ("Correct", "Incorrect", "Incomplete")
    path = tmp_path / "journal.json"
    live = {"BTC": [{
        "as_of": str(closes.index[-5].date()),
        "horizon_days": 1,
        "horizon_label": "1D",
        "predicted_return_pct": 1.0,
        "predicted_price": float(closes.iloc[-1]),
        "model": "Test",
        "trustworthy": False,
    }]}
    doc = update_journal({"BTC": btc_fixture}, live, path=str(path))
    assert "summary" in doc
    assert path.exists()
