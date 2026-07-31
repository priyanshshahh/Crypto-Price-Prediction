"""Unit tests for competitive market-overview helpers.

Callers: pytest. Module under test: crypto_pipeline.market_overview.
"""

from crypto_pipeline.features import add_features
from crypto_pipeline.market_overview import (
    enrich_live_row,
    prediction_scorecard,
    risk_analysis,
    technical_overview,
)


def test_technical_overview_has_bias(btc_fixture):
    feat = add_features(btc_fixture, horizons=(1,))
    ov = technical_overview(feat)
    assert ov["bias"] in ("bullish", "bearish", "neutral")
    assert 0 <= ov["scale_0_100"] <= 100
    assert ov["signals"]


def test_risk_analysis_levels(btc_fixture):
    feat = add_features(btc_fixture, horizons=(1,))
    risk = risk_analysis(feat)
    assert risk["level"] in ("low", "medium", "high")


def test_scorecard_and_enrich(btc_fixture):
    feat = add_features(btc_fixture, horizons=(1,))
    sc = prediction_scorecard(0.64, 0.3, True, n_test=100)
    assert sc["holdout_correct_est"] == 64
    assert sc["grade"] == "strong"
    row = {
        "horizon_label": "1D",
        "predicted_return_pct": 1.2,
        "direction_prob_up": 0.55,
        "model": "Ridge",
        "holdout_directional_accuracy": 0.52,
        "wf_pass_frac": 0.2,
        "holdout_beats_persistence": False,
        "trustworthy": False,
    }
    out = enrich_live_row(row, feat, n_test=40)
    assert "reasoning" in out and len(out["reasoning"]) >= 4
    assert out["technical_bias"] in ("bullish", "bearish", "neutral")
    assert out["risk_level"] in ("low", "medium", "high")
    assert "scorecard" in out
