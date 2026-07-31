"""Walk-forward ML ledger + adaptive weight gates."""

import numpy as np
import pandas as pd

from crypto_pipeline.features import add_features
from crypto_pipeline.journal import _is_ma_seed, replace_with_walkforward_ledger
from crypto_pipeline.walkforward_ledger import (
    adaptive_weight_plan,
    analyze_wrong_clusters,
    error_aware_weights,
    walkforward_ledger_for_horizon,
)


def test_ma_seed_detector():
    assert _is_ma_seed({"model": "MA50/200 retrospective seed"})
    assert not _is_ma_seed({"model": "WalkForward-LightGBM"})


def test_walkforward_ledger_smoke(btc_fixture):
    feat = add_features(btc_fixture, horizons=(1,))
    # small fixture — may return few/no rows; must not crash
    rows = walkforward_ledger_for_horizon(
        feat, 1, symbol="BTC",
        min_train=80, dense_days=20, dense_step=5,
        archive_days=0, archive_step=5,
    )
    for r in rows:
        assert r["source"] == "walkforward_ml"
        assert r["verdict"] in ("Correct", "Incorrect", "Incomplete")
        assert "retrospective" not in r["model"].lower()


def test_wrong_clusters_and_adaptive(tmp_path):
    resolved = []
    for i in range(40):
        resolved.append({
            "symbol": "BTC",
            "horizon_days": 1,
            "as_of": f"2026-01-{i+1:02d}" if i < 28 else f"2026-02-{i-27:02d}",
            "verdict": "Incorrect" if i >= 30 else "Correct",
            "source": "walkforward_ml",
        })
    # force a streak at end
    for i in range(5):
        resolved.append({
            "symbol": "BTC", "horizon_days": 1,
            "as_of": f"2026-03-{i+1:02d}",
            "verdict": "Incorrect", "source": "walkforward_ml",
        })
    wc = analyze_wrong_clusters(resolved)
    assert wc["cells"]
    plan = adaptive_weight_plan(resolved, wc)
    assert "eligible_cells" in plan
    dates = pd.date_range("2025-01-01", periods=100, freq="D")
    w = error_aware_weights(dates, error_as_of=["2025-02-15"], boost=2.0, neighbor_days=2)
    assert len(w) == 100
    assert abs(w.sum() - 100) < 1e-6


def test_replace_strips_seeds(tmp_path):
    path = tmp_path / "j.json"
    path.write_text('{"pending":[],"resolved":[{"model":"MA50/200 retrospective seed","verdict":"Correct","symbol":"BTC","as_of":"2026-01-01","horizon_days":1}],"summary":{}}')
    out = replace_with_walkforward_ledger([
        {"symbol": "BTC", "as_of": "2026-06-01", "horizon_days": 1,
         "verdict": "Correct", "source": "walkforward_ml", "model": "WalkForward-LightGBM"},
    ], path=str(path))
    assert all(r.get("source") == "walkforward_ml" for r in out["resolved"])
    assert not any(_is_ma_seed(r) for r in out["resolved"])
