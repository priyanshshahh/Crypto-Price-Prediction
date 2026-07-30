"""Feature-engineering tests: columns, target alignment (no leakage), windows."""

import numpy as np
import pytest

from crypto_pipeline.features import add_features, build_windows, feature_matrix


@pytest.fixture(scope="module")
def feat(btc_fixture):
    return add_features(btc_fixture)


def test_expected_columns_and_no_nan(feat):
    for col in ["Return_1d", "MA7", "MA30", "MA90", "MACD", "BB_width",
                "Volatility", "RSI", "Volume_MA7", "Target_Close_1d", "Target_Return_1d"]:
        assert col in feat.columns
    assert not feat.isna().any().any()


def test_target_is_next_day_close(feat, btc_fixture):
    """Target_Return_1d at day t must equal Close[t+1]/Close[t] - 1 (future, not past)."""
    close = btc_fixture["Close"]
    for t in feat.index[:20]:
        t_next = t + (close.index[1] - close.index[0])
        if t_next in close.index:
            expected = close[t_next] / close[t] - 1
            assert feat.loc[t, "Target_Return_1d"] == pytest.approx(expected)


def test_feature_matrix_excludes_targets(feat):
    X = feature_matrix(feat)
    assert not any(c.startswith("Target_") for c in X.columns)
    assert len(X) == len(feat)


def test_rsi_bounded(feat):
    assert feat["RSI"].between(0, 100).all()


def test_build_windows_shapes_and_alignment():
    values = np.arange(100, dtype=float)
    X, y = build_windows(values, look_back=10)
    assert X.shape == (90, 10, 1)
    assert y.shape == (90,)
    # window i must end strictly before its target: X[i][-1] == values[i+9], y[i] == values[i+10]
    assert X[0, -1, 0] == 9.0 and y[0] == 10.0
    assert X[-1, -1, 0] == 98.0 and y[-1] == 99.0


def test_build_windows_too_short_raises():
    with pytest.raises(ValueError):
        build_windows(np.arange(5, dtype=float), look_back=10)
