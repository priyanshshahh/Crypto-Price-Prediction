"""Data-layer tests: API parsers (canned payloads, no network), validation,
cache round-trip, and the committed real-data fixture."""

import numpy as np
import pandas as pd
import pytest

from crypto_pipeline.data import (OHLCV_COLUMNS, load_ohlcv_csv, parse_coinbase,
                                  parse_kraken, save_ohlcv_csv, validate_ohlcv)

COINBASE_PAYLOAD = [  # [time, low, high, open, close, volume], newest first
    [1783296000, 61250.0, 64658.85, 63580.45, 64052.45, 9870.813],
    [1783209600, 62384.08, 63940.79, 63086.46, 63580.44, 3932.928],
    [1783123200, 62274.16, 63410.0, 62520.22, 63086.45, 3145.816],
]

KRAKEN_PAYLOAD = {
    "error": [],
    "result": {
        "XXBTZUSD": [
            [1721088000, "64764.3", "65416.3", "62466.0", "65088.7", "64315.2", "3185.16", 46923],
            [1721174400, "65077.1", "66100.3", "63853.1", "64120.0", "64968.4", "1952.56", 30598],
        ],
        "last": 1721174400,
    },
}


def test_parse_coinbase_shape_and_order():
    df = parse_coinbase(COINBASE_PAYLOAD)
    assert list(df.columns) == OHLCV_COLUMNS
    assert len(df) == 3
    assert df.index.is_monotonic_increasing          # sorted ascending
    assert df["Close"].iloc[-1] == pytest.approx(64052.45)
    assert df["Low"].iloc[-1] == pytest.approx(61250.0)


def test_parse_kraken_shape_and_types():
    df = parse_kraken(KRAKEN_PAYLOAD)
    assert list(df.columns) == OHLCV_COLUMNS
    assert len(df) == 2
    assert df["Open"].dtype == float
    assert df["Close"].iloc[0] == pytest.approx(65088.7)


def test_parse_kraken_error_raises():
    with pytest.raises(ValueError):
        parse_kraken({"error": ["EGeneral:Invalid arguments"], "result": {}})


def test_validate_rejects_bad_data(btc_fixture):
    good = btc_fixture.copy()
    validate_ohlcv(good)  # passes

    with pytest.raises(ValueError):  # NaN
        bad = good.copy(); bad.iloc[0, 0] = np.nan; validate_ohlcv(bad)
    with pytest.raises(ValueError):  # non-positive price
        bad = good.copy(); bad.iloc[0, 3] = -1.0; validate_ohlcv(bad)
    with pytest.raises(ValueError):  # unsorted dates
        validate_ohlcv(good.iloc[::-1])
    with pytest.raises(ValueError):  # too few rows
        validate_ohlcv(good.head(5))


def test_cache_round_trip(tmp_path, btc_fixture):
    path = tmp_path / "BTC_daily.csv"
    save_ohlcv_csv(btc_fixture, str(path))
    loaded = load_ohlcv_csv(str(path))
    pd.testing.assert_frame_equal(loaded, btc_fixture)


def test_fixture_is_real_and_sane(btc_fixture):
    """The committed fixture is real BTC-USD data: sane price range, daily cadence."""
    assert len(btc_fixture) >= 200
    assert btc_fixture["Close"].between(1_000, 500_000).all()   # plausible BTC prices
    gaps = btc_fixture.index.to_series().diff().dropna()
    assert (gaps == pd.Timedelta(days=1)).mean() > 0.95          # daily candles
    assert (btc_fixture["High"] >= btc_fixture["Low"]).all()
