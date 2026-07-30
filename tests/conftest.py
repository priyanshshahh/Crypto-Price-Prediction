import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FIXTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


@pytest.fixture(scope="session")
def btc_fixture() -> pd.DataFrame:
    """Small sample of REAL BTC-USD daily OHLCV committed for offline tests."""
    from crypto_pipeline.data import load_ohlcv_csv
    return load_ohlcv_csv(os.path.join(FIXTURE_DIR, "BTC_sample.csv"))
