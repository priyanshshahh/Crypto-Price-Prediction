"""Crypto price prediction pipeline package.

Modules:
    data        -- real OHLCV fetching (Coinbase Exchange / Kraken public APIs) + caching
    features    -- technical-indicator feature engineering and LSTM windowing
    models      -- regression, ARIMA and LSTM with honest time-ordered evaluation
    clustering  -- market-regime clustering (KMeans, Agglomerative, GMM, DBSCAN)
    report      -- results/metrics.json + dashboard JSON assembly and schema validation
"""

__version__ = "2.0.0"

SYMBOLS = {
    "BTC": "Bitcoin",
    "ETH": "Ethereum",
    "DOGE": "Dogecoin",
}

RANDOM_STATE = 42
