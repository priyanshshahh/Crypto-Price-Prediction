"""Crypto price prediction pipeline package.

Modules:
    data        -- real OHLCV fetching (Coinbase Exchange / Kraken public APIs) + caching
    external    -- Fear & Greed, Binance Vision funding, FRED macro (free)
    features    -- technical indicators, multi-horizon targets, LSTM windowing
    models      -- regression, ARIMA and LSTM with honest time-ordered evaluation
    horizons    -- multi-horizon (1/7/30d) price models + walk-forward
    paper       -- cost-aware paper trading (long/flat + long/short)
    cv          -- purged/embargo walk-forward splits
    clustering  -- market-regime clustering (KMeans, Agglomerative, GMM, DBSCAN)
    report      -- results/metrics.json + dashboard JSON assembly and schema validation
"""

__version__ = "2.1.0"

SYMBOLS = {
    "BTC": "Bitcoin",
    "ETH": "Ethereum",
    "DOGE": "Dogecoin",
}

RANDOM_STATE = 42
