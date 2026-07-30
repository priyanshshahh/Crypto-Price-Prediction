# Architecture

End-to-end ML pipeline forecasting daily cryptocurrency prices. This doc describes what the
code actually does, traced from `pipeline.py` outward.

## Folder map

```
Crypto-Price-Prediction/
├── pipeline.py                    one-command orchestrator; parses CLI flags, wires data→features→models→report
├── crypto_pipeline/               Python package for ML pipeline steps
│   ├── __init__.py                 exports SYMBOLS, RANDOM_STATE (42)
│   ├── data.py                     Coinbase/Kraken API fetchers; parse/validate OHLCV; local cache in data/raw/
│   ├── features.py                 Technical indicators (RSI, MACD, Bollinger); LSTM windowing (no look-ahead)
│   ├── models.py                   Regressions (Ridge/Lasso/ElasticNet/SVR/RF/GBR); ARIMA walk-forward; LSTM train-only scaling
│   ├── clustering.py               Market-regime clustering: KMeans / Agglomerative / GMM / DBSCAN
│   └── report.py                   JSON builders: metrics.json (provenance + results), dashboard JSON (schema-validated)
├── tests/                          pytest suite (23 tests); fixtures/ has real-data CSV sample
├── src/                            React dashboard (Vite, TypeScript)
│   ├── components/
│   │   ├── Dashboard.jsx            main page; reads static crypto_data.json or real Supabase (optional)
│   │   ├── ModelPerformance.jsx     regression results table; persistence baseline reference line
│   │   ├── TimeSeriesModels.jsx     ARIMA/LSTM predictions, forecasts, directional accuracy
│   │   └── ClusteringAnalysis.jsx   market regimes (PCA scatter plots, silhouette scores)
│   ├── lib/supabase.ts             mock client (default) or real Supabase via env vars
│   └── App.tsx                     entry point
├── public/data/crypto_data.json    static dashboard data (written by pipeline.py / report.py)
├── results/                        committed metrics.json + per-symbol CSVs + PNG visualizations
├── data/raw/                       raw API pulls cached locally (gitignored; 24h TTL)
├── supabase/                       migrations (unused in static-JSON mode; optional)
├── docs/
│   ├── PROJECT-NOTES.md            rebuild audit log + evaluation policy
│   └── ARCHITECTURE.md             this file
├── README.md                       model card; honest limitations; setup/run
├── requirements.txt                Python (TensorFlow needs <= 3.12)
├── package.json                    React/Vite dependencies
└── DEPLOY.md                       Vercel deployment steps
```

## Pipeline architecture

### Data pipeline

```
pipeline.py --days 730
  │
  ├─→ fetch_ohlcv(symbol)
  │      Tries Coinbase (paginated 300/req, 0.4s between)
  │      Falls back to Kraken (single batch)
  │      On API fail, uses stale cache (better than nothing)
  │      Returns df[Date, Open, High, Low, Close, Volume]
  │
  ├─→ add_features(df)
  │      Returns (prev close): used for returns calculation
  │      Technical indicators: RSI, MACD, Bollinger Bands, SMA, EMA
  │      Lags: price change, volume change (no forward-looking)
  │      Target: next-day return (log-return to close, 1-step ahead)
  │
  ├─→ run_regressions(feat)
  │      Chronological 80/20 split (no shuffling)
  │      Scalers fit on TRAIN only (no leakage)
  │      6 models: Ridge, Lasso, ElasticNet, SVR, RandomForest, GradientBoosting
  │      Returns: directional accuracy, returns RMSE/R2/MAE, price RMSE/R2/MAE
  │      vs. persistence baseline (predict zero return)
  │
  ├─→ run_arima(close)
  │      AIC-based order selection on train data
  │      Walk-forward 1-step-ahead on test window
  │      Selected order applied to full series, then one-step predictions extracted
  │      Returns: ARIMA order, price/returns metrics, directional accuracy
  │
  ├─→ run_lstm(close, epochs=40)
  │      MinMaxScaler fit on train closes only
  │      60-day look-back window (no forward info)
  │      2-layer LSTM (50 units each) with Dropout(0.2)
  │      Early stopping: patience=5 (usually stops at 13–15 epochs)
  │      Returns: epochs_ran, price/returns metrics, directional accuracy
  │
  ├─→ run_clustering(feat)
  │      Features: return, volatility, RSI, MACD, volume normalized
  │      4 algorithms tested: KMeans (k=2–6), Agglomerative, GMM, DBSCAN
  │      Selects best k via silhouette score sweep
  │      PCA reduction to 2D for visualization
  │      Returns: silhouette score, algorithm rankings
  │
  └─→ build_metrics_json() → results/metrics.json
         Provenance: generated_at, seed, data source, date range, eval policy
         Per-symbol: all model results, baselines, test date ranges
```

### Evaluation policy (applies everywhere)

- **Chronological 80/20 split**: test window = last ~20% of data, never shuffled
- **Scalers fit on TRAIN only**: StandardScaler / MinMaxScaler fit before accessing test data
- **Primary metrics = RETURNS-based**: because price-level R² ≈ autocorrelation (predict yesterday's price → R² > 0.9)
- **Persistence baseline alongside every model**: "tomorrow's close = today's close" (zero predicted return)
- **Fixed seed (42)**: NumPy / scikit-learn / TensorFlow for reproducibility
- **ARIMA**: walk-forward one-step-ahead, not a single 146-day extrapolation
- **LSTM**: epochs capped at 40; early stopping with patience=5; look-back=60

## Dashboard data flow

```
pipeline.py → report.build_dashboard_json()
  │
  ├─→ Serializes per-symbol model results, forecasts, clustering labels
  ├─→ Schema validation (checks required fields, types, ranges)
  └─→ Writes public/data/crypto_data.json

frontend (React)
  │
  ├─→ lib/supabase.ts loads JSON
  │    Default: fetch /data/crypto_data.json (mock Supabase)
  │    Optional: set VITE_SUPABASE_URL + VITE_SUPABASE_ANON_KEY → real Supabase
  │
  └─→ Components render:
       - ModelPerformance: regression table + directional accuracy + persistence baseline
       - TimeSeriesModels: ARIMA/LSTM predictions (one-step-ahead on test), forecasts (30-day)
       - ClusteringAnalysis: regime PCA scatter plots + silhouette scores per algorithm
```

## Model specifications (from `results/metrics.json`, run 2026-07-06)

**Data**: Real daily OHLCV, Coinbase Exchange (primary), Kraken (fallback); date range 2024-07-06 → 2026-07-06; ~731 candles per symbol (BTC, ETH, DOGE).

**Evaluation split**: Chronological 80/20; test window ≈129 days (2026-02-27 → 2026-07-05).

**Regression models**:
- Ridge, Lasso, ElasticNet (linear + L1/L2 regularization)
- SVR (support vector regression, non-linear)
- RandomForest (200 trees, max_depth=6)
- GradientBoosting (200 trees, max_depth=3)

**Time-series models**:
- ARIMA: order selection via AIC on train; walk-forward predictions on test
  - All three symbols selected (0,1,1) — essentially a random walk
- LSTM: 60-day look-back, 2 layers (50 units + Dropout=0.2), early stopping (patience=5)
  - Epochs capped at 40; typically stopped at 13–15 epochs

**Clustering**:
- KMeans, Agglomerative, GMM, DBSCAN
- Silhouette score sweep to find optimal k (2–6)
- Features: return, volatility, RSI, MACD, volume

**Result**: No regression model beats persistence RMSE on next-day returns; directional accuracy hovers ≈50%; clustering finds real regime structure (silhouettes 0.24–0.62).

## Test layout (`tests/`, 23 tests)

| File | Covers |
|---|---|
| `conftest.py` | Shared fixture: real 250-row BTC sample (`fixtures/BTC_sample.csv`) |
| `test_data.py` | Coinbase/Kraken parsers on canned payloads (no network); OHLCV validation; cache round-trip; data freshness logic |
| `test_features.py` | Feature columns, technical indicator math, target alignment (no look-ahead leakage), window building for LSTM |
| `test_models.py` | Regression smoke test (all 6 models, metrics finite); ARIMA order selection and walk-forward logic; LSTM on fixture with tiny epochs; time-split chronology |
| `test_schema.py` | Dashboard JSON schema validation; committed artifacts (metrics.json, crypto_data.json) conform to schema |

Run with `python -m pytest tests/ -q`. All tests use the committed 250-row BTC fixture or synthetic test data — no network calls, no external APIs.

## Environment setup

```bash
# Python (requires <= 3.12 for TensorFlow)
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Node.js (React dashboard)
npm ci
npm run build
```

## Known limitations

- Tiny dataset (~731 daily candles) for LSTM training; poor generalization.
- Daily technical indicators contain little exploitable signal; results reflect that.
- No hyperparameter search or ensemble methods (would invite overfitting the small holdout).
- LSTM results vary slightly across hardware (TensorFlow non-determinism even with seed 42).
- Coinbase returns the still-forming candle for "today" — acceptable for daily modeling, documented in `data.py`.
- No transaction cost, slippage, or backtest—directional accuracy is not a trading claim.
