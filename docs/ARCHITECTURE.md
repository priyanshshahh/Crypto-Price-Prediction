# Architecture

**Last updated:** 2026-07-31  
Public product docs for this open-source crypto forecasting stack.

## Modes

| Mode | Command | Role |
|------|---------|------|
| **Production** | `pipeline.py --mode production` | 1d / 7d / 30d models, externals, paper book, walk-forward journal, LSTM/GRU paths |
| **Classic** | `pipeline.py` | Next-day regressions, ARIMA, LSTM/GRU, clustering |

```bash
.venv/bin/python pipeline.py --mode production
.venv/bin/python pipeline.py --mode production --skip-lstm
.venv/bin/python pipeline.py --mode production --skip-wf-ledger
```

## Package map

```
pipeline.py
crypto_pipeline/
  data.py                 Coinbase / Kraken OHLCV + cache
  features.py             TA + multi-horizon targets + external merge
  models.py               Regressions, ARIMA (log-return), LSTM, GRU
  alpha.py                Tree ensembles, direction calibration, conformal bands
  tune.py                 Optuna LightGBM + stacking helpers
  horizons.py             Multi-horizon train/eval, live forecasts, adaptive weights
  external.py             Fear&Greed, funding, FRED, Trends, on-chain proxies, …
  fundamentals.py         ETH staking APY / TVL scenario helpers
  prediction_markets.py   Prediction-market crowd Yes snapshots
  quant.py                EWMA vol + Monte Carlo paths
  walkforward_ledger.py   Dense leak-free skill ledger
  journal.py              Pending + resolved Correct/Incorrect
  cv.py                   Chronological holdout + expanding WF folds
  paper.py                Long/flat and long/short paper books
  clustering.py           Regime clustering
  report.py               metrics.json + dashboard JSON
src/                      React dashboard (Vite)
public/data/crypto_data.json
results/                  metrics, horizons, prediction_journal
```

## Production data flow

```
fetch OHLCV → load externals → add_features(1,7,30)
  → walk-forward ledger (optional) → wrong-cluster / adaptive plan
  → run_all_horizons → live_price_forecasts
  → clustering → classic regressions + ARIMA/LSTM/GRU
  → metrics.json + horizons.json + crypto_data.json
```

## Dashboard

Static JSON by default (mock Supabase client). Tabs: **Overview**, **Models**, **Forecast**, **Clustering**.

Optional Supabase: set `VITE_SUPABASE_URL` + `VITE_SUPABASE_ANON_KEY`.

## Trustworthy gate

A horizon is Trustworthy only when **all** hold:

1. Beat persistence on return RMSE  
2. Walk-forward pass fraction ≥ 0.55  
3. Direction gate (≥ 0.55 or high-confidence subset)

Otherwise the UI shows **Low confidence**.
