# Crypto Price Prediction & Analysis

End-to-end ML pipeline forecasting daily prices for **Bitcoin**, **Ethereum**, and **Dogecoin**
from **real OHLCV market data**, plus an interactive React dashboard.

- **Data**: real daily candles from the Coinbase Exchange public API (Kraken as automatic fallback) — free, keyless, cached locally
- **Models**: regularized regressions (Ridge/Lasso/ElasticNet), SVR, Random Forest, Gradient Boosting; ARIMA and LSTM time-series models; K-Means / Agglomerative / GMM / DBSCAN market-regime clustering
- **Evaluation**: chronological holdout, walk-forward one-step ARIMA, persistence baselines, returns-based metrics — no shuffled splits, no leakage, no cherry-picking
- **Outputs**: `results/metrics.json` (with provenance), `public/data/crypto_data.json` (drives the dashboard), PNG charts

## Honest results (measured 2026-07-06, data 2024-07-06 → 2026-07-06, seed 42)

The headline first: **daily crypto returns are close to unpredictable from technical
indicators alone, and this project says so.** Directional accuracy hovers around 50% and no
model decisively beats the naive persistence baseline (tomorrow's close = today's close).

An earlier version of this README quoted R² ≈ 0.98 for "price prediction". That number was
real but **misleading**: next-day *price levels* are strongly autocorrelated, so even
"predict yesterday's price" scores R² > 0.9. All primary metrics below are therefore
**returns-based**, with the persistence baseline alongside every model.

### Next-day return prediction (chronological 80/20 holdout, test ≈ 129 days)

| Crypto | Best directional accuracy | Best return RMSE (model) | Persistence RMSE (baseline) | Up-day share (baseline) |
|--------|---------------------------|--------------------------|------------------------------|--------------------------|
| BTC    | 57.4% (SVR)               | 0.02143 (Lasso)          | **0.02088**                  | 50.4% |
| ETH    | 50.8% (Ridge)             | 0.02926 (Lasso)          | **0.02829**                  | 50.4% |
| DOGE   | 53.9% (GradientBoosting)  | 0.02888 (Lasso)          | **0.02631**                  | 49.6% |

No regression model beats the persistence RMSE — the honest conclusion. (SVR's 57.4%
directional accuracy on BTC comes with the worst RMSE; it is not a free lunch.)

### Time-series models — one-step-ahead price RMSE on the holdout

| Crypto | ARIMA (order) | ARIMA RMSE | LSTM RMSE | Persistence RMSE | ARIMA dir. acc | LSTM dir. acc |
|--------|---------------|------------|-----------|-------------------|----------------|----------------|
| BTC    | (0,1,1)       | 1477.72    | 3837.32   | **1472.24**       | 45.6% | 42.9% |
| ETH    | (0,1,1)       | **59.538** | 116.76    | 59.540            | 52.7% | 54.1% |
| DOGE   | (0,1,1)       | 0.00304    | 0.01020   | **0.00293**       | 49.3% | 49.3% |

ARIMA collapses to essentially a random-walk model — (0,1,1) tracking persistence — which is
exactly what efficient-market theory predicts for daily data. The LSTM (capped at 40 epochs
with early stopping; it stopped at 13–15) underperforms persistence.

### Market-regime clustering (silhouette scores)

| Crypto | Best algorithm | k | Silhouette |
|--------|----------------|---|------------|
| BTC    | DBSCAN         | 3 | 0.363 |
| ETH    | KMeans         | 3 | 0.267 |
| DOGE   | Agglomerative  | 2 | 0.625 |

Clustering on (return, volatility, RSI, MACD, volume) daily vectors does find structure —
distinct calm/volatile regimes — and is the most defensible unsupervised result here.

Every number above comes from a run executed on this repository; full metrics with
provenance (data source, date range, seed, evaluation policy) are committed in
[`results/metrics.json`](results/metrics.json).

## Architecture

```
pipeline.py                 one-command orchestrator
crypto_pipeline/
  data.py                   Coinbase/Kraken fetchers, parsing, validation, data/raw/ cache
  features.py               technical indicators + LSTM windowing (no look-ahead)
  models.py                 regressions, ARIMA (walk-forward), LSTM (train-only scaling)
  clustering.py             KMeans / Agglomerative / GMM / DBSCAN regime clustering
  report.py                 metrics.json + dashboard JSON assembly, schema validation
tests/                      pytest suite (23 tests) + committed real-data fixture
src/                        React (Vite) dashboard
public/data/crypto_data.json  static data consumed by the dashboard
results/                    metrics.json, per-symbol CSVs, PNG charts
```

## Setup & run

```bash
# Python (TensorFlow needs Python <= 3.12)
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt

# one command: fetch real data -> train -> evaluate -> write results + dashboard JSON
.venv/bin/python pipeline.py            # ~3-5 min on CPU (LSTM epochs capped at 40 + early stopping)
# options: --days 730  --force-refresh  --skip-lstm  --epochs N

# tests
.venv/bin/python -m pytest tests/ -q
```

Raw pulls are cached in `data/raw/` (gitignored, refreshed after 24h). A 250-row sample of
real BTC data is committed at `tests/fixtures/BTC_sample.csv` so tests run offline.

## Dashboard

The React dashboard reads the **static JSON** produced by the pipeline
(`public/data/crypto_data.json`) via a mock Supabase-compatible client — this is the
default and requires **zero backend**. If you set `VITE_SUPABASE_URL` /
`VITE_SUPABASE_ANON_KEY`, it transparently switches to a real Supabase project
(migrations in `supabase/`), but that is optional.

```bash
npm ci
npm run dev      # local dev
npm run build    # production build (verified passing)
```

Deployment steps (Vercel, free tier): see [DEPLOY.md](DEPLOY.md).

A Tableau dashboard built on the same outputs:
https://us-east-1.online.tableau.com/t/priyanshshah-7ae2fd725b/views/Dasboard/Dashboard1

## Evaluation policy (what "honest" means here)

- Chronological 80/20 split; never shuffled. Test window ≈ last 129 days.
- Scalers (StandardScaler / MinMaxScaler) fit on **train only** — the original LSTM fit its
  scaler on the full series, which was look-ahead leakage; fixed.
- ARIMA evaluated **walk-forward one-step-ahead** (train-fitted parameters applied over the
  growing history), not a single 146-day-ahead extrapolation.
- Every model is compared against the **persistence baseline**.
- Fixed seed (42) for NumPy / scikit-learn / TensorFlow; LSTM epoch cap documented.

## Limitations

- Daily technical indicators contain little exploitable signal; results reflect that. This
  project demonstrates pipeline & evaluation engineering, not alpha.
- ~2 years of daily data (≈731 candles, Coinbase/Kraken public API caps) — small for LSTMs.
- No transaction-cost or strategy backtest; directional accuracy is not a trading claim.
- LSTM results vary slightly across hardware even with fixed seeds (TF non-determinism).
- The Jupyter notebooks (`cryp_*.ipynb`, `crypt_*.ipynb`) are the original exploratory
  work, kept for history; `pipeline.py` + `crypto_pipeline/` is the maintained path.

## Author

**Priyansh Shah** — Stony Brook University, B.S. Applied Mathematics and Statistics
[priyansh.shah@stonybrook.edu](mailto:priyansh.shah@stonybrook.edu) · [LinkedIn](https://linkedin.com/in/priyansh-shah)

## License

MIT (see LICENSE). Educational project — not financial advice.
