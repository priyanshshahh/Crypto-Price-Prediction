# Crypto Price Prediction

Open-source, production-oriented pipeline and dashboard for **Bitcoin**, **Ethereum**, and **Dogecoin** price forecasting at **1-day / 1-week / 1-month** horizons.

Real market data · honest evaluation · paper trading only · MIT licensed.

**Not financial advice.** No model predicts crypto with certainty. This project optimizes for *measurable* skill (vs persistence baselines and walk-forward tests), not marketing claims.

## Features

| Area | What ships |
|------|------------|
| **Multi-horizon forecasts** | 1D / 1W / 1M return → price targets with p10–p90 bands and P(up) |
| **ML stack** | LightGBM, XGBoost, stacked / regime models; LSTM & GRU path charts |
| **Market context** | Technical indicators, funding, on-chain proxies, macro (FRED), Fear & Greed, Trends |
| **Fundamentals (ETH)** | DeFi TVL, staking-yield proxy, scenario weights |
| **Prediction-market odds** | Crowd Yes probability vs model P(up) |
| **Quant risk** | EWMA volatility + Monte Carlo scenario bands |
| **Skill journal** | Walk-forward Correct / Incorrect ledger (purge by horizon) |
| **Gates** | Trustworthy only if beat persistence **and** walk-forward ≥55% **and** direction gate |
| **Regimes** | KMeans / DBSCAN / Agglomerative / GMM clustering |
| **Dashboard** | React (Vite) — overview, models, forecast, clustering |

## Quick start

```bash
# Python ≤ 3.12 (TensorFlow)
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Optional keys — copy .env.example → .env.local
.venv/bin/python pipeline.py --mode production
.venv/bin/python -m pytest tests/ -q

npm ci && npm run dev    # http://localhost:5173
npm run build            # production static build
```

Faster runs: `--skip-lstm` (no deep path charts), `--skip-wf-ledger` (skip dense walk-forward journal refresh).

Deploy: see [DEPLOY.md](DEPLOY.md).

## Honest results (snapshot 2026-07-30)

Primary metric: **return RMSE vs persistence** (predicting “no change”).  
Price-level R² ≈ 0.95 is **not** skill — persistence alone scores ~0.95 on prices.

| Asset | Horizon | Highlight |
|-------|---------|-----------|
| **BTC** | **30d** | RegimeLGBM beat persistence; ~64% direction (WF still weak) |
| BTC / ETH / DOGE | 1d / 7d | Generally **do not** beat persistence |

**Trustworthy** requires all three gates. Most live cells show **Low confidence** by design.

Walk-forward journal hit rate on recent dense folds is often near coin-flip (~40%) — that is the honest ledger.

Full numbers: [`docs/METHODS.md`](docs/METHODS.md), committed [`results/metrics.json`](results/metrics.json).

## Repository layout

```
pipeline.py                 classic + production orchestrator
crypto_pipeline/            data, features, models, horizons, journal, …
src/                        React dashboard
public/data/crypto_data.json
results/                    metrics, horizons, prediction_journal
docs/                       ARCHITECTURE, METHODS, DATA
tests/                      pytest (offline fixtures)
DEPLOY.md                   Vercel deploy
```

## Documentation

| Doc | Purpose |
|-----|---------|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Code map and data flow |
| [docs/METHODS.md](docs/METHODS.md) | Models, evaluation, gates, results |
| [docs/DATA.md](docs/DATA.md) | Data sources and API keys |
| [DEPLOY.md](DEPLOY.md) | Vercel deployment |

## Evaluation principles

- Chronological splits only (never shuffle time)  
- Scalers fit on **train only**  
- Persistence baseline beside every model  
- Walk-forward with purge / embargo  
- Fixed seed **42**  

## Author

**Priyansh Shah** — Stony Brook University, B.S. Applied Mathematics and Statistics  
[priyansh.shah@stonybrook.edu](mailto:priyansh.shah@stonybrook.edu) · [LinkedIn](https://linkedin.com/in/priyansh-shah)

## License

MIT — see [LICENSE](LICENSE). Educational / research use. Paper trading only.
