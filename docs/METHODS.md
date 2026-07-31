# Methods & evaluation

**Last updated:** 2026-07-31

## Why returns, not price R²

Next-day **price** is highly autocorrelated. Predicting “tomorrow’s close ≈ today’s close” (persistence) yields price R² ≈ 0.95. That looks impressive and means almost nothing.

We optimize and report **forward returns** (and return RMSE vs persistence). Directional accuracy is secondary and must beat a coin-flip with gates.

## Models

### Tabular (primary for 1D / 1W / 1M)

Ridge, Lasso, ElasticNet, HistGradientBoosting, LightGBM, XGBoost, Optuna-tuned LightGBM, stacked blend, regime (high/low vol) LightGBM.

### Sequence / classical paths

- **ARIMA** on log-returns, compounded path + bands  
- **LSTM** and **GRU** (train-only MinMax scaling, early stopping)

### Context features

Moving averages, RSI, MACD, Bollinger, ATR, funding, Fear & Greed, FRED macro (incl. VIX), Google Trends, BTC hashrate/tx, ETH TVL/gas/staking proxy, prediction-market crowd Yes.

## Evaluation

| Rule | Detail |
|------|--------|
| Split | Chronological 80/20 (never shuffle) |
| Scaling | Fit on train only |
| Baseline | Persistence (zero predicted return) |
| Walk-forward | Expanding folds with purge = horizon |
| Seed | 42 |

### Trustworthy badge

Beat persistence **and** walk-forward pass ≥ 55% **and** direction gate.

### Walk-forward journal

For each as-of date, train only on rows whose horizon labels are fully resolved (`purge = horizon`), predict, then mark Correct / Incorrect on direction. Recent window: dense day-by-day; older window: coarser step.

Adaptive sample weights near past Incorrect dates apply **only if** a micro-validation RMSE gate passes (avoids memorizing a single crash).

## Results snapshot (2026-07-30)

| Asset | Horizon | Note |
|-------|---------|------|
| BTC | 30d | Best cell: beat persistence, ~64% direction; WF still below gate |
| Short horizons | 1d / 7d | Typically lose to persistence |

Live walk-forward journal hit rate on dense recent folds ≈ **40%** — honest, near coin-flip on hard regimes.

See `results/metrics.json` and `results/prediction_journal.json`.

## Paper trading

Long/flat and long/short books with fee + slippage assumptions. Educational only.
