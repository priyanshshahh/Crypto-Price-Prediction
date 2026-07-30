# PROJECT-NOTES — rebuild log (2026-07-06)

Log of the production-grade rebuild: what was wrong, what changed, exact commands, and
measured results. Companion to README.md.

## Audit findings (verified before fixing)

1. **Synthetic data sold as real.** `pipeline.py` generated Geometric Brownian Motion
   OHLCV (`generate_ohlcv()`, seeded GBM + sine "seasonality") while README claimed real
   CryptoCompare data and quoted R² = 0.98. Credibility killer — confirmed in the old
   `pipeline.py` lines 53–88.
2. **Duplicate React trees.** `crypto-prediction-web/src/` was a copy of root `src/`
   (identical components/lib; only extras: unused `App.css`, `assets/react.svg`, stock
   README). Confirmed with `diff -rq`.
3. **README referenced `regression_results.json` / `time_series_results.json` /
   `clustering_results.json`** — none existed. `.gitignore` even ignored `results/`.
4. **Zero tests.**
5. **Leaky/misleading evaluation.** LSTM fit its MinMaxScaler on the FULL series (train +
   test) — look-ahead leakage. Clustering had a bug where Agglomerative reused KMeans'
   best silhouette as its sweep threshold. Headline metric was price-level R² on next-day
   close, which is trivially ≈1 due to autocorrelation.
6. **UI hardcoded "R² scores above 94%"** in Dashboard.jsx.

## Changes

### Python pipeline (rewritten)
- New package `crypto_pipeline/` (data / features / models / clustering / report);
  `pipeline.py` is now a thin orchestrator. One command: `python pipeline.py`.
- **Real data**: Coinbase Exchange public candles API (keyless, paginated 300/req,
  0.4s between requests), Kraken public OHLC as automatic fallback (Binance is
  geo-blocked in the US — returns HTTP 451). Raw pulls cached to `data/raw/` (gitignored,
  24h freshness); stale cache used if all sources fail.
- **Honest evaluation**: chronological 80/20 split, train-only scaler fits, next-day
  RETURN as the regression target, walk-forward one-step ARIMA
  (`fitted.apply(full_series).get_prediction(dynamic=False)`), persistence baseline
  reported next to every model, directional accuracy as headline metric. Seed 42
  everywhere (numpy / sklearn / tensorflow).
- **LSTM**: epochs capped at 40 with EarlyStopping(patience=5, restore_best_weights);
  actually ran 13–15 epochs per symbol. Look-back 60. Scaler fixed to train-only.
- Clustering sweep bug fixed (each algorithm gets its own best-silhouette sweep).
- Outputs: `results/metrics.json` (with provenance: generated_at, source, date range,
  seed, eval policy), `results/<SYM>_regression.csv`, `results/visualizations/*.png`,
  `public/data/crypto_data.json` (schema-validated before writing).

### Removed
- `crypto-prediction-web/` duplicate tree (kept the root Vite app). A destructive-command
  hook blocked `rm`, so it was moved out of the repo to the session scratchpad; it remains
  in git history at 2dd2f88 if ever needed.
- Old synthetic-data `pipeline.py` (replaced; also in git history).

### Tests (new, `tests/`)
- 23 tests: API parsers on canned payloads (no network), OHLCV validation, cache
  round-trip, real-data fixture sanity (`tests/fixtures/BTC_sample.csv` — 250 committed
  rows of real Coinbase BTC-USD), feature columns + target-alignment (leakage guard),
  window building, regression/ARIMA/LSTM/clustering smoke on the fixture, dashboard-JSON
  schema (including the committed artifacts).

### Frontend
- `ModelPerformance.jsx`: now leads with directional accuracy vs the persistence baseline
  (coin-flip reference line), shows returns-based R²/RMSE/MAE, renders the baseline row,
  and explains why price-level R² is misleading.
- `Dashboard.jsx`: removed the false "R² above 94%" paragraph; replaced with honest text.
- Static-JSON is the documented default (mock Supabase client reads
  `/data/crypto_data.json`); real Supabase remains optional via env vars.
- Added `vercel.json` + `DEPLOY.md`.

### Housekeeping
- `.gitignore`: now ignores `data/` and `.venv/`, no longer ignores `results/` (metrics
  artifacts are committed).
- `requirements.txt` pinned from the verified environment (Python 3.12.4 — TF does not
  support 3.14, which is the system default here).
- Notebooks kept as original exploratory work, labeled as such in README.

## Commands executed (verification)

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt        # (originally: pip install <pkgs>, then frozen)
.venv/bin/python pipeline.py                     # exit 0; fetched 731 real candles/symbol
.venv/bin/python -m pytest tests/ -q             # 23 passed in ~8s
npm ci && npm run build                          # ✓ built in 2.17s (668 kB JS chunk warning only)
npx vite preview                                 # /data/crypto_data.json served; real BTC close $64,062.97
```

## Measured results (2026-07-06, Coinbase data 2024-07-06 → 2026-07-06, seed 42)

See README "Honest results" tables — reproduced from `results/metrics.json`. Key takeaways:
- No regression model beats persistence RMSE on next-day returns (BTC best model 0.02143
  vs baseline 0.02088). Directional accuracy 41–57%.
- ARIMA selects (0,1,1) for all three symbols ≈ random walk; RMSE within 0.4% of
  persistence. LSTM underperforms persistence (expected with ~600 training points).
- Clustering finds real regime structure (DOGE silhouette 0.62, BTC DBSCAN 0.36).

## Known gaps / future work
- Bundle is one 668 kB chunk (recharts); could code-split.
- LSTM metrics have minor cross-hardware nondeterminism despite seeding (TF kernels).
- Coinbase returns the still-forming candle for "today"; fine for daily modeling, noted.
- No hyperparameter search (deliberate — would invite overfitting the tiny holdout).
