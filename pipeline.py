"""
End-to-end Cryptocurrency Price Prediction Pipeline (REAL data)
================================================================
One command: `python pipeline.py`

Steps:
  1. Fetch real daily OHLCV for BTC / ETH / DOGE from free keyless APIs
     (Coinbase Exchange, Kraken fallback); cached in data/raw/ (gitignored).
  2. Feature engineering (returns, MAs, MACD, Bollinger, RSI, volatility).
  3. Regression models (Ridge, Lasso, ElasticNet, SVR, RF, GBR) predicting
     NEXT-DAY RETURNS with a chronological 80/20 holdout.
  4. Time-series models: ARIMA (walk-forward one-step-ahead) and LSTM
     (train-only scaling, capped epochs + early stopping).
  5. Market-regime clustering (KMeans, Agglomerative, GMM, DBSCAN).
  6. Save honest metrics (vs persistence baseline) + provenance to results/,
     and dashboard data to public/data/crypto_data.json.

Flags:
  --days N        history length to fetch (default 730)
  --force-refresh ignore the local data cache
  --skip-lstm     skip the LSTM (e.g. if TensorFlow isn't installed)
  --epochs N      LSTM epoch cap (default 40, EarlyStopping patience=5)
  --mode MODE     classic | production (multi-horizon + external + paper)
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from crypto_pipeline import RANDOM_STATE, SYMBOLS
from crypto_pipeline.clustering import run_clustering
from crypto_pipeline.data import fetch_ohlcv
from crypto_pipeline.external import load_external_frame
from crypto_pipeline.features import add_features
from crypto_pipeline.horizons import live_price_forecasts, run_all_horizons
from crypto_pipeline.models import run_arima, run_gru, run_lstm, run_regressions
from crypto_pipeline.report import (build_dashboard_json, build_metrics_json,
                                    validate_dashboard_json)

try:
    from dotenv import load_dotenv
    _root = os.path.dirname(os.path.abspath(__file__))
    load_dotenv(os.path.join(_root, ".env.local"))
    load_dotenv(os.path.join(_root, ".env"))
except ImportError:
    pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE_DIR, "results")
VIS_DIR = os.path.join(RESULTS, "visualizations")
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PUBLIC_DATA = os.path.join(BASE_DIR, "public", "data")


def plot_ts_eval(sym: str, close: pd.Series, res: dict, model_name: str) -> None:
    split = len(close) - len(res["actual_price"])
    plt.figure(figsize=(10, 4))
    plt.plot(close.index[max(0, split - 90):split], close.iloc[max(0, split - 90):split],
             label="Train (last 90d)", color="#6b7280")
    plt.plot(close.index[split:], res["actual_price"], label="Actual", color="#111827")
    plt.plot(close.index[split:], res["pred_price"], label=f"{model_name} 1-step-ahead",
             color="#ef4444", alpha=0.8)
    plt.legend()
    plt.title(f"{sym} {model_name} — one-step-ahead holdout predictions")
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, f"{sym}_{model_name.lower()}_eval.png"), dpi=110)
    plt.close()


def plot_clustering(sym: str, clus: dict) -> None:
    coords, labels = clus["pca_coords"], clus["kmeans_labels"]
    plt.figure(figsize=(7, 5))
    sc = plt.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", s=18, alpha=0.6)
    plt.colorbar(sc, label="KMeans regime")
    plt.title(f"{sym} market regimes (PCA projection)")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, f"{sym}_kmeans_pca.png"), dpi=110)
    plt.close()


SLIPPAGE = {"BTC": 10.0, "ETH": 10.0, "DOGE": 20.0}


def _json_safe(obj):
    """Drop large arrays for summary JSON."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()
                if k not in ("best_pred", "actual", "equity", "test_index")}
    if isinstance(obj, list):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def run_production(args) -> int:
    """Multi-horizon price prediction + free external features + paper modes."""
    print("=" * 64)
    print("  Production mode — multi-horizon price prediction")
    print("  Horizons: 1d / 7d / 30d (day · week · month) · paper: long/flat + long/short")
    print("=" * 64)

    frames, metas, featured = {}, {}, {}
    horizon_all, clustering_all = {}, {}
    btc_returns = None

    # Fetch all spot first (for BTC lead-lag on ETH/DOGE)
    for sym, name in SYMBOLS.items():
        print(f"\n── Fetch {name} ({sym}) " + "─" * 36)
        df, meta = fetch_ohlcv(sym, days=args.days, cache_dir=DATA_DIR,
                               force=args.force_refresh)
        frames[sym], metas[sym] = df, meta
        print(f"      {meta['rows']} candles · {meta['source']} · "
              f"{meta['start']} → {meta['end']}")
        if sym == "BTC":
            btc_returns = df["Close"].pct_change(1).rename("BTC_Return_1d")

    externals = {}
    for sym, name in SYMBOLS.items():
        print(f"\n── Features {name} ({sym}) " + "─" * 36)
        print("[1/4] External features (Fear&Greed, funding, FRED, Polymarket, ...) ...")
        ext = load_external_frame(sym, cache_dir=DATA_DIR, force=args.force_refresh)
        externals[sym] = ext
        if len(ext):
            print(f"      external cols: {list(ext.columns)}")
        else:
            print("      no external series loaded (continuing with spot+tech)")

        cross = None
        if sym != "BTC" and btc_returns is not None:
            cross = btc_returns.to_frame()

        feat = add_features(frames[sym], horizons=(1, 7, 30), external=ext, cross_asset=cross)
        featured[sym] = feat
        print(f"      {len(feat)} rows · {feat.shape[1]} cols after features")

    # Dense walk-forward ML journal (replaces MA seed) → wrong-cluster + adaptive plan
    adaptive_plan = None
    wrong_clusters = {}
    journal_path = os.path.join(RESULTS, "prediction_journal.json")
    if not getattr(args, "skip_wf_ledger", False):
        print("\n[1b] Walk-forward ML skill ledger (purge=horizon, no look-ahead) ...")
        from crypto_pipeline.walkforward_ledger import (
            adaptive_weight_plan, analyze_wrong_clusters, build_walkforward_ledger,
        )
        from crypto_pipeline.journal import replace_with_walkforward_ledger
        wf_rows = build_walkforward_ledger(
            featured, horizons=(1, 7, 30),
            dense_days=60, dense_step=1,
            archive_days=120, archive_step=5,
        )
        journal = replace_with_walkforward_ledger(wf_rows, path=journal_path)
        wrong_clusters = analyze_wrong_clusters(journal.get("resolved") or [])
        adaptive_plan = adaptive_weight_plan(
            journal.get("resolved") or [], wrong_clusters,
        )
        journal["wrong_clusters"] = wrong_clusters
        journal["adaptive_plan"] = {
            "eligible_cells": adaptive_plan.get("eligible_cells"),
            "gate": adaptive_plan.get("gate"),
            "boost": adaptive_plan.get("boost"),
        }
        with open(journal_path, "w") as f:
            json.dump(journal, f, indent=2)
        print(f"      ledger hit_rate={journal['summary'].get('hit_rate')} "
              f"n={journal['summary'].get('n_resolved')} "
              f"retrain_focus={wrong_clusters.get('retrain_focus')}")
    else:
        print("\n[1b] Walk-forward ledger skipped (--skip-wf-ledger)")

    for sym, name in SYMBOLS.items():
        print(f"\n── Model {name} ({sym}) " + "─" * 36)
        feat = featured[sym]
        ext = externals[sym]
        cross = None
        if sym != "BTC" and btc_returns is not None:
            cross = btc_returns.to_frame()

        print("[2/4] Multi-horizon models + paper book (gated adaptive weights) ...")
        hz = run_all_horizons(
            feat,
            horizons=(1, 7, 30),
            symbol=sym,
            fee_bps=50.0,
            slippage_bps=SLIPPAGE[sym],
            adaptive_plan=adaptive_plan,
        )
        horizon_all[sym] = hz
        for row in hz["summary"]:
            adapt = "adapt" if row.get("adaptive_weights_applied") else "base"
            print(
                f"      H={row['horizon']:>2}d  {row['best_model']:<20s} "
                f"rmse={row['ret_rmse']:.5f} (pers {row['persistence_rmse']:.5f}) "
                f"beat={row['beats_persistence']}  dir={row['directional_accuracy']:.3f}  "
                f"paper={row['paper_best_mode']} sharpe={row['paper_net_sharpe']:+.2f}  "
                f"wf_pass={row['wf_pass_frac']:.0%}  w={adapt}"
            )

        live = live_price_forecasts(
            frames[sym], feat, hz["by_horizon"],
            symbol=sym, external=ext, cross_asset=cross, horizons=(1, 7, 30),
        )
        horizon_all[sym]["live_forecasts"] = live
        print("[2b] Live price forecasts ...")
        for r in live:
            trust = "TRUST" if r["trustworthy"] else "low-confidence"
            print(
                f"      {r['horizon_label']}: ${r['current_price']:,.2f} → "
                f"${r['predicted_price']:,.2f} ({r['predicted_return_pct']:+.2f}%) "
                f"[{r['model']}] {trust}"
            )

        print("[3/4] Regime clustering ...")
        clus = run_clustering(feat)
        clustering_all[sym] = clus
        plot_clustering(sym, clus)

    # Also keep classic 1d regressions for dashboard compatibility
    regression_all, ts_all = {}, {}
    for sym in SYMBOLS:
        feat1 = featured[sym]
        reg = run_regressions(feat1)
        regression_all[sym] = reg
        close = frames[sym]["Close"]
        arima = run_arima(close)
        ts_all[sym] = {"ARIMA": arima}
        if not args.skip_lstm:
            try:
                ts_all[sym]["LSTM"] = run_lstm(close, epochs=args.epochs)
                ts_all[sym]["GRU"] = run_gru(close, epochs=args.epochs)
            except ImportError:
                pass

    metrics = build_metrics_json(SYMBOLS, metas, regression_all, ts_all,
                                 clustering_all, seed=RANDOM_STATE,
                                 lstm_config={})
    metrics["production"] = {
        "config": "configs/price_v1.yaml",
        "horizons": _json_safe({s: horizon_all[s]["summary"] for s in SYMBOLS}),
        "costs": {"fee_bps_round_trip": 50, "slippage_bps": SLIPPAGE},
        "paper_note": "Both long_flat and long_short evaluated; best by net Sharpe",
    }
    # Full horizon detail (without huge equity curves)
    with open(os.path.join(RESULTS, "horizons.json"), "w") as f:
        json.dump(_json_safe({s: horizon_all[s] for s in SYMBOLS}), f, indent=2)
    with open(os.path.join(RESULTS, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    hz_rows = []
    for sym in SYMBOLS:
        for row in horizon_all[sym].get("live_forecasts", []):
            hz_rows.append({**row, "symbol": sym})

    # Merge live forecasts into journal pending; keep WF-ML resolved ledger
    from crypto_pipeline.journal import update_journal
    live_map = {s: horizon_all[s].get("live_forecasts", []) for s in SYMBOLS}
    journal = update_journal(frames, live_map, path=os.path.join(RESULTS, "prediction_journal.json"))
    if wrong_clusters:
        journal["wrong_clusters"] = wrong_clusters
    if adaptive_plan:
        journal["adaptive_plan"] = {
            "eligible_cells": adaptive_plan.get("eligible_cells"),
            "gate": adaptive_plan.get("gate"),
            "boost": adaptive_plan.get("boost"),
        }
        with open(os.path.join(RESULTS, "prediction_journal.json"), "w") as f:
            json.dump(journal, f, indent=2)
    metrics["production"]["journal"] = journal.get("summary", {})
    metrics["production"]["wrong_clusters"] = wrong_clusters
    metrics["production"]["adaptive_plan"] = journal.get("adaptive_plan") or {}

    dashboard = build_dashboard_json(
        SYMBOLS, frames, regression_all, ts_all, clustering_all,
        horizon_forecasts=hz_rows,
    )
    dashboard["production_summary"] = metrics["production"]["horizons"]
    dashboard["prediction_journal"] = journal
    dashboard["prediction_journal_resolved"] = journal.get("resolved") or []
    dashboard["prediction_journal_pending"] = journal.get("pending") or []
    dashboard["prediction_journal_summary"] = [journal.get("summary") or {}]
    with open(os.path.join(RESULTS, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    validate_dashboard_json(dashboard)
    with open(os.path.join(PUBLIC_DATA, "crypto_data.json"), "w") as f:
        json.dump(dashboard, f, indent=2)

    print("\n" + "=" * 64)
    print(f"  metrics    -> {os.path.join(RESULTS, 'metrics.json')}")
    print(f"  horizons   -> {os.path.join(RESULTS, 'horizons.json')}")
    print(f"  dashboard  -> {os.path.join(PUBLIC_DATA, 'crypto_data.json')}")
    print("  Production pipeline complete.")
    print("=" * 64)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=int, default=730)
    ap.add_argument("--force-refresh", action="store_true")
    ap.add_argument("--skip-lstm", action="store_true")
    ap.add_argument("--skip-wf-ledger", action="store_true",
                    help="Skip dense walk-forward ML journal (faster)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--mode", choices=("classic", "production"), default="classic")
    args = ap.parse_args(argv)

    np.random.seed(RANDOM_STATE)
    for d in (RESULTS, VIS_DIR, DATA_DIR, PUBLIC_DATA):
        os.makedirs(d, exist_ok=True)

    if args.mode == "production":
        if args.days == 730:
            args.days = 1460  # need longer history for 30d targets
        return run_production(args)

    print("=" * 64)
    print("  Crypto Price Prediction — end-to-end pipeline (real data)")
    print("=" * 64)

    frames, metas, featured = {}, {}, {}
    regression_all, ts_all, clustering_all = {}, {}, {}

    for sym, name in SYMBOLS.items():
        print(f"\n── {name} ({sym}) " + "─" * 40)

        print("[1/4] Fetching real OHLCV data ...")
        df, meta = fetch_ohlcv(sym, days=args.days, cache_dir=DATA_DIR,
                               force=args.force_refresh)
        frames[sym], metas[sym] = df, meta
        print(f"      {meta['rows']} daily candles from {meta['source']} "
              f"({meta['start']} → {meta['end']})")

        feat = add_features(df)
        featured[sym] = feat
        print(f"      {len(feat)} rows after feature engineering ({feat.shape[1]} cols)")

        print("[2/4] Regression models (target: next-day return) ...")
        reg = run_regressions(feat)
        regression_all[sym] = reg
        base = reg["baseline_persistence"]
        for m in reg["models"]:
            print(f"      {m['name']:<17s} dir.acc={m['directional_accuracy']:.3f}  "
                  f"ret.RMSE={m['returns']['rmse']:.5f}  ret.R2={m['returns']['r2']:+.4f}")
        print(f"      {'Persistence base':<17s} up-day share={base['up_day_share']:.3f}  "
              f"ret.RMSE={base['returns']['rmse']:.5f}")

        print("[3/4] Time-series models ...")
        close = df["Close"]
        arima = run_arima(close)
        plot_ts_eval(sym, close, arima, "ARIMA")
        print(f"      ARIMA{tuple(arima['order'])} price.RMSE={arima['price']['rmse']:.4f} "
              f"(persistence {arima['baseline_persistence']['price']['rmse']:.4f})  "
              f"dir.acc={arima['directional_accuracy']:.3f}")
        ts_all[sym] = {"ARIMA": arima}

        if not args.skip_lstm:
            try:
                lstm = run_lstm(close, epochs=args.epochs)
                plot_ts_eval(sym, close, lstm, "LSTM")
                print(f"      LSTM ({lstm['epochs_ran']} epochs) "
                      f"price.RMSE={lstm['price']['rmse']:.4f} "
                      f"(persistence {lstm['baseline_persistence']['price']['rmse']:.4f})  "
                      f"dir.acc={lstm['directional_accuracy']:.3f}")
                ts_all[sym]["LSTM"] = lstm
                gru = run_gru(close, epochs=args.epochs)
                plot_ts_eval(sym, close, gru, "GRU")
                print(f"      GRU ({gru['epochs_ran']} epochs) "
                      f"price.RMSE={gru['price']['rmse']:.4f} "
                      f"(persistence {gru['baseline_persistence']['price']['rmse']:.4f})  "
                      f"dir.acc={gru['directional_accuracy']:.3f}")
                ts_all[sym]["GRU"] = gru
            except ImportError:
                print("      LSTM/GRU skipped: TensorFlow not installed")

        print("[4/4] Market-regime clustering ...")
        clus = run_clustering(feat)
        clustering_all[sym] = clus
        plot_clustering(sym, clus)
        for r in clus["results"]:
            print(f"      {r['algorithm']:<14s} k={r['optimal_clusters']}  "
                  f"silhouette={r['silhouette_score']:.4f}")

    # ── Persist results ──────────────────────────────────────────────────────
    lstm_cfg = {"look_back": 60, "epoch_cap": args.epochs,
                "early_stopping_patience": 5} if not args.skip_lstm else {}
    metrics = build_metrics_json(SYMBOLS, metas, regression_all, ts_all,
                                 clustering_all, seed=RANDOM_STATE, lstm_config=lstm_cfg)
    with open(os.path.join(RESULTS, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    dashboard = build_dashboard_json(SYMBOLS, frames, regression_all, ts_all, clustering_all)
    validate_dashboard_json(dashboard)
    with open(os.path.join(PUBLIC_DATA, "crypto_data.json"), "w") as f:
        json.dump(dashboard, f, indent=2)

    # per-symbol CSV summaries
    for sym in SYMBOLS:
        rows = [{"model": m["name"], "directional_accuracy": m["directional_accuracy"],
                 "returns_rmse": m["returns"]["rmse"], "returns_r2": m["returns"]["r2"],
                 "price_rmse": m["price"]["rmse"]}
                for m in regression_all[sym]["models"]]
        pd.DataFrame(rows).to_csv(os.path.join(RESULTS, f"{sym}_regression.csv"), index=False)

    print("\n" + "=" * 64)
    print(f"  metrics       -> {os.path.join(RESULTS, 'metrics.json')}")
    print(f"  dashboard     -> {os.path.join(PUBLIC_DATA, 'crypto_data.json')}")
    print(f"  visualizations-> {VIS_DIR}")
    print("  Pipeline complete.")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
