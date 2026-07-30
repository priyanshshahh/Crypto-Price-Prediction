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
from crypto_pipeline.features import add_features
from crypto_pipeline.models import run_arima, run_lstm, run_regressions
from crypto_pipeline.report import (build_dashboard_json, build_metrics_json,
                                    validate_dashboard_json)

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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=int, default=730)
    ap.add_argument("--force-refresh", action="store_true")
    ap.add_argument("--skip-lstm", action="store_true")
    ap.add_argument("--epochs", type=int, default=40)
    args = ap.parse_args(argv)

    np.random.seed(RANDOM_STATE)
    for d in (RESULTS, VIS_DIR, DATA_DIR, PUBLIC_DATA):
        os.makedirs(d, exist_ok=True)

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
            except ImportError:
                print("      LSTM skipped: TensorFlow not installed")

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
