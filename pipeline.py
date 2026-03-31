"""
End-to-end Cryptocurrency Price Prediction Pipeline
=====================================================
Generates synthetic OHLCV data, engineers features, and runs:
  - Regression models  (Ridge, Lasso, ElasticNet, SVR, RF, GBR)
  - Time-series models (ARIMA, LSTM)
  - Clustering analysis (KMeans, DBSCAN, Agglomerative, GMM)

Results are saved to:
  - results/crypto_data.json   (dashboard-ready)
  - results/regression/        (per-symbol CSV metrics)
  - results/time_series/       (forecasts CSV)
  - results/clustering/        (cluster metrics CSV)
  - results/visualizations/    (PNG charts)
"""

import os
import json
import warnings
import uuid
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ── Directories ────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
RESULTS   = os.path.join(BASE_DIR, "results")
REG_DIR   = os.path.join(RESULTS, "regression")
TS_DIR    = os.path.join(RESULTS, "time_series")
CLUS_DIR  = os.path.join(RESULTS, "clustering")
VIS_DIR   = os.path.join(RESULTS, "visualizations")
DATA_DIR  = os.path.join(BASE_DIR, "public", "data")

for d in (RESULTS, REG_DIR, TS_DIR, CLUS_DIR, VIS_DIR, DATA_DIR):
    os.makedirs(d, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

CRYPTOS = {
    "BTC":  {"name": "Bitcoin",  "start_price": 28000.0, "volatility": 0.025},
    "ETH":  {"name": "Ethereum", "start_price": 1800.0,  "volatility": 0.030},
    "DOGE": {"name": "Dogecoin", "start_price": 0.07,    "volatility": 0.055},
}

# ── 1. Synthetic data generation ───────────────────────────────────────────────

def generate_ohlcv(symbol: str, n_days: int = 730) -> pd.DataFrame:
    """Geometric Brownian Motion OHLCV with realistic intra-day spreads."""
    cfg  = CRYPTOS[symbol]
    mu   = 0.0003          # slight upward drift
    sig  = cfg["volatility"]
    p0   = cfg["start_price"]

    dates   = pd.date_range(end=datetime.today().date(), periods=n_days, freq="D")
    returns = np.random.normal(mu, sig, n_days)
    closes  = p0 * np.exp(np.cumsum(returns))

    # Add cyclical + seasonal signals
    t      = np.arange(n_days)
    trend  = 0.15 * np.sin(2 * np.pi * t / 365)
    closes = closes * (1 + trend)
    closes = np.maximum(closes, p0 * 0.01)

    highs   = closes * (1 + np.abs(np.random.normal(0, 0.01, n_days)))
    lows    = closes * (1 - np.abs(np.random.normal(0, 0.01, n_days)))
    opens   = np.roll(closes, 1)
    opens[0] = closes[0]
    base_vol = cfg["start_price"] * 1e6 if symbol == "BTC" else cfg["start_price"] * 5e7
    volumes  = np.abs(np.random.normal(base_vol, base_vol * 0.3, n_days))

    df = pd.DataFrame({
        "Date":   dates,
        "Open":   opens,
        "High":   highs,
        "Low":    lows,
        "Close":  closes,
        "Volume": volumes,
    })
    df.set_index("Date", inplace=True)
    return df


# ── 2. Feature engineering ────────────────────────────────────────────────────

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators used by the regression models."""
    d = df.copy()

    # Returns
    d["Return_1d"]  = d["Close"].pct_change(1)
    d["Return_7d"]  = d["Close"].pct_change(7)
    d["Return_30d"] = d["Close"].pct_change(30)

    # Moving averages
    d["MA7"]   = d["Close"].rolling(7).mean()
    d["MA30"]  = d["Close"].rolling(30).mean()
    d["MA90"]  = d["Close"].rolling(90).mean()
    d["EMA12"] = d["Close"].ewm(span=12, adjust=False).mean()
    d["EMA26"] = d["Close"].ewm(span=26, adjust=False).mean()
    d["MACD"]  = d["EMA12"] - d["EMA26"]

    # Bollinger bands
    d["BB_mid"]   = d["Close"].rolling(20).mean()
    d["BB_upper"] = d["BB_mid"] + 2 * d["Close"].rolling(20).std()
    d["BB_lower"] = d["BB_mid"] - 2 * d["Close"].rolling(20).std()
    d["BB_width"] = d["BB_upper"] - d["BB_lower"]

    # Volatility
    d["Volatility"] = d["Return_1d"].rolling(30).std()

    # RSI
    delta    = d["Close"].diff()
    gain     = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss     = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs       = gain / loss.replace(0, np.nan)
    d["RSI"] = 100 - 100 / (1 + rs)

    # Volume
    d["Volume_MA7"] = d["Volume"].rolling(7).mean()

    # Target
    d["Target_Next_Day"]   = d["Close"].shift(-1)
    d["Target_Next_Week"]  = d["Close"].shift(-7)
    d["Target_Next_Month"] = d["Close"].shift(-30)

    d.dropna(inplace=True)
    return d


# ── 3. Regression models ──────────────────────────────────────────────────────

def run_regression(symbol: str, df: pd.DataFrame) -> list[dict]:
    """Train multiple regressors and return list of model-metric dicts."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.linear_model import Ridge, Lasso, ElasticNet
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.svm import SVR
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    drop_cols = [c for c in df.columns if c.startswith("Target_")]
    X = df.drop(columns=drop_cols, errors="ignore").select_dtypes(include=[np.number])
    y = df["Target_Next_Day"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, shuffle=False)

    scaler = StandardScaler()

    MODELS = {
        "Ridge":      Ridge(alpha=1.0),
        "Lasso":      Lasso(alpha=0.01, max_iter=10000),
        "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000),
        "SVR":        SVR(C=1.0, epsilon=0.1),
        "RF":         RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE),
        "GBR":        GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_STATE),
    }

    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)

    results = []
    for name, model in MODELS.items():
        model.fit(X_tr_sc, y_tr)
        y_pred = model.predict(X_te_sc)
        r2   = float(r2_score(y_te, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_te, y_pred)))
        mae  = float(mean_absolute_error(y_te, y_pred))
        results.append({"name": name, "r2": r2, "rmse": rmse, "mae": mae})
        print(f"  {symbol} {name:12s}  R²={r2:.4f}  RMSE={rmse:.2f}  MAE={mae:.2f}")

    # Save metrics CSV
    pd.DataFrame(results).to_csv(os.path.join(REG_DIR, f"{symbol}_metrics.csv"), index=False)

    # Plot R² comparison
    df_plot = pd.DataFrame(results).sort_values("r2")
    ax = df_plot.plot.barh(x="name", y="r2", legend=False,
                           title=f"{symbol} Regression R² Scores",
                           figsize=(8, 4), color="#3B82F6")
    ax.set_xlabel("R²")
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, f"{symbol}_regression_r2.png"))
    plt.close()

    return results


# ── 4. ARIMA ──────────────────────────────────────────────────────────────────

def run_arima(symbol: str, ts: pd.Series, steps: int = 30) -> dict:
    """Fit ARIMA on 80 % of ts, evaluate on hold-out, forecast `steps` days."""
    from statsmodels.tsa.arima.model import ARIMA
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    split  = int(len(ts) * 0.8)
    train  = ts.iloc[:split]
    test   = ts.iloc[split:]

    # Try several orders, pick the best AIC
    best_aic, best_order, best_model = np.inf, (1, 1, 1), None
    for p in (1, 2):
        for d in (0, 1):
            for q in (1, 2):
                try:
                    m = ARIMA(train, order=(p, d, q)).fit()
                    if m.aic < best_aic:
                        best_aic, best_order, best_model = m.aic, (p, d, q), m
                except Exception:
                    pass

    if best_model is None:
        best_model = ARIMA(train, order=(1, 1, 1)).fit()
        best_order = (1, 1, 1)

    preds = best_model.forecast(steps=len(test))
    r2   = float(r2_score(test.values, preds.values))
    rmse = float(np.sqrt(mean_squared_error(test.values, preds.values)))
    mae  = float(mean_absolute_error(test.values, preds.values))

    # Future forecast
    fc_vals = best_model.forecast(steps=steps)
    last_date = ts.index[-1]
    fc_dates  = pd.date_range(start=last_date + timedelta(days=1),
                              periods=steps, freq="D")
    forecasts = [(d.strftime("%Y-%m-%d"), float(v))
                 for d, v in zip(fc_dates, fc_vals)]

    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(train, label="Train")
    plt.plot(test, label="Test")
    plt.plot(preds.index if hasattr(preds, "index") else test.index,
             preds.values, label=f"ARIMA{best_order}", color="red")
    plt.legend()
    plt.title(f"{symbol} ARIMA Fit")
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, f"{symbol}_arima_fit.png"))
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(ts.iloc[-90:], label="Last 90 days")
    plt.plot(fc_dates, fc_vals, label="ARIMA Forecast", linestyle="--", color="red")
    plt.legend()
    plt.title(f"{symbol} ARIMA 30-day Forecast")
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, f"{symbol}_arima_forecast.png"))
    plt.close()

    print(f"  {symbol} ARIMA{best_order}  R²={r2:.4f}  RMSE={rmse:.2f}  MAE={mae:.2f}")
    return {"order": best_order, "r2": r2, "rmse": rmse, "mae": mae, "forecasts": forecasts}


# ── 5. LSTM ───────────────────────────────────────────────────────────────────

def run_lstm(symbol: str, ts: pd.Series, steps: int = 30) -> dict:
    """LSTM with look_back=60. Falls back to ARIMA-style random walk if TF absent."""
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        tf.get_logger().setLevel("ERROR")
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

        look_back = 60
        arr = ts.values.reshape(-1, 1)
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(arr)

        X_all, y_all = [], []
        for i in range(look_back, len(scaled)):
            X_all.append(scaled[i - look_back:i, 0])
            y_all.append(scaled[i, 0])
        X_all = np.array(X_all).reshape(-1, look_back, 1)
        y_all = np.array(y_all)

        split   = int(len(X_all) * 0.8)
        X_tr, X_te = X_all[:split], X_all[split:]
        y_tr, y_te = y_all[:split], y_all[split:]

        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(look_back, 1)),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(1),
        ])
        model.compile(optimizer="adam", loss="mse")
        model.fit(X_tr, y_tr, validation_data=(X_te, y_te),
                  epochs=30, batch_size=32, verbose=0)

        y_pred    = model.predict(X_te, verbose=0)
        y_te_inv  = scaler.inverse_transform(y_te.reshape(-1, 1)).flatten()
        y_pred_inv = scaler.inverse_transform(y_pred).flatten()

        r2   = float(r2_score(y_te_inv, y_pred_inv))
        rmse = float(np.sqrt(mean_squared_error(y_te_inv, y_pred_inv)))
        mae  = float(mean_absolute_error(y_te_inv, y_pred_inv))

        # Auto-regressive future forecast
        last_seq = X_te[-1].flatten().copy()
        preds_sc = []
        for _ in range(steps):
            p = model.predict(last_seq.reshape(1, look_back, 1), verbose=0)[0, 0]
            preds_sc.append(p)
            last_seq = np.roll(last_seq, -1)
            last_seq[-1] = p
        fc_vals = scaler.inverse_transform(
            np.array(preds_sc).reshape(-1, 1)
        ).flatten()

        print(f"  {symbol} LSTM          R²={r2:.4f}  RMSE={rmse:.2f}  MAE={mae:.2f}")

    except ImportError:
        print(f"  {symbol} LSTM: TensorFlow not installed — using statistical fallback")
        last_val = float(ts.iloc[-1])
        drift    = float(ts.pct_change().dropna().mean())
        sigma    = float(ts.pct_change().dropna().std())

        preds_vals = []
        cur = last_val
        for _ in range(steps):
            cur = cur * (1 + np.random.normal(drift, sigma))
            preds_vals.append(cur)
        fc_vals = np.array(preds_vals)

        # Evaluate on last 20% as hold-out
        split = int(len(ts) * 0.8)
        y_te_inv = ts.iloc[split:].values
        y_pred_inv = np.array([ts.iloc[split - 1] * (1 + drift) ** (i + 1)
                                for i in range(len(y_te_inv))])
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        r2   = float(r2_score(y_te_inv, y_pred_inv))
        rmse = float(np.sqrt(mean_squared_error(y_te_inv, y_pred_inv)))
        mae  = float(mean_absolute_error(y_te_inv, y_pred_inv))

    # Build forecast series
    last_date = ts.index[-1]
    fc_dates  = pd.date_range(start=last_date + timedelta(days=1),
                              periods=steps, freq="D")
    forecasts = [(d.strftime("%Y-%m-%d"), float(v))
                 for d, v in zip(fc_dates, fc_vals)]

    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(ts.iloc[-90:], label="Last 90 days")
    plt.plot(fc_dates, fc_vals, label="LSTM Forecast", linestyle="--", color="green")
    plt.legend()
    plt.title(f"{symbol} LSTM 30-day Forecast")
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, f"{symbol}_lstm_forecast.png"))
    plt.close()

    return {"r2": r2, "rmse": rmse, "mae": mae, "forecasts": forecasts}


# ── 6. Clustering ─────────────────────────────────────────────────────────────

def run_clustering(symbol: str, df: pd.DataFrame) -> list[dict]:
    """KMeans, DBSCAN, Agglomerative, GMM. Returns list of algorithm metrics."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.mixture import GaussianMixture
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score

    features = ["Return_1d", "Volatility", "RSI", "MACD", "Volume_MA7"]
    X = df[features].dropna()
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    results = []

    # ── KMeans: try k=2..6, pick best silhouette
    best_k, best_sil = 4, -1
    for k in range(2, 7):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_sc)
        sil = silhouette_score(X_sc, labels)
        if sil > best_sil:
            best_sil, best_k = sil, k
    km = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
    km_labels = km.fit_predict(X_sc)
    results.append({"algorithm": "KMeans", "optimal_clusters": best_k,
                    "silhouette_score": float(best_sil)})

    # ── Agglomerative
    for k in range(2, 7):
        ag = AgglomerativeClustering(n_clusters=k)
        labels = ag.fit_predict(X_sc)
        sil = silhouette_score(X_sc, labels)
        if sil > best_sil:
            best_sil, best_k = sil, k
    ag = AgglomerativeClustering(n_clusters=best_k)
    ag_labels = ag.fit_predict(X_sc)
    results.append({"algorithm": "Agglomerative", "optimal_clusters": best_k,
                    "silhouette_score": float(silhouette_score(X_sc, ag_labels))})

    # ── GMM
    best_k_gmm, best_sil_gmm = 4, -1
    for k in range(2, 7):
        gm = GaussianMixture(n_components=k, random_state=RANDOM_STATE)
        labels = gm.fit_predict(X_sc)
        sil = silhouette_score(X_sc, labels)
        if sil > best_sil_gmm:
            best_sil_gmm, best_k_gmm = sil, k
    gm = GaussianMixture(n_components=best_k_gmm, random_state=RANDOM_STATE)
    gm_labels = gm.fit_predict(X_sc)
    results.append({"algorithm": "GMM", "optimal_clusters": best_k_gmm,
                    "silhouette_score": float(best_sil_gmm)})

    # ── DBSCAN (eps auto-tuned)
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=5).fit(X_sc)
    distances, _ = nbrs.kneighbors(X_sc)
    eps = float(np.percentile(distances[:, -1], 90))
    db = DBSCAN(eps=eps, min_samples=5)
    db_labels = db.fit_predict(X_sc)
    n_clusters_db = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    if n_clusters_db >= 2:
        sil_db = float(silhouette_score(X_sc, db_labels))
    else:
        n_clusters_db, sil_db = 2, 0.0
    results.append({"algorithm": "DBSCAN", "optimal_clusters": n_clusters_db,
                    "silhouette_score": sil_db})

    # PCA scatter plot coloured by KMeans labels
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_sc)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=km_labels, cmap="tab10",
                          alpha=0.6, s=20)
    plt.colorbar(scatter, label="Cluster")
    plt.title(f"{symbol} KMeans Clustering (PCA)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, f"{symbol}_kmeans_pca.png"))
    plt.close()

    # Save CSV
    pd.DataFrame(results).to_csv(os.path.join(CLUS_DIR, f"{symbol}_clustering.csv"),
                                 index=False)

    print(f"  {symbol} Clustering results:")
    for r in results:
        print(f"    {r['algorithm']:15s}  k={r['optimal_clusters']}  sil={r['silhouette_score']:.4f}")

    return results


# ── 7. Assemble dashboard JSON ─────────────────────────────────────────────────

def make_uuid() -> str:
    return str(uuid.uuid4())


def build_dashboard_json(crypto_frames: dict, crypto_ids: dict,
                         regression_all: dict, ts_all: dict,
                         clustering_all: dict) -> dict:
    """Build the JSON structure expected by the React dashboard."""

    crypto_list = [
        {"id": crypto_ids[sym], "symbol": sym, "name": CRYPTOS[sym]["name"],
         "created_at": datetime.utcnow().isoformat()}
        for sym in CRYPTOS
    ]

    # price_history — last 90 days
    price_history = []
    for sym, df in crypto_frames.items():
        for date, row in df.tail(90).iterrows():
            price_history.append({
                "id": make_uuid(),
                "crypto_id": crypto_ids[sym],
                "date": date.strftime("%Y-%m-%d"),
                "open":   round(float(row["Open"]),   6),
                "high":   round(float(row["High"]),   6),
                "low":    round(float(row["Low"]),    6),
                "close":  round(float(row["Close"]),  6),
                "volume": round(float(row["Volume"]), 2),
            })

    # regression_models
    regression_models = []
    for sym, models in regression_all.items():
        best_r2 = max(m["r2"] for m in models)
        for m in models:
            regression_models.append({
                "id": make_uuid(),
                "crypto_id": crypto_ids[sym],
                "model_name": m["name"],
                "r2_score": round(m["r2"], 6),
                "rmse": round(m["rmse"], 4),
                "mae": round(m["mae"], 4),
                "is_best": m["r2"] == best_r2,
            })

    # forecasts
    forecasts = []
    for sym, ts_data in ts_all.items():
        for model_type in ("ARIMA", "LSTM"):
            for date_str, price in ts_data[model_type]["forecasts"]:
                forecasts.append({
                    "id": make_uuid(),
                    "crypto_id": crypto_ids[sym],
                    "model_type": model_type,
                    "forecast_date": date_str,
                    "predicted_price": round(float(price), 6),
                })

    # clustering_results
    clustering_results = []
    for sym, algs in clustering_all.items():
        best_sil = max(a["silhouette_score"] for a in algs)
        for a in algs:
            clustering_results.append({
                "id": make_uuid(),
                "crypto_id": crypto_ids[sym],
                "algorithm": a["algorithm"],
                "optimal_clusters": a["optimal_clusters"],
                "silhouette_score": round(a["silhouette_score"], 6),
                "is_best": a["silhouette_score"] == best_sil,
            })

    return {
        "cryptocurrencies": crypto_list,
        "price_history": price_history,
        "regression_models": regression_models,
        "forecasts": forecasts,
        "clustering_results": clustering_results,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Crypto Price Prediction — End-to-End Pipeline")
    print("=" * 60)

    crypto_ids    = {sym: make_uuid() for sym in CRYPTOS}
    crypto_frames = {}
    crypto_featured = {}
    regression_all  = {}
    ts_all          = {}
    clustering_all  = {}

    for sym in CRYPTOS:
        name = CRYPTOS[sym]["name"]
        print(f"\n{'─'*50}")
        print(f"  Processing {name} ({sym})")
        print(f"{'─'*50}")

        # 1 — Generate data
        print(f"\n[1/4] Generating synthetic OHLCV data …")
        df_raw = generate_ohlcv(sym, n_days=730)
        crypto_frames[sym] = df_raw

        df_feat = add_features(df_raw)
        crypto_featured[sym] = df_feat
        print(f"  {len(df_feat)} rows with {df_feat.shape[1]} features")

        # 2 — Regression
        print(f"\n[2/4] Running regression models …")
        regression_all[sym] = run_regression(sym, df_feat)

        # 3 — Time series
        print(f"\n[3/4] Running time-series models …")
        ts = df_raw["Close"]
        arima_res = run_arima(sym, ts, steps=30)
        lstm_res  = run_lstm(sym, ts,  steps=30)
        ts_all[sym] = {"ARIMA": arima_res, "LSTM": lstm_res}

        # 4 — Clustering
        print(f"\n[4/4] Running clustering analysis …")
        clustering_all[sym] = run_clustering(sym, df_feat)

    # Assemble & save dashboard JSON
    dashboard_json = build_dashboard_json(
        crypto_frames, crypto_ids, regression_all, ts_all, clustering_all
    )

    json_path = os.path.join(DATA_DIR, "crypto_data.json")
    with open(json_path, "w") as f:
        json.dump(dashboard_json, f, indent=2)
    print(f"\n{'='*60}")
    print(f"  Dashboard data saved to: {json_path}")

    # Summary
    print("\n  Regression summary (best model per crypto):")
    for sym, models in regression_all.items():
        best = max(models, key=lambda m: m["r2"])
        print(f"    {sym}  best={best['name']:12s}  R²={best['r2']:.4f}  RMSE={best['rmse']:.2f}")

    print("\n  Time-series summary:")
    for sym, ts_data in ts_all.items():
        ar = ts_data["ARIMA"]
        ls = ts_data["LSTM"]
        print(f"    {sym}  ARIMA R²={ar['r2']:.4f}  LSTM R²={ls['r2']:.4f}")

    print("\n  Clustering summary (best silhouette per crypto):")
    for sym, algs in clustering_all.items():
        best = max(algs, key=lambda a: a["silhouette_score"])
        print(f"    {sym}  best={best['algorithm']:15s}  sil={best['silhouette_score']:.4f}")

    print(f"\n  Visualizations saved to: {VIS_DIR}")
    print("  Pipeline complete ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
