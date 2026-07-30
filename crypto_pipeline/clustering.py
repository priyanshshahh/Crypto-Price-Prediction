"""Market-regime clustering on daily feature vectors.

Each trading day is a point in (Return_1d, Volatility, RSI, MACD, Volume_MA7)
space; clusters correspond to market regimes (calm accumulation, high-vol
selloff, ...). KMeans / Agglomerative / GMM sweep k=2..6 and keep the best
silhouette; DBSCAN auto-tunes eps from the 5-NN distance distribution.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RANDOM_STATE = 42
CLUSTER_FEATURES = ["Return_1d", "Volatility", "RSI", "MACD", "Volume_MA7"]


def run_clustering(df_feat: pd.DataFrame, k_range: range = range(2, 7)) -> dict:
    from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    from sklearn.mixture import GaussianMixture
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    X = df_feat[CLUSTER_FEATURES].dropna()
    X_sc = StandardScaler().fit_transform(X)

    results = []

    def sweep(make_model):
        best = (-1.0, None, None)  # (silhouette, k, labels)
        for k in k_range:
            labels = make_model(k).fit_predict(X_sc)
            sil = float(silhouette_score(X_sc, labels))
            if sil > best[0]:
                best = (sil, k, labels)
        return best

    sil, k, km_labels = sweep(lambda k: KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10))
    results.append({"algorithm": "KMeans", "optimal_clusters": int(k), "silhouette_score": sil})

    sil, k, _ = sweep(lambda k: AgglomerativeClustering(n_clusters=k))
    results.append({"algorithm": "Agglomerative", "optimal_clusters": int(k), "silhouette_score": sil})

    sil, k, _ = sweep(lambda k: GaussianMixture(n_components=k, random_state=RANDOM_STATE))
    results.append({"algorithm": "GMM", "optimal_clusters": int(k), "silhouette_score": sil})

    # DBSCAN with eps at the 90th percentile of 5-NN distances
    nbrs = NearestNeighbors(n_neighbors=5).fit(X_sc)
    distances, _ = nbrs.kneighbors(X_sc)
    eps = float(np.percentile(distances[:, -1], 90))
    db_labels = DBSCAN(eps=eps, min_samples=5).fit_predict(X_sc)
    n_db = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    sil_db = float(silhouette_score(X_sc, db_labels)) if n_db >= 2 else 0.0
    results.append({"algorithm": "DBSCAN", "optimal_clusters": int(max(n_db, 1)),
                    "silhouette_score": sil_db})

    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_sc)

    return {"results": results, "kmeans_labels": km_labels, "pca_coords": coords,
            "n_points": int(len(X))}
