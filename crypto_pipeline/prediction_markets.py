"""Prediction-market crowd probabilities (Polymarket) — video strategy #6.

Callers: external.load_external_frame (snapshot series), horizons live enrich,
  pipeline production. Public Gamma API (keyless). Schema: Polymarket_Yes_Prob
  daily series + live meta dict. User: learn 8 strategies; implement Polymarket.
"""

from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd
import requests

_UA = {"User-Agent": "crypto-price-prediction-pipeline"}
_TIMEOUT = 45

SEARCH_QUERIES = {
    "BTC": ["bitcoin price", "bitcoin 100k", "bitcoin above"],
    "ETH": ["ethereum price", "ethereum above", "eth 5000"],
    "DOGE": ["dogecoin", "doge price"],
}


def _parse_yes_price(outcome_prices: Any) -> float | None:
    if outcome_prices is None:
        return None
    if isinstance(outcome_prices, str):
        try:
            outcome_prices = json.loads(outcome_prices)
        except json.JSONDecodeError:
            return None
    if not isinstance(outcome_prices, (list, tuple)) or not outcome_prices:
        return None
    try:
        return float(outcome_prices[0])
    except (TypeError, ValueError):
        return None


def fetch_polymarket_snapshot(symbol: str, *, limit_events: int = 5) -> dict[str, Any]:
    """Search Polymarket for active crypto markets; average Yes probs.

    Returns crowd_yes in [0,1] when at least one market parses, else empty.
    """
    queries = SEARCH_QUERIES.get(symbol, [symbol.lower()])
    yes_prices: list[float] = []
    samples: list[dict[str, Any]] = []
    sess = requests.Session()
    for q in queries:
        try:
            r = sess.get(
                "https://gamma-api.polymarket.com/public-search",
                params={"q": q, "limit_per_type": limit_events},
                timeout=_TIMEOUT,
                headers=_UA,
            )
            if r.status_code != 200:
                continue
            for ev in r.json().get("events") or []:
                title = ev.get("title") or ""
                for m in ev.get("markets") or []:
                    if m.get("closed"):
                        continue
                    yes = _parse_yes_price(m.get("outcomePrices"))
                    if yes is None:
                        continue
                    # Skip near-certain / empty liquidity noise when possible
                    liq = float(m.get("liquidity") or 0)
                    if liq and liq < 100:
                        continue
                    yes_prices.append(yes)
                    samples.append({
                        "title": title,
                        "question": m.get("question") or title,
                        "yes_prob": yes,
                        "liquidity": liq,
                    })
        except Exception:  # noqa: BLE001
            continue

    if not yes_prices:
        return {"available": False, "crowd_yes": None, "n_markets": 0, "samples": []}

    # Prefer mid-range markets (informative); still average all collected
    crowd = float(sum(yes_prices) / len(yes_prices))
    samples = sorted(samples, key=lambda s: -float(s.get("liquidity") or 0))[:5]
    return {
        "available": True,
        "crowd_yes": round(crowd, 4),
        "n_markets": len(yes_prices),
        "samples": samples,
        "as_of": pd.Timestamp.utcnow().isoformat(),
    }


def polymarket_series(symbol: str, cache_dir: str = "data/raw",
                      force: bool = False) -> pd.Series:
    """Append today's Polymarket crowd Yes into a cached daily series."""
    path = os.path.join(cache_dir, "external", f"{symbol}_polymarket.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    today = pd.Timestamp.utcnow().normalize().tz_localize(None)

    hist = pd.Series(dtype=float, name="Polymarket_Yes_Prob")
    if not force and os.path.exists(path):
        hist = pd.read_csv(path, parse_dates=["Date"], index_col="Date")["Polymarket_Yes_Prob"]
        hist.index = pd.to_datetime(hist.index).tz_localize(None)

    snap = fetch_polymarket_snapshot(symbol)
    if snap.get("available") and snap.get("crowd_yes") is not None:
        hist.loc[today] = float(snap["crowd_yes"])
        # also stash meta beside series
        meta_path = path.replace(".csv", "_meta.json")
        with open(meta_path, "w") as f:
            json.dump(snap, f, indent=2)
        hist.sort_index().to_csv(path, index_label="Date", header=["Polymarket_Yes_Prob"])
    return hist.sort_index().rename("Polymarket_Yes_Prob")
