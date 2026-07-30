"""Real daily OHLCV data fetching for BTC / ETH / DOGE.

Sources (both free and keyless):
    1. Coinbase Exchange public candles API (primary)
       https://api.exchange.coinbase.com/products/<PRODUCT>/candles
       Max 300 candles per request -> paginated.
    2. Kraken public OHLC API (fallback)
       https://api.kraken.com/0/public/OHLC
       Returns up to 720 daily candles in one request.

Raw pulls are cached to data/raw/<SYMBOL>_daily.csv (gitignored) with a
sidecar <SYMBOL>_meta.json recording source and fetch time, so repeated
pipeline runs do not hammer the APIs.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

COINBASE_PRODUCTS = {"BTC": "BTC-USD", "ETH": "ETH-USD", "DOGE": "DOGE-USD"}
KRAKEN_PAIRS = {"BTC": "XBTUSD", "ETH": "ETHUSD", "DOGE": "XDGUSD"}

_COINBASE_URL = "https://api.exchange.coinbase.com/products/{product}/candles"
_KRAKEN_URL = "https://api.kraken.com/0/public/OHLC"

_DAY = 86400
_REQUEST_TIMEOUT = 30
_MAX_CANDLES_PER_REQ = 300  # Coinbase hard limit


# ── Parsers (pure functions, unit-tested) ─────────────────────────────────────

def parse_coinbase(payload: list) -> pd.DataFrame:
    """Parse Coinbase candles payload: [[time, low, high, open, close, volume], ...]."""
    if not payload:
        return pd.DataFrame(columns=OHLCV_COLUMNS)
    df = pd.DataFrame(payload, columns=["time", "low", "high", "open", "close", "volume"])
    df["Date"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_localize(None)
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low",
                            "close": "Close", "volume": "Volume"})
    df = df[["Date"] + OHLCV_COLUMNS].astype({c: float for c in OHLCV_COLUMNS})
    df = df.drop_duplicates(subset="Date").sort_values("Date").set_index("Date")
    return df


def parse_kraken(payload: dict) -> pd.DataFrame:
    """Parse Kraken OHLC response body (the full JSON dict)."""
    if payload.get("error"):
        raise ValueError(f"Kraken API error: {payload['error']}")
    result = payload.get("result", {})
    pair_keys = [k for k in result if k != "last"]
    if not pair_keys:
        return pd.DataFrame(columns=OHLCV_COLUMNS)
    rows = result[pair_keys[0]]
    df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close",
                                     "vwap", "volume", "count"])
    df["Date"] = pd.to_datetime(df["time"].astype(float), unit="s", utc=True).dt.tz_localize(None)
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low",
                            "close": "Close", "volume": "Volume"})
    df = df[["Date"] + OHLCV_COLUMNS].astype({c: float for c in OHLCV_COLUMNS})
    df = df.drop_duplicates(subset="Date").sort_values("Date").set_index("Date")
    return df


# ── Validation ────────────────────────────────────────────────────────────────

def validate_ohlcv(df: pd.DataFrame, min_rows: int = 50) -> pd.DataFrame:
    """Sanity-check an OHLCV frame. Raises ValueError on bad data."""
    if list(df.columns) != OHLCV_COLUMNS:
        raise ValueError(f"Unexpected columns: {list(df.columns)}")
    if len(df) < min_rows:
        raise ValueError(f"Too few rows: {len(df)} < {min_rows}")
    if df[OHLCV_COLUMNS].isna().any().any():
        raise ValueError("NaN values in OHLCV data")
    if not df.index.is_monotonic_increasing:
        raise ValueError("Dates are not sorted ascending")
    if df.index.duplicated().any():
        raise ValueError("Duplicate dates")
    if (df[["Open", "High", "Low", "Close"]] <= 0).any().any():
        raise ValueError("Non-positive prices")
    if (df["High"] < df["Low"]).any():
        raise ValueError("High < Low rows present")
    return df


# ── Network fetchers ──────────────────────────────────────────────────────────

def fetch_coinbase(symbol: str, days: int = 730, session: requests.Session | None = None) -> pd.DataFrame:
    """Fetch ~`days` daily candles from Coinbase Exchange, paginating 300 at a time."""
    product = COINBASE_PRODUCTS[symbol]
    sess = session or requests.Session()
    end = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    start_target = end - timedelta(days=days)

    frames = []
    window_end = end
    while window_end > start_target:
        window_start = max(start_target, window_end - timedelta(days=_MAX_CANDLES_PER_REQ))
        resp = sess.get(
            _COINBASE_URL.format(product=product),
            params={
                "granularity": _DAY,
                "start": window_start.isoformat(),
                "end": window_end.isoformat(),
            },
            timeout=_REQUEST_TIMEOUT,
            headers={"User-Agent": "crypto-price-prediction-pipeline"},
        )
        resp.raise_for_status()
        frames.append(parse_coinbase(resp.json()))
        window_end = window_start
        time.sleep(0.4)  # stay well under Coinbase's public rate limit (10 req/s)

    df = pd.concat(frames)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    return validate_ohlcv(df)


def fetch_kraken(symbol: str, days: int = 720, session: requests.Session | None = None) -> pd.DataFrame:
    """Fetch daily candles from Kraken (single request; API caps at ~720 candles)."""
    sess = session or requests.Session()
    since = int((datetime.now(timezone.utc) - timedelta(days=min(days, 720))).timestamp())
    resp = sess.get(
        _KRAKEN_URL,
        params={"pair": KRAKEN_PAIRS[symbol], "interval": 1440, "since": since},
        timeout=_REQUEST_TIMEOUT,
        headers={"User-Agent": "crypto-price-prediction-pipeline"},
    )
    resp.raise_for_status()
    df = parse_kraken(resp.json())
    # Kraken's most recent candle is the still-forming day; drop it.
    if len(df) and df.index[-1].date() >= datetime.now(timezone.utc).date():
        df = df.iloc[:-1]
    return validate_ohlcv(df)


# ── Caching front-end ─────────────────────────────────────────────────────────

def cache_paths(symbol: str, cache_dir: str) -> tuple[str, str]:
    return (os.path.join(cache_dir, f"{symbol}_daily.csv"),
            os.path.join(cache_dir, f"{symbol}_meta.json"))


def load_ohlcv_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    return validate_ohlcv(df[OHLCV_COLUMNS])


def save_ohlcv_csv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index_label="Date")


def fetch_ohlcv(symbol: str, days: int = 730, cache_dir: str = "data/raw",
                force: bool = False, max_cache_age_hours: float = 24.0) -> tuple[pd.DataFrame, dict]:
    """Return (df, meta) for a symbol, using the local cache when fresh.

    meta = {"source": ..., "fetched_at": ..., "start": ..., "end": ..., "rows": ...}
    """
    csv_path, meta_path = cache_paths(symbol, cache_dir)

    if not force and os.path.exists(csv_path) and os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        fetched_at = datetime.fromisoformat(meta["fetched_at"])
        age_h = (datetime.now(timezone.utc) - fetched_at).total_seconds() / 3600
        if age_h <= max_cache_age_hours:
            return load_ohlcv_csv(csv_path), meta

    last_err = None
    for source, fetcher in (("coinbase", fetch_coinbase), ("kraken", fetch_kraken)):
        try:
            df = fetcher(symbol, days=days)
            meta = {
                "symbol": symbol,
                "source": source,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "start": df.index[0].strftime("%Y-%m-%d"),
                "end": df.index[-1].strftime("%Y-%m-%d"),
                "rows": len(df),
            }
            save_ohlcv_csv(df, csv_path)
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
            return df, meta
        except Exception as err:  # noqa: BLE001 - try next source
            last_err = err
            print(f"  [data] {source} failed for {symbol}: {err}")

    # Final fallback: stale cache is better than nothing.
    if os.path.exists(csv_path) and os.path.exists(meta_path):
        print(f"  [data] all sources failed for {symbol}; using stale cache")
        with open(meta_path) as f:
            meta = json.load(f)
        return load_ohlcv_csv(csv_path), meta

    raise RuntimeError(f"Could not fetch OHLCV for {symbol}: {last_err}")
