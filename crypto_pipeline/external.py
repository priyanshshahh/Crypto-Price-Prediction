"""Free external market features: Fear & Greed, funding, FRED macro.

All adapters cache under data/raw/external/ and degrade gracefully
(missing series → NaN columns filled forward/back limited, then 0 for z-scores
only after explicit flag). Callers: features.add_features / pipeline production mode.
"""

from __future__ import annotations

import io
import json
import os
import zipfile
import pandas as pd
import requests

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env.local"))
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
except ImportError:
    pass

_REQUEST_TIMEOUT = 45
_UA = {"User-Agent": "crypto-price-prediction-pipeline"}

FUNDING_SYMBOLS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "DOGE": "DOGEUSDT"}


def _cache_dir(base: str) -> str:
    path = os.path.join(base, "external")
    os.makedirs(path, exist_ok=True)
    return path


def fetch_fear_greed(cache_dir: str = "data/raw", force: bool = False,
                     limit: int = 0) -> pd.Series:
    """Daily Crypto Fear & Greed index from alternative.me (free, keyless)."""
    cache = os.path.join(_cache_dir(cache_dir), "fear_greed.csv")
    if not force and os.path.exists(cache):
        s = pd.read_csv(cache, parse_dates=["Date"], index_col="Date")["value"]
        return s.sort_index()

    url = "https://api.alternative.me/fng/"
    params = {"limit": limit or 0, "format": "json"}
    resp = requests.get(url, params=params, timeout=_REQUEST_TIMEOUT, headers=_UA)
    resp.raise_for_status()
    payload = resp.json().get("data", [])
    rows = []
    for item in payload:
        day = pd.to_datetime(int(item["timestamp"]), unit="s", utc=True).tz_localize(None).normalize()
        rows.append({"Date": day, "value": float(item["value"])})
    df = pd.DataFrame(rows).drop_duplicates("Date").sort_values("Date").set_index("Date")
    df.to_csv(cache, index_label="Date")
    return df["value"]


def fetch_funding_binance_vision(symbol: str, cache_dir: str = "data/raw",
                                 force: bool = False,
                                 months: int | None = 24) -> pd.Series:
    """Historical perpetual funding from data.binance.vision monthly ZIPs (free)."""
    pair = FUNDING_SYMBOLS[symbol]
    cache = os.path.join(_cache_dir(cache_dir), f"{symbol}_funding.csv")
    if not force and os.path.exists(cache):
        s = pd.read_csv(cache, parse_dates=["Date"], index_col="Date")["funding"]
        return s.sort_index()

    end = pd.Timestamp.utcnow().normalize().tz_localize(None)
    start = end - pd.DateOffset(months=months or 24)
    months_idx = pd.period_range(start, end, freq="M")
    frames: list[pd.DataFrame] = []
    sess = requests.Session()
    for per in months_idx:
        yyyymm = f"{per.year}-{per.month:02d}"
        name = f"{pair}-fundingRate-{yyyymm}.zip"
        url = (
            "https://data.binance.vision/data/futures/um/monthly/fundingRate/"
            f"{pair}/{name}"
        )
        try:
            r = sess.get(url, timeout=_REQUEST_TIMEOUT, headers=_UA)
            if r.status_code != 200:
                continue
            with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
                csv_name = zf.namelist()[0]
                raw = pd.read_csv(zf.open(csv_name))
            cols = {c.lower(): c for c in raw.columns}
            time_col = cols.get("calc_time") or cols.get("fundingtime") or list(raw.columns)[0]
            rate_col = cols.get("fundingrate") or cols.get("last_funding_rate") or list(raw.columns)[-1]
            part = pd.DataFrame({
                "Date": pd.to_datetime(raw[time_col], unit="ms", utc=True).dt.tz_localize(None),
                "funding": pd.to_numeric(raw[rate_col], errors="coerce"),
            })
            frames.append(part)
        except Exception as err:  # noqa: BLE001
            print(f"  [external] funding {pair} {yyyymm}: {err}")

    if not frames:
        empty = pd.Series(dtype=float, name="funding")
        empty.index.name = "Date"
        return empty

    df = pd.concat(frames, ignore_index=True)
    df = df.dropna().drop_duplicates("Date").sort_values("Date").set_index("Date")
    daily = df["funding"].resample("1D").mean().dropna()
    daily.to_frame("funding").to_csv(cache, index_label="Date")
    return daily


def fetch_funding_perpfinder(symbol: str, cache_dir: str = "data/raw",
                             force: bool = False) -> pd.Series:
    """Live funding snapshot — best-effort free fallback (Vision preferred for history)."""
    cache = os.path.join(_cache_dir(cache_dir), f"{symbol}_funding_live.json")
    if not force and os.path.exists(cache):
        with open(cache) as f:
            payload = json.load(f)
        if "rate" in payload and "date" in payload:
            idx = pd.to_datetime([payload["date"]])
            return pd.Series([payload["rate"]], index=idx, name="funding")

    url = "https://perpfinder.com/api/data/funding-rates"
    try:
        r = requests.get(url, timeout=_REQUEST_TIMEOUT, headers=_UA)
        r.raise_for_status()
        rows = r.json().get("rows") or r.json()
        if not isinstance(rows, list):
            return pd.Series(dtype=float, name="funding")
        rates = []
        for row in rows:
            asset = str(row.get("asset") or row.get("symbol") or "").upper()
            if symbol not in asset and asset not in (symbol, f"{symbol}USDT"):
                continue
            rate = row.get("fundingRate") or row.get("rate") or row.get("funding")
            if rate is None:
                continue
            rates.append(float(rate))
        if not rates:
            return pd.Series(dtype=float, name="funding")
        avg = float(sum(rates) / len(rates))
        today = pd.Timestamp.utcnow().normalize().tz_localize(None)
        with open(cache, "w") as f:
            json.dump({"date": str(today.date()), "rate": avg, "source": "perpfinder"}, f)
        return pd.Series([avg], index=[today], name="funding")
    except Exception as err:  # noqa: BLE001
        print(f"  [external] perpfinder funding {symbol}: {err}")
        return pd.Series(dtype=float, name="funding")


def fetch_fred_series(series_id: str, api_key: str, cache_dir: str = "data/raw",
                      force: bool = False) -> pd.Series:
    """FRED daily/business-day series (free API key)."""
    cache = os.path.join(_cache_dir(cache_dir), f"fred_{series_id}.csv")
    if not force and os.path.exists(cache):
        s = pd.read_csv(cache, parse_dates=["Date"], index_col="Date")["value"]
        return s.sort_index()

    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": "2018-01-01",
    }
    r = requests.get(url, params=params, timeout=_REQUEST_TIMEOUT, headers=_UA)
    r.raise_for_status()
    obs = r.json().get("observations", [])
    rows = []
    for o in obs:
        if o.get("value") in (".", "", None):
            continue
        rows.append({"Date": pd.to_datetime(o["date"]), "value": float(o["value"])})
    df = pd.DataFrame(rows).drop_duplicates("Date").sort_values("Date").set_index("Date")
    df.to_csv(cache, index_label="Date")
    return df["value"]


def load_external_frame(symbol: str, cache_dir: str = "data/raw", force: bool = False,
                        fred_api_key: str | None = None,
                        fred_series: list[str] | None = None) -> pd.DataFrame:
    """Build a Date-indexed frame of external features for one symbol."""
    parts: dict[str, pd.Series] = {}

    try:
        fg = fetch_fear_greed(cache_dir=cache_dir, force=force)
        parts["FearGreed"] = fg
        parts["FearGreed_chg"] = fg.diff()
    except Exception as err:  # noqa: BLE001
        print(f"  [external] fear_greed failed: {err}")

    funding = fetch_funding_binance_vision(symbol, cache_dir=cache_dir, force=force)
    if funding.empty:
        funding = fetch_funding_perpfinder(symbol, cache_dir=cache_dir, force=force)
    if not funding.empty:
        parts["Funding"] = funding
        parts["Funding_z30"] = (funding - funding.rolling(30).mean()) / funding.rolling(30).std()

    key = fred_api_key or os.environ.get("FRED_API_KEY")
    if key:
        # Extra free macro: VIX + fed funds (risk / liquidity) — prediction skill
        for sid in (fred_series or ["DTWEXBGS", "DGS10", "VIXCLS", "DFF"]):
            try:
                s = fetch_fred_series(sid, key, cache_dir=cache_dir, force=force)
                parts[f"FRED_{sid}"] = s
                parts[f"FRED_{sid}_chg"] = s.diff()
            except Exception as err:  # noqa: BLE001
                print(f"  [external] FRED {sid}: {err}")

    # BTC mcap strength proxy from CoinGecko (free / demo key)
    try:
        dom = fetch_btc_dominance_coingecko(cache_dir=cache_dir, force=force)
        if not dom.empty:
            parts["BTC_Dominance"] = dom
            parts["BTC_Dominance_chg"] = dom.diff()
    except Exception as err:  # noqa: BLE001
        print(f"  [external] coingecko dominance: {err}")

    # CMC keyless (trial-pro-api) — no key required
    try:
        cmc = fetch_cmc_keyless(cache_dir=cache_dir, force=force)
        for col in cmc.columns:
            parts[col] = cmc[col]
    except Exception as err:  # noqa: BLE001
        print(f"  [external] CMC keyless: {err}")

    # Free BTC on-chain (mempool.space + blockchain.info charts)
    if symbol == "BTC":
        try:
            oc = fetch_btc_onchain_free(cache_dir=cache_dir, force=force)
            for col in oc.columns:
                parts[col] = oc[col]
        except Exception as err:  # noqa: BLE001
            print(f"  [external] BTC on-chain: {err}")

    # Free ETH DeFi TVL proxy (DefiLlama, keyless)
    if symbol == "ETH":
        try:
            tvl = fetch_defillama_eth_tvl(cache_dir=cache_dir, force=force)
            if not tvl.empty:
                parts["ETH_TVL"] = tvl
                parts["ETH_TVL_chg"] = tvl.pct_change()
        except Exception as err:  # noqa: BLE001
            print(f"  [external] DefiLlama ETH TVL: {err}")
        try:
            gas = fetch_etherscan_gas_snapshot(cache_dir=cache_dir, force=force)
            if not gas.empty:
                for col in gas.columns:
                    parts[col] = gas[col]
        except Exception as err:  # noqa: BLE001
            print(f"  [external] Etherscan gas: {err}")

    # Google Trends interest (SMU paper: search volume > sentiment polarity)
    try:
        trends = fetch_google_trends(symbol, cache_dir=cache_dir, force=force)
        if not trends.empty:
            parts["GoogleTrends"] = trends
            parts["GoogleTrends_chg"] = trends.diff()
            parts["GoogleTrends_z30"] = (
                (trends - trends.rolling(30).mean()) / trends.rolling(30).std()
            )
    except Exception as err:  # noqa: BLE001
        print(f"  [external] Google Trends: {err}")

    # Polymarket crowd Yes (prediction markets — snapshot series)
    try:
        from .prediction_markets import polymarket_series
        pm = polymarket_series(symbol, cache_dir=cache_dir, force=force)
        if not pm.empty:
            parts["Polymarket_Yes_Prob"] = pm
            parts["Polymarket_Yes_chg"] = pm.diff()
    except Exception as err:  # noqa: BLE001
        print(f"  [external] Polymarket: {err}")

    # ETH staking yield proxy (Lido via DefiLlama) — fundamentals layer
    if symbol == "ETH":
        try:
            from .fundamentals import fetch_lido_steth_apy
            apy = fetch_lido_steth_apy(cache_dir=cache_dir, force=force)
            if not apy.empty:
                parts["ETH_Staking_APY"] = apy
        except Exception as err:  # noqa: BLE001
            print(f"  [external] ETH staking APY: {err}")

    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, axis=1).sort_index()
    out.index = pd.to_datetime(out.index).tz_localize(None)
    return out


TRENDS_KEYWORDS = {
    "BTC": "Bitcoin",
    "ETH": "Ethereum",
    "DOGE": "Dogecoin",
}


def fetch_google_trends(symbol: str, cache_dir: str = "data/raw",
                        force: bool = False, years: int = 4) -> pd.Series:
    """Daily Google Trends interest for the asset keyword (pytrends, keyless).

    Weekly Trends points are forward-filled to daily. Degrades to empty Series
    if pytrends is missing or Google rate-limits.
    """
    cache = os.path.join(_cache_dir(cache_dir), f"{symbol}_google_trends.csv")
    if not force and os.path.exists(cache):
        s = pd.read_csv(cache, parse_dates=["Date"], index_col="Date")["interest"]
        return s.sort_index()

    try:
        from pytrends.request import TrendReq
    except ImportError:
        print("  [external] pytrends not installed — skip Google Trends")
        return pd.Series(dtype=float)

    keyword = TRENDS_KEYWORDS.get(symbol, symbol)
    end = pd.Timestamp.utcnow().normalize().tz_localize(None)
    start = end - pd.DateOffset(years=years)
    timeframe = f"{start.date()} {end.date()}"
    try:
        pytrends = TrendReq(hl="en-US", tz=0, timeout=(10, 30))
        pytrends.build_payload([keyword], cat=0, timeframe=timeframe, geo="", gprop="")
        raw = pytrends.interest_over_time()
    except Exception as err:  # noqa: BLE001
        print(f"  [external] Google Trends fetch failed: {err}")
        return pd.Series(dtype=float)

    if raw is None or raw.empty or keyword not in raw.columns:
        return pd.Series(dtype=float)

    weekly = raw[keyword].astype(float)
    weekly.index = pd.to_datetime(weekly.index).tz_localize(None).normalize()
    daily_idx = pd.date_range(weekly.index.min(), end, freq="D")
    daily = weekly.reindex(daily_idx).ffill()
    daily.name = "interest"
    daily.to_frame().to_csv(cache, index_label="Date")
    return daily


COINGECKO_IDS = {"BTC": "bitcoin", "ETH": "ethereum", "DOGE": "dogecoin"}
_CMC_KEYLESS = "https://pro-api.coinmarketcap.com/trial-pro-api"


def fetch_btc_dominance_coingecko(cache_dir: str = "data/raw", force: bool = False,
                                  days: int = 365) -> pd.Series:
    """BTC market-cap strength proxy via CoinGecko market_chart (free)."""
    cache = os.path.join(_cache_dir(cache_dir), "btc_dominance.csv")
    if not force and os.path.exists(cache):
        s = pd.read_csv(cache, parse_dates=["Date"], index_col="Date")["dominance"]
        return s.sort_index()

    headers = dict(_UA)
    cg_key = os.environ.get("COINGECKO_API_KEY")
    if cg_key:
        headers["x-cg-demo-api-key"] = cg_key

    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    # Demo tier rejects some long windows (e.g. 730); try requested then fall back.
    last_err = None
    caps = []
    for d in (days, 365, 180):
        try:
            r = requests.get(url, params={"vs_currency": "usd", "days": d},
                             timeout=_REQUEST_TIMEOUT, headers=headers)
            r.raise_for_status()
            caps = r.json().get("market_caps") or []
            if caps:
                break
        except Exception as err:  # noqa: BLE001
            last_err = err
    if not caps:
        raise RuntimeError(f"CoinGecko market_caps empty ({last_err})")
    df = pd.DataFrame(caps, columns=["ts", "mcap"])
    df["Date"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_localize(None).dt.normalize()
    daily = df.groupby("Date")["mcap"].last()
    z = (daily - daily.rolling(30).mean()) / daily.rolling(30).std()
    z = z.rename("dominance")
    z.to_frame("dominance").to_csv(cache, index_label="Date")
    return z.dropna()


def fetch_cmc_keyless(cache_dir: str = "data/raw", force: bool = False) -> pd.DataFrame:
    """CMC Trial Pro (keyless): global metrics snapshot + Fear&Greed history.

    Base: https://pro-api.coinmarketcap.com/trial-pro-api
    Rate-limited; cached under data/raw/external/.
    """
    cache = os.path.join(_cache_dir(cache_dir), "cmc_keyless.csv")
    if not force and os.path.exists(cache):
        hist = pd.read_csv(cache, parse_dates=["Date"], index_col="Date")
        age_h = (pd.Timestamp.utcnow().tz_localize(None) - hist.index.max()).total_seconds() / 3600
        if age_h <= 24:
            return hist

    frames: list[pd.DataFrame] = []

    # Fear & Greed historical (daily) — keyless max ~500; use 365
    try:
        r = requests.get(
            f"{_CMC_KEYLESS}/v3/fear-and-greed/historical",
            params={"limit": 365},
            timeout=_REQUEST_TIMEOUT,
            headers={**_UA, "Accept": "application/json"},
        )
        r.raise_for_status()
        rows = []
        for item in r.json().get("data") or []:
            ts = item.get("timestamp")
            day = pd.to_datetime(int(ts), unit="s", utc=True).tz_localize(None).normalize()
            rows.append({"Date": day, "CMC_FearGreed": float(item["value"])})
        if rows:
            frames.append(pd.DataFrame(rows).drop_duplicates("Date").set_index("Date"))
    except Exception as err:  # noqa: BLE001
        print(f"  [external] CMC FearGreed hist: {err}")

    # Global metrics latest — append as today's row (deep history needs paid)
    r2 = requests.get(
        f"{_CMC_KEYLESS}/v1/global-metrics/quotes/latest",
        timeout=_REQUEST_TIMEOUT,
        headers={**_UA, "Accept": "application/json"},
    )
    r2.raise_for_status()
    data = r2.json()["data"]
    q = data["quote"]["USD"]
    today = pd.Timestamp.utcnow().normalize().tz_localize(None)
    snap = pd.DataFrame([{
        "Date": today,
        "CMC_total_mcap": float(q.get("total_market_cap") or 0),
        "CMC_btc_dominance": float(data.get("btc_dominance") or 0),
        "CMC_eth_dominance": float(data.get("eth_dominance") or 0),
    }]).set_index("Date")
    frames.append(snap)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, axis=1).sort_index()
    # merge with prior cache so dominance snapshots accumulate over days
    if os.path.exists(cache):
        prev = pd.read_csv(cache, parse_dates=["Date"], index_col="Date")
        out = pd.concat([prev, out])
        out = out[~out.index.duplicated(keep="last")].sort_index()
    out.to_csv(cache, index_label="Date")
    return out


def fetch_btc_onchain_free(cache_dir: str = "data/raw", force: bool = False) -> pd.DataFrame:
    """Free BTC on-chain: mempool.space hashrate + blockchain.info tx count."""
    cache = os.path.join(_cache_dir(cache_dir), "btc_onchain.csv")
    if not force and os.path.exists(cache):
        hist = pd.read_csv(cache, parse_dates=["Date"], index_col="Date")
        age_h = (pd.Timestamp.utcnow().tz_localize(None) - hist.index.max()).total_seconds() / 3600
        if age_h <= 24:
            return hist

    parts: dict[str, pd.Series] = {}

    r = requests.get(
        "https://mempool.space/api/v1/mining/hashrate/3y",
        timeout=_REQUEST_TIMEOUT,
        headers=_UA,
    )
    r.raise_for_status()
    hrs = r.json().get("hashrates") or []
    if hrs:
        df = pd.DataFrame(hrs)
        df["Date"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_localize(None).dt.normalize()
        parts["BTC_Hashrate"] = df.groupby("Date")["avgHashrate"].last()
        parts["BTC_Hashrate_chg"] = parts["BTC_Hashrate"].pct_change()

    r2 = requests.get(
        "https://api.blockchain.info/charts/n-transactions",
        params={"timespan": "2years", "format": "json"},
        timeout=_REQUEST_TIMEOUT,
        headers=_UA,
    )
    r2.raise_for_status()
    vals = r2.json().get("values") or []
    if vals:
        df2 = pd.DataFrame(vals)
        df2["Date"] = pd.to_datetime(df2["x"], unit="s", utc=True).dt.tz_localize(None).dt.normalize()
        parts["BTC_TxCount"] = df2.groupby("Date")["y"].last()
        parts["BTC_TxCount_chg"] = parts["BTC_TxCount"].pct_change()

    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, axis=1).sort_index()
    out.to_csv(cache, index_label="Date")
    return out


def fetch_defillama_eth_tvl(cache_dir: str = "data/raw", force: bool = False) -> pd.Series:
    """Ethereum chain TVL history from DefiLlama (keyless)."""
    cache = os.path.join(_cache_dir(cache_dir), "eth_tvl.csv")
    if not force and os.path.exists(cache):
        s = pd.read_csv(cache, parse_dates=["Date"], index_col="Date")["tvl"]
        return s.sort_index()

    r = requests.get(
        "https://api.llama.fi/v2/historicalChainTvl/Ethereum",
        timeout=_REQUEST_TIMEOUT,
        headers=_UA,
    )
    r.raise_for_status()
    rows = r.json()
    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["date"], unit="s", utc=True).dt.tz_localize(None).dt.normalize()
    s = df.groupby("Date")["tvl"].last().rename("tvl")
    s.to_frame("tvl").to_csv(cache, index_label="Date")
    return s


def fetch_etherscan_gas_snapshot(cache_dir: str = "data/raw", force: bool = False) -> pd.DataFrame:
    """Etherscan free gas oracle snapshot (historical daily stats are Pro-only).

    Accumulates one row per calendar day in cmc-style cache so the series grows
    over time. Requires ETHERSCAN_API_KEY in env.
    """
    key = os.environ.get("ETHERSCAN_API_KEY", "").strip()
    if not key:
        return pd.DataFrame()

    cache = os.path.join(_cache_dir(cache_dir), "eth_gas_oracle.csv")
    today = pd.Timestamp.utcnow().normalize().tz_localize(None)
    if not force and os.path.exists(cache):
        hist = pd.read_csv(cache, parse_dates=["Date"], index_col="Date")
        if len(hist) and hist.index[-1].date() == today.date():
            return hist

    r = requests.get(
        "https://api.etherscan.io/v2/api",
        params={"chainid": 1, "module": "gastracker", "action": "gasoracle", "apikey": key},
        timeout=_REQUEST_TIMEOUT,
        headers=_UA,
    )
    r.raise_for_status()
    payload = r.json()
    if str(payload.get("status")) != "1":
        raise RuntimeError(payload.get("result") or payload.get("message"))
    res = payload["result"]
    part = pd.DataFrame([{
        "Date": today,
        "ETH_GasSafe": float(res.get("SafeGasPrice") or 0),
        "ETH_GasPropose": float(res.get("ProposeGasPrice") or 0),
        "ETH_GasFast": float(res.get("FastGasPrice") or 0),
    }]).set_index("Date")
    if os.path.exists(cache):
        hist = pd.read_csv(cache, parse_dates=["Date"], index_col="Date")
        hist = pd.concat([hist, part])
        hist = hist[~hist.index.duplicated(keep="last")].sort_index()
    else:
        hist = part
    hist.to_csv(cache, index_label="Date")
    return hist
