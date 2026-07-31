"""Fundamentals proxies — video strategy #1 (Token Terminal / staking / TVL).

Callers: external.load_external_frame, market_overview / live enrich.
Free: DefiLlama yields (Lido stETH APY) + TVL already in external.
No DCF oracle — scenario tree is qualitative weights for UI/docs.
"""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
import requests

_UA = {"User-Agent": "crypto-price-prediction-pipeline"}
_TIMEOUT = 60


def fetch_lido_steth_apy(cache_dir: str = "data/raw", force: bool = False) -> pd.Series:
    """Lido stETH pool APY from DefiLlama yields (ETH staking yield proxy)."""
    path = os.path.join(cache_dir, "external", "eth_lido_apy.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    today = pd.Timestamp.utcnow().normalize().tz_localize(None)

    hist = pd.Series(dtype=float, name="ETH_Staking_APY")
    if not force and os.path.exists(path):
        hist = pd.read_csv(path, parse_dates=["Date"], index_col="Date")["ETH_Staking_APY"]
        hist.index = pd.to_datetime(hist.index).tz_localize(None)
        if today in hist.index and not force:
            return hist.sort_index()

    r = requests.get("https://yields.llama.fi/pools", timeout=_TIMEOUT, headers=_UA)
    r.raise_for_status()
    pools = r.json().get("data") or []
    candidates = [
        p for p in pools
        if str(p.get("project", "")).lower() == "lido"
        and "ETH" in str(p.get("symbol", "")).upper()
    ]
    if not candidates:
        return hist.sort_index()
    best = max(candidates, key=lambda p: float(p.get("tvlUsd") or 0))
    apy = float(best.get("apy") or best.get("apyBase") or 0)
    if apy > 0:
        hist.loc[today] = apy
        hist.sort_index().to_csv(path, index_label="Date", header=["ETH_Staking_APY"])
    return hist.sort_index()


def fundamental_snapshot(symbol: str, feat: pd.DataFrame | None = None) -> dict[str, Any]:
    """UI-facing fundamentals card: TVL/staking/scenario tree weights."""
    out: dict[str, Any] = {
        "symbol": symbol,
        "available": False,
        "staking_apy": None,
        "tvl": None,
        "tvl_chg_7d": None,
        "scenario_tree": [],
        "summary": "Fundamentals thin for this asset on free data.",
    }
    if feat is None or feat.empty:
        return out

    row = feat.iloc[-1]
    if "ETH_Staking_APY" in feat.columns and pd.notna(row.get("ETH_Staking_APY")):
        out["staking_apy"] = float(row["ETH_Staking_APY"])
        out["available"] = True
    if "ETH_TVL" in feat.columns and pd.notna(row.get("ETH_TVL")):
        out["tvl"] = float(row["ETH_TVL"])
        out["available"] = True
        if len(feat) >= 8 and "ETH_TVL" in feat.columns:
            past = feat["ETH_TVL"].iloc[-8]
            if pd.notna(past) and past > 0:
                out["tvl_chg_7d"] = float(row["ETH_TVL"] / past - 1.0)

    # Scenario tree (video): weigh bullish vs bearish drivers — qualitative
    bull = 0.5
    notes = []
    if out.get("tvl_chg_7d") is not None:
        if out["tvl_chg_7d"] > 0.02:
            bull += 0.1
            notes.append("TVL rising (demand / DeFi activity)")
        elif out["tvl_chg_7d"] < -0.02:
            bull -= 0.1
            notes.append("TVL contracting (liquidity risk)")
    if out.get("staking_apy") is not None:
        notes.append(f"Staking yield proxy ~{out['staking_apy']:.2f}% APY (Lido)")
        if out["staking_apy"] >= 3.0:
            bull += 0.05
        elif out["staking_apy"] <= 1.5:
            bull -= 0.05
    if "FearGreed" in feat.columns and pd.notna(row.get("FearGreed")):
        fg = float(row["FearGreed"])
        if fg >= 75:
            bull -= 0.1
            notes.append("Extreme greed — sentiment overheating")
        elif fg <= 25:
            bull += 0.05
            notes.append("Extreme fear — possible accumulation zone")

    bull = max(0.15, min(0.85, bull))
    out["scenario_tree"] = [
        {"name": "bullish", "weight": round(bull, 2),
         "note": "TVL/staking/sentiment support demand"},
        {"name": "base", "weight": round(1.0 - abs(bull - 0.5) * 0.5, 2),
         "note": "Range / chop — no clear fundamental impulse"},
        {"name": "bearish", "weight": round(1.0 - bull, 2),
         "note": "Liquidity contraction / unlock-like sell pressure (proxy)"},
    ]
    # renormalize first+third for display simplicity
    b = out["scenario_tree"][0]["weight"]
    be = out["scenario_tree"][2]["weight"]
    s = b + be
    if s > 0:
        out["scenario_tree"][0]["weight"] = round(b / s, 2)
        out["scenario_tree"][2]["weight"] = round(be / s, 2)
        out["scenario_tree"][1]["weight"] = round(1.0 - out["scenario_tree"][0]["weight"]
                                                   - out["scenario_tree"][2]["weight"], 2)
    out["notes"] = notes
    out["summary"] = (
        f"Fundamentals scenario bull weight ≈ {out['scenario_tree'][0]['weight']:.0%} "
        f"(free proxies only — not a DCF)."
    )
    return out
