"""Quant risk layer — video strategy #7 (vol / Monte Carlo paths).

Callers: horizons.live_price_forecasts enrich. EWMA vol (GARCH-lite) +
Monte Carlo return paths → scenario p10/p50/p90. No network.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from . import RANDOM_STATE


def ewma_vol(returns: pd.Series, span: int = 30) -> float:
    """Annualization-free daily EWMA volatility (GARCH-lite)."""
    r = returns.dropna().astype(float)
    if len(r) < 10:
        return float(r.std()) if len(r) else 0.02
    return float(r.ewm(span=span, adjust=False).std().iloc[-1])


def monte_carlo_paths(
    last_price: float,
    mu_daily: float,
    vol_daily: float,
    horizon_days: int,
    n_paths: int = 1000,
    seed: int = RANDOM_STATE,
) -> dict[str, Any]:
    """Simulate GBM-like paths; return price quantiles at horizon."""
    rng = np.random.default_rng(seed + int(horizon_days))
    vol = max(float(vol_daily), 1e-6)
    shocks = rng.normal(loc=mu_daily, scale=vol, size=(n_paths, max(horizon_days, 1)))
    log_cum = shocks.sum(axis=1)
    terminal = last_price * np.exp(log_cum)
    terminal = np.maximum(terminal, 0.0)
    p10, p50, p90 = np.quantile(terminal, [0.1, 0.5, 0.9])
    ret = terminal / last_price - 1.0
    return {
        "n_paths": n_paths,
        "vol_daily": round(vol, 6),
        "mu_daily": round(float(mu_daily), 6),
        "price_p10": float(p10),
        "price_p50": float(p50),
        "price_p90": float(p90),
        "return_p10_pct": float(np.quantile(ret, 0.1) * 100),
        "return_p50_pct": float(np.quantile(ret, 0.5) * 100),
        "return_p90_pct": float(np.quantile(ret, 0.9) * 100),
        "prob_up": float((ret > 0).mean()),
    }


def quant_overlay(close: pd.Series, horizon_days: int,
                  pred_return: float | None = None) -> dict[str, Any]:
    """Combine EWMA vol + MC; drift = model pred or historical mean."""
    rets = close.pct_change(1).dropna()
    vol = ewma_vol(rets)
    mu = float(pred_return) if pred_return is not None else float(rets.tail(90).mean())
    # pred_return is horizon cumulative; convert to daily drift approx
    h = max(int(horizon_days), 1)
    mu_daily = (1.0 + mu) ** (1.0 / h) - 1.0 if pred_return is not None else mu
    last = float(close.iloc[-1])
    mc = monte_carlo_paths(last, mu_daily, vol, h)
    mc["method"] = "EWMA-vol + Monte Carlo"
    return mc
