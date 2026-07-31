"""Cost-aware paper trading: long/flat and long/short modes.

Live-standard defaults from configs/price_v1.yaml:
  fee 50 bps round-trip, slippage 10/10/20 bps (BTC/ETH/DOGE).
Select best mode by net Sharpe after costs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PaperConfig:
    fee_bps_round_trip: float = 50.0
    slippage_bps: float = 10.0
    initial_equity: float = 100_000.0
    target_ann_vol: float = 0.10
    min_abs_pred: float = 0.015
    mode: str = "long_flat"  # or long_short


def _cost_frac(cfg: PaperConfig) -> float:
    return (cfg.fee_bps_round_trip + 2 * cfg.slippage_bps) / 10_000.0


def run_paper(
    actual_returns: np.ndarray,
    predicted_returns: np.ndarray,
    cfg: PaperConfig,
    realized_vol: np.ndarray | None = None,
) -> dict:
    """Simulate positions from predicted H-day returns mapped to daily book."""
    actual = np.asarray(actual_returns, dtype=float)
    pred = np.asarray(predicted_returns, dtype=float)
    n = len(actual)
    if n == 0:
        return {"net_sharpe": 0.0, "gross_sharpe": 0.0, "max_drawdown": 0.0,
                "n_trades": 0, "hit_rate": 0.5, "mode": cfg.mode, "equity": []}

    if cfg.mode == "long_short":
        pos = np.where(pred >= cfg.min_abs_pred, 1.0,
                       np.where(pred <= -cfg.min_abs_pred, -1.0, 0.0))
    else:
        pos = np.where(pred >= cfg.min_abs_pred, 1.0, 0.0)

    if realized_vol is not None and len(realized_vol) == n:
        vol = np.maximum(np.asarray(realized_vol, dtype=float), 1e-6)
        scale = (cfg.target_ann_vol / np.sqrt(365.0)) / vol
        scale = np.clip(scale, 0.0, 3.0)
    else:
        scale = np.ones(n)
    pos = pos * scale

    turnover = np.abs(np.diff(pos, prepend=0.0))
    cost = turnover * _cost_frac(cfg)
    gross = pos * actual
    net = gross - cost

    equity = cfg.initial_equity * np.cumprod(1.0 + net)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = float(dd.min()) if len(dd) else 0.0

    def _sharpe(x: np.ndarray) -> float:
        if len(x) < 2 or np.std(x) < 1e-12:
            return 0.0
        return float(np.mean(x) / np.std(x) * np.sqrt(365.0))

    active = pos != 0
    hit = 0.5
    if active.any():
        hit = float((np.sign(pred[active]) == np.sign(actual[active])).mean())

    return {
        "mode": cfg.mode,
        "net_sharpe": _sharpe(net),
        "gross_sharpe": _sharpe(gross),
        "max_drawdown": max_dd,
        "n_trades": int((turnover > 0).sum()),
        "hit_rate": hit,
        "mean_net_return": float(np.mean(net)),
        "final_equity": float(equity[-1]),
        "equity": equity.tolist(),
        "cost_drag": float(np.mean(cost)),
    }


def select_best_mode(
    actual_returns: np.ndarray,
    predicted_returns: np.ndarray,
    *,
    fee_bps_round_trip: float = 50.0,
    slippage_bps: float = 10.0,
    min_abs_pred: float = 0.015,
    realized_vol: np.ndarray | None = None,
) -> dict:
    """Run long_flat and long_short; return the better net Sharpe result + both."""
    results = {}
    for mode in ("long_flat", "long_short"):
        cfg = PaperConfig(
            fee_bps_round_trip=fee_bps_round_trip,
            slippage_bps=slippage_bps,
            min_abs_pred=min_abs_pred,
            mode=mode,
        )
        results[mode] = run_paper(actual_returns, predicted_returns, cfg, realized_vol)
    best = max(results.values(), key=lambda r: r["net_sharpe"])
    return {"best": best, "modes": results}
