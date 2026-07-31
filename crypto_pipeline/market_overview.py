"""Market overview helpers for the forecast UI.

Technical bias scale, risk summary, holdout scorecard, and evidence bullets.
Callers: crypto_pipeline.horizons.live_price_forecasts; tests.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def technical_overview(feat: pd.DataFrame) -> dict[str, Any]:
    """Bullish vs bearish technical scale from indicator votes.

    Votes from RSI, MACD, MA50/200 cross, short return. Score in [-1, +1].
    """
    row = feat.iloc[-1]
    votes: list[tuple[str, float, str]] = []

    rsi = float(row.get("RSI", np.nan))
    if np.isfinite(rsi):
        if rsi >= 70:
            votes.append(("RSI", -0.6, f"RSI {rsi:.0f} overbought"))
        elif rsi <= 30:
            votes.append(("RSI", 0.6, f"RSI {rsi:.0f} oversold"))
        elif rsi >= 55:
            votes.append(("RSI", 0.25, f"RSI {rsi:.0f} mildly bullish"))
        elif rsi <= 45:
            votes.append(("RSI", -0.25, f"RSI {rsi:.0f} mildly bearish"))
        else:
            votes.append(("RSI", 0.0, f"RSI {rsi:.0f} neutral"))

    macd = float(row.get("MACD", np.nan))
    if np.isfinite(macd):
        side = 0.35 if macd > 0 else -0.35
        votes.append(("MACD", side, f"MACD {'positive' if macd > 0 else 'negative'}"))

    ma_cross = float(row.get("MA50_vs_MA200", np.nan))
    if np.isfinite(ma_cross):
        if ma_cross > 0.01:
            votes.append(("MA cross", 0.45, "MA50 above MA200 (golden-cross side)"))
        elif ma_cross < -0.01:
            votes.append(("MA cross", -0.45, "MA50 below MA200 (death-cross side)"))
        else:
            votes.append(("MA cross", 0.0, "MA50 ≈ MA200"))

    ret = float(row.get("Return_1d", np.nan))
    if np.isfinite(ret):
        votes.append(("1d return", float(np.clip(ret * 8.0, -0.3, 0.3)),
                      f"Last day {ret * 100:+.2f}%"))

    fg = float(row.get("FearGreed", np.nan)) if "FearGreed" in feat.columns else np.nan
    if np.isfinite(fg):
        if fg >= 75:
            votes.append(("Fear&Greed", -0.2, f"Fear&Greed {fg:.0f} extreme greed"))
        elif fg <= 25:
            votes.append(("Fear&Greed", 0.2, f"Fear&Greed {fg:.0f} extreme fear"))
        else:
            votes.append(("Fear&Greed", 0.0, f"Fear&Greed {fg:.0f}"))

    if not votes:
        return {
            "bias": "neutral",
            "score": 0.0,
            "scale_0_100": 50,
            "signals": [],
            "summary": "Insufficient indicators for a technical overview.",
        }

    score = float(np.clip(np.mean([v[1] for v in votes]), -1.0, 1.0))
    if score >= 0.2:
        bias = "bullish"
    elif score <= -0.2:
        bias = "bearish"
    else:
        bias = "neutral"
    scale = int(round((score + 1.0) * 50.0))
    signals = [{"name": n, "vote": v, "note": note} for n, v, note in votes]
    return {
        "bias": bias,
        "score": round(score, 4),
        "scale_0_100": scale,
        "signals": signals,
        "summary": f"Technical overview: {bias} ({scale}/100 bullish scale).",
    }


def risk_analysis(feat: pd.DataFrame) -> dict[str, Any]:
    """Risk summary from realized vol + ATR."""
    row = feat.iloc[-1]
    vol = float(row.get("Volatility", np.nan))
    atr = float(row.get("ATR14", np.nan))
    close = float(row.get("Close", np.nan))
    atr_pct = (atr / close) if (np.isfinite(atr) and np.isfinite(close) and close > 0) else np.nan

    vol_series = feat["Volatility"].dropna().tail(90) if "Volatility" in feat.columns else pd.Series(dtype=float)
    vol_pctile = float(vol_series.rank(pct=True).iloc[-1]) if len(vol_series) >= 10 else 0.5

    if vol_pctile >= 0.8 or (np.isfinite(atr_pct) and atr_pct >= 0.05):
        level = "high"
    elif vol_pctile <= 0.3:
        level = "low"
    else:
        level = "medium"

    return {
        "level": level,
        "volatility": None if not np.isfinite(vol) else round(vol, 6),
        "atr_pct": None if not np.isfinite(atr_pct) else round(atr_pct * 100.0, 3),
        "vol_percentile_90d": round(vol_pctile, 3),
        "summary": f"Risk: {level} (90d vol percentile {vol_pctile:.0%}).",
    }


def prediction_scorecard(
    directional_accuracy: float,
    wf_pass_frac: float,
    beats_persistence: bool,
    n_test: int | None = None,
) -> dict[str, Any]:
    """Holdout accuracy disclosure (not a live guarantee)."""
    hit = float(directional_accuracy)
    n_ok = int(round(hit * n_test)) if n_test else None
    grade = "strong" if hit >= 0.58 and beats_persistence else (
        "mixed" if hit >= 0.52 else "weak"
    )
    return {
        "holdout_hit_rate": round(hit, 4),
        "holdout_n": n_test,
        "holdout_correct_est": n_ok,
        "wf_pass_frac": round(float(wf_pass_frac), 4),
        "beats_persistence": bool(beats_persistence),
        "grade": grade,
        "disclaimer": (
            "Hit rate = chronological holdout directional accuracy for this horizon — "
            "not a guarantee of the live forecast. Correct/Incorrect labels apply only "
            "after the horizon resolves ."
        ),
    }


def build_reasoning(
    *,
    horizon_label: str,
    pred_return_pct: float,
    p_up: float,
    model: str,
    tech: dict[str, Any],
    risk: dict[str, Any],
    scorecard: dict[str, Any],
    trustworthy: bool,
) -> list[str]:
    """Evidence bullets for a live forecast row (plain language)."""
    direction = "up" if pred_return_pct >= 0 else "down"
    return [
        f"{horizon_label}: model {model} projects {pred_return_pct:+.2f}% ({direction}).",
        f"Calibrated P(up) = {p_up:.0%}; confidence from |P(up)−50%|.",
        tech.get("summary", "Technical overview unavailable."),
        risk.get("summary", "Risk overview unavailable."),
        (
            f"Holdout direction hit rate {scorecard['holdout_hit_rate']:.0%}"
            + (f" (≈{scorecard['holdout_correct_est']}/{scorecard['holdout_n']})"
               if scorecard.get("holdout_n") else "")
            + f"; walk-forward pass {scorecard['wf_pass_frac']:.0%}; "
            f"persistence beat={'yes' if scorecard['beats_persistence'] else 'no'}."
        ),
        (
            "Trustworthy gate: PASSED."
            if trustworthy
            else "Trustworthy gate: FAILED — treat as research / low-confidence only."
        ),
    ]


def enrich_live_row(
    row: dict[str, Any],
    feat_live: pd.DataFrame,
    *,
    n_test: int | None = None,
) -> dict[str, Any]:
    """Attach overview / risk / scorecard / reasoning to a live forecast row."""
    tech = technical_overview(feat_live)
    risk = risk_analysis(feat_live)
    scorecard = prediction_scorecard(
        directional_accuracy=float(row.get("holdout_directional_accuracy", 0.5)),
        wf_pass_frac=float(row.get("wf_pass_frac", 0.0)),
        beats_persistence=bool(row.get("holdout_beats_persistence", False)),
        n_test=n_test,
    )
    reasoning = build_reasoning(
        horizon_label=str(row.get("horizon_label", "")),
        pred_return_pct=float(row.get("predicted_return_pct", 0.0)),
        p_up=float(row.get("direction_prob_up", 0.5)),
        model=str(row.get("model", "")),
        tech=tech,
        risk=risk,
        scorecard=scorecard,
        trustworthy=bool(row.get("trustworthy", False)),
    )
    out = dict(row)
    out["technical_bias"] = tech["bias"]
    out["technical_scale"] = tech["scale_0_100"]
    out["technical_summary"] = tech["summary"]
    out["technical_signals"] = tech["signals"]
    out["risk_level"] = risk["level"]
    out["risk_summary"] = risk["summary"]
    out["atr_pct"] = risk["atr_pct"]
    out["scorecard"] = scorecard
    out["reasoning"] = reasoning
    return out
