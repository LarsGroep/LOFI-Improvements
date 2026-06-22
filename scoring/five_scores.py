"""
Heuristic five-score engine for LOFI artist intelligence.

Scores: Momentum, Growth, Market Relevance, Future Potential, Confidence.
All outputs are 0-100. Inputs are flat profile + ml_features dicts already
loaded from Supabase — no DB calls here.

Principle (CLAUDE.md): acceleration over level. sp_listeners_accel (second
derivative of growth rate) is the primary signal for Growth and Future Potential.
"""
from __future__ import annotations

import math


def _clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    try:
        return max(lo, min(hi, float(v)))
    except (TypeError, ValueError):
        return 0.0


def _pct_to_score(pct: float | None, scale: float = 30.0) -> float:
    """Logistic mapping: pct=0 → 50, pct=+scale → ~75, pct=-scale → ~25."""
    if pct is None:
        return 50.0
    return _clamp(50.0 + 50.0 * math.tanh(float(pct) / scale))


def _rank_to_score(rank: float | None, max_rank: float = 200_000.0) -> float:
    """Lower rank number = higher score. 0 means unranked — treated as missing."""
    if rank is None or rank == 0:
        return 0.0
    return _clamp(100.0 * (1.0 - float(rank) / max_rank))


def compute_five_scores(profile: dict, ml: dict) -> dict:
    """
    profile  — from artist_chartmetric_flat
    ml       — ml_features dict from artist_chartmetric

    Returns dict with keys: momentum, growth, market_relevance,
    future_potential, confidence, breakdown.
    """
    # ── raw inputs ───────────────────────────────────────────────────────────
    sp_30   = ml.get("sp_listeners_30d_pct")
    sp_90   = ml.get("sp_listeners_90d_pct")
    sp_180  = ml.get("sp_listeners_180d_pct")
    accel   = ml.get("sp_listeners_accel")
    xpm     = ml.get("cross_platform_momentum_30d")
    plat_g  = ml.get("platforms_growing_30d")
    cpp_30  = ml.get("cpp_score_30d_pct")
    cpp_90  = ml.get("cpp_score_90d_pct")
    cpp_cur = ml.get("cpp_score_current")

    cm_score     = profile.get("cm_artist_score")
    cm_rank      = profile.get("cm_artist_rank")
    fan_rank     = profile.get("fan_base_rank")
    career_stage = profile.get("career_stage_score")
    career_trend = profile.get("career_trend_score")
    sp_listeners = profile.get("spotify_listeners")
    ig_followers = profile.get("instagram_followers")

    # ── 1. Momentum — cross-platform traction right now ──────────────────
    m_sp30  = _pct_to_score(sp_30, scale=20.0)
    m_xpm   = _pct_to_score(xpm, scale=25.0)
    m_plat  = _clamp(plat_g / 5.0 * 100.0) if plat_g is not None else 50.0
    m_cpp30 = _pct_to_score(cpp_30, scale=8.0)

    momentum = _clamp(
        0.35 * m_sp30 + 0.30 * m_xpm + 0.20 * m_plat + 0.15 * m_cpp30
    )

    # ── 2. Growth — acceleration signal (second derivative primary) ───────
    g_accel = _clamp(50.0 + (accel or 0.0) * 2.0)
    g_sp30  = _pct_to_score(sp_30, scale=20.0)
    g_trend = _clamp((career_trend or 0.0) * 10.0 + 50.0)

    growth = _clamp(0.50 * g_accel + 0.30 * g_sp30 + 0.20 * g_trend)

    # ── 3. Market Relevance — current standing ───────────────────────────
    r_cm    = _clamp(float(cm_score or 0))
    r_rank  = _rank_to_score(cm_rank)
    r_fan   = _rank_to_score(fan_rank)
    r_cpp   = _clamp((float(cpp_cur or 0) / 10.0) * 100.0)

    market_relevance = _clamp(
        0.35 * r_cm + 0.25 * r_rank + 0.25 * r_fan + 0.15 * r_cpp
    )

    # ── 4. Future Potential — long-term trajectory ───────────────────────
    f_180   = _pct_to_score(sp_180, scale=60.0)
    f_accel = _clamp(50.0 + (accel or 0.0) * 1.5)
    f_stage = _clamp((career_stage or 0.0) * 10.0 + 50.0)
    f_cpp90 = _pct_to_score(cpp_90, scale=15.0)

    future_potential = _clamp(
        0.35 * f_180 + 0.30 * f_accel + 0.20 * f_stage + 0.15 * f_cpp90
    )

    # ── 5. Confidence — data coverage quality ────────────────────────────
    fields = [
        sp_30, sp_90, sp_180, accel, xpm, plat_g, cpp_cur,
        cm_score, cm_rank, ig_followers, sp_listeners, career_stage, career_trend,
    ]
    filled = sum(1 for f in fields if f is not None)
    confidence = _clamp(filled / len(fields) * 100.0)

    return {
        "momentum":         round(momentum, 1),
        "growth":           round(growth, 1),
        "market_relevance": round(market_relevance, 1),
        "future_potential": round(future_potential, 1),
        "confidence":       round(confidence, 1),
        "breakdown": {
            "m_sp30d":           round(m_sp30, 1),
            "m_cross_platform":  round(m_xpm, 1),
            "m_platforms_pct":   round(m_plat, 1),
            "m_cpp30d":          round(m_cpp30, 1),
            "g_acceleration":    round(g_accel, 1),
            "g_sp30d":           round(g_sp30, 1),
            "g_career_trend":    round(g_trend, 1),
            "r_cm_score":        round(r_cm, 1),
            "r_cm_rank":         round(r_rank, 1),
            "r_fan_rank":        round(r_fan, 1),
            "r_cpp_current":     round(r_cpp, 1),
            "f_sp180d":          round(f_180, 1),
            "f_accel":           round(f_accel, 1),
            "f_career_stage":    round(f_stage, 1),
            "f_cpp90d":          round(f_cpp90, 1),
            "data_fields_filled": filled,
            "data_fields_total":  len(fields),
        },
    }
