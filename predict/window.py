"""
Booking-window economics — the price of waiting.

FEEDBACK.md item 3: "book now / monitor / too early" is qualitative; fees
inflect AFTER growth inflects, and the whole game is booking in the gap.
This module projects the artist's audience forward with damped growth, maps
audience to fee via the same sub-linear elasticity the fee model uses, and
prices the difference: "booking now vs in 4 months ≈ €X, at Y% occupancy
risk they outgrow the room."

All math is pure and injectable — no data loads here.
"""
from __future__ import annotations

import os


def _monthly_growth(ml: dict) -> float:
    """Best monthly listener growth-rate estimate from the ml_features the
    dashboard already computes: 30d pct is primary, 90d pct fills gaps."""
    g30 = ml.get("sp_listeners_30d_pct")
    g90 = ml.get("sp_listeners_90d_pct")
    if g30 is not None:
        return float(g30) / 100.0
    if g90 is not None:
        return (1.0 + float(g90) / 100.0) ** (1.0 / 3.0) - 1.0
    return 0.0


def project_listeners(listeners: float, ml: dict, months: int,
                      damping: float = 0.85) -> float:
    """Compound the current monthly rate forward, geometrically damped —
    hype curves flatten; acceleration (the second derivative the scoring
    engine leans on) slows or speeds the damping a notch."""
    if not listeners or listeners <= 0:
        return 0.0
    rate = _monthly_growth(ml)
    accel = ml.get("sp_listeners_accel")
    if accel is not None:
        damping = min(0.95, max(0.70, damping + float(accel) / 200.0))
    out = float(listeners)
    r = rate
    for _ in range(max(0, months)):
        out *= max(0.2, 1.0 + r)      # a month can't erase >80% of an audience
        r *= damping
    return out


def _elasticity() -> float:
    try:
        return float(os.environ.get("LOFI_FEE_ELASTICITY", "0.7"))
    except ValueError:
        return 0.7


def booking_window(listeners: float | None, ml: dict, fee_base: float | None,
                   draw_base: float | None = None, capacity: float | None = None,
                   horizons: tuple[int, ...] = (3, 6)) -> dict:
    """The decision artefact: projected fee at each horizon, the euro cost of
    waiting, and the risk they outgrow the room (projected draw > capacity).

    Verdicts:
      book_now   growth is compounding and waiting is priced (>10% fee drift)
      no_rush    flat/negative trajectory — waiting is free or profitable
      monitor    growing, but the fee drift is small
      act_fast   projected draw crosses capacity inside the horizon window
    """
    if not listeners or listeners <= 0 or not ml:
        return {"method": "insufficient"}

    ela = _elasticity()
    out: dict = {"horizons": {}, "method": "modelled"}
    max_ratio = 1.0
    outgrow_month = None
    for m in horizons:
        proj = project_listeners(listeners, ml, m)
        ratio = proj / listeners if listeners else 1.0
        max_ratio = max(max_ratio, ratio)
        h: dict = {"projected_listeners": round(proj),
                   "audience_ratio": round(ratio, 2)}
        if fee_base:
            fee_then = fee_base * ratio ** ela
            h["projected_fee"] = round(fee_then)
            h["wait_cost"] = round(fee_then - fee_base)
        if draw_base and capacity:
            proj_draw = draw_base * ratio ** 0.85   # draw tracks audience closely
            h["projected_draw"] = round(proj_draw)
            if proj_draw > capacity and outgrow_month is None:
                outgrow_month = m
        out["horizons"][f"{m}m"] = h

    growth = _monthly_growth(ml)
    if outgrow_month is not None:
        verdict = "act_fast"
        note = (f"projected draw crosses capacity within ~{outgrow_month} "
                "months — after that they play bigger rooms")
    elif growth <= 0.005:
        verdict = "no_rush"
        note = "flat or cooling trajectory — waiting costs nothing"
    elif fee_base and max_ratio ** ela - 1.0 > 0.10:
        verdict = "book_now"
        drift = round(fee_base * (max_ratio ** ela - 1.0))
        note = f"waiting to the far horizon prices in ≈ €{drift} of fee drift"
    else:
        verdict = "monitor"
        note = "growing, but modelled fee drift is small — timing is not the risk"

    out.update({"verdict": verdict, "note": note,
                "monthly_growth_pct": round(growth * 100.0, 1)})
    return out
