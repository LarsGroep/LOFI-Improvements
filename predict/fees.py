"""
Fee (gage) prediction — draw-adjusted comparable pricing.

FEEDBACK.md item 1: the Airtable join knows what LOFI paid historically;
this module predicts what an artist should cost TODAY and flags an agent
quote that is far above modelled fair value.

Method (deliberately simple and auditable, ~tens of gage samples):
  1. The artist's OWN past LOFI gages are the anchor when they exist.
  2. Otherwise: quantiles of genre-matched comparable gages, scaled by how
     the artist's predicted draw compares to the comparables' typical draw,
     with sub-linear elasticity (fees grow slower than draw — a 2x draw is
     not a 2x fee). Elasticity default 0.7, overridable via LOFI_FEE_ELASTICITY.
Every estimate carries its provenance (n samples, method) so the booker —
and the LLM that explains it — can judge how hard to lean on it.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from predict.draw import quantile  # noqa: E402

_QS = (0.25, 0.5, 0.75)    # low / base / high — tighter than draw: fees are negotiated
_MIN_COMPARABLES = 3


def _elasticity() -> float:
    try:
        return float(os.environ.get("LOFI_FEE_ELASTICITY", "0.7"))
    except ValueError:
        return 0.7


def estimate_fee(own_gages: list[float], comparable_gages: list[float],
                 draw_ratio: float | None = None) -> dict:
    """Pure core. `draw_ratio` = artist's predicted draw / comparables' typical
    draw; applied sub-linearly to the comparable range only (own gages are
    already this artist's market)."""
    own = [g for g in (own_gages or []) if g and g > 0]
    comps = [g for g in (comparable_gages or []) if g and g > 0]

    if own:
        rng = {"low": quantile(own, _QS[0]), "base": quantile(own, _QS[1]),
               "high": quantile(own, _QS[2])}
        method = "own_gages"
    elif len(comps) >= _MIN_COMPARABLES:
        rng = {"low": quantile(comps, _QS[0]), "base": quantile(comps, _QS[1]),
               "high": quantile(comps, _QS[2])}
        method = "comparables"
        if draw_ratio and draw_ratio > 0:
            adj = min(2.0, max(0.5, draw_ratio)) ** _elasticity()
            rng = {k: v * adj for k, v in rng.items()}
            method = "comparables_draw_adjusted"
    else:
        return {"low": None, "base": None, "high": None,
                "method": "insufficient",
                "n_own": len(own), "n_comparables": len(comps)}

    rng = {k: round(v) for k, v in rng.items()}
    rng.update({"method": method, "n_own": len(own), "n_comparables": len(comps)})
    return rng


def estimate_fee_for(name: str, genres, draw: dict | None = None,
                     booking_history: list[dict] | None = None,
                     comparables: list[dict] | None = None,
                     comp_draw_base: float | None = None) -> dict:
    """Loader wrapper over the Airtable layer (graceful-empty when Airtable is
    not configured). `draw` is the estimate from predict.draw; comparables'
    typical draw comes from the same genre-matched events corpus."""
    from scout.airtable import load_booking_history, load_comparables
    if booking_history is None:
        booking_history = load_booking_history(name)
    if comparables is None:
        comparables = load_comparables(name, {}, limit=10, genres=list(genres or []))

    own_gages = [b.get("gage") for b in (booking_history or []) if b.get("gage")]
    comp_gages = [b.get("gage") for b in (comparables or []) if b.get("gage")]

    draw_ratio = None
    if draw and draw.get("base") and comp_draw_base:
        draw_ratio = draw["base"] / comp_draw_base
    return estimate_fee(own_gages, comp_gages, draw_ratio=draw_ratio)


def quote_check(quote: float, est: dict) -> dict:
    """Is an agent's quote fair against the modelled range? The negotiation
    artefact: 'the model prices this at 2.5-4k; 9k is ~2.6x fair value'."""
    if not quote or quote <= 0 or not est or not est.get("base"):
        return {"verdict": "no_model", "ratio": None}
    ratio = quote / est["base"]
    if quote <= (est.get("high") or est["base"]):
        verdict = "fair"
    elif ratio <= 2.0:
        verdict = "above_range"
    else:
        verdict = "well_above_range"
    return {"verdict": verdict, "ratio": round(ratio, 2),
            "fair_range": [est.get("low"), est.get("high")]}


# ── booking economics: fee + draw → expected door margin ─────────────────────

def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, "") or default)
    except ValueError:
        return default


def booking_margin(fee: dict, draw: dict,
                   ticket_price: float | None = None,
                   variable_cost_pct: float | None = None) -> dict:
    """Predicted margin per booking = door revenue − fee, per scenario.
    Venue economics via env: LOFI_TICKET_PRICE (default 15), and
    LOFI_VARIABLE_COST_PCT (default 0.35) for per-head variable costs."""
    if ticket_price is None:
        ticket_price = _env_float("LOFI_TICKET_PRICE", 15.0)
    if variable_cost_pct is None:
        variable_cost_pct = _env_float("LOFI_VARIABLE_COST_PCT", 0.35)
    if (not fee or fee.get("base") is None
            or not draw or draw.get("base") is None):
        return {"method": "insufficient"}

    net = ticket_price * (1 - variable_cost_pct)
    scen = {
        "conservative": draw["conservative"] * net - fee["high"],
        "base": draw["base"] * net - fee["base"],
        "high": draw["high"] * net - fee["low"],
    }
    return {k: round(v) for k, v in scen.items()} | {
        "ticket_price": ticket_price,
        "net_per_head": round(net, 2),
        "method": "modelled",
    }
