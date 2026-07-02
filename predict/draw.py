"""
Calibrated ticket-draw estimation from LOFI's own event corpus.

Replaces "the LLM reasons over comparables and picks a number" with empirical
quantiles over matched events — a statistical model that can be backtested,
so the conservative/base/high range carries a measured error bar instead of
vibes. The LLM's job becomes explaining this number (see agents/core.py),
not inventing its own.

Method: the artist's OWN past LOFI draw is the strongest signal; with few own
events it is shrunk toward the genre-matched comparable pool (James-Stein
style weight n/(n+K)). `backtest()` runs leave-one-artist-out over the corpus
and reports interval coverage + median absolute percentage error, so "80% of
actuals fall inside the range" is a claim we verify, not assert.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scout.lofi_events import (  # noqa: E402
    load_artist_lofi_history, load_comparable_events)

_QS = (0.2, 0.5, 0.8)      # conservative / base / high
_SHRINK_K = 3.0            # own-history weight = n_own / (n_own + K)
_MIN_COMPARABLES = 3       # below this a range is not grounded — say so


def quantile(values: list[float], q: float) -> float | None:
    """Linear-interpolated quantile, dependency-free."""
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    if len(vals) == 1:
        return float(vals[0])
    pos = q * (len(vals) - 1)
    lo, hi = int(pos), min(int(pos) + 1, len(vals) - 1)
    return float(vals[lo] + (vals[hi] - vals[lo]) * (pos - lo))


def _range_from(values: list[float]) -> dict | None:
    qs = [quantile(values, q) for q in _QS]
    if any(v is None for v in qs):
        return None
    return {"conservative": round(qs[0]), "base": round(qs[1]),
            "high": round(qs[2])}


def estimate_draw(own_tickets: list[int], comp_tickets: list[int]) -> dict:
    """Pure core: quantile ranges from own + comparable ticket counts, shrunk
    together by how much own history exists. Returns method 'insufficient'
    when neither side is grounded enough for a defensible range."""
    own_rng = _range_from(own_tickets) if own_tickets else None
    comp_rng = (_range_from(comp_tickets)
                if len(comp_tickets) >= _MIN_COMPARABLES else None)

    if own_rng and comp_rng:
        w = len(own_tickets) / (len(own_tickets) + _SHRINK_K)
        blended = {k: round(w * own_rng[k] + (1 - w) * comp_rng[k])
                   for k in own_rng}
        method = "own_history+comparables"
        rng = blended
    elif own_rng:
        rng, method = own_rng, "own_history"
    elif comp_rng:
        rng, method = comp_rng, "comparables"
    else:
        return {"conservative": None, "base": None, "high": None,
                "method": "insufficient",
                "n_own": len(own_tickets), "n_comparables": len(comp_tickets)}

    rng.update({"method": method, "n_own": len(own_tickets),
                "n_comparables": len(comp_tickets)})
    return rng


def estimate_draw_for(name: str, genres, own: dict | None = None,
                      comps: list[dict] | None = None) -> dict:
    """Loader wrapper: pull own LOFI history + genre-matched comparables from
    the events corpus, then estimate. `own`/`comps` are injectable for tests."""
    if own is None:
        own = load_artist_lofi_history(name)
    if comps is None:
        comps = load_comparable_events(name, genres, limit=20)
    own_tix = [e["actual_tickets"] for e in (own.get("events") or [])
               if e.get("actual_tickets")]
    comp_tix = [e["actual_tickets"] for e in (comps or [])
                if e.get("actual_tickets")]
    return estimate_draw(own_tix, comp_tix)


# ── calibration: is the range honest? ─────────────────────────────────────────

def backtest(events: list[dict], min_events_per_artist: int = 1) -> dict:
    """Leave-one-artist-out over the corpus: predict each artist's events from
    comparables only (their own rows held out) and measure how often the
    actual landed inside [conservative, high], plus the median APE of `base`.
    This is the measured error bar the UI shows next to every estimate."""
    from scout.genre import norm as _norm
    from scout.genre import parse_genres as _pg

    by_artist: dict[str, list[dict]] = {}
    for e in events:
        by_artist.setdefault(_norm(e.get("artist_name")), []).append(e)

    n = covered = 0
    apes: list[float] = []
    for key, own in by_artist.items():
        if len(own) < min_events_per_artist:
            continue
        want = set()
        for e in own:
            want |= _pg(e.get("genre"))
        pool = [e["actual_tickets"] for e in events
                if _norm(e.get("artist_name")) != key
                and (not want or (want & _pg(e.get("genre"))))]
        est = estimate_draw([], pool)
        if est["method"] == "insufficient":
            continue
        for e in own:
            actual = e.get("actual_tickets")
            if not actual:
                continue
            n += 1
            if est["conservative"] <= actual <= est["high"]:
                covered += 1
            if est["base"]:
                apes.append(abs(actual - est["base"]) / actual * 100.0)

    apes.sort()
    return {
        "n_events": n,
        "interval_coverage_pct": round(covered / n * 100.0, 1) if n else None,
        "median_ape_pct": round(apes[len(apes) // 2], 1) if apes else None,
        "target_coverage_pct": round((_QS[2] - _QS[0]) * 100.0, 1),
    }
