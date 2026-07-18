"""
Venue-tier ladder — the most booking-relevant curve there is (§5.4).

Climbing 100-cap → 300-cap rooms over two quarters is a career change-point no
streaming platform shows. `ladder_trend` splits an artist's dated events into
first/second half and compares median capacity when capacities are known; when
they aren't, it falls back to a weaker proxy (events-per-month + unique venues)
and says which basis it used, so nothing is ever passed off as harder than it
is.

The pure function is the deliverable this phase. The agent stays parked: RA
event rows need a per-artist Supabase sweep that isn't wired yet, so it returns
an honest "skipped" rather than hammering a loader that doesn't exist.
"""
from __future__ import annotations

import datetime as _dt
from statistics import median

from osk.blackboard import parse_ts
from osk.manifest import AgentManifest, Budget

DEFAULT_MIN_EVENTS = 6
DEFAULT_MIN_SPAN_DAYS = 120


def _as_date(value) -> _dt.date | None:
    if isinstance(value, _dt.datetime):
        return value.date()
    if isinstance(value, _dt.date):
        return value
    ts = parse_ts(value)
    return ts.date() if ts else None


# ── detection (pure) ──────────────────────────────────────────────────────────

def ladder_trend(events: list[dict], min_events: int = DEFAULT_MIN_EVENTS,
                 min_span_days: int = DEFAULT_MIN_SPAN_DAYS) -> dict | None:
    """`events`: [{date, venue, city, capacity|None}]. Needs ≥ min_events dated
    events spanning ≥ min_span_days. Returns a trend dict (basis "capacity" or
    "proxy") or None when there simply isn't enough to say anything honest."""
    rows = []
    for e in events or []:
        d = _as_date((e or {}).get("date"))
        if d is None:
            continue
        cap = (e or {}).get("capacity")
        rows.append({"date": d, "venue": (e or {}).get("venue"),
                     "city": (e or {}).get("city"),
                     "capacity": float(cap) if cap else None})
    if len(rows) < min_events:
        return None
    rows.sort(key=lambda r: r["date"])
    span_days = (rows[-1]["date"] - rows[0]["date"]).days
    if span_days < min_span_days:
        return None

    mid = len(rows) // 2
    first, second = rows[:mid], rows[mid:]
    cap_first = [r["capacity"] for r in first if r["capacity"]]
    cap_second = [r["capacity"] for r in second if r["capacity"]]

    if len(cap_first) >= 2 and len(cap_second) >= 2:
        m1, m2 = median(cap_first), median(cap_second)
        return {
            "basis": "capacity",
            "first_median": round(m1), "second_median": round(m2),
            "rising": m2 > m1,
            "value": round(m2 / m1, 3) if m1 else None,
            "n_events": len(rows), "span_days": span_days,
            "explain": (f"median venue capacity {m1:.0f}→{m2:.0f} across "
                        f"{len(rows)} shows in {span_days}d"),
        }

    # weaker proxy — activity + venue spread, explicitly labelled
    def _rate(half):
        span = (half[-1]["date"] - half[0]["date"]).days
        months = max(span / 30.0, 1.0 / 30.0)
        return len(half) / months

    r1, r2 = _rate(first), _rate(second)
    v1 = len({r["venue"] for r in first if r["venue"]})
    v2 = len({r["venue"] for r in second if r["venue"]})
    return {
        "basis": "proxy",
        "first_rate": round(r1, 3), "second_rate": round(r2, 3),
        "first_venues": v1, "second_venues": v2,
        "rising": r2 > r1 or v2 > v1,
        "value": round(r2 / r1, 3) if r1 else None,
        "n_events": len(rows), "span_days": span_days,
        "explain": (f"activity {r1:.1f}→{r2:.1f} shows/mo, unique venues "
                    f"{v1}→{v2} across {span_days}d (no capacity data)"),
    }


# ── agent shell ───────────────────────────────────────────────────────────────

class VenueLadderDetector:
    manifest = AgentManifest(
        name="venue_ladder_detector",
        description="median venue-capacity trend from RA event history",
        reads=("signal",), writes=("signal",),
        schedule="daily@05:20", budget=Budget(max_runs_per_day=2))

    def run(self, ctx) -> str:
        # RA event rows are not reachable in one sweep yet (needs the per-artist
        # loaders of scout/context.py, Phase C+). Park honestly rather than
        # hammer a loader that isn't there; ladder_trend is the deliverable.
        return "skipped: needs per-artist RA sweep (Phase C+)"
