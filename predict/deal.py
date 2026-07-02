"""
The deal sheet — every model's answer to one booking question, in one dict.

This is the artefact FEEDBACK.md asked for: predicted fee next to predicted
draw next to predicted margin, with slot fit, booking-window economics,
trajectory twins and (optionally) routing. Framework-agnostic: the Streamlit
panel renders it, `build_validation_view` embeds it so the LLM anchors its
numbers to the models instead of inventing them, and tests can call it with
injected parts.

Every section degrades independently: no Airtable → fee says insufficient,
no events zip → draw says insufficient; nothing here ever raises.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from predict import draw as _draw          # noqa: E402
from predict import fees as _fees          # noqa: E402
from predict import slots as _slots        # noqa: E402
from predict import twins as _twins        # noqa: E402
from predict import window as _window      # noqa: E402
from predict.routing import routing_signal  # noqa: E402


def _safe(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception:
        return {"method": "error"}


def build_deal_sheet(name: str, genres, profile: dict | None = None,
                     ml: dict | None = None,
                     flat_profiles: list[dict] | None = None,
                     ml_by_id: dict | None = None,
                     include_routing: bool = False) -> dict:
    """One artist → the full economic picture. `flat_profiles`/`ml_by_id`
    enable the trajectory-twins section (the Scout page has them loaded
    anyway); `include_routing` gates the external Bandsintown call so it only
    runs on demand, never on page load."""
    profile, ml = profile or {}, ml or {}

    from scout.lofi_events import load_artist_lofi_history, load_comparable_events
    own = _safe(load_artist_lofi_history, name) or {}
    comps = _safe(load_comparable_events, name, genres, limit=20) or []
    if isinstance(comps, dict):        # _safe error sentinel
        comps = []

    draw = _safe(_draw.estimate_draw_for, name, genres, own=own, comps=comps)

    comp_tix = [e.get("actual_tickets") for e in comps if e.get("actual_tickets")]
    comp_draw_base = _draw.quantile(comp_tix, 0.5) if comp_tix else None
    fee = _safe(_fees.estimate_fee_for, name, genres, draw=draw,
                comp_draw_base=comp_draw_base)

    margin = _safe(_fees.booking_margin, fee, draw)
    slot = _safe(_slots.slot_fit, draw)
    window = _safe(_window.booking_window,
                   profile.get("spotify_listeners"), ml,
                   fee.get("base") if isinstance(fee, dict) else None,
                   draw_base=draw.get("base") if isinstance(draw, dict) else None,
                   capacity=_slots.capacity())

    sheet = {
        "artist_name": name,
        "draw_estimate": draw,          # tickets: conservative/base/high + basis
        "fee_estimate": fee,            # EUR: low/base/high + basis
        "margin_estimate": margin,      # EUR per scenario (door − fee)
        "slot_fit": slot,               # headliner/support/… vs LOFI_CAPACITY
        "booking_window": window,       # cost of waiting + outgrow risk
    }
    if flat_profiles is not None:
        sheet["trajectory_twins"] = _safe(
            _twins.twins_for, profile, ml, flat_profiles, ml_by_id or {}, 4)
    if include_routing:
        sheet["routing"] = _safe(routing_signal, name)
    return sheet


def deal_sheet_for_candidate(c: dict, flat_by_id: dict | None = None,
                             ml: dict | None = None,
                             flat_profiles: list[dict] | None = None) -> dict:
    """Convenience for the Scout page's candidate dicts."""
    profile = (flat_by_id or {}).get(c.get("artist_id")) or {}
    return build_deal_sheet(
        c.get("artist_name") or "", c.get("genres") or [], profile=profile,
        ml=(ml or {}).get(c.get("artist_id")) or {},
        flat_profiles=flat_profiles, ml_by_id=ml)
