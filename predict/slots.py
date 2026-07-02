"""
Slot-level fit — "book" is not one decision.

FEEDBACK.md item 9: a headliner at full capacity and an 01:00 support slot
are different bets. Map the calibrated draw estimate onto the room and say
WHICH slot the artist is a bet for, with the occupancy math shown.

Venue capacity via LOFI_CAPACITY (default 500), injectable for tests.
"""
from __future__ import annotations

import os

_SLOT_LADDER = [
    (0.75, "headliner", "their base draw carries the room"),
    (0.45, "co_headliner", "strong draw — pair with one comparable name"),
    (0.20, "support", "adds real bodies, needs a headliner above them"),
    (0.0, "opener", "booking is a bet on the trajectory, not tonight's draw"),
]


def capacity() -> float:
    try:
        return float(os.environ.get("LOFI_CAPACITY", "500"))
    except ValueError:
        return 500.0


def slot_fit(draw: dict, cap: float | None = None) -> dict:
    """draw = the estimate from predict.draw. Returns the recommended slot,
    the occupancy share it's based on, and the too-big guard (conservative
    draw already over the room = they've outgrown us)."""
    if not draw or draw.get("base") is None:
        return {"slot": None, "method": "insufficient"}
    cap = cap or capacity()
    if cap <= 0:
        return {"slot": None, "method": "insufficient"}

    if (draw.get("conservative") or 0) > cap * 1.15:
        return {"slot": "too_big", "occupancy_base": round(draw["base"] / cap, 2),
                "note": "even the conservative draw overfills the room — "
                        "they've likely outgrown this capacity", "method": "modelled"}

    share = draw["base"] / cap
    for threshold, slot, note in _SLOT_LADDER:
        if share >= threshold:
            return {"slot": slot, "occupancy_base": round(share, 2),
                    "note": note, "method": "modelled"}
    return {"slot": None, "method": "insufficient"}
