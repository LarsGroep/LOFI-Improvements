"""
First-signal combinatorics — the classic ignition pattern (§5.5).

A label signing + a first editorial playlist + first press inside 90 days is
the co-occurrence that precedes a breakout far more often than any single
metric moving. This detector counts DISTINCT milestone kinds within a rolling
90-day window per artist; ≥2 distinct kinds fires an ignition signal whose
value is the count and whose explain names the combo and its dates.

Idempotent by construction: the agent remembers the last count emitted per
artist (ctx.state) and only re-emits when the combo GROWS — a third milestone
landing is news, the same two milestones seen again tomorrow are not.
"""
from __future__ import annotations

import datetime as _dt
import os

from osk.detectors._common import HONEST_EMPTY, id_index, signal_views
from osk.manifest import AgentManifest, Budget

# The milestone categories. sc_plays_velocity counts only above a floor (a
# trickle of plays is not a milestone); the others count on presence.
MILESTONE_KINDS = ("label_signing", "playlist_add", "press_mention",
                   "sc_plays_velocity")

DEFAULT_WINDOW_DAYS = 90
DEFAULT_MIN_KINDS = 2


def _sc_min() -> float:
    try:
        return float(os.environ.get("LOFI_OS_IGNITION_SC_MIN", "0.0"))
    except ValueError:
        return 0.0


# ── detection (pure) ──────────────────────────────────────────────────────────

def detect_ignition(signals: list[dict], now: _dt.datetime,
                    window_days: int = DEFAULT_WINDOW_DAYS,
                    sc_min: float = 0.0,
                    min_kinds: int = DEFAULT_MIN_KINDS) -> list[dict]:
    """`signals` are dicts {artist, kind, value, when}. Returns one dict per
    artist with ≥ min_kinds distinct milestones inside the window, carrying the
    earliest date seen for each kind."""
    horizon = now - _dt.timedelta(days=window_days)
    by_artist: dict = {}
    for s in signals or []:
        key, kind, when = s.get("artist"), s.get("kind"), s.get("when")
        if not key or kind not in MILESTONE_KINDS or when is None:
            continue
        if when < horizon or when > now:
            continue
        if kind == "sc_plays_velocity" and not (float(s.get("value") or 0.0) > sc_min):
            continue
        seen = by_artist.setdefault(key, {})
        if kind not in seen or when < seen[kind]:
            seen[kind] = when

    out = []
    for key, seen in by_artist.items():
        if len(seen) < min_kinds:
            continue
        combo = sorted(seen)
        dates = {k: seen[k].date().isoformat() for k in combo}
        explain = ("ignition: "
                   + " + ".join(f"{k} ({dates[k]})" for k in combo)
                   + f" within {window_days}d")
        out.append({"artist": key, "value": float(len(seen)),
                    "kinds": combo, "dates": dates, "explain": explain})
    out.sort(key=lambda r: r["value"], reverse=True)
    return out


# ── agent shell ───────────────────────────────────────────────────────────────

class IgnitionDetector:
    manifest = AgentManifest(
        name="ignition_detector",
        description="first-signal combinatorics over a 90-day window",
        reads=("signal",), writes=("signal",),
        schedule="daily@05:10", budget=Budget(max_runs_per_day=2))

    def run(self, ctx) -> str:
        since = (ctx.now - _dt.timedelta(days=DEFAULT_WINDOW_DAYS)
                 ).isoformat(timespec="seconds")
        views = signal_views(ctx.read(["signal"], since=since, limit=5000),
                             kinds=MILESTONE_KINDS)
        if not views:
            return HONEST_EMPTY
        idx = id_index(views)
        signals = [{"artist": v["key"], "kind": v["kind"], "value": v["value"],
                    "when": v["when"]} for v in views]
        results = detect_ignition(signals, ctx.now, sc_min=_sc_min())

        emitted_state = ctx.state_get("emitted") or {}
        new_state = dict(emitted_state)
        emitted = 0
        for r in results:
            key = r["artist"]
            prev = emitted_state.get(key)
            if prev is not None and r["value"] <= prev:
                continue                        # combo hasn't grown — idempotent
            meta = idx[key]
            ctx.emit("signal", {
                "source": "ignition_detector", "kind": "ignition",
                "value": r["value"], "inputs": meta["ids"],
                "explain": r["explain"], "kinds": r["kinds"],
                "dates": r["dates"],
                "observed_at": ctx.now.date().isoformat(),
            }, artist_id=meta["artist_id"], artist_name=meta["artist_name"])
            new_state[key] = r["value"]
            emitted += 1
        ctx.state_set("emitted", new_state)
        return f"{emitted} new ignition signal(s) ({len(results)} artists ≥2 kinds)"
