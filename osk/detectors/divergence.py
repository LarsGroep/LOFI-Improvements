"""
Cross-platform divergence — the highest-EV detector (docs/agentic_os.md §5.1).

Underground heat (SoundCloud velocity, press mentions, Partyflock growth, a
label signing as a strong binary) z-scored against the mainstream metric
(Spotify-listener growth, when the Scout candidates happen to be loaded this
tick). A big POSITIVE gap = heat the mainstream number hasn't priced in — the
cheap-booking window. A NEGATIVE gap = playlist-inflated numbers the Skeptic
will want to see later, so those are emitted too, with value < 0.

Detection is a pure function over aggregated per-artist vectors; the agent is
a thin shell that reads signals, optionally borrows mainstream growth from
ctx.shared, and emits. No network of its own — signals accrue nightly.
"""
from __future__ import annotations

import datetime as _dt
import os

from osk.detectors._common import (
    HONEST_EMPTY, LISTENER_KINDS, id_index, signal_views, zmap)
from osk.manifest import AgentManifest, Budget

# Components of the underground vector. playlist_add is deliberately absent —
# it is a mainstream/editorial signal, not underground heat; artists carrying
# only playlist adds still count toward the population, just at component 0.
UNDERGROUND_COMPONENTS = ("sc_plays_velocity", "press_mention",
                          "pf_fans_delta", "label_signing")

DEFAULT_MIN_GAP = 1.0


def _min_gap() -> float:
    try:
        return float(os.environ.get("LOFI_OS_DIVERGENCE_MIN", DEFAULT_MIN_GAP))
    except ValueError:
        return DEFAULT_MIN_GAP


# ── detection (pure) ──────────────────────────────────────────────────────────

def detect_divergence(signals: list[dict], mainstream: dict | None = None,
                      min_gap: float = DEFAULT_MIN_GAP) -> list[dict]:
    """`signals` are flat dicts {artist, kind, value}; `mainstream` maps
    artist → Spotify-listener growth (absent = unknown). Returns one emission
    dict per artist whose |divergence| ≥ min_gap, sorted best gap first."""
    mainstream = mainstream or {}
    agg: dict = {}
    for s in signals or []:
        key, kind = s.get("artist"), s.get("kind")
        if not key or kind not in LISTENER_KINDS:
            continue
        d = agg.setdefault(key, {c: 0.0 for c in UNDERGROUND_COMPONENTS})
        val = s.get("value")
        if kind == "press_mention":
            d["press_mention"] += 1.0
        elif kind == "label_signing":
            d["label_signing"] = 1.0            # strong binary
        elif kind == "sc_plays_velocity":
            d["sc_plays_velocity"] += float(val or 0.0)
        elif kind == "pf_fans_delta":
            d["pf_fans_delta"] += float(val or 0.0)
        # playlist_add: registers the artist in the population, no component

    if len(agg) < 2:                            # nothing to z-score against
        return []

    zc = {c: zmap({a: agg[a][c] for a in agg}) for c in UNDERGROUND_COMPONENTS}
    mz = zmap(mainstream)

    out = []
    for a in agg:
        u = sum(zc[c][a] for c in UNDERGROUND_COMPONENTS) / len(UNDERGROUND_COMPONENTS)
        m = mz.get(a, 0.0)
        gap = u - m
        if abs(gap) < min_gap:
            continue
        if gap >= 0:
            explain = (f"underground signals outrun the mainstream metric by "
                       f"{gap:.1f}σ — cheap-booking window")
        else:
            explain = (f"mainstream metric outpaces underground by {abs(gap):.1f}σ "
                       f"— possible playlist-inflated (flagged for the Skeptic)")
        out.append({
            "artist": a,
            "value": round(gap, 3),
            "components": {c: round(zc[c][a], 3) for c in UNDERGROUND_COMPONENTS},
            "mainstream_present": a in mainstream,
            "explain": explain,
        })
    out.sort(key=lambda r: r["value"], reverse=True)
    return out


def mainstream_from_shared(ctx) -> dict:
    """Spotify-listener growth borrowed from the Scout candidates *iff* another
    agent already loaded them this tick. The detector never sweeps Supabase
    itself — absent is a fine, honest answer."""
    cands = ctx.shared.get("candidates")
    if not cands:
        return {}
    out = {}
    for c in cands:
        key = c.get("artist_id") or c.get("artist_name")
        g = c.get("growth")
        if key and g is not None:
            out[key] = float(g)
    return out


# ── agent shell ───────────────────────────────────────────────────────────────

class DivergenceDetector:
    manifest = AgentManifest(
        name="divergence_detector",
        description="cross-platform underground-vs-mainstream heat gap",
        reads=("signal",), writes=("signal",),
        schedule="daily@05:00", budget=Budget(max_runs_per_day=2))

    def run(self, ctx) -> str:
        since = (ctx.now - _dt.timedelta(days=30)).isoformat(timespec="seconds")
        views = signal_views(ctx.read(["signal"], since=since, limit=5000),
                             kinds=LISTENER_KINDS)
        population = {v["key"] for v in views}
        if len(population) < 2:
            return HONEST_EMPTY
        idx = id_index(views)
        signals = [{"artist": v["key"], "kind": v["kind"], "value": v["value"]}
                   for v in views]
        results = detect_divergence(signals, mainstream_from_shared(ctx),
                                    min_gap=_min_gap())
        for r in results:
            meta = idx[r["artist"]]
            ctx.emit("signal", {
                "source": "divergence_detector", "kind": "divergence",
                "value": r["value"], "inputs": meta["ids"],
                "explain": r["explain"], "components": r["components"],
                "mainstream_present": r["mainstream_present"],
                "observed_at": ctx.now.date().isoformat(),
            }, artist_id=meta["artist_id"], artist_name=meta["artist_name"])
        return (f"{len(results)} divergence signal(s) over "
                f"{len(population)} artists")
