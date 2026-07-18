"""
Shared plumbing for the detectors (and the sentinel that weighs their output).

The detectors are pure-Python processes over the `signal` spine of the
blackboard. This module holds the small, boring things they all need — the
signal-kind vocabulary, a Record→view normaliser, a population z-score — so
the detection math in each module stays clean and testable. No I/O here.
"""
from __future__ import annotations

import datetime as _dt
import math

from osk.blackboard import parse_ts

# Raw listener signals (the Phase B contract). Detectors read these.
LISTENER_KINDS = ("sc_plays_velocity", "playlist_add", "label_signing",
                  "press_mention", "pf_fans_delta")

# Derived signals the detectors emit. The sentinel weighs these too.
DERIVED_KINDS = ("divergence", "ignition", "venue_ladder", "scene_centrality",
                 "fee_lag")


def artist_key(record) -> str | None:
    """Stable per-artist key: id when we have it (post-database artists),
    else the name (the pre-database window §4 cares about). None = un-keyable."""
    return record.artist_id or record.artist_name or None


def record_when(record) -> _dt.datetime | None:
    """When the signal was observed — payload.observed_at wins, created_at
    is the fallback. Always tz-aware (parse_ts guarantees it)."""
    payload = record.payload or {}
    return parse_ts(payload.get("observed_at")) or parse_ts(record.created_at)


def signal_views(records, kinds: tuple[str, ...] | None = None) -> list[dict]:
    """Normalise blackboard Records into the flat dicts the pure functions
    take: {id, key, artist_id, artist_name, kind, value, when, explain}.
    `kinds` optionally filters by payload kind (e.g. only listener signals)."""
    out = []
    for r in records or []:
        if r.kind != "signal":
            continue
        payload = r.payload or {}
        kind = payload.get("kind")
        if kinds is not None and kind not in kinds:
            continue
        key = artist_key(r)
        if not key:
            continue
        out.append({
            "id": r.id,
            "key": key,
            "artist_id": r.artist_id,
            "artist_name": r.artist_name or key,
            "kind": kind,
            "value": payload.get("value"),
            "when": record_when(r),
            "explain": payload.get("explain"),
            "source": payload.get("source"),
        })
    return out


def zmap(values_by_key: dict) -> dict:
    """Population z-score of a {key: value} map. <2 members → all zeros (no
    variance to speak of); zero variance → sd 1.0 (mirrors predict/twins)."""
    keys = list(values_by_key)
    if len(keys) < 2:
        return {k: 0.0 for k in keys}
    vals = [float(values_by_key[k]) for k in keys]
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    sd = math.sqrt(var) or 1.0
    return {k: (float(values_by_key[k]) - mean) / sd for k in keys}


def id_index(views: list[dict]) -> dict:
    """key → {artist_id, artist_name, ids:[record ids]} for emission wiring."""
    idx: dict = {}
    for v in views:
        e = idx.setdefault(v["key"], {
            "artist_id": v["artist_id"], "artist_name": v["artist_name"],
            "ids": []})
        if v["id"] is not None:
            e["ids"].append(v["id"])
    return idx


HONEST_EMPTY = "no data yet — signals accrue nightly"
