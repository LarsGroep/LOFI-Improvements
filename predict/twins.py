"""
Trajectory twins — path-alike artists, not sound-alike name lists.

FEEDBACK.md item 4: the persuasive booking artefact is "this artist looks
like Toman a year ago — booked for €X, sold out". Last.fm/Chartmetric
similar-artists match SOUND; this matches the SHAPE of the trajectory
(audience size + 30/90/180d growth + acceleration + CPP movement) against
artists LOFI actually booked, and attaches their real LOFI outcome
(draw from the events corpus, gage from Airtable).

Nearest-neighbour on z-scored features with missing-value masking — small
pool, no training step, fully explainable: every twin shows WHY it matched.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# feature key → (source dict, readable label). Log-scale the size feature so
# a 1M-listener headliner doesn't dwarf every growth signal.
_FEATURES = [
    ("log_listeners", "audience size"),
    ("sp_listeners_30d_pct", "30d growth"),
    ("sp_listeners_90d_pct", "90d growth"),
    ("sp_listeners_180d_pct", "180d growth"),
    ("sp_listeners_accel", "acceleration"),
    ("cpp_score_30d_pct", "CPP 30d"),
]


def trajectory_vector(profile: dict, ml: dict) -> dict[str, float | None]:
    """The shape of an artist's trajectory as a feature dict (None = missing)."""
    ml = ml or {}
    listeners = (profile or {}).get("spotify_listeners")
    return {
        "log_listeners": (math.log10(listeners + 1)
                          if listeners and listeners > 0 else None),
        "sp_listeners_30d_pct": ml.get("sp_listeners_30d_pct"),
        "sp_listeners_90d_pct": ml.get("sp_listeners_90d_pct"),
        "sp_listeners_180d_pct": ml.get("sp_listeners_180d_pct"),
        "sp_listeners_accel": ml.get("sp_listeners_accel"),
        "cpp_score_30d_pct": ml.get("cpp_score_30d_pct"),
    }


def _zstats(pool: list[dict]) -> dict[str, tuple[float, float]]:
    stats = {}
    for key, _ in _FEATURES:
        vals = [v[key] for v in pool if v.get(key) is not None]
        if len(vals) < 2:
            stats[key] = (0.0, 1.0)
            continue
        mean = sum(vals) / len(vals)
        var = sum((x - mean) ** 2 for x in vals) / len(vals)
        stats[key] = (mean, math.sqrt(var) or 1.0)
    return stats


def _distance(a: dict, b: dict, stats: dict) -> float | None:
    """Masked z-scored euclidean: only dimensions both artists have; needs at
    least 3 shared dimensions to count as comparable at all."""
    total, dims = 0.0, 0
    for key, _ in _FEATURES:
        va, vb = a.get(key), b.get(key)
        if va is None or vb is None:
            continue
        mean, sd = stats[key]
        total += ((va - mean) / sd - (vb - mean) / sd) ** 2
        dims += 1
    if dims < 3:
        return None
    return math.sqrt(total / dims)


def find_twins(target: dict, pool: list[dict], k: int = 5) -> list[dict]:
    """Rank pool entries (each: {name, vector, outcome}) by trajectory
    similarity to `target` (a trajectory_vector). Similarity in (0, 1]."""
    vectors = [p["vector"] for p in pool] + [target]
    stats = _zstats(vectors)
    scored = []
    for p in pool:
        d = _distance(target, p["vector"], stats)
        if d is None:
            continue
        scored.append((d, p))
    scored.sort(key=lambda t: t[0])
    out = []
    for d, p in scored[:k]:
        out.append({
            "name": p.get("name"),
            "similarity": round(1.0 / (1.0 + d), 3),
            "outcome": p.get("outcome") or {},
        })
    return out


def booked_pool(flat_profiles: list[dict], ml_by_id: dict) -> list[dict]:
    """The comparison pool: artists LOFI has booked, with their trajectory
    vector and real LOFI outcome (own draw + last gage) attached."""
    from scout.airtable import load_artist_record
    from scout.lofi_events import load_artist_lofi_history

    pool = []
    for p in flat_profiles or []:
        if not p.get("lofi_booked"):
            continue
        name = p.get("artist_name") or ""
        if not name:
            continue
        hist = load_artist_lofi_history(name, limit=3)
        agg = hist.get("aggregate") or {}
        rec = load_artist_record(name) or {}
        pool.append({
            "name": name,
            "vector": trajectory_vector(p, (ml_by_id or {}).get(p.get("artist_id"))),
            "outcome": {
                "avg_tickets": agg.get("avg_tickets"),
                "events_played": agg.get("event_count"),
                "last_fee_paid": rec.get("last_fee_paid"),
                "fee_range": rec.get("fee_range"),
            },
        })
    return pool


def twins_for(candidate_profile: dict, candidate_ml: dict,
              flat_profiles: list[dict], ml_by_id: dict, k: int = 5) -> list[dict]:
    """Loader wrapper: 'who did LOFI book that looked like THIS on the way up?'"""
    target = trajectory_vector(candidate_profile, candidate_ml)
    return find_twins(target, booked_pool(flat_profiles, ml_by_id), k=k)
