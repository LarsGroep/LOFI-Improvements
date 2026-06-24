"""
Deterministic scout ranking — framework-agnostic, no LLM.

Reuses scoring.five_scores (the same engine as the dashboard) + the XGBoost
forecast (ml/models/predictions.csv) + the LOFI-feel taxonomy. Produces a ranked
shortlist of unbooked, on-feel artists with a rule-based Dutch rationale.

The LLM layer (Phase 2) will later replace `explain_nl` with richer reasoning;
the ranking + filters here remain the deterministic backbone.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import yaml

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scoring.five_scores import compute_five_scores  # noqa: E402
from scout.genre import norm as _norm  # noqa: E402

_PREDICTIONS_PATH = _ROOT / "ml" / "models" / "predictions.csv"
_TAXONOMY_PATH = _ROOT / "scoring" / "lofi_feel_taxonomy.yaml"


# ── Loading ──────────────────────────────────────────────────────────────────

def load_predictions(path: Path = _PREDICTIONS_PATH) -> dict[str, float]:
    out: dict[str, float] = {}
    if not path.exists():
        return out
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                out[(row.get("artist_id") or "").strip()] = float(
                    row["predicted_growth_90d"])
            except (KeyError, TypeError, ValueError):
                continue
    return out


def load_taxonomy(path: Path = _TAXONOMY_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ── Genre helpers ────────────────────────────────────────────────────────────

def parse_genres(g) -> list[str]:
    if not g:
        return []
    if isinstance(g, list):
        return [str(x).strip().lower() for x in g if x]
    if isinstance(g, str):
        s = g.strip()
        if s.startswith("["):
            import json
            try:
                return [str(x).strip().lower() for x in json.loads(s) if x]
            except Exception:
                pass
        return [p.strip().lower() for p in s.split(",") if p.strip()]
    return []


def genre_match(genres: list[str], taxonomy: dict) -> dict:
    """Normalised genre matching. Tiers match on normalised equality; the
    disqualifying set matches as a substring, so spelled/combined variants
    (hip-hop, hyperpop, k-pop, "korean hip-hop/rap") are still caught."""
    g = taxonomy.get("genres", {})
    t1 = {_norm(x) for x in g.get("tier_1", [])}
    t2 = {_norm(x) for x in g.get("tier_2", [])}
    dq = {_norm(x) for x in g.get("disqualifying", [])}
    pairs = [(_norm(x), x) for x in genres if x]
    return {
        "tier1": [o for n, o in pairs if n in t1],
        "tier2": [o for n, o in pairs if n in t2],
        "disqualifying": [o for n, o in pairs if any(d and d in n for d in dq)],
    }


def is_disqualified(genres: list[str], taxonomy: dict) -> bool:
    """Off-feel: has a disqualifying genre and no core/adjacent match."""
    m = genre_match(genres, taxonomy)
    return bool(m["disqualifying"]) and not (m["tier1"] or m["tier2"])


# ── Candidate construction ───────────────────────────────────────────────────

def build_candidates(flat_profiles: list[dict], ml_by_id: dict[str, dict],
                     predictions: dict[str, float]) -> list[dict]:
    """Compute five scores per *unbooked* artist and attach the forecast."""
    candidates: list[dict] = []
    for p in flat_profiles:
        aid = p.get("artist_id")
        if not aid or p.get("lofi_booked"):
            continue
        ml = ml_by_id.get(aid) or {}
        scores = compute_five_scores(p, ml)
        candidates.append({
            "artist_id": aid,
            "artist_name": p.get("artist_name") or "Onbekend",
            "genres": parse_genres(p.get("genres")),
            "career_status": p.get("career_status") or p.get("career_stage") or "",
            "spotify_listeners": p.get("spotify_listeners"),
            "cm_artist_score": p.get("cm_artist_score"),
            "momentum": scores["momentum"],
            "growth": scores["growth"],
            "market_relevance": scores["market_relevance"],
            "future_potential": scores["future_potential"],
            "confidence": scores["confidence"],
            "forecast_90d": predictions.get(aid),
        })
    return candidates


# ── Ranking + explanation ────────────────────────────────────────────────────

def rank_score(c: dict) -> float:
    """Blend trajectory signals with the XGBoost forecast. Future potential and
    growth lead; the forecast and current momentum round it out."""
    fc = c.get("forecast_90d")
    fc_norm = 50.0 if fc is None else max(0.0, min(100.0, float(fc)))
    return (
        0.35 * c["future_potential"]
        + 0.30 * c["growth"]
        + 0.20 * fc_norm
        + 0.15 * c["momentum"]
    )


def explain_nl(c: dict, taxonomy: dict) -> str:
    """Rule-based Dutch one-liner for the booking team — the deterministic
    stand-in for the LLM rationale that arrives in Phase 2."""
    reasons: list[str] = []

    if c["growth"] >= 70:
        reasons.append("sterke groeiversnelling")
    elif c["growth"] >= 55:
        reasons.append("gestage groei")

    if c["momentum"] >= 75:
        reasons.append("hoog momentum")

    if c["future_potential"] >= 75 and not any("groei" in r for r in reasons):
        reasons.append("hoog potentieel")

    # Forecast — honest about direction, so a decline isn't hidden.
    fc = c.get("forecast_90d")
    if fc is not None:
        if fc >= 25:
            reasons.append(f"forecast +{fc:.0f}%")
        elif fc <= -15:
            reasons.append(f"let op: forecast {fc:.0f}%")

    m = genre_match(c["genres"], taxonomy)
    if m["tier1"]:
        reasons.append(f"kerngenre ({m['tier1'][0]})")
    elif m["tier2"]:
        reasons.append(f"aanverwant ({m['tier2'][0]})")
    elif c["genres"]:
        reasons.append("genre buiten kernprofiel")

    if not reasons:
        reasons.append("nog niet geboekt, in profiel")

    text = ", ".join(reasons[:4])
    return text[0].upper() + text[1:] + "."


def rank_candidates(candidates: list[dict], taxonomy: dict, *,
                    genres: list[str] | None = None,
                    min_confidence: float = 30.0,
                    query: str | None = None,
                    drop_disqualified: bool = True,
                    core_only: bool = False,
                    sort_by: str = "rank",
                    top_n: int | None = None) -> list[dict]:
    """Filter + sort, attaching `rank` and `waarom` to each surviving candidate."""
    sel_genres = {g.lower() for g in (genres or [])}
    q = (query or "").strip().lower()

    pool = []
    for c in candidates:
        if c["confidence"] < min_confidence:
            continue
        if drop_disqualified and is_disqualified(c["genres"], taxonomy):
            continue
        if core_only:
            m = genre_match(c["genres"], taxonomy)
            if not (m["tier1"] or m["tier2"]):
                continue
        if sel_genres and not (set(c["genres"]) & sel_genres):
            continue
        if q and q not in c["artist_name"].lower():
            continue
        pool.append(c)

    for c in pool:
        if "rank" not in c:  # deterministic — skip if already ranked (pre-ranked pool)
            c["rank"] = round(rank_score(c), 1)
            c["waarom"] = explain_nl(c, taxonomy)

    keymap = {
        "rank": "rank",
        "growth": "growth",
        "future_potential": "future_potential",
        "momentum": "momentum",
        "forecast": "forecast_90d",
        "market_relevance": "market_relevance",
    }
    key = keymap.get(sort_by, "rank")
    pool.sort(key=lambda c: (c.get(key) is not None, c.get(key) or 0),
              reverse=True)
    return pool[:top_n] if top_n else pool
