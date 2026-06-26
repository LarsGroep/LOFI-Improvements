"""
LOFI Feel Scorer — hybrid taxonomy + embedding + neighboring-network scoring.

Three components (no LLM):
  taxonomy   (35%) — genre/label/agency tier matching against lofi_feel_taxonomy.yaml
  embedding  (35%) — cosine distance to centroid of all 173 booked artist embeddings
  neighboring (30%) — how many LOFI-booked artists list this artist as a
                      CM similar-artist or neighboring-artist (network signal)

Usage:
    python scoring/lofi_scorer.py [--limit N] [--dry-run] [--status pending|all]

Scores stored on tinder.artists.lofi_feel:
    {
        "score":            0-100  (weighted composite),
        "taxonomy_score":   0-100,
        "embedding_score":  0-100,
        "neighboring_score":0-100,
        "matched":          [str],   # taxonomy hits
        "disqualified":     bool,
        "scored_at":        ISO timestamp,
    }
"""
from __future__ import annotations

import argparse
import os
import io
import re
import sys
import time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from datetime import datetime, timezone
from pathlib import Path

import yaml

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv(_ROOT / ".env")

_TAXONOMY_PATH = Path(__file__).parent / "lofi_feel_taxonomy.yaml"


def _load_taxonomy() -> dict:
    with open(_TAXONOMY_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Taxonomy scorer ────────────────────────────────────────────────────────────

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", " ", s.lower()).strip()


def _match(value: str, tier: list[str]) -> bool:
    v = _norm(value)
    return any(_norm(t) in v or v in _norm(t) for t in tier)


def score_taxonomy(cm: dict, taxonomy: dict) -> dict:
    """Return {score: 0-100, matched: [...], disqualified: bool}."""
    genres = [g.lower() for g in (cm.get("genres") or [])]
    label  = (cm.get("record_label") or "").lower()
    agency = (cm.get("booking_agent") or "").lower()

    g_cfg = taxonomy.get("genres", {})
    l_cfg = taxonomy.get("labels", {})
    a_cfg = taxonomy.get("agencies", {})
    tw    = taxonomy.get("taxonomy_weights", {})

    matched = []
    disqualified = False

    g_score = 0
    disq_list = g_cfg.get("disqualifying", [])
    disq_hits: list[str] = []  # collect all disqualifying genre matches
    for g in genres:
        if any(_norm(d) in _norm(g) or _norm(g) in _norm(d) for d in disq_list):
            disq_hits.append(g)

    # Disqualify only if a bad genre appears in the first 3 positions OR
    # 2+ disqualifying tags are present anywhere in the genre list.
    # This prevents one secondary tag (e.g. "emo rap" on an acidcore artist)
    # from triggering a full disqualify.
    primary_disq = [g for g in disq_hits if genres.index(g) < 3]
    if primary_disq or len(disq_hits) >= 2:
        disqualified = True
        for g in disq_hits:
            matched.append(f"DISQUALIFY: {g}")

    for g in genres:
        if any(_norm(t) in _norm(g) or _norm(g) in _norm(t)
               for t in g_cfg.get("tier_1", [])):
            g_score = max(g_score, 100)
            matched.append(f"genre tier-1: {g}")
        elif any(_norm(t) in _norm(g) or _norm(g) in _norm(t)
                 for t in g_cfg.get("tier_2", [])):
            g_score = max(g_score, 60)
            matched.append(f"genre tier-2: {g}")

    l_score = 0
    if label:
        if any(_norm(t) in _norm(label) or _norm(label) in _norm(t)
               for t in l_cfg.get("tier_1", [])):
            l_score = 100
            matched.append(f"label tier-1: {label}")
        elif any(_norm(t) in _norm(label) or _norm(label) in _norm(t)
                 for t in l_cfg.get("tier_2", [])):
            l_score = 60
            matched.append(f"label tier-2: {label}")

    a_score = 0
    if agency:
        if any(_norm(t) in _norm(agency) or _norm(agency) in _norm(t)
               for t in a_cfg.get("tier_1", [])):
            a_score = 100
            matched.append(f"agency tier-1: {agency}")
        elif any(_norm(t) in _norm(agency) or _norm(agency) in _norm(t)
                 for t in a_cfg.get("tier_2", [])):
            a_score = 80
            matched.append(f"agency tier-2: {agency}")
        elif any(_norm(t) in _norm(agency) or _norm(agency) in _norm(t)
                 for t in a_cfg.get("tier_3", [])):
            a_score = 60
            matched.append(f"agency tier-3: {agency}")

    composite = (
        g_score * tw.get("genre",  0.50)
      + l_score * tw.get("label",  0.30)
      + a_score * tw.get("agency", 0.20)
    )
    if disqualified:
        composite = max(0.0, composite - 50)

    return {"score": round(composite), "matched": matched, "disqualified": disqualified}


# ── Embedding scorer ───────────────────────────────────────────────────────────

def score_embedding(cosine_dist: float | None) -> int:
    """Cosine distance [0, 2] → score [0, 100]. Closer to booked centroid = higher."""
    if cosine_dist is None:
        return -1
    clamped = max(0.0, min(1.0, cosine_dist / 0.8))
    return round((1.0 - clamped) * 100)


# ── Neighboring network scorer ─────────────────────────────────────────────────

def score_neighboring(booked_neighbor_count: int, booked_similar_count: int) -> int:
    """Score based on how many LOFI-booked artists reference this artist.

    neighbor references are weighted higher (career-stage proximity + genre filter
    already applied at queue time) than generic similar-artist references.

    Formula: each neighbor ref = 25 pts, each similar ref = 10 pts, cap 100.
    """
    raw = booked_neighbor_count * 25 + booked_similar_count * 10
    return min(100, raw)


# ── Composite ─────────────────────────────────────────────────────────────────

def compute_composite(
    taxonomy_score: int,
    embedding_score: int,
    neighboring_score: int,
    weights: dict,
) -> int:
    """Weighted composite; components scored -1 are skipped (unavailable)."""
    total_w = 0.0
    total   = 0.0
    for score, key in [
        (taxonomy_score,   "taxonomy"),
        (embedding_score,  "embedding"),
        (neighboring_score,"neighboring"),
    ]:
        if score >= 0:
            w = weights.get(key, 0.0)
            total   += score * w
            total_w += w
    if total_w == 0:
        return 0
    return round(total / total_w)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Score artists for LOFI feel (no LLM)")
    parser.add_argument("--limit",  type=int, default=0,
                        help="Max artists to score (0 = all)")
    parser.add_argument("--status", default="pending",
                        help="candidate_status filter: pending|accepted|all (default: pending)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print scores, do not write to DB")
    parser.add_argument("--no-autopromote", action="store_true",
                        help="Skip auto-promotion of high-scoring artists")
    parser.add_argument("--threshold", type=int, default=0,
                        help="Override auto-promote threshold (0 = use taxonomy YAML value)")
    args = parser.parse_args()

    from supabase import create_client
    sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

    taxonomy  = _load_taxonomy()
    weights   = taxonomy.get("weights", {"taxonomy": 0.35, "embedding": 0.35, "neighboring": 0.30})
    threshold = args.threshold or taxonomy.get("auto_promote_threshold", 60)

    # Fetch artists + CM data + reference counts
    q = (
        sb.schema("tinder").table("artists")
        .select("id, name, slug, booked_similar_count, booked_neighbor_count, artist_chartmetric(*)")
    )
    if args.status != "all":
        q = q.eq("candidate_status", args.status)
    q = q.not_.is_("chartmetric_id", "null")
    if args.limit > 0:
        q = q.limit(args.limit)

    rows = q.execute().data or []
    print(f"Artists to score: {len(rows)}  (status={args.status})")

    # Load centroid-based cosine distances from artist_embeddings (keyed by artist_id)
    profile_dist: dict[str, float] = {}
    try:
        offset = 0
        while True:
            batch = (
                sb.schema("tinder").table("artist_embeddings")
                .select("artist_id, cosine_dist")
                .range(offset, offset + 999)
                .execute().data or []
            )
            for r in batch:
                if r.get("cosine_dist") is not None:
                    profile_dist[r["artist_id"]] = r["cosine_dist"]
            if len(batch) < 1000:
                break
            offset += 1000
        print(f"Embedding distances loaded for {len(profile_dist)} profiles")
    except Exception as e:
        print(f"Embedding distances unavailable ({e})")

    done = errors = 0
    start = time.time()
    promoted_ids: list[str] = []  # collect for auto-promote

    for i, row in enumerate(rows, 1):
        name = row["name"]
        slug = row.get("slug") or re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")

        raw_cm = row.get("artist_chartmetric") or {}
        cm = raw_cm[0] if isinstance(raw_cm, list) else raw_cm
        if not cm:
            continue

        try:
            tax = score_taxonomy(cm, taxonomy)
            emb = score_embedding(profile_dist.get(row["id"]))
            nbr = score_neighboring(
                row.get("booked_neighbor_count") or 0,
                row.get("booked_similar_count")  or 0,
            )
            composite = compute_composite(tax["score"], emb, nbr, weights)

            payload = {
                "score":             composite,
                "taxonomy_score":    tax["score"],
                "embedding_score":   emb,
                "neighboring_score": nbr,
                "matched":           tax["matched"],
                "disqualified":      tax["disqualified"],
                "booked_neighbor_count": row.get("booked_neighbor_count") or 0,
                "booked_similar_count":  row.get("booked_similar_count")  or 0,
                "scored_at":         datetime.now(timezone.utc).isoformat(),
            }

            if args.dry_run:
                promote_flag = " → PROMOTE" if composite >= threshold and not tax["disqualified"] else ""
                print(
                    f"  [{i}/{len(rows)}] {name:<32}  "
                    f"composite={composite:3}  tax={tax['score']:3}  "
                    f"emb={emb:3}  nbr={nbr:3}  "
                    f"{'DISQ' if tax['disqualified'] else ''}{promote_flag}"
                )
            else:
                sb.schema("tinder").table("artists").update(
                    {"lofi_feel": payload}
                ).eq("id", row["id"]).execute()
                if composite >= threshold and not tax["disqualified"]:
                    promoted_ids.append(row["id"])
                if i % 50 == 0 or i == len(rows):
                    elapsed = time.time() - start
                    print(f"  [{i}/{len(rows)}]  done={done}  elapsed={elapsed:.0f}s")
                done += 1

        except Exception as e:
            errors += 1
            print(f"  [{i}/{len(rows)}] ERROR {name}: {e}")

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.0f}s — {done} scored, {errors} errors")

    # ── Auto-promote ─────────────────────────────────────────────────────────
    if args.dry_run or args.no_autopromote or args.status != "pending":
        if promoted_ids:
            print(f"Would promote {len(promoted_ids)} artists (score >= {threshold}) — skipped (dry-run or --no-autopromote)")
        return

    if not promoted_ids:
        print(f"No artists scored >= {threshold} — nothing to promote")
        return

    print(f"\nAuto-promoting {len(promoted_ids)} artists (score >= {threshold}) → accepted + needs_scraping")
    promote_errors = 0
    for artist_id in promoted_ids:
        try:
            sb.schema("tinder").table("artists").update({
                "candidate_status": "accepted",
                "needs_scraping":   True,
                "updated_at":       datetime.now(timezone.utc).isoformat(),
            }).eq("id", artist_id).execute()
        except Exception as e:
            promote_errors += 1
            print(f"  promote error {artist_id}: {e}")

    print(f"Promoted {len(promoted_ids) - promote_errors} artists  |  {promote_errors} errors")


if __name__ == "__main__":
    main()
