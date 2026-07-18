"""
Artist-name resolution — link a name a listener found to a Supabase artist_id.

The OS deliberately tracks names *before* they exist in Chartmetric/Supabase
(docs/agentic_os.md §4: "that pre-database window is where the deepest hidden
gems live"). So resolution is optional and conservative: exact match after
light normalisation only. No fuzzy guessing — a wrong artist_id silently
mis-attributes a signal, which is worse than an honest None.

The index is built from the flat Chartmetric profiles the tick already loaded
(ctx.shared["flat_profiles"]); with no Supabase configured there is no index
and every name resolves to None, which is a valid, expected state.
"""
from __future__ import annotations

import unicodedata


# ── normalisation (pure) ──────────────────────────────────────────────────────

def normalise(name: str) -> str:
    """Lowercase, strip accents, drop punctuation, collapse whitespace.
    'Kölsch', 'kolsch' and 'KOLSCH ' all fold to the same key — but nothing
    fuzzier than that."""
    if not name:
        return ""
    decomposed = unicodedata.normalize("NFKD", str(name))
    stripped = "".join(c for c in decomposed
                       if not unicodedata.combining(c))
    kept = [c if (c.isalnum() or c.isspace()) else " " for c in stripped]
    return " ".join("".join(kept).lower().split())


def build_index(profiles) -> dict:
    """{normalised name -> artist_id} from flat profiles. Later rows lose to
    earlier ones on a collision (deterministic, first-seen wins)."""
    index: dict[str, str] = {}
    for p in profiles or []:
        aid = p.get("artist_id")
        key = normalise(p.get("artist_name") or "")
        if aid and key and key not in index:
            index[key] = str(aid)
    return index


def resolve(name: str, index: dict) -> str | None:
    """Exact (normalised) match only; else None."""
    if not index:
        return None
    return index.get(normalise(name))


# ── per-tick convenience ──────────────────────────────────────────────────────

def index_from_ctx(ctx) -> dict:
    """Build the index once per tick and cache it on ctx.shared, so a morning
    where three listeners resolve names pays for the index once."""
    if "artist_index" in ctx.shared:
        return ctx.shared["artist_index"]
    index = build_index(ctx.shared.get("flat_profiles") or [])
    ctx.shared["artist_index"] = index
    return index


def match_in_title(title: str, index: dict) -> tuple[str | None, str | None]:
    """Find a known artist whose normalised name appears as a whole token-run
    inside a press headline. Returns (artist_name, artist_id) or (None, None).
    Conservative on purpose — substring-of-a-word matches are rejected."""
    if not title or not index:
        return None, None
    norm_title = f" {normalise(title)} "
    for key, aid in index.items():
        if key and f" {key} " in norm_title:
            return key, aid
    return None, None
