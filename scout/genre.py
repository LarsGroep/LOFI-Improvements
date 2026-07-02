"""
Shared genre helpers — one normalisation codec for the agent modules.

The Airtable layer (scout/airtable.py), the LOFI ticketing corpus
(scout/lofi_events.py) and the ranking (scout/ranking.py) all need to normalise
and split genre values the same way. Keeping that in one place stops the three
from silently diverging as the taxonomy evolves.
"""
from __future__ import annotations


def norm(s) -> str:
    """Normalised token for matching: lowercase, alphanumerics only."""
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


def genre_list(g) -> list[str]:
    """Readable genre names (case preserved) from a list or delimited string."""
    if isinstance(g, list):
        return [str(x).strip() for x in g if str(x).strip()]
    if isinstance(g, str):
        return [p.strip() for p in str(g).replace(";", ",").split(",") if p.strip()]
    return []


def parse_genres(g) -> set[str]:
    """Normalised genre tokens for overlap matching."""
    return {norm(x) for x in genre_list(g)}


# ── broad genre buckets (so the team — and the models — think in genres, not
# subgenres). Ordered: more specific buckets first. The first keyword a
# subgenre matches wins; "electronic/electronica" is checked before "electro"
# so it doesn't get swallowed. Anything unmatched falls into "Other".
BROAD_RULES = [
    ("House", ["house"]),
    ("Techno", ["techno", "schranz", "minimal"]),
    ("Disco / Nu-Disco", ["disco", "italo", "boogie"]),
    ("Garage / UKG", ["garage", "ukg", "2-step", "bassline"]),
    ("Trance", ["trance", "psy"]),
    ("Drum & Bass", ["drum and bass", "drum & bass", "dnb", "jungle"]),
    ("Dubstep / Bass", ["dubstep", "bass music", "wonky"]),
    ("Electronic", ["electronic", "electronica", "idm", "leftfield", "left field"]),
    ("Electro", ["electro"]),
    ("Ambient / Downtempo", ["ambient", "downtempo", "lo-fi", "lofi", "balearic",
                             "chill"]),
    ("Hip-Hop / Rap", ["hip hop", "hip-hop", "rap", "trap", "grime"]),
    ("Pop", ["pop"]),
    ("Hard", ["hardstyle", "hardcore", "hard techno", "gabber"]),
]


def broad_genres(genres) -> list[str]:
    """Map an artist's subgenres to the broad families they belong to."""
    found: list[str] = []
    for g in (genres or []):
        gl = str(g).lower()
        for bucket, kws in BROAD_RULES:
            if any(k in gl for k in kws):
                if bucket not in found:
                    found.append(bucket)
                break
        else:
            if "Other" not in found:
                found.append("Other")
    return found
