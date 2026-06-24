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
