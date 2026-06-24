"""
Supabase read layer for the Scout — framework-agnostic (no Streamlit, no LLM).

Phase 1 of the agent is fully compliant with the Verwerkersovereenkomst: it only
reads from Lofi's existing Supabase and computes/displays inside the dashboard.
Nothing leaves for an external LLM here.
"""
from __future__ import annotations

import json
import os

from supabase import create_client

_SCHEMA = "tinder"


def make_client():
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set.")
    return create_client(url, key)


def fetch_all(client, table: str, columns: str = "*", schema: str = _SCHEMA,
              page_size: int = 1000) -> list[dict]:
    """Paginated select — mirrors the training scripts to avoid the implicit
    1000-row cap / statement timeouts."""
    rows: list[dict] = []
    start = 0
    while True:
        page = (
            client.schema(schema).table(table).select(columns)
            .range(start, start + page_size - 1).execute().data
        ) or []
        rows.extend(page)
        if len(page) < page_size:
            break
        start += page_size
    return rows


def load_flat_profiles(client) -> list[dict]:
    """All artist rows from the flat Chartmetric view (select * to stay robust
    to schema changes — same approach the dashboard uses per artist)."""
    return fetch_all(client, "artist_chartmetric_flat", "*")


def load_ml_features(client) -> dict[str, dict]:
    """artist_id -> ml_features dict (the momentum/growth signals)."""
    out: dict[str, dict] = {}
    for r in fetch_all(client, "artist_chartmetric", "artist_id, ml_features"):
        ml = r.get("ml_features") or {}
        if isinstance(ml, str):
            try:
                ml = json.loads(ml)
            except Exception:
                ml = {}
        out[r["artist_id"]] = ml
    return out
