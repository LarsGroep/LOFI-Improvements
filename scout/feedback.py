"""
Booker-in-the-loop — capture booker feedback and feed it back into the AI.

The booker's tacit knowledge (real-world results, model corrections, industry
intel) is LOFI's most trustworthy, non-scrapeable signal. This module captures
it (LLM-structured when live, plain note otherwise) into the existing
`tinder.artist_feedback` table — no migration — and reads it back so every future
AI judgement is grounded in it.
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agents.core import structure_feedback  # noqa: E402


@lru_cache(maxsize=1)
def _client():
    from scout.data import make_client
    return make_client()


def load_feedback(artist_id: str, limit: int = 20) -> list[dict]:
    """Recent booker feedback for an artist. Graceful-empty if there's no DB."""
    try:
        return (
            _client().schema("tinder").table("artist_feedback").select("*")
            .eq("artist_id", artist_id).order("created_at", desc=True)
            .limit(limit).execute().data
        ) or []
    except Exception:
        return []


def capture_feedback(artist_id: str, raw_text: str) -> dict:
    """Structure (LLM or fallback) + store. Returns the structured interpretation
    so the UI can show the booker how it was understood (also a trust check)."""
    s = structure_feedback(raw_text)
    category = s.get("category", "other")
    row = {
        "artist_id": artist_id,
        "feedback_type": "general_note",  # always-allowed value; no CHECK risk
        "field_key": f"{category}:{s.get('field_key', 'note')}"[:120],
        "field_value": (s.get("field_value") or "")[:200],
        "event_ref": (s.get("event_ref") or "").strip() or None,
        "notes": ((s.get("summary") or raw_text).strip()
                  + "\n---\n" + raw_text.strip()),
        "created_by": "booker_llm",
    }
    _client().schema("tinder").table("artist_feedback").insert(row).execute()
    return s
