"""
Per-artist context assembly for the chat — the minimised "model view" that is
the only thing Claude sees. Joins the two worlds:
  - dashboard / Supabase: scores, forecast, public platform metrics, genres
  - Lofi Airtable (read-only): this artist's booking history + comparables

Framework-agnostic (no Streamlit) so it can be unit-tested.
"""
from __future__ import annotations

import datetime as _dt
import sys
from functools import lru_cache
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scoring.five_scores import compute_five_scores  # noqa: E402
from scout.airtable import (  # noqa: E402
    load_artist_record, load_booking_history, load_comparables)
from scout.lofi_events import (  # noqa: E402
    load_artist_lofi_history, load_comparable_events)
from scout.ranking import load_predictions, parse_genres  # noqa: E402


@lru_cache(maxsize=1)
def _predictions() -> dict:
    return load_predictions()


def _trim_feedback(rows: list[dict] | None, limit: int = 10) -> list[dict]:
    out = []
    for r in (rows or [])[:limit]:
        out.append({
            "type": r.get("field_key") or r.get("feedback_type"),
            "value": r.get("field_value"),
            "note": r.get("notes"),
            "date": r.get("created_at"),
        })
    return out


def _similar_artists(profile: dict, ext: dict) -> dict:
    """Scene adjacency we already load in the dashboard (Last.fm + Chartmetric).
    This is the grounded comparables source for the chat."""
    lfm = profile.get("lfm_similar_artists") or []
    lfm_names = [str(n) for n in lfm if n] if isinstance(lfm, list) else []
    cm_names = []
    for r in (ext.get("related_artists") or []):
        if isinstance(r, dict):
            cm_names.append(r.get("name") or r.get("artist_name") or "")
        elif isinstance(r, str):
            cm_names.append(r)
    return {"lastfm": lfm_names[:25],
            "chartmetric": [n for n in cm_names if n][:25]}


_NL_CITIES = {"amsterdam", "rotterdam", "utrecht", "eindhoven", "groningen",
              "nijmegen", "haarlem", "tilburg", "arnhem", "maastricht",
              "den haag", "the hague"}
_NL_COUNTRIES = {"NL", "NETHERLANDS", "NEDERLAND", "BE", "BELGIUM"}


def _iter_rows(df_or_list):
    """Accept a pandas DataFrame or a list of dicts (keeps this testable)."""
    if df_or_list is None:
        return []
    if hasattr(df_or_list, "iterrows"):
        if getattr(df_or_list, "empty", False):
            return []
        return [row for _, row in df_or_list.iterrows()]
    return list(df_or_list)


def _is_nl(city, country) -> bool:
    return (str(country or "").upper().strip() in _NL_COUNTRIES
            or str(city or "").lower().strip() in _NL_CITIES)


def _show_summary(ra_df, pf_data: dict | None) -> dict:
    """Live-performance history — the signal that identifies DJ-led artists and
    NL/Amsterdam draw, from RA events + Partyflock."""
    pf = pf_data or {}
    rows = _iter_rows(ra_df)
    summary = {
        "ra_events_loaded": len(rows),
        "pf_past_performances": pf.get("pf_past_performances"),
        "pf_upcoming_performances": pf.get("pf_upcoming_performances"),
        "pf_total_performances": pf.get("pf_total_performances"),
    }
    if not rows:
        return summary
    today = _dt.date.today().isoformat()
    past = upcoming = nl = ams = 0
    recent = []
    for row in rows:
        d = row.get("date")
        ds = d.isoformat() if isinstance(d, _dt.date) else str(d or "")[:10]
        if ds and ds >= today:
            upcoming += 1
        else:
            past += 1
        city, country, venue = row.get("city"), row.get("country"), row.get("venue")
        if _is_nl(city, country):
            nl += 1
            if "amsterdam" in str(city or "").lower():
                ams += 1
        if len(recent) < 8 and venue:
            recent.append(f"{venue}" + (f" ({city})" if city else ""))
    summary.update({
        "ra_past_shows": past, "ra_upcoming_shows": upcoming,
        "ra_nl_shows": nl, "ra_amsterdam_shows": ams,
        "recent_venues": recent,
    })
    return summary


def _nl_signal(nl_score, pf_data: dict | None) -> dict:
    pf = pf_data or {}
    return {
        "nl_audience_score": nl_score,         # 0-100, computed by the dashboard
        "partyflock_fans": pf.get("pf_fans"),  # NL-platform following
    }


def _milestones(vdf, limit: int = 8) -> list[dict]:
    out = []
    for r in _iter_rows(vdf)[:limit]:
        out.append({
            "type": r.get("event_type"),
            "date": str(r.get("event_date") or ""),
            "confirmed": r.get("confirmed"),
            "details": r.get("details"),
        })
    return out


def build_artist_view(artist_id: str, name: str, profile: dict,
                      ml: dict | None, ext: dict | None = None,
                      ra_df=None, pf_data: dict | None = None, vdf=None,
                      nl_score=None,
                      booker_feedback: list[dict] | None = None) -> dict:
    """Allow-listed view of one artist for the LLM (data minimisation)."""
    scores = compute_five_scores(profile or {}, ml or {})
    if booker_feedback is None:
        from scout.feedback import load_feedback
        booker_feedback = load_feedback(artist_id)
    lofi = load_artist_record(name)  # Artists-table row (genre, last fee, draw)
    lofi_genres = (lofi or {}).get("genres")
    return {
        "artist_id": artist_id,
        "name": name,
        "genres": parse_genres((profile or {}).get("genres")),
        "career_status": (profile or {}).get("career_status")
        or (profile or {}).get("career_stage"),
        "scores": {k: scores.get(k) for k in (
            "momentum", "growth", "market_relevance",
            "future_potential", "confidence")},
        "forecast_90d": _predictions().get(artist_id),
        "metrics": {
            "spotify_listeners": (profile or {}).get("spotify_listeners"),
            "spotify_followers": (profile or {}).get("spotify_followers"),
            "instagram_followers": (profile or {}).get("instagram_followers"),
            "cm_artist_score": (profile or {}).get("cm_artist_score"),
            "cm_artist_rank": (profile or {}).get("cm_artist_rank"),
        },
        # booker-in-the-loop — LOFI's most trustworthy, non-scrapeable signal
        "booker_feedback": _trim_feedback(booker_feedback),
        # scene adjacency we already load (Last.fm + Chartmetric) — grounds comparables
        "similar_artists": _similar_artists(profile or {}, ext or {}),
        # live-performance history — identifies DJ-led artists + NL/Amsterdam draw
        "show_history": _show_summary(ra_df, pf_data),
        "nl_signal": _nl_signal(nl_score, pf_data),
        "milestones": _milestones(vdf),
        # second world (Airtable) — [] / None until configured; gage-approved (opt 1)
        "lofi_record": lofi,
        "booking_history": load_booking_history(name),
        "comparables": load_comparables(name, profile or {}, genres=lofi_genres),
    }


def build_validation_view(artist_id: str, name: str, profile: dict,
                          ml: dict | None, ext: dict | None = None,
                          ra_df=None, pf_data: dict | None = None, vdf=None,
                          nl_score=None,
                          booker_feedback: list[dict] | None = None) -> dict:
    """The chat view PLUS the LOFI ticketing corpus (own history + genre-matched
    comparable events) used to anchor ticket/draw estimates. Adds a `grounding`
    block so the trust-first rule (no ticket number without real evidence) can be
    enforced in code, not just prompted."""
    view = build_artist_view(artist_id, name, profile, ml, ext=ext, ra_df=ra_df,
                             pf_data=pf_data, vdf=vdf, nl_score=nl_score,
                             booker_feedback=booker_feedback)

    # genre source for comparable retrieval: LOFI Sound > dashboard genres
    genres = ((view.get("lofi_record") or {}).get("genres")
              or list(view.get("genres") or []))
    own = load_artist_lofi_history(name)
    comps = load_comparable_events(name, genres)

    has_own = bool(own.get("aggregate") or own.get("events"))
    view["lofi_event_history"] = own            # own LOFI draw (CSV corpus)
    view["comparable_lofi_events"] = comps      # genre-matched past LOFI events
    # trust-first: a ticket range needs own-history OR >=3 comparable events
    view["grounding"] = {
        "has_own_lofi_history": has_own,
        "n_comparable_events": len(comps),
        "ticket_estimate_allowed": has_own or len(comps) >= 3,
    }
    return view
