"""
Tour-routing signal — an artist already routed through the EU is a cheaper,
warmer booking than a fly-in.

FEEDBACK.md item 8. Optional: needs BANDSINTOWN_APP_ID (their public API is
free for this use). Without it every call returns the graceful-empty signal
and the deal sheet simply omits routing. Only the artist's public name goes
into the request — no LOFI-internal data leaves (same rule as web search).
"""
from __future__ import annotations

import datetime as _dt
import os
from urllib.parse import quote

_EU = {"netherlands", "belgium", "germany", "france", "united kingdom", "uk",
       "spain", "italy", "portugal", "austria", "switzerland", "denmark",
       "sweden", "norway", "finland", "poland", "czech republic", "czechia",
       "ireland", "hungary", "romania", "croatia", "greece", "luxembourg"}


def _app_id() -> str:
    return os.environ.get("BANDSINTOWN_APP_ID", "")


def upcoming_events(artist_name: str, app_id: str | None = None) -> list[dict]:
    """Public upcoming dates from Bandsintown; [] when unconfigured/unknown."""
    app_id = app_id or _app_id()
    if not app_id or not artist_name:
        return []
    import httpx
    url = (f"https://rest.bandsintown.com/artists/{quote(artist_name)}/events"
           f"?app_id={quote(app_id)}")
    try:
        resp = httpx.get(url, timeout=15)
        data = resp.json()
        return data if isinstance(data, list) else []
    except Exception:
        return []


def routing_signal(artist_name: str, window_days: int = 120,
                   events: list[dict] | None = None) -> dict:
    """EU/NL presence inside the booking window. `events` injectable for tests."""
    if events is None:
        events = upcoming_events(artist_name)
    if not events:
        return {"method": "unavailable", "eu_shows": None, "nl_shows": None}

    horizon = (_dt.date.today() + _dt.timedelta(days=window_days)).isoformat()
    eu = nl = 0
    next_eu = None
    for e in events:
        venue = e.get("venue") or {}
        country = str(venue.get("country") or "").lower()
        date = str(e.get("datetime") or "")[:10]
        if not date or date > horizon:
            continue
        if country in _EU:
            eu += 1
            if next_eu is None or date < next_eu:
                next_eu = date
            if country in ("netherlands",):
                nl += 1
    note = ("already routed through the EU in the window — a warmer, "
            "likely cheaper booking than a fly-in" if eu
            else "no EU dates in the window — expect fly-in economics")
    return {"method": "bandsintown", "window_days": window_days,
            "eu_shows": eu, "nl_shows": nl, "next_eu_date": next_eu,
            "note": note}
