"""
LOFI historical ticketing corpus — the retrieval source for ticket/draw
reasoning in the validation panel.

NOT a trained model: ~1k past artist-events are a *retrieval corpus*, not a
regressor. Two grains, loaded from the repo CSVs (no network):

  artist_events_clean.csv  one row per artist-per-event — actual_tickets,
                           total_visitors, occupancy_rate, genre, event_type.
                           This is the comparable-events pool.
  artists_clean.csv        one row per artist — avg_tickets / avg_visitors /
                           avg_occupancy / event_count. The artist's OWN draw,
                           the strongest single signal when LOFI has booked them.

Two contamination guards (verified against the data): the corpus mixes past
events (real actuals) with future/cancelled rows whose actual_tickets is 0, so
we keep only rows with actual_tickets > 0 AND a parseable past date. Everything
is graceful-empty: missing zip / parse error → [].
"""
from __future__ import annotations

import csv
import datetime as _dt
import io
import zipfile
from functools import lru_cache
from pathlib import Path

_ROOT = Path(__file__).parent.parent
_ZIP = _ROOT / "lineup_recommender" / "data" / "lofi" / "artists+events_clean.zip"
_EVENTS_CSV = "artist_events_clean.csv"
_ARTISTS_CSV = "artists_clean.csv"


def _norm(s) -> str:
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


def _parse_genres(g) -> set[str]:
    if isinstance(g, list):
        return {_norm(x) for x in g if x}
    if isinstance(g, str):
        return {_norm(x) for x in g.replace(";", ",").split(",") if x.strip()}
    return set()


def _to_int(x):
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return None


def _to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _parse_date(s) -> str:
    """LOFI dates look like '1/1/2024 12:00am' → ISO 'YYYY-MM-DD' (or '')."""
    s = str(s or "").strip()
    if not s:
        return ""
    head = s.split(" ")[0]
    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%d/%m/%Y"):
        try:
            return _dt.datetime.strptime(head, fmt).date().isoformat()
        except ValueError:
            continue
    return ""


def _today() -> str:
    return _dt.date.today().isoformat()


def _read_csv(name: str) -> list[dict]:
    if not _ZIP.exists():
        return []
    try:
        with zipfile.ZipFile(_ZIP) as z, z.open(name) as f:
            return list(csv.DictReader(
                io.TextIOWrapper(f, encoding="utf-8", errors="replace")))
    except Exception:
        return []


# ── parsing ──────────────────────────────────────────────────────────────────

def _parse_event(r: dict) -> dict | None:
    """One past artist-event with real ticket numbers, or None if unusable."""
    tickets = _to_int(r.get("actual_tickets"))
    if not tickets or tickets <= 0:          # guard 1: drop future/0-ticket rows
        return None
    date = _parse_date(r.get("event_date"))
    if date and date >= _today():            # guard 2: drop future dates
        return None
    occ = _to_float(r.get("occupancy_rate"))
    return {
        "artist_name": (r.get("artist_name") or "").strip(),
        "event_name": (r.get("event_name") or "").strip(),
        "event_date": date,
        "genre": (r.get("genre") or "").strip(),
        "event_type": (r.get("event_type") or "").strip(),
        "actual_tickets": tickets,
        "total_visitors": _to_int(r.get("total_visitors")),
        "occupancy_rate": round(occ, 2) if occ is not None else None,
    }


@lru_cache(maxsize=1)
def _events() -> tuple:
    return tuple(e for e in (_parse_event(r) for r in _read_csv(_EVENTS_CSV)) if e)


def _parse_aggregate(r: dict) -> dict:
    return {
        "artist_name": (r.get("artist_name") or "").strip(),
        "event_count": _to_int(r.get("event_count")),
        "avg_tickets": _to_int(r.get("avg_tickets")),
        "avg_visitors": _to_int(r.get("avg_visitors")),
        "avg_occupancy_rate": (round(_to_float(r.get("avg_occupancy_rate")), 2)
                               if _to_float(r.get("avg_occupancy_rate")) is not None
                               else None),
        "genres": [g.strip() for g in (r.get("genres") or "").split(",") if g.strip()],
        "first_event_date": _parse_date(r.get("first_event_date")),
        "last_event_date": _parse_date(r.get("last_event_date")),
    }


@lru_cache(maxsize=1)
def _aggregates() -> dict:
    out = {}
    for r in _read_csv(_ARTISTS_CSV):
        a = _parse_aggregate(r)
        if a["artist_name"]:
            out[_norm(a["artist_name"])] = a
    return out


# ── public loaders ───────────────────────────────────────────────────────────

def load_artist_lofi_history(name: str, limit: int = 12,
                             events=None, aggregates=None) -> dict:
    """This artist's OWN LOFI track record: their per-event rows (newest first)
    plus the pre-computed aggregate. Strongest signal when it exists."""
    key = _norm(name)
    src = events if events is not None else _events()
    own = [e for e in src if _norm(e["artist_name"]) == key]
    own.sort(key=lambda e: e.get("event_date") or "", reverse=True)
    aggs = aggregates if aggregates is not None else _aggregates()
    return {"events": own[:limit], "aggregate": aggs.get(key)}


def load_comparable_events(name: str, genres, event_type: str | None = None,
                           limit: int = 8, events=None) -> list[dict]:
    """Genre-matched past LOFI events (excluding this artist) for ticket/draw
    benchmarking. Fallback chain: genre overlap → event_type → recency."""
    key = _norm(name)
    want = {_norm(x) for x in genres} if genres else set()
    et = _norm(event_type) if event_type else ""
    src = events if events is not None else _events()

    scored = []
    for e in src:
        if _norm(e["artist_name"]) == key:
            continue
        overlap = len(want & _parse_genres(e.get("genre"))) if want else 0
        if want and overlap == 0:
            continue                       # when genres are known, require a match
        type_match = 1 if et and _norm(e.get("event_type")) == et else 0
        recency = e.get("event_date") or ""
        scored.append((overlap, type_match, recency, e))
    # genre overlap, then event_type, then recency — the brief's fallback order
    scored.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)

    # dedupe by event (b2b nights repeat once per artist); keep best-ranked row
    out, seen = [], set()
    for *_, e in scored:
        ek = _norm(e.get("event_name")) or id(e)
        if ek in seen:
            continue
        seen.add(ek)
        out.append(e)
        if len(out) >= limit:
            break
    return out
