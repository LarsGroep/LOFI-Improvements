"""
Read-only Airtable — LOFI's own booking history + artist economics, the
"second world" that grounds the chat's fee / draw reasoning. Gage data into the
LLM is option-1 approved.

Two tables, joined on the linked-record IDs:

  - "Bookings / Line-up"  one row per artist-per-event: Artist fee (the gage),
                          Total artist cost, Booking status + lookups that
                          denormalise the event (Eventname / Doors open) and the
                          artist (real name). The `Artist` field is a link, so it
                          arrives as record IDs — we resolve those against …
  - "Artists"             one row per artist: Sound (genre), Fee range,
                          Last AF paid by Lofi, Total visitors — last event,
                          Momentum, Fit Lofi, Spotify listeners.

Everything is read-only and graceful-empty: not configured, or any error →
loaders return [] / None, so the dashboard never breaks.

    AIRTABLE_TOKEN, AIRTABLE_BASE_ID                      (required)
    AIRTABLE_BOOKINGS_TABLE   default "Bookings / Line-up"
    AIRTABLE_ARTISTS_TABLE    default "Artists"
    AIRTABLE_F_*  / AIRTABLE_AR_*   per-field name overrides (defaults below)

Run `python -m scout.airtable`        to dump the schema, or
    `python -m scout.airtable verify` to check the join resolves.
"""
from __future__ import annotations

import os
from functools import lru_cache
from urllib.parse import quote

from scout.genre import genre_list as _genre_list
from scout.genre import norm as _norm
from scout.genre import parse_genres as _parse_genres

_API_ROOT = "https://api.airtable.com/v0"
_META_ROOT = "https://api.airtable.com/v0/meta"

# ── field maps (real LOFI schema; override individually via env) ──────────────

# "Bookings / Line-up" — one row per artist per event.
_BK = {
    "artist_link": "Artist",                        # link → Artists (record IDs)
    "real_name":   "Artist real name (from Artist)",  # lookup
    "event":       "Eventname (from Events)",          # lookup
    "date":        "Doors open (from Events)",          # lookup
    "gage":        "Artist fee",                     # currency — the gage
    "total_cost":  "Total artist cost",             # formula
    "status":      "Booking status",
    "area":        "Area",
}

# "Artists" — one row per artist.
_AR = {
    "stage_name":  "Artist_ID 🔑",   # primary — the booking/stage name
    "real_name":   "Artist real name",
    "sound":       "Sound",          # multipleSelects → genre
    "fee_range":   "Fee range",
    "last_fee":    "Last AF paid by Lofi",
    "visitors":    "Total visitors — last event",
    "avg_ticket":  "Avg ticket price — last event",
    "avg_bar":     "Avg bar spend — last event",
    "momentum":    "Momentum",
    "fit":         "Fit Lofi",
    "spotify":     "Spotify maandelijkse luisteraars",
}


def _token() -> str:
    return os.environ.get("AIRTABLE_TOKEN", "")


def _base() -> str:
    return os.environ.get("AIRTABLE_BASE_ID", "")


def _bookings_table() -> str:
    return os.environ.get("AIRTABLE_BOOKINGS_TABLE", "Bookings / Line-up")


def _artists_table() -> str:
    return os.environ.get("AIRTABLE_ARTISTS_TABLE", "Artists")


def _configured() -> bool:
    return bool(_token() and _base())


def _bk(key: str) -> str:
    return os.environ.get(f"AIRTABLE_F_{key.upper()}", _BK[key])


def _ar(key: str) -> str:
    return os.environ.get(f"AIRTABLE_AR_{key.upper()}", _AR[key])


# ── small value helpers ──────────────────────────────────────────────────────

def _value(v):
    """Flatten Airtable values (links / lookups / attachments arrive as lists)."""
    if isinstance(v, list):
        names = []
        for x in v:
            names.append(x.get("name") or x.get("id") or "" if isinstance(x, dict)
                         else str(x))
        joined = ", ".join(n for n in names if n)
        return joined
    return v


def _num(v):
    """First numeric value (currency/number fields, or a lookup wrapping one)."""
    if isinstance(v, list):
        v = next((x for x in v if x is not None), None)
    return v if isinstance(v, (int, float)) else None


def _short_date(v) -> str:
    return str(_value(v) or "")[:10]


# ── fetch ────────────────────────────────────────────────────────────────────

def _fetch_records(table: str) -> list[dict]:
    import httpx  # lazy — only when configured

    url = f"{_API_ROOT}/{_base()}/{quote(table, safe='')}"
    headers = {"Authorization": f"Bearer {_token()}"}
    out, offset = [], None
    with httpx.Client(timeout=30.0) as client:
        while True:
            params = {"pageSize": 100}
            if offset:
                params["offset"] = offset
            r = client.get(url, headers=headers, params=params)
            r.raise_for_status()
            data = r.json()
            out.extend(data.get("records", []))
            offset = data.get("offset")
            if not offset:
                break
    return out


# ── parse: Artists ───────────────────────────────────────────────────────────

def _parse_artist(rec: dict) -> dict:
    f = rec.get("fields", {})
    return {
        "record_id": rec.get("id"),
        "stage_name": _value(f.get(_ar("stage_name"))),
        "real_name": _value(f.get(_ar("real_name"))),
        "genres": _genre_list(f.get(_ar("sound"))),
        "fee_range": _value(f.get(_ar("fee_range"))),
        "last_fee_paid": _num(f.get(_ar("last_fee"))),
        "last_event_visitors": _num(f.get(_ar("visitors"))),
        "avg_ticket_price": _num(f.get(_ar("avg_ticket"))),
        "avg_bar_spend": _num(f.get(_ar("avg_bar"))),
        "momentum": _value(f.get(_ar("momentum"))),
        "fit_lofi": _value(f.get(_ar("fit"))),
        "spotify_listeners": _num(f.get(_ar("spotify"))),
    }


@lru_cache(maxsize=1)
def _artists() -> tuple:
    if not _configured():
        return ()
    try:
        return tuple(_parse_artist(r) for r in _fetch_records(_artists_table()))
    except Exception:
        return ()


def _artists_by_id() -> dict:
    return {a["record_id"]: a for a in _artists() if a.get("record_id")}


# ── parse: Bookings / Line-up (resolves the Artist link) ─────────────────────

def _parse_booking(rec: dict, by_id: dict) -> dict:
    f = rec.get("fields", {})
    ids = [i for i in (f.get(_bk("artist_link")) or []) if isinstance(i, str)]

    stages, genres = [], []
    for rid in ids:
        a = by_id.get(rid)
        if not a:
            continue
        if a.get("stage_name"):
            stages.append(a["stage_name"])
        for g in (a.get("genres") or []):
            if g not in genres:
                genres.append(g)  # readable, deduped, order preserved

    real = _value(f.get(_bk("real_name")))
    artist = ", ".join(stages) if stages else real
    return {
        "artist": artist,            # stage name(s), resolved from the link
        "real_name": real,
        "event": _value(f.get(_bk("event"))),
        "date": _short_date(f.get(_bk("date"))),
        "gage": _num(f.get(_bk("gage"))),
        "total_cost": _num(f.get(_bk("total_cost"))),
        "status": _value(f.get(_bk("status"))),
        "area": _value(f.get(_bk("area"))),
        "genre": ", ".join(genres),
    }


@lru_cache(maxsize=1)
def _all_bookings() -> tuple:
    if not _configured():
        return ()
    try:
        by_id = _artists_by_id()
        return tuple(_parse_booking(r, by_id)
                     for r in _fetch_records(_bookings_table()))
    except Exception:
        return ()


def _matches(booking: dict, key: str) -> bool:
    """A booking is 'this artist' if the (normalised) query name equals any
    resolved stage name (b2b lines join several) or the real name."""
    if not key:
        return False
    cand = {_norm(p) for p in str(booking.get("artist", "")).split(",")}
    cand.add(_norm(booking.get("real_name", "")))
    return key in cand


# ── public loaders (used by scout/context.build_artist_view) ─────────────────

def load_artist_record(artist_name: str, artists=None) -> dict | None:
    """The Artists-table row for this artist: genre, fee range, last fee paid,
    last-event draw — the richest single-row LOFI grounding."""
    src = artists if artists is not None else _artists()
    key = _norm(artist_name)
    for a in src:
        if _norm(a.get("stage_name", "")) == key or _norm(a.get("real_name", "")) == key:
            return {k: v for k, v in a.items()
                    if k != "record_id" and v not in (None, "", [])}
    return None


def load_booking_history(artist_name: str, limit: int = 20,
                         bookings=None) -> list[dict]:
    """Past LOFI bookings of THIS artist (gage, event, date, outcome),
    newest first."""
    src = bookings if bookings is not None else _all_bookings()
    key = _norm(artist_name)
    hits = [b for b in src if _matches(b, key)]
    hits.sort(key=lambda b: b.get("date") or "", reverse=True)
    return hits[:limit]


def load_comparables(artist_name: str, profile: dict, limit: int = 5,
                     bookings=None, genres=None) -> list[dict]:
    """Similar past bookings (genre overlap) for fee/draw benchmarking. `genres`
    (the LOFI Sound of the target artist) wins; else the dashboard profile."""
    src = bookings if bookings is not None else _all_bookings()
    key = _norm(artist_name)
    want = {_norm(x) for x in genres} if genres else _parse_genres(
        (profile or {}).get("genres"))

    scored = []
    for b in src:
        if _matches(b, key):
            continue  # exclude the artist themselves
        overlap = len(want & _parse_genres(b.get("genre"))) if want else 0
        if want and overlap == 0:
            continue  # when we know the genres, only genre-matched comparables
        if b.get("gage") is None:
            continue  # a comparable is only useful if it carries a fee
        scored.append((overlap, b))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [b for _, b in scored[:limit]]


# ── introspection (run locally to discover / verify the schema) ──────────────

# Tables we care about for the resolver — keeps the output small + pasteable.
_FOCUS_DEFAULT = "booking,artist,event,financial,ticket,budget"


def _print_sample(client, headers, table_ref: str, n: int = 2,
                  indent: str = "      ") -> None:
    """Print n sample rows (field: truncated value) — shows how links resolve."""
    try:
        r = client.get(f"{_API_ROOT}/{_base()}/{quote(table_ref, safe='')}",
                       headers=headers, params={"pageSize": n})
        if r.status_code != 200:
            print(f"{indent}(sample unavailable: {r.status_code})")
            return
        recs = r.json().get("records", [])[:n]
        if not recs:
            print(f"{indent}(no rows)")
            return
        print(f"{indent}sample:")
        for rec in recs:
            for k, v in rec.get("fields", {}).items():
                print(f"{indent}  {k}: {str(_value(v))[:55]}")
            print(f"{indent}  --")
    except Exception as e:  # noqa: BLE001
        print(f"{indent}(sample failed: {e})")


def _introspect() -> None:
    """`python -m scout.airtable` — discover the schema."""
    import httpx

    if not _token():
        print("Set AIRTABLE_TOKEN (and ideally AIRTABLE_BASE_ID) in your .env first.")
        return
    headers = {"Authorization": f"Bearer {_token()}"}
    keywords = [k.strip().lower() for k in
                os.environ.get("AIRTABLE_FILTER", _FOCUS_DEFAULT).split(",")
                if k.strip()]

    with httpx.Client(timeout=30.0) as client:
        if not _base():
            r = client.get(f"{_META_ROOT}/bases", headers=headers)
            if r.status_code != 200:
                print(f"Could not list bases ({r.status_code}). Set "
                      "AIRTABLE_BASE_ID and re-run."); return
            print("Bases the token can see (set one as AIRTABLE_BASE_ID):")
            for b in r.json().get("bases", []):
                print(f"  {b['id']}   {b['name']}")
            return

        r = client.get(f"{_META_ROOT}/bases/{_base()}/tables", headers=headers)
        if r.status_code != 200:
            table = _bookings_table()
            print(f"Metadata API unavailable ({r.status_code}). Sampling "
                  f"'{table}':")
            _print_sample(client, headers, table, n=3, indent="  ")
            return

        tables = r.json().get("tables", [])
        print(f"Base {_base()} — {len(tables)} tables:\n")
        for t in tables:
            print(f"  - {t['name']}")

        matched = [t for t in tables
                   if any(k in t["name"].lower() for k in keywords)]
        print(f"\n=== Detail + samples for {len(matched)} relevant tables "
              f"(filter: {', '.join(keywords)}) ===")
        for t in matched:
            print(f"\nTABLE: {t['name']}")
            for fld in t.get("fields", []):
                print(f"    - {fld['name']}  ({fld.get('type')})")
            _print_sample(client, headers, t["id"], n=2)
        print("\nPaste this back and I'll map the fields + build the resolver.")


def _verify() -> None:
    """`python -m scout.airtable verify` — confirm the two-table join resolves."""
    if not _configured():
        print("Set AIRTABLE_TOKEN and AIRTABLE_BASE_ID in your .env first.")
        return
    artists, bookings = _artists(), _all_bookings()
    print(f"Artists table  ({_artists_table()}): {len(artists)} rows")
    print(f"Bookings table ({_bookings_table()}): {len(bookings)} rows")
    if not artists and not bookings:
        print("\nNothing loaded — check the table names / token scope.")
        return

    resolved = sum(1 for b in bookings if b.get("artist"))
    with_fee = sum(1 for b in bookings if b.get("gage") is not None)
    print(f"  bookings with a resolved artist name: {resolved}/{len(bookings)}")
    print(f"  bookings carrying a fee (gage):       {with_fee}/{len(bookings)}")

    print("\nSample resolved bookings:")
    for b in bookings[:3]:
        print(f"  • {b['artist']}  — €{b['gage']} @ {b['event']} "
              f"({b['date']})  [{b['genre']}]")

    if artists:
        a = next((x for x in artists if x.get("stage_name")), artists[0])
        nm = a.get("stage_name") or a.get("real_name")
        print(f"\nSample artist record — {nm}:")
        print(f"  genres={a.get('genres')}  fee_range={a.get('fee_range')}  "
              f"last_fee={a.get('last_fee_paid')}  "
              f"last_visitors={a.get('last_event_visitors')}")
        hist = load_booking_history(nm)
        print(f"  booking history rows for {nm}: {len(hist)}")


if __name__ == "__main__":
    import sys
    try:
        from pathlib import Path as _Path
        from dotenv import load_dotenv
        load_dotenv(_Path(__file__).parent.parent / ".env")
    except Exception:
        pass
    if len(sys.argv) > 1 and sys.argv[1] == "verify":
        _verify()
    else:
        _introspect()
