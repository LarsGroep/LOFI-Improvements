"""
Read-only Airtable — LOFI booking history + comparables for the AI chat.

The "second world": Lofi's own bookings (gages, events, outcomes) ground the
chat's fee/draw reasoning. Gage data into the LLM is option-1 approved.

Field names vary per base, so the mapping is configurable via env (set after
running the introspector: `python -m scout.airtable`). Everything is read-only
and graceful-empty: if not configured or on any error, loaders return [].

    AIRTABLE_TOKEN, AIRTABLE_BASE_ID, AIRTABLE_BOOKINGS_TABLE   (required)
    AIRTABLE_F_ARTIST / _GAGE / _DATE / _EVENT / _VENUE / _GENRE / _OUTCOME
        (field-name overrides; defaults below)
"""
from __future__ import annotations

import os
from functools import lru_cache
from urllib.parse import quote

_API_ROOT = "https://api.airtable.com/v0"
_META_ROOT = "https://api.airtable.com/v0/meta"

_DEFAULT_FIELDS = {
    "artist": "Artist", "event": "Event", "venue": "Venue", "date": "Date",
    "gage": "Fee", "genre": "Genre", "outcome": "Tickets sold",
}


def _token() -> str:
    return os.environ.get("AIRTABLE_TOKEN", "")


def _base() -> str:
    return os.environ.get("AIRTABLE_BASE_ID", "")


def _bookings_table() -> str:
    return os.environ.get("AIRTABLE_BOOKINGS_TABLE", "")


def _configured() -> bool:
    return bool(_token() and _base() and _bookings_table())


def _mapping() -> dict:
    return {k: os.environ.get(f"AIRTABLE_F_{k.upper()}", v)
            for k, v in _DEFAULT_FIELDS.items()}


def _norm(s) -> str:
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


def _parse_genres(g) -> set[str]:
    if isinstance(g, list):
        return {_norm(x) for x in g if x}
    if isinstance(g, str):
        return {_norm(x) for x in g.replace(";", ",").split(",") if x.strip()}
    return set()


def _value(v):
    """Flatten Airtable values (linked records / attachments come as lists)."""
    if isinstance(v, list):
        names = []
        for x in v:
            names.append(x.get("name") or x.get("id") or "" if isinstance(x, dict)
                         else str(x))
        return ", ".join(n for n in names if n)
    return v


# ── fetch + parse ────────────────────────────────────────────────────────────

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


def _record_to_booking(rec: dict, m: dict) -> dict:
    f = rec.get("fields", {})
    return {k: _value(f.get(m[k])) for k in _DEFAULT_FIELDS}


@lru_cache(maxsize=1)
def _all_bookings() -> tuple:
    """Parsed bookings, cached for the session (read-only reference data)."""
    if not _configured():
        return ()
    try:
        m = _mapping()
        return tuple(_record_to_booking(r, m)
                     for r in _fetch_records(_bookings_table()))
    except Exception:
        return ()


# ── public loaders (used by scout/context.build_artist_view) ─────────────────

def load_booking_history(artist_name: str, limit: int = 20,
                         bookings=None) -> list[dict]:
    """Past LOFI bookings of THIS artist (gage, event, date, outcome)."""
    src = bookings if bookings is not None else _all_bookings()
    key = _norm(artist_name)
    return [b for b in src if _norm(b.get("artist", "")) == key][:limit]


def load_comparables(artist_name: str, profile: dict, limit: int = 5,
                     bookings=None) -> list[dict]:
    """Similar past bookings (genre overlap) for fee/draw benchmarking."""
    src = bookings if bookings is not None else _all_bookings()
    key = _norm(artist_name)
    genres = _parse_genres((profile or {}).get("genres"))

    scored = []
    for b in src:
        if _norm(b.get("artist", "")) == key:
            continue  # exclude the artist themselves
        overlap = len(genres & _parse_genres(b.get("genre"))) if genres else 0
        if genres and overlap == 0:
            continue  # when we know the genres, only genre-matched comparables
        scored.append((overlap, b))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [b for _, b in scored[:limit]]


# ── introspection (run locally to discover the schema) ───────────────────────

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
    """`python -m scout.airtable` — discover the schema.

    Lists every table name, then full fields + sample rows for the tables that
    matter (booking/artist/event/financial/ticket/budget). Override the keyword
    filter with AIRTABLE_FILTER (comma-separated, e.g. "book,artist,event")."""
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
            if not table:
                print(f"Metadata API unavailable ({r.status_code}). Set "
                      "AIRTABLE_BOOKINGS_TABLE and re-run to sample a table.")
                return
            print(f"Sample of '{table}':")
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


if __name__ == "__main__":
    try:
        from pathlib import Path as _Path
        from dotenv import load_dotenv
        load_dotenv(_Path(__file__).parent.parent / ".env")
    except Exception:
        pass
    _introspect()
