"""
A polite, cached fetcher — the one door every listener uses to touch the web.

Why this exists: listeners re-run nightly over the same label pages, playlists
and feeds. Re-fetching each time is rude to the source and slow for us, so a
snapshot accrues in a SQLite cache (docs/agentic_os.md §3.1: "snapshots
accruing is the whole game") and re-analysis never re-fetches fresh bytes.

Design rules (all load-bearing):
  - Identifying User-Agent, one request, 20s timeout, no retries.
  - Conditional requests: an `etag` from a prior fetch is replayed as
    If-None-Match; a 304 just refreshes `fetched_at` and returns the cached body.
  - `fetch()` NEVER raises. A network error returns (0, "") — but keeps any
    stale cache and, with allow_stale=True (the default), serves it. Honest
    degradation over a hard crash (README philosophy).

The cache DB lives in osk/data/ (gitignored, venue-private — same policy as the
blackboard), overridable via LOFI_OS_SCRAPE_DB.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import os
import sqlite3
from pathlib import Path

_DEFAULT_DB = Path(__file__).parent / "data" / "scrape_cache.db"

USER_AGENT = "LOFI-Scout/0.1"
TIMEOUT_SECONDS = 20
DEFAULT_MAX_AGE = 21600          # 6h — a nightly sweep never double-fetches

_SCHEMA_SQL = """
create table if not exists scrape_cache (
  url_hash text primary key,
  url text,
  fetched_at text,
  status integer,
  body text,
  etag text);
"""


# ── pure helpers (separable, unit-testable) ──────────────────────────────────

def cache_key(url: str) -> str:
    """Stable primary key for a URL — sha256 hex, so the table never grows a
    column-width problem on long query strings."""
    return hashlib.sha256(url.strip().encode("utf-8")).hexdigest()


def _parse_ts(value) -> _dt.datetime | None:
    if not value:
        return None
    try:
        ts = _dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return ts if ts.tzinfo else ts.replace(tzinfo=_dt.timezone.utc)


def is_fresh(fetched_at, now: _dt.datetime, max_age_seconds: int) -> bool:
    """True iff a cache row fetched at `fetched_at` is still within max_age.
    Unparseable/missing timestamp → stale (re-fetch)."""
    ts = _parse_ts(fetched_at)
    if ts is None:
        return False
    return (now - ts).total_seconds() <= max_age_seconds


def _utcnow() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


def _iso(now: _dt.datetime) -> str:
    return now.isoformat(timespec="seconds")


# ── cache I/O ─────────────────────────────────────────────────────────────────

def _db_path() -> Path:
    return Path(os.environ.get("LOFI_OS_SCRAPE_DB") or _DEFAULT_DB)


def _connect():
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.executescript(_SCHEMA_SQL)
    return db


def _read_row(db, key: str):
    return db.execute("select * from scrape_cache where url_hash=?",
                      (key,)).fetchone()


def _write_row(db, key, url, now_iso, status, body, etag) -> None:
    db.execute(
        "insert into scrape_cache (url_hash, url, fetched_at, status, body, "
        "etag) values (?,?,?,?,?,?) on conflict(url_hash) do update set "
        "url=excluded.url, fetched_at=excluded.fetched_at, "
        "status=excluded.status, body=excluded.body, etag=excluded.etag",
        (key, url, now_iso, status, body, etag))
    db.commit()


def _touch(db, key, now_iso) -> None:
    """A 304 — the cached body is still current, just refresh fetched_at."""
    db.execute("update scrape_cache set fetched_at=? where url_hash=?",
               (now_iso, key))
    db.commit()


# ── the one network primitive (monkeypatched in tests) ───────────────────────

def _http_get(url: str, headers: dict, timeout: int):
    """Single GET, no retries. Returns (status, body, etag). Isolated so tests
    can replace exactly this and stay offline."""
    import httpx
    with httpx.Client(timeout=timeout, follow_redirects=True,
                      headers=headers) as client:
        resp = client.get(url)
        return resp.status_code, resp.text, resp.headers.get("etag", "") or ""


# ── the public door ──────────────────────────────────────────────────────────

def fetch(url: str, max_age_seconds: int = DEFAULT_MAX_AGE,
          headers: dict | None = None, allow_stale: bool = True):
    """Return (status, body) for `url`, from cache when fresh.

    - Fresh cache hit → returned without a network call.
    - Stale (or absent) → one conditional GET (If-None-Match when an etag is
      known). 304 refreshes the timestamp and serves the cached body; 2xx/other
      replaces the row.
    - Network error → (0, "") unless allow_stale and a prior row exists, in
      which case the stale body is served. Never raises.
    """
    now = _utcnow()
    key = cache_key(url)
    db = _connect()
    try:
        row = _read_row(db, key)
        if row is not None and is_fresh(row["fetched_at"], now, max_age_seconds):
            return int(row["status"] or 0), row["body"] or ""

        req_headers = {"User-Agent": USER_AGENT, **(headers or {})}
        if row is not None and row["etag"]:
            req_headers["If-None-Match"] = row["etag"]

        try:
            status, body, etag = _http_get(url, req_headers, TIMEOUT_SECONDS)
        except Exception:
            # network died — degrade, don't crash
            if allow_stale and row is not None:
                return int(row["status"] or 0), row["body"] or ""
            return 0, ""

        if status == 304 and row is not None:
            _touch(db, key, _iso(now))
            return int(row["status"] or 0), row["body"] or ""

        # keep the last known good etag if the server omitted one on a 200
        etag = etag or (row["etag"] if row is not None else "")
        _write_row(db, key, url, _iso(now), status, body, etag)
        return status, body
    finally:
        db.close()
