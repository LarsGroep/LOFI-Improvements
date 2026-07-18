"""
The blackboard — agents' only IPC, and the OS's filesystem.

Agents never call each other; they emit typed records and read what others
emitted. Two interchangeable backends behind one API:

  - SQLite (default): zero-dependency, VPS-friendly, lives in osk/data/os.db
    (gitignored, venue-private — same policy as predict/data/).
  - Supabase: the `agentic` schema (deploy/migrations/001_agentic_schema.sql),
    for when the dashboard should read the feed too. Select with
    LOFI_OS_BACKEND=supabase.

Both store the same three tables: records (the feed), agent_runs (syslog),
state (per-agent key/value, e.g. watchlist snapshots).
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path

_DEFAULT_DB = Path(__file__).parent / "data" / "os.db"


def _utcnow() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def parse_ts(value) -> _dt.datetime | None:
    if not value:
        return None
    try:
        ts = _dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return ts if ts.tzinfo else ts.replace(tzinfo=_dt.timezone.utc)


@dataclass(frozen=True)
class Record:
    id: int | None
    kind: str
    payload: dict
    emitted_by: str
    artist_id: str | None = None
    artist_name: str | None = None
    created_at: str = ""


# ── SQLite backend (default) ─────────────────────────────────────────────────

_SCHEMA_SQL = """
create table if not exists records (
  id integer primary key autoincrement,
  kind text not null,
  artist_id text,
  artist_name text,
  payload text not null default '{}',
  emitted_by text not null,
  created_at text not null);
create index if not exists idx_records_kind on records(kind, created_at desc);
create index if not exists idx_records_artist on records(artist_id);
create table if not exists agent_runs (
  id integer primary key autoincrement,
  agent text not null,
  trigger text,
  started_at text not null,
  finished_at text,
  outcome text,
  error text,
  llm_calls integer not null default 0,
  tokens_in integer not null default 0,
  tokens_out integer not null default 0,
  records_emitted integer not null default 0);
create index if not exists idx_runs_agent on agent_runs(agent, started_at desc);
create table if not exists state (
  key text primary key,
  value text,
  updated_at text not null);
"""


class SQLiteBlackboard:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or os.environ.get("LOFI_OS_DB") or _DEFAULT_DB)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path)
        self._db.row_factory = sqlite3.Row
        self._db.execute("pragma journal_mode=wal")
        self._db.executescript(_SCHEMA_SQL)

    def describe(self) -> str:
        return f"sqlite:{self.path}"

    # records
    def emit(self, kind: str, payload: dict, emitted_by: str,
             artist_id: str | None = None,
             artist_name: str | None = None) -> int:
        cur = self._db.execute(
            "insert into records (kind, artist_id, artist_name, payload, "
            "emitted_by, created_at) values (?,?,?,?,?,?)",
            (kind, artist_id, artist_name,
             json.dumps(payload, ensure_ascii=False), emitted_by, _utcnow()))
        self._db.commit()
        return int(cur.lastrowid)

    def read(self, kinds: list[str] | None = None,
             artist_id: str | None = None, since: str | None = None,
             limit: int = 100) -> list[Record]:
        sql, args = "select * from records where 1=1", []
        if kinds:
            sql += f" and kind in ({','.join('?' * len(kinds))})"
            args += list(kinds)
        if artist_id:
            sql += " and artist_id = ?"
            args.append(artist_id)
        if since:
            sql += " and created_at >= ?"
            args.append(since)
        sql += " order by id desc limit ?"
        args.append(limit)
        out = []
        for r in self._db.execute(sql, args):
            out.append(Record(
                id=r["id"], kind=r["kind"],
                payload=json.loads(r["payload"] or "{}"),
                emitted_by=r["emitted_by"], artist_id=r["artist_id"],
                artist_name=r["artist_name"], created_at=r["created_at"]))
        return out

    # syslog
    def run_start(self, agent: str, trigger: str) -> int:
        cur = self._db.execute(
            "insert into agent_runs (agent, trigger, started_at) values (?,?,?)",
            (agent, trigger, _utcnow()))
        self._db.commit()
        return int(cur.lastrowid)

    def run_finish(self, run_id: int, outcome: str, error: str | None = None,
                   llm_calls: int = 0, tokens_in: int = 0,
                   tokens_out: int = 0, records_emitted: int = 0) -> None:
        self._db.execute(
            "update agent_runs set finished_at=?, outcome=?, error=?, "
            "llm_calls=?, tokens_in=?, tokens_out=?, records_emitted=? "
            "where id=?",
            (_utcnow(), outcome, error, llm_calls, tokens_in, tokens_out,
             records_emitted, run_id))
        self._db.commit()

    def runs(self, agent: str | None = None, limit: int = 50) -> list[dict]:
        sql, args = "select * from agent_runs", []
        if agent:
            sql += " where agent = ?"
            args.append(agent)
        sql += " order by id desc limit ?"
        args.append(limit)
        return [dict(r) for r in self._db.execute(sql, args)]

    def last_run_at(self, agent: str) -> _dt.datetime | None:
        r = self._db.execute(
            "select max(started_at) as ts from agent_runs where agent=?",
            (agent,)).fetchone()
        return parse_ts(r["ts"] if r else None)

    def runs_since(self, agent: str, since: str) -> int:
        r = self._db.execute(
            "select count(*) as n from agent_runs where agent=? and "
            "started_at >= ?", (agent, since)).fetchone()
        return int(r["n"])

    # state
    def state_get(self, key: str):
        r = self._db.execute("select value from state where key=?",
                             (key,)).fetchone()
        return json.loads(r["value"]) if r and r["value"] is not None else None

    def state_set(self, key: str, value) -> None:
        self._db.execute(
            "insert into state (key, value, updated_at) values (?,?,?) "
            "on conflict(key) do update set value=excluded.value, "
            "updated_at=excluded.updated_at",
            (key, json.dumps(value, ensure_ascii=False), _utcnow()))
        self._db.commit()

    def close(self) -> None:
        self._db.close()


# ── Supabase backend (opt-in) ────────────────────────────────────────────────

class SupabaseBlackboard:
    """Same API over the `agentic` schema. Apply
    deploy/migrations/001_agentic_schema.sql to the project first."""

    _SCHEMA = "agentic"

    def __init__(self, client=None):
        if client is None:
            from scout.data import make_client
            client = make_client()
        self._c = client

    def describe(self) -> str:
        return f"supabase:{os.environ.get('SUPABASE_URL', '?')}/agentic"

    def _t(self, table: str):
        return self._c.schema(self._SCHEMA).table(table)

    def emit(self, kind, payload, emitted_by, artist_id=None,
             artist_name=None) -> int:
        row = self._t("records").insert({
            "kind": kind, "artist_id": artist_id, "artist_name": artist_name,
            "payload": payload, "emitted_by": emitted_by,
            "created_at": _utcnow()}).execute().data
        return int(row[0]["id"]) if row else 0

    def read(self, kinds=None, artist_id=None, since=None,
             limit=100) -> list[Record]:
        q = self._t("records").select("*")
        if kinds:
            q = q.in_("kind", list(kinds))
        if artist_id:
            q = q.eq("artist_id", artist_id)
        if since:
            q = q.gte("created_at", since)
        rows = q.order("id", desc=True).limit(limit).execute().data or []
        return [Record(id=r.get("id"), kind=r["kind"],
                       payload=r.get("payload") or {},
                       emitted_by=r.get("emitted_by") or "",
                       artist_id=r.get("artist_id"),
                       artist_name=r.get("artist_name"),
                       created_at=str(r.get("created_at") or ""))
                for r in rows]

    def run_start(self, agent, trigger) -> int:
        row = self._t("agent_runs").insert({
            "agent": agent, "trigger": trigger,
            "started_at": _utcnow()}).execute().data
        return int(row[0]["id"]) if row else 0

    def run_finish(self, run_id, outcome, error=None, llm_calls=0,
                   tokens_in=0, tokens_out=0, records_emitted=0) -> None:
        self._t("agent_runs").update({
            "finished_at": _utcnow(), "outcome": outcome, "error": error,
            "llm_calls": llm_calls, "tokens_in": tokens_in,
            "tokens_out": tokens_out, "records_emitted": records_emitted,
        }).eq("id", run_id).execute()

    def runs(self, agent=None, limit=50) -> list[dict]:
        q = self._t("agent_runs").select("*")
        if agent:
            q = q.eq("agent", agent)
        return q.order("id", desc=True).limit(limit).execute().data or []

    def last_run_at(self, agent) -> _dt.datetime | None:
        rows = (self._t("agent_runs").select("started_at").eq("agent", agent)
                .order("started_at", desc=True).limit(1).execute().data)
        return parse_ts(rows[0]["started_at"]) if rows else None

    def runs_since(self, agent, since) -> int:
        rows = (self._t("agent_runs").select("id").eq("agent", agent)
                .gte("started_at", since).execute().data)
        return len(rows or [])

    def state_get(self, key):
        rows = (self._t("state").select("value").eq("key", key)
                .limit(1).execute().data)
        return rows[0]["value"] if rows else None

    def state_set(self, key, value) -> None:
        self._t("state").upsert({
            "key": key, "value": value, "updated_at": _utcnow()}).execute()

    def close(self) -> None:
        pass


# ── factory ──────────────────────────────────────────────────────────────────

def open_blackboard():
    """LOFI_OS_BACKEND=supabase → Supabase; anything else → SQLite."""
    backend = os.environ.get("LOFI_OS_BACKEND", "sqlite").strip().lower()
    if backend == "supabase":
        return SupabaseBlackboard()
    return SQLiteBlackboard()
