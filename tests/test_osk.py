"""
Kernel tests — pure functions + SQLite blackboard on tmp paths, fake agents.
No network, no Supabase, no LLM (same policy as tests/test_predict.py).
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from osk.blackboard import SQLiteBlackboard  # noqa: E402
from osk.manifest import AgentManifest, Budget  # noqa: E402
from osk.orchestrator import run_agent, tick  # noqa: E402
from osk.registry import Registry, default_registry  # noqa: E402
from osk.scheduler import is_due, last_slot, validate_schedule  # noqa: E402

UTC = dt.timezone.utc


def _t(*args):
    return dt.datetime(*args, tzinfo=UTC)


# ── scheduler ────────────────────────────────────────────────────────────────

def test_schedule_validation():
    for ok in ("hourly", "daily", "daily@03:10", "weekly@mon 04:00",
               "every:15m", "every:6h"):
        validate_schedule(ok)
    for bad in ("daily@25:00", "weekly@xyz 04:00", "every:0m", "sometimes"):
        with pytest.raises(ValueError):
            validate_schedule(bad)


def test_daily_slot_and_dueness():
    now = _t(2026, 7, 18, 8, 0)
    assert last_slot("daily@03:10", now) == _t(2026, 7, 18, 3, 10)
    # before today's slot time → slot is yesterday's
    assert last_slot("daily@03:10", _t(2026, 7, 18, 1, 0)) == \
        _t(2026, 7, 17, 3, 10)
    assert is_due("daily@03:10", None, now)                      # never ran
    assert is_due("daily@03:10", _t(2026, 7, 17, 3, 11), now)    # ran yesterday
    assert not is_due("daily@03:10", _t(2026, 7, 18, 3, 12), now)  # ran today


def test_weekly_slot():
    # 2026-07-18 is a Saturday; most recent Mon 04:00 is 2026-07-13
    now = _t(2026, 7, 18, 8, 0)
    assert last_slot("weekly@mon 04:00", now) == _t(2026, 7, 13, 4, 0)
    assert is_due("weekly@mon 04:00", _t(2026, 7, 6, 4, 1), now)
    assert not is_due("weekly@mon 04:00", _t(2026, 7, 13, 4, 1), now)


def test_interval_and_event_only():
    now = _t(2026, 7, 18, 8, 0)
    assert is_due("every:15m", _t(2026, 7, 18, 7, 44), now)
    assert not is_due("every:15m", _t(2026, 7, 18, 7, 50), now)
    assert not is_due(None, None, now)   # event-only agents are never time-due


# ── blackboard ───────────────────────────────────────────────────────────────

def test_blackboard_roundtrip(tmp_path):
    bb = SQLiteBlackboard(tmp_path / "os.db")
    bb.emit("signal", {"kind": "x", "v": 1}, "tester", artist_id="a1",
            artist_name="Toman")
    bb.emit("alert", {"reasons": ["momentum"]}, "tester", artist_id="a1")
    assert [r.kind for r in bb.read()] == ["alert", "signal"]
    assert bb.read(kinds=["alert"])[0].payload["reasons"] == ["momentum"]
    assert len(bb.read(artist_id="a1")) == 2

    bb.state_set("k", {"nested": True})
    assert bb.state_get("k") == {"nested": True}
    assert bb.state_get("missing") is None

    run_id = bb.run_start("tester", "cli")
    bb.run_finish(run_id, "ok", records_emitted=2)
    assert bb.runs("tester")[0]["outcome"] == "ok"
    assert bb.last_run_at("tester") is not None
    bb.close()


# ── manifests + registry ─────────────────────────────────────────────────────

def test_manifest_rejects_unknown_kind_and_llm_mismatch():
    with pytest.raises(ValueError):
        AgentManifest(name="x", writes=("gossip",)).validate()
    with pytest.raises(ValueError):
        AgentManifest(name="x", llm=False,
                      budget=Budget(max_llm_calls_per_run=3)).validate()


class _FakeAgent:
    manifest = AgentManifest(name="fake", writes=("signal",),
                             schedule="daily@00:00")

    def run(self, ctx):
        ctx.emit("signal", {"hello": "world"})
        return "did the thing"


class _CrashingAgent:
    manifest = AgentManifest(name="crasher", schedule="daily@00:00")

    def run(self, ctx):
        raise RuntimeError("boom")


def test_registry_validates():
    reg = Registry()
    reg.register(_FakeAgent())
    with pytest.raises(ValueError):
        reg.register(_FakeAgent())          # duplicate name
    with pytest.raises(TypeError):
        reg.register(object())              # no manifest


def test_default_registry_ships_the_roster():
    names = default_registry().names()
    for expected in ("backtest_reporter", "forecast_logger",
                     "rank_weights_refit", "watchlist_sentinel",
                     "soundcloud_listener", "playlist_listener",
                     "label_radar", "press_listener"):
        assert expected in names


# ── orchestrator ─────────────────────────────────────────────────────────────

def test_run_agent_audits_and_enforces_manifest(tmp_path):
    bb = SQLiteBlackboard(tmp_path / "os.db")
    res = run_agent(_FakeAgent(), bb, trigger="cli")
    assert res["outcome"] == "did the thing"
    run = bb.runs("fake")[0]
    assert run["outcome"] == "did the thing"
    assert run["records_emitted"] == 1
    assert bb.read(kinds=["signal"])[0].emitted_by == "fake"

    # a crash is captured, never raised
    res = run_agent(_CrashingAgent(), bb)
    assert res["outcome"] == "error" and "boom" in res["error"]
    assert bb.runs("crasher")[0]["error"].endswith("boom")
    bb.close()


class _Overreacher:
    manifest = AgentManifest(name="overreacher", writes=("signal",))

    def run(self, ctx):
        ctx.emit("verdict", {"v": "book_now"})   # not in manifest.writes


def test_context_blocks_undeclared_writes_and_llm(tmp_path):
    bb = SQLiteBlackboard(tmp_path / "os.db")
    res = run_agent(_Overreacher(), bb)
    assert res["outcome"] == "error" and "may not emit" in res["error"]
    assert bb.read(kinds=["verdict"]) == []      # nothing leaked through

    from osk.orchestrator import Context
    ctx = Context(_FakeAgent.manifest, bb, {}, _t(2026, 7, 18, 8, 0))
    with pytest.raises(PermissionError):
        _ = ctx.llm                              # llm=False agents get no egress
    bb.close()


def test_tick_runs_due_agents_once(tmp_path):
    bb = SQLiteBlackboard(tmp_path / "os.db")
    reg = Registry()
    reg.register(_FakeAgent())
    now = _t(2026, 7, 18, 8, 0)
    assert [r["agent"] for r in tick(reg, bb, now=now)] == ["fake"]
    assert tick(reg, bb, now=now) == []          # not due again today
    # force ignores the schedule
    assert tick(reg, bb, now=now, force="fake")[0]["outcome"] == "did the thing"
    bb.close()


def test_tick_parks_over_budget_agents(tmp_path):
    class Greedy:
        manifest = AgentManifest(name="greedy", schedule="every:1m",
                                 budget=Budget(max_runs_per_day=1))

        def run(self, ctx):
            return "ran"

    bb = SQLiteBlackboard(tmp_path / "os.db")
    reg = Registry()
    reg.register(Greedy())
    # run_start stamps real wall-clock time, so anchor the fake `now` to it —
    # a fixed date would drift out of agreement with the audit log.
    t0 = dt.datetime.now(UTC)
    assert tick(reg, bb, now=t0)[0]["outcome"] == "ran"
    res = tick(reg, bb, now=t0 + dt.timedelta(minutes=5))
    assert res[0]["outcome"] == "parked"
    assert "budget" in res[0]["error"]
    bb.close()


def test_kill_switch(tmp_path, monkeypatch):
    bb = SQLiteBlackboard(tmp_path / "os.db")
    reg = Registry()
    reg.register(_FakeAgent())
    monkeypatch.setenv("LOFI_OS_PAUSED", "1")
    assert tick(reg, bb, now=_t(2026, 7, 18, 8, 0)) == []
    # force still works while paused (deliberate: operators can test agents)
    assert tick(reg, bb, now=_t(2026, 7, 18, 8, 0), force="fake")
    bb.close()


def test_builtin_agents_skip_gracefully_without_supabase(tmp_path, monkeypatch):
    from osk.agents_builtin import BUILTIN_AGENTS
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_KEY", raising=False)
    bb = SQLiteBlackboard(tmp_path / "os.db")
    for cls in BUILTIN_AGENTS:
        res = run_agent(cls(), bb, trigger="cli")
        assert res["outcome"].startswith("skipped"), res
        assert res["error"] is None
    bb.close()
