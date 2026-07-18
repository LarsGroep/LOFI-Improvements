"""
Detector + sentinel tests — pure functions and a real SQLite blackboard on
tmp paths, synthetic signal fixtures only. No network, no Supabase, no
Streamlit imports (same policy as tests/test_osk.py / test_predict.py).
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from osk.blackboard import SQLiteBlackboard  # noqa: E402
from osk.detectors import DETECTOR_AGENTS  # noqa: E402
from osk.detectors.divergence import DivergenceDetector, detect_divergence  # noqa: E402
from osk.detectors.feelag import heat_rising, is_fee_lag  # noqa: E402
from osk.detectors.ignition import IgnitionDetector, detect_ignition  # noqa: E402
from osk.detectors.ladder import ladder_trend  # noqa: E402
from osk.detectors.scene import build_graph, centrality_delta  # noqa: E402
from osk.orchestrator import Context, run_agent  # noqa: E402
from osk.sentinel import Sentinel, compute_heat, match_watches  # noqa: E402

UTC = dt.timezone.utc
NOW = dt.datetime(2026, 7, 18, 6, 0, tzinfo=UTC)


def _iso(days_ago: int) -> str:
    return (NOW - dt.timedelta(days=days_ago)).date().isoformat()


def _emit_signal(bb, kind, artist, value=None, days_ago=0, artist_id=None):
    """Write a Phase-B-shaped signal record straight to the blackboard."""
    bb.emit("signal", {
        "source": "test", "kind": kind, "value": value, "evidence": {},
        "observed_at": _iso(days_ago),
    }, emitted_by="test_listener", artist_id=artist_id, artist_name=artist)


# ── divergence math ───────────────────────────────────────────────────────────

def test_divergence_positive_gap():
    # A has strong underground velocity; B and C are flat. A must diverge up.
    signals = ([{"artist": "A", "kind": "sc_plays_velocity", "value": 100.0}]
               + [{"artist": "A", "kind": "press_mention", "value": None}]
               + [{"artist": "B", "kind": "sc_plays_velocity", "value": 1.0}]
               + [{"artist": "C", "kind": "playlist_add", "value": None}])
    out = detect_divergence(signals, mainstream={}, min_gap=0.5)
    top = out[0]
    assert top["artist"] == "A"
    assert top["value"] > 0
    assert "cheap-booking window" in top["explain"]


def test_divergence_negative_when_mainstream_outpaces():
    # A has huge Spotify growth but no underground signals → negative gap.
    signals = [{"artist": "A", "kind": "playlist_add", "value": None},
               {"artist": "B", "kind": "sc_plays_velocity", "value": 80.0},
               {"artist": "C", "kind": "press_mention", "value": None}]
    out = detect_divergence(signals, mainstream={"A": 90.0, "B": 1.0, "C": 1.0},
                            min_gap=0.5)
    a = next(r for r in out if r["artist"] == "A")
    assert a["value"] < 0
    assert a["mainstream_present"] is True
    assert "playlist-inflated" in a["explain"]


def test_divergence_needs_population():
    assert detect_divergence([{"artist": "A", "kind": "press_mention"}]) == []


# ── ignition distinct-kind logic + idempotence ────────────────────────────────

def test_ignition_needs_two_distinct_kinds():
    sigs = [{"artist": "A", "kind": "label_signing", "value": None,
             "when": NOW - dt.timedelta(days=10)},
            {"artist": "A", "kind": "press_mention", "value": None,
             "when": NOW - dt.timedelta(days=5)},
            {"artist": "B", "kind": "press_mention", "value": None,
             "when": NOW - dt.timedelta(days=5)},
            {"artist": "B", "kind": "press_mention", "value": None,
             "when": NOW - dt.timedelta(days=4)}]
    out = detect_ignition(sigs, NOW)
    keys = {r["artist"] for r in out}
    assert keys == {"A"}                          # B has only one distinct kind
    assert out[0]["value"] == 2.0
    assert "label_signing" in out[0]["explain"]


def test_ignition_sc_threshold_gates():
    sigs = [{"artist": "A", "kind": "playlist_add", "value": None,
             "when": NOW - dt.timedelta(days=3)},
            {"artist": "A", "kind": "sc_plays_velocity", "value": 2.0,
             "when": NOW - dt.timedelta(days=2)}]
    assert detect_ignition(sigs, NOW, sc_min=5.0) == []   # sc below floor
    assert detect_ignition(sigs, NOW, sc_min=1.0)         # above floor → fires


def test_ignition_idempotent_until_combo_grows(tmp_path):
    bb = SQLiteBlackboard(tmp_path / "os.db")
    _emit_signal(bb, "label_signing", "Toman", days_ago=10)
    _emit_signal(bb, "press_mention", "Toman", days_ago=5)
    agent = IgnitionDetector()

    r1 = run_agent(agent, bb, now=NOW, trigger="cli")
    assert "1 new ignition" in r1["outcome"]
    # second run, same combo → no new emission (idempotent)
    r2 = run_agent(agent, bb, now=NOW, trigger="cli")
    assert "0 new ignition" in r2["outcome"]
    # combo grows to 3 kinds → re-emit
    _emit_signal(bb, "playlist_add", "Toman", days_ago=1)
    r3 = run_agent(agent, bb, now=NOW, trigger="cli")
    assert "1 new ignition" in r3["outcome"]
    ignitions = [r for r in bb.read(kinds=["signal"])
                 if r.payload.get("kind") == "ignition"]
    assert len(ignitions) == 2
    bb.close()


# ── ladder_trend ──────────────────────────────────────────────────────────────

def _ev(day, venue, cap=None):
    return {"date": (dt.date(2026, 1, 1) + dt.timedelta(days=day)),
            "venue": venue, "city": "Utrecht", "capacity": cap}


def test_ladder_capacity_basis_rising():
    events = [_ev(0, "v1", 100), _ev(20, "v2", 120), _ev(40, "v3", 110),
              _ev(150, "v4", 300), _ev(170, "v5", 320), _ev(190, "v6", 280)]
    out = ladder_trend(events)
    assert out["basis"] == "capacity"
    assert out["rising"] is True
    assert out["second_median"] > out["first_median"]


def test_ladder_proxy_basis_when_no_capacity():
    events = [_ev(0, "v1"), _ev(60, "v1"), _ev(90, "v2"),
              _ev(130, "v3"), _ev(140, "v4"), _ev(150, "v5")]
    out = ladder_trend(events)
    assert out["basis"] == "proxy"
    assert "no capacity data" in out["explain"]


def test_ladder_insufficient():
    assert ladder_trend([_ev(0, "v1", 100), _ev(10, "v2", 200)]) is None
    # 6 events but spanning < 120 days
    tight = [_ev(i * 5, f"v{i}", 100) for i in range(6)]
    assert ladder_trend(tight) is None


# ── scene pure functions ──────────────────────────────────────────────────────

def test_build_graph_is_weighted_and_undirected():
    g = build_graph([("a", "b"), ("a", "b"), ("b", "c")])
    assert g["a"]["b"] == 2 and g["b"]["a"] == 2
    assert g["b"]["c"] == 1
    assert build_graph([("a", "a")]) == {}        # self-loops dropped


def test_centrality_delta_toward_core():
    core = {"core1", "core2"}
    then = build_graph([("x", "other")])
    now = build_graph([("x", "core1"), ("x", "core2"), ("x", "other")])
    delta = centrality_delta(then, now, core)
    assert delta["x"] > 0                          # pulled toward the core


# ── heat decay + weights ──────────────────────────────────────────────────────

def _hv(key, kind, days_ago, value=None, name=None):
    return {"key": key, "artist_name": name or key, "kind": kind,
            "value": value, "when": NOW - dt.timedelta(days=days_ago),
            "explain": None}


def test_heat_older_signals_count_less():
    fresh = compute_heat([_hv("A", "press_mention", 0)], NOW)["A"]["heat"]
    old = compute_heat([_hv("A", "press_mention", 40)], NOW)["A"]["heat"]
    assert fresh > old > 0


def test_heat_label_signing_outlasts_press():
    label = compute_heat([_hv("A", "label_signing", 90)], NOW)["A"]["heat"]
    press = compute_heat([_hv("B", "press_mention", 90)], NOW)["B"]["heat"]
    assert label > press                           # weight 3 + half-life 120


def test_heat_derived_kinds_weigh_double():
    div = compute_heat([_hv("A", "divergence", 0)], NOW)["A"]["heat"]
    press = compute_heat([_hv("B", "press_mention", 0)], NOW)["B"]["heat"]
    assert abs(div - 2 * press) < 1e-6


# ── feelag helpers ────────────────────────────────────────────────────────────

def test_heat_rising_positive_only():
    sigs = [{"artist": "A", "kind": "divergence", "value": 2.0,
             "when": NOW - dt.timedelta(days=3)},
            {"artist": "B", "kind": "divergence", "value": -1.5,
             "when": NOW - dt.timedelta(days=3)},
            {"artist": "C", "kind": "ignition", "value": 2.0,
             "when": NOW - dt.timedelta(days=3)}]
    rising = heat_rising(sigs, NOW)
    assert set(rising) == {"A", "C"}               # B's negative gap excluded


def test_is_fee_lag_requires_comparable_static_basis():
    assert is_fee_lag({"method": "comparables", "base": 3000}, None) is True
    assert is_fee_lag({"method": "comparables", "base": 3000}, 3050) is True
    assert is_fee_lag({"method": "comparables", "base": 3000}, 5000) is False
    assert is_fee_lag({"method": "own_gages", "base": 3000}, None) is False
    assert is_fee_lag({"method": "insufficient", "base": None}, None) is False


# ── sentinel promotion + cooldown + re-hearing (real blackboard) ──────────────

def _sentinel_signals(bb):
    # Enough heat on one artist to clear the default 4.0 threshold:
    # a label signing (weight 3) + ignition (weight 2) + divergence (weight 2).
    _emit_signal(bb, "label_signing", "Rising", days_ago=1)
    bb.emit("signal", {"source": "ignition_detector", "kind": "ignition",
                       "value": 2.0, "explain": "ignition: label + press"},
            emitted_by="ignition_detector", artist_name="Rising")
    bb.emit("signal", {"source": "divergence_detector", "kind": "divergence",
                       "value": 1.5, "explain": "underground outrun"},
            emitted_by="divergence_detector", artist_name="Rising")


def test_sentinel_promotes_and_respects_cooldown(tmp_path):
    bb = SQLiteBlackboard(tmp_path / "os.db")
    _sentinel_signals(bb)
    agent = Sentinel()

    r1 = run_agent(agent, bb, now=NOW, trigger="cli")
    assert "promoted 1" in r1["outcome"]
    cands = bb.read(kinds=["candidate"])
    alerts = bb.read(kinds=["alert"])
    assert len(cands) == 1 and len(alerts) == 1
    assert cands[0].payload["trigger_signals"]

    # a day later the same artist is in cooldown → no second promotion
    r2 = run_agent(agent, bb, now=NOW + dt.timedelta(days=1), trigger="cli")
    assert "none crossed" in r2["outcome"] or "promoted 0" in r2["outcome"]
    assert len(bb.read(kinds=["candidate"])) == 1

    # past the 30-day cooldown → promotes again
    r3 = run_agent(agent, bb, now=NOW + dt.timedelta(days=31), trigger="cli")
    assert "promoted 1" in r3["outcome"]
    bb.close()


def test_sentinel_rehearing_promotes_below_threshold(tmp_path):
    bb = SQLiteBlackboard(tmp_path / "os.db")
    # A lone press mention — nowhere near the heat threshold on its own.
    _emit_signal(bb, "press_mention", "Sleeper", days_ago=1)
    # Standing watch from a (future) Judge dissent: any press ≥ 0 re-hears.
    bb.state_set("sentinel:watches", [
        {"artist_name": "Sleeper", "kind": "press_mention",
         "threshold": 0, "reason": "NL footprint objection from March"}])

    res = run_agent(Sentinel(), bb, now=NOW, trigger="cli")
    assert "promoted 1" in res["outcome"]
    cand = bb.read(kinds=["candidate"])[0]
    assert any("re-hearing" in t for t in cand.payload["trigger_signals"])
    bb.close()


def test_match_watches_picks_highest_value():
    sigs = [{"key": "A", "artist_name": "A", "kind": "sc_plays_velocity",
             "value": 50.0},
            {"key": "A", "artist_name": "A", "kind": "sc_plays_velocity",
             "value": 80.0}]
    watches = [{"artist_name": "A", "kind": "sc_plays_velocity",
                "threshold": 40, "reason": "watch"}]
    out = match_watches(watches, sigs)
    assert out["A"]["value"] == 80.0


# ── honest-empty / skipped on an empty blackboard ─────────────────────────────

def test_all_detectors_honest_on_empty_blackboard(tmp_path, monkeypatch):
    monkeypatch.delenv("AIRTABLE_TOKEN", raising=False)
    monkeypatch.delenv("AIRTABLE_BASE_ID", raising=False)
    bb = SQLiteBlackboard(tmp_path / "os.db")
    for cls in DETECTOR_AGENTS:
        res = run_agent(cls(), bb, now=NOW, trigger="cli")
        assert res["error"] is None
        assert (res["outcome"].startswith("no data")
                or res["outcome"].startswith("skipped")), (cls, res)
    # sentinel too
    res = run_agent(Sentinel(), bb, now=NOW, trigger="cli")
    assert res["outcome"].startswith("no signals")
    bb.close()


def test_context_state_namespacing_for_ignition(tmp_path):
    """Idempotence state is namespaced under the agent — sanity that Context
    reads/writes what the agent expects."""
    bb = SQLiteBlackboard(tmp_path / "os.db")
    ctx = Context(IgnitionDetector.manifest, bb, {}, NOW)
    ctx.state_set("emitted", {"X": 2.0})
    assert bb.state_get("ignition_detector:emitted") == {"X": 2.0}
    assert ctx.state_get("emitted") == {"X": 2.0}
    bb.close()
