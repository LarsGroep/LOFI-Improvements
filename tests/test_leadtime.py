"""
Phase E tests — learned promotion threshold + detection lead time.

Pure functions on synthetic data + the ThresholdLearner agent end-to-end on a
tmp-path SQLite blackboard. No network, no Supabase, no LLM, no Streamlit
(same policy as tests/test_osk.py / tests/test_predict.py).
Run: python -m pytest tests/ -q
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from osk.blackboard import SQLiteBlackboard  # noqa: E402
from osk.orchestrator import run_agent  # noqa: E402
from osk.threshold_learner import (  # noqa: E402
    ThresholdLearner,
    build_samples,
    learn_threshold,
)
from predict.leadtime import evaluate_leadtime, evaluate_verdicts  # noqa: E402

UTC = dt.timezone.utc


def _iso(days_ago: int, now: dt.datetime | None = None) -> str:
    now = now or dt.datetime(2026, 7, 18, 12, 0, tzinfo=UTC)
    return (now - dt.timedelta(days=days_ago)).isoformat(timespec="seconds")


# ── learn_threshold ───────────────────────────────────────────────────────────

def test_learn_threshold_picks_f1_maximising_cutoff():
    # Heats ≥ 5 are the bookings; heats < 5 are noise. The clean separator at
    # cutoff 5.0 gives precision=recall=F1=1.0 — no other cutpoint beats it.
    samples = ([{"heat": h, "positive": True} for h in (5.0, 6.0, 7.0, 8.0)]
               + [{"heat": h, "positive": False} for h in
                  (1.0, 2.0, 3.0, 4.0, 4.5, 4.8)])
    out = learn_threshold(samples)
    assert out is not None
    assert out["threshold"] == 5.0
    assert out["f1"] == 1.0
    assert out["n"] == 10 and out["positives"] == 4


def test_learn_threshold_insufficiency_gate():
    # < 10 samples → None even though perfectly separable
    few = ([{"heat": 6.0, "positive": True}] * 3
           + [{"heat": 1.0, "positive": False}] * 4)
    assert learn_threshold(few) is None
    # ≥ 10 samples but < 3 positives → None
    thin_pos = ([{"heat": 6.0, "positive": True}] * 2
                + [{"heat": 1.0, "positive": False}] * 10)
    assert learn_threshold(thin_pos) is None


def test_learn_threshold_clamps_to_range():
    # Perfect separator sits at 0.5 (below the 2.0 floor) → clamps up to 2.0
    lo = ([{"heat": h, "positive": True} for h in (0.5, 0.6, 0.7, 0.8)]
          + [{"heat": 0.1, "positive": False} for _ in range(6)])
    assert learn_threshold(lo)["threshold"] == 2.0
    # Perfect separator at 12.0 (above the 10.0 ceiling) → clamps down to 10.0
    hi = ([{"heat": h, "positive": True} for h in (12.0, 13.0, 14.0, 15.0)]
          + [{"heat": 5.0, "positive": False} for _ in range(6)])
    assert learn_threshold(hi)["threshold"] == 10.0


# ── evaluate_leadtime ─────────────────────────────────────────────────────────

def _history(artist_id, series):
    """series = [(days_ago, listeners)] → forecast_log-shaped rows."""
    return [{"date": _iso(d)[:10], "artist_id": artist_id,
             "spotify_listeners": v} for d, v in series]


def test_leadtime_exact_median_and_excludes_late_promotions():
    bar = 75000.0
    # A crossed 60 days ago (logged below at 90d, above at 60d); promoted at 90d
    #   → lead 30. B crossed 40 days ago, promoted at 70d → lead 30.
    # C crossed 20 days ago but was promoted at 10d (AFTER crossing) → excluded.
    history = (_history("A", [(90, 40000), (60, 90000)])
               + _history("B", [(70, 50000), (40, 80000)])
               + _history("C", [(50, 60000), (20, 90000)]))
    promotions = [
        {"artist_id": "A", "artist_name": "Ava", "promoted_at": _iso(90)},
        {"artist_id": "B", "artist_name": "Ben", "promoted_at": _iso(70)},
        {"artist_id": "C", "artist_name": "Cid", "promoted_at": _iso(10)},
    ]
    rep = evaluate_leadtime(promotions, history, bar)
    assert rep["n_promoted"] == 3
    assert rep["n_crossed"] == 2
    assert rep["median_lead_days"] == 30.0
    assert {a["artist_name"] for a in rep["per_artist"]} == {"Ava", "Ben"}


def test_leadtime_requires_a_prior_below_reading():
    # Artist is only ever logged ABOVE the bar → we never watched the crossing.
    history = _history("Z", [(60, 90000), (30, 95000)])
    promotions = [{"artist_id": "Z", "artist_name": "Zoe",
                   "promoted_at": _iso(90)}]
    rep = evaluate_leadtime(promotions, history, 75000.0)
    assert rep["n_crossed"] == 0
    assert "bar" in rep["note"]


def test_leadtime_honest_empty_note():
    rep = evaluate_leadtime([], [], 75000.0)
    assert rep["n_promoted"] == 0 and rep["n_crossed"] == 0
    assert "note" in rep


# ── evaluate_verdicts ─────────────────────────────────────────────────────────

def test_verdicts_maturity_gate_and_precision():
    verdicts = [
        {"artist_name": "Ava", "verdict": "book_now", "created_at": _iso(120)},
        {"artist_name": "Ben", "verdict": "act_fast", "created_at": _iso(100)},
        {"artist_name": "Cid", "verdict": "book_now", "created_at": _iso(95)},
        {"artist_name": "Dot", "verdict": "monitor", "created_at": _iso(200)},
        # immature (< 90 days) — must be ignored entirely
        {"artist_name": "Eli", "verdict": "book_now", "created_at": _iso(10)},
    ]
    outcomes = {"Ava": True, "Ben": True, "Cid": False, "Dot": True, "Eli": True}
    rep = evaluate_verdicts(verdicts, outcomes)
    # 4 mature & scorable (Eli excluded by age)
    assert rep["n_mature"] == 4
    # positive verdicts: Ava, Ben, Cid → 2 booked / 3 = 66.7% precision
    assert rep["n_positive_verdicts"] == 3
    assert rep["precision_pct"] == 66.7
    # booked artists: Ava, Ben, Dot; flagged positive: Ava, Ben → recall 2/3
    assert rep["recall_pct"] == 66.7
    assert rep["buckets"]["book_now"] == {"n": 2, "booked": 1}


def test_verdicts_honest_empty_note():
    verdicts = [{"artist_name": "Ava", "verdict": "book_now",
                 "created_at": _iso(10)}]           # too fresh
    rep = evaluate_verdicts(verdicts, {"Ava": True})
    assert rep["n_mature"] == 0 and "note" in rep


# ── build_samples (labelling) ─────────────────────────────────────────────────

class _Rec:
    def __init__(self, artist_id, artist_name, payload):
        self.artist_id = artist_id
        self.artist_name = artist_name
        self.payload = payload


def test_build_samples_labels_from_verdicts_and_booked():
    matured = [_Rec("a1", "Ava", {"priority": 6.0}),
               _Rec("a2", "Ben", {"priority": 3.0}),
               _Rec("a3", "Cid", {"priority": 5.0})]
    verdicts = [_Rec("a1", "Ava", {"verdict": "book_now"})]
    booked = {"a3": True}                            # Cid booked via Supabase
    samples = build_samples(matured, verdicts, booked)
    by_heat = {s["heat"]: s["positive"] for s in samples}
    assert by_heat == {6.0: True, 3.0: False, 5.0: True}


# ── ThresholdLearner agent end-to-end ─────────────────────────────────────────

def _seed_candidate(bb, key, name, priority, days_ago):
    bb.emit("candidate", {"priority": priority}, "sentinel",
            artist_id=key, artist_name=name)
    # backdate created_at so the maturity gate (≥60d) sees it
    bb._db.execute("update records set created_at=? where id="
                   "(select max(id) from records)", (_iso(days_ago),))
    bb._db.commit()


def _run(bb):
    now = dt.datetime(2026, 7, 18, 12, 0, tzinfo=UTC)
    return run_agent(ThresholdLearner(), bb, now=now, trigger="cli")


def test_agent_emits_clamped_threshold(tmp_path, monkeypatch):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_KEY", raising=False)
    bb = SQLiteBlackboard(tmp_path / "os.db")
    # 4 positives at high heat, 6 negatives at low heat, all matured (90d old).
    for i in range(4):
        _seed_candidate(bb, f"pos{i}", f"Pos{i}", 6.0 + i, 90)
    for i in range(6):
        _seed_candidate(bb, f"neg{i}", f"Neg{i}", 1.0 + i * 0.2, 90)
    # verdicts make the 4 high-heat artists positive
    for i in range(4):
        bb.emit("verdict", {"verdict": "book_now"}, "judge",
                artist_id=f"pos{i}", artist_name=f"Pos{i}")

    res = _run(bb)
    assert res["error"] is None
    assert "learned promotion threshold" in res["outcome"]

    sigs = [r for r in bb.read(kinds=["signal"], limit=50)
            if r.payload.get("kind") == "promotion_threshold"]
    assert len(sigs) == 1
    val = sigs[0].payload["value"]
    assert 2.0 <= val <= 10.0
    assert "F1=" in sigs[0].payload["explain"]
    # the handoff: the Sentinel's state key is set for its next run
    assert bb.state_get("sentinel:promotion_threshold") == val
    bb.close()


def test_agent_thin_labels_no_emission(tmp_path, monkeypatch):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_KEY", raising=False)
    bb = SQLiteBlackboard(tmp_path / "os.db")
    # only 3 matured promotions → below the 10-sample floor
    for i in range(3):
        _seed_candidate(bb, f"c{i}", f"Cand{i}", 5.0, 90)

    res = _run(bb)
    assert res["error"] is None
    assert res["outcome"] == "not enough matured promotions (3/10) — " \
                             "keeping current threshold"
    sigs = [r for r in bb.read(kinds=["signal"], limit=50)
            if r.payload.get("kind") == "promotion_threshold"]
    assert sigs == []
    assert bb.state_get("sentinel:promotion_threshold") is None
    bb.close()
