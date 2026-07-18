"""
Phase D tests — the Deliberation Chamber, the dissent → re-hearing hook, and the
Dossier Curator. No network, no live LLM: LOFI_LLM_ENABLED is forced off so the
full protocol runs on deterministic, fact-derived mock output (same policy as
tests/test_osk.py). SQLiteBlackboard on tmp_path + the kernel Context.
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from osk.blackboard import SQLiteBlackboard  # noqa: E402
from osk.chamber import (  # noqa: E402
    VERDICTS, Chamber, chamber_view, strike_claims)
from osk.curator import DossierCurator  # noqa: E402
from osk.orchestrator import Context, run_agent  # noqa: E402
from osk.sentinel import Sentinel  # noqa: E402

UTC = dt.timezone.utc


@pytest.fixture(autouse=True)
def _force_mock(monkeypatch):
    """Guarantee mock mode + no external-data paths for every test here."""
    for var in ("LOFI_LLM_ENABLED", "ANTHROPIC_API_KEY", "SUPABASE_URL",
                "SUPABASE_KEY", "AIRTABLE_TOKEN"):
        monkeypatch.delenv(var, raising=False)


def _bb(tmp_path):
    return SQLiteBlackboard(tmp_path / "os.db")


def _ctx(cls, bb, now=None):
    return Context(cls.manifest, bb, {}, now or dt.datetime.now(UTC))


def _seed_candidate(bb, aid="a1", name="Toman"):
    bb.emit("candidate", {"priority": 5.0, "heat": 5.0,
                          "trigger_signals": ["ignition (3)", "press mention"]},
            "sentinel", artist_id=aid, artist_name=name)


def _seed_signals(bb, aid="a1", name="Toman"):
    bb.emit("signal", {"kind": "ignition", "value": 3.0, "explain": "ignition"},
            "ignition_detector", artist_id=aid, artist_name=name)
    bb.emit("signal", {"kind": "press_mention", "value": 1.0},
            "press_listener", artist_id=aid, artist_name=name)


# ── chamber_view minimisation ────────────────────────────────────────────────

def test_chamber_view_minimises_and_pointers(tmp_path):
    brief = {
        "artist_id": "a1", "artist_name": "Toman",
        "candidate": {"priority": 5.0, "trigger_signals": ["x"],
                      "secret": "LEAK-CAND"},
        "signals": [{"key": "a1", "kind": "ignition", "value": 3.0,
                     "when": dt.datetime.now(UTC), "explain": "e",
                     "secret": "LEAK-SIG"}],
        "heat": {"heat": 4.2, "top_reasons": ["ignition"], "secret": "LEAK-HEAT"},
        "model_estimates": None,
    }
    view = chamber_view(brief)
    import json
    blob = json.dumps(view)
    assert "LEAK" not in blob                      # nothing non-allow-listed leaks
    ids = [e["id"] for e in view["evidence"]]
    assert ids[:3] == ["E1", "E2", "E3"]           # stable pointer ids
    assert view["evidence"][0]["kind"] == "candidate"
    assert view["evidence"][1]["kind"] == "signal"


# ── evidence-pointer striking ────────────────────────────────────────────────

def test_strike_claims_drops_ungrounded():
    claims = [
        {"text": "grounded", "evidence": ["E1", "E99"]},   # keep, prune E99
        {"text": "phantom", "evidence": ["E99"]},          # dropped
        {"text": "empty", "evidence": []},                 # dropped
    ]
    out = strike_claims(claims, {"E1", "E2"})
    texts = [c["text"] for c in out]
    assert texts == ["grounded"]
    assert out[0]["evidence"] == ["E1"]


# ── the full mock debate ─────────────────────────────────────────────────────

def test_full_mock_debate(tmp_path):
    bb = _bb(tmp_path)
    _seed_candidate(bb)
    _seed_signals(bb)
    ctx = _ctx(Chamber, bb)
    outcome = Chamber().run(ctx)

    assert outcome.startswith("[mock]")
    assert len(bb.read(kinds=["brief"])) == 1
    args = bb.read(kinds=["argument"])
    assert len(args) == 3                          # advocate, skeptic, judge
    roles = {a.payload["role"] for a in args}
    assert roles == {"advocate", "skeptic", "judge"}

    verds = bb.read(kinds=["verdict"])
    assert len(verds) == 1
    vp = verds[0].payload
    assert vp["verdict"] in VERDICTS
    assert vp["dissent"] and vp["dissent"]["kind"] == "ignition"
    assert 0 < ctx.llm_calls <= 5                  # hard-capped protocol

    # every stored claim carries at least one evidence pointer
    for a in args:
        for c in a.payload["claims"]:
            assert c["evidence"]


# ── unheard-candidate selection ──────────────────────────────────────────────

def test_recent_verdict_skips_candidate(tmp_path):
    bb = _bb(tmp_path)
    # heard: has a verdict already → skipped
    _seed_candidate(bb, aid="a1", name="Toman")
    bb.emit("verdict", {"debate_id": "old", "verdict": "monitor",
                        "dissent": {}}, "chamber", artist_id="a1",
            artist_name="Toman")
    # unheard: fresh candidate with signals → deliberated
    _seed_candidate(bb, aid="a2", name="Rising")
    _seed_signals(bb, aid="a2", name="Rising")

    run_agent(Chamber(), bb, trigger="cli")
    briefed = {b.artist_name for b in bb.read(kinds=["brief"])}
    assert "Rising" in briefed
    assert "Toman" not in briefed


# ── sentinel dissent → re-hearing watch ──────────────────────────────────────

def test_sentinel_promotes_on_dissent_watch(tmp_path):
    bb = _bb(tmp_path)
    now = dt.datetime.now(UTC)
    # A Judge dissent watching `ignition` above 2.0 — nowhere near heat threshold.
    bb.emit("verdict", {"debate_id": "d1", "verdict": "monitor",
                        "dissent": {"kind": "ignition", "threshold": 2.0,
                                    "condition": "re-hear if ignition > 2",
                                    "reason": "strongest signal"}},
            "chamber", artist_id="a1", artist_name="Toman")
    # A future ignition signal above the dissent threshold trips the re-hearing.
    bb.emit("signal", {"kind": "ignition", "value": 5.0},
            "ignition_detector", artist_id="a1", artist_name="Toman")

    res = run_agent(Sentinel(), bb, now=now, trigger="cli")
    assert "promoted 1" in res["outcome"]
    cand = bb.read(kinds=["candidate"])[0]
    assert cand.artist_name == "Toman"
    assert any("re-hearing" in t for t in cand.payload["trigger_signals"])


# ── curator: facts, diff, material-change gating, revisions ──────────────────

def test_curator_material_gating_and_revisions(tmp_path):
    bb = _bb(tmp_path)
    now = dt.datetime.now(UTC)
    _seed_candidate(bb)
    _seed_signals(bb)

    # first run — no prior facts → emits rev 1, narrative None in mock mode
    DossierCurator().run(_ctx(DossierCurator, bb, now))
    d1 = bb.read(kinds=["dossier_update"])
    assert len(d1) == 1
    assert d1[0].payload["revision"] == 1
    assert d1[0].payload["narrative"] is None
    assert d1[0].payload["facts"]["signal_counts"].get("ignition") == 1

    # second run, identical facts (same `now`) → no material change, no record
    DossierCurator().run(_ctx(DossierCurator, bb, now))
    assert len(bb.read(kinds=["dossier_update"])) == 1

    # a new signal kind is material → new record at rev 2
    bb.emit("signal", {"kind": "label_signing", "value": 1.0},
            "label_radar", artist_id="a1", artist_name="Toman")
    DossierCurator().run(_ctx(DossierCurator, bb, now))
    d3 = bb.read(kinds=["dossier_update"])
    assert len(d3) == 2
    assert max(p.payload["revision"] for p in d3) == 2
    newest = d3[0].payload
    assert any("label_signing" in line for line in newest["diff"])
