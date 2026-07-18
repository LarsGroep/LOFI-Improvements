"""
Agent OS console — the blackboard, made human (docs/agentic_os.md §6).

Two tabs:
  - Feed : the blackboard newest-first, with a heat leaderboard on top and
           filters by kind/artist/agent. Candidates and alerts stand out.
  - Ops  : backend + pause status, the agent_runs syslog, and a last-run
           summary per registered agent (the `python -m osk status` view).

Rendering only — every number comes from the blackboard, read-only. The
console must render honest empty states with NO Supabase and NO signals, so
every data access is guarded; nothing here ever writes.
"""
from __future__ import annotations

import datetime as _dt

import streamlit as st

# ── badge styling per record kind ─────────────────────────────────────────────

_KIND_BADGE = {
    "signal": "🛰️ signal", "candidate": "⭐ candidate", "alert": "⚡ alert",
    "brief": "📄 brief", "argument": "💬 argument", "verdict": "⚖️ verdict",
    "dossier_update": "📇 dossier",
}
_HEAT_WINDOW_DAYS = 90

# verdict → (label, colour), aligned with scout/validation.py + predict/window.py
_VERDICT_STYLE = {
    "book_now": ("✅ Book now", "#1DB954"),
    "act_fast": ("⚡ Act fast", "#f59e0b"),
    "monitor": ("👀 Monitor", "#FF9900"),
    "too_early": ("⏳ Too early", "#3b82f6"),
    "not_a_fit": ("⛔ Not a fit", "#e05252"),
}


def _open_bb():
    """Read-only blackboard handle, or (None, error-string)."""
    try:
        from osk.blackboard import open_blackboard
        return open_blackboard(), None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def _short_payload(payload: dict, limit: int = 140) -> str:
    if not isinstance(payload, dict):
        return str(payload)[:limit]
    if payload.get("explain"):
        return str(payload["explain"])
    bits = []
    for k in ("kind", "source", "value", "reasons", "priority"):
        if k in payload and payload[k] is not None:
            bits.append(f"{k}={payload[k]}")
    text = ", ".join(bits) or str(payload)
    return text[:limit] + ("…" if len(text) > limit else "")


# ── heat leaderboard ──────────────────────────────────────────────────────────

def _render_heat(bb, now: _dt.datetime) -> None:
    st.subheader("🔥 Heat leaderboard", divider=True)
    try:
        from osk.detectors._common import signal_views
        from osk.sentinel import compute_heat
        since = (now - _dt.timedelta(days=_HEAT_WINDOW_DAYS)
                 ).isoformat(timespec="seconds")
        views = signal_views(bb.read(kinds=["signal"], since=since, limit=5000))
    except Exception as exc:  # noqa: BLE001
        st.info(f"Heat unavailable: {exc}")
        return
    if not views:
        st.info("No signals yet — the leaderboard fills as listeners and "
                "detectors accrue signals nightly.")
        return
    heat = compute_heat(views, now)
    top = sorted(heat.values(), key=lambda h: h["heat"], reverse=True)[:15]
    for h in top:
        cols = st.columns([3, 1])
        cols[0].markdown(f"**{h['artist_name']}** — "
                         + "; ".join(h["top_reasons"][:2]))
        cols[1].metric("heat", f"{h['heat']:.2f}")


# ── Feed tab ──────────────────────────────────────────────────────────────────

def _render_feed(bb) -> None:
    now = _dt.datetime.now(_dt.timezone.utc)
    _render_heat(bb, now)

    st.subheader("📡 Feed", divider=True)
    try:
        records = bb.read(limit=200)
    except Exception as exc:  # noqa: BLE001
        st.info(f"Feed unavailable: {exc}")
        return
    if not records:
        st.info("The blackboard is empty. Signals appear here as the listeners "
                "and detectors run — nothing to show yet, honestly.")
        return

    kinds = sorted({r.kind for r in records})
    agents = sorted({r.emitted_by for r in records if r.emitted_by})
    c1, c2, c3 = st.columns(3)
    pick_kinds = c1.multiselect("Kind", kinds, default=kinds)
    artist_q = c2.text_input("Artist contains").strip().lower()
    pick_agents = c3.multiselect("Emitted by", agents, default=agents)

    shown = 0
    for r in records:
        if r.kind not in pick_kinds:
            continue
        if pick_agents and r.emitted_by not in pick_agents:
            continue
        if artist_q and artist_q not in (r.artist_name or "").lower():
            continue
        shown += 1
        badge = _KIND_BADGE.get(r.kind, r.kind)
        who = r.artist_name or "—"
        ts = str(r.created_at)[:16]
        line = f"`{ts}`  **{badge}**  ·  {who}  ·  _{r.emitted_by}_"
        body = _short_payload(r.payload)
        if r.kind == "candidate":
            with st.container():
                st.error(f"{line}\n\n{body}")
        elif r.kind == "alert":
            with st.container():
                st.warning(f"{line}\n\n{body}")
        else:
            st.markdown(f"{line}  —  {body}")
    if not shown:
        st.caption("No records match the current filters.")


# ── Ops tab ───────────────────────────────────────────────────────────────────

def _display_registry():
    """Builtin roster + Phase C detectors + sentinel, for the status view.
    Display-only — the live registry is assembled by the kernel elsewhere."""
    from osk.registry import Registry
    reg = Registry()
    try:
        from osk.agents_builtin import BUILTIN_AGENTS
        from osk.detectors import DETECTOR_AGENTS
        from osk.sentinel import Sentinel
        for cls in (*BUILTIN_AGENTS, *DETECTOR_AGENTS, Sentinel):
            try:
                reg.register(cls())
            except Exception:  # noqa: BLE001 — a duplicate/name clash is non-fatal
                pass
    except Exception:  # noqa: BLE001
        pass
    return reg


def _render_ops(bb) -> None:
    st.subheader("🩺 Backend", divider=True)
    try:
        from osk import budget
        paused = budget.paused()
    except Exception:  # noqa: BLE001
        paused = False
    c1, c2 = st.columns(2)
    try:
        c1.metric("Backend", bb.describe())
    except Exception:  # noqa: BLE001
        c1.metric("Backend", "unknown")
    c2.metric("Paused", "YES (LOFI_OS_PAUSED)" if paused else "no")

    st.subheader("📜 Agent runs (last 50)", divider=True)
    try:
        runs = bb.runs(limit=50)
    except Exception as exc:  # noqa: BLE001
        runs = []
        st.info(f"Run log unavailable: {exc}")
    if runs:
        import pandas as pd
        rows = [{
            "agent": r.get("agent"),
            "started_at": str(r.get("started_at"))[:16],
            "outcome": r.get("outcome") or "running",
            "emitted": r.get("records_emitted"),
            "error": (r.get("error") or "")[:60],
        } for r in runs]
        st.dataframe(pd.DataFrame(rows), use_container_width=True,
                     hide_index=True)
    else:
        st.info("No agent runs recorded yet.")

    st.subheader("🗂️ Registered agents", divider=True)
    reg = _display_registry()
    for agent in reg.all():
        m = agent.manifest
        try:
            last_runs = bb.runs(m.name, limit=1)
        except Exception:  # noqa: BLE001
            last_runs = []
        last = last_runs[0] if last_runs else None
        sched = m.schedule or "event-only"
        if last:
            summary = (f"last {str(last.get('started_at'))[:16]} → "
                       f"{last.get('outcome') or 'running'}")
        else:
            summary = "never ran"
        st.markdown(f"- **{m.name}** · `{sched}` · {summary}")


# ── Chamber tab ───────────────────────────────────────────────────────────────

def _group_debates(records) -> dict:
    """Group brief + argument + verdict records by payload.debate_id."""
    debates: dict = {}
    for r in records:
        did = (r.payload or {}).get("debate_id")
        if not did:
            continue
        d = debates.setdefault(did, {"artist": None, "order": 0,
                                     "brief": None, "args": [], "verdict": None})
        if r.artist_name:
            d["artist"] = r.artist_name
        if (r.id or 0) > d["order"]:
            d["order"] = r.id or 0
        if r.kind == "verdict":
            d["verdict"] = r.payload
        elif r.kind == "brief":
            d["brief"] = r.payload
        else:
            d["args"].append(r.payload or {})
    return debates


def _render_chamber(bb) -> None:
    st.subheader("⚖️ Deliberation Chamber", divider=True)
    st.caption("Structured adversarial review — Advocate, Skeptic, Judge — over "
               "one shared, minimised evidence pack. Newest debates first.")

    if st.button("🧠 Deliberate now"):
        try:
            from osk.blackboard import open_blackboard
            from osk.chamber import Chamber
            from osk.orchestrator import run_agent
            wb = open_blackboard()
            try:
                res = run_agent(Chamber(), wb, trigger="ui")
            finally:
                try:
                    wb.close()
                except Exception:  # noqa: BLE001
                    pass
            st.success(f"Chamber ran: {res.get('outcome')}")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not run the chamber: {exc}")
        st.rerun()

    try:
        records = bb.read(kinds=["brief", "argument", "verdict"], limit=600)
    except Exception as exc:  # noqa: BLE001
        st.info(f"Chamber feed unavailable: {exc}")
        return
    debates = _group_debates(records)
    if not debates:
        st.info("No debates yet. Run the chamber with the button above, or wait "
                "for the daily 06:30 deliberation.")
        return

    for _, d in sorted(debates.items(), key=lambda kv: kv[1]["order"],
                       reverse=True):
        v = d["verdict"] or {}
        verdict = v.get("verdict") or "—"
        label, color = _VERDICT_STYLE.get(verdict, (verdict, "#888888"))
        conf = v.get("confidence")
        title = f"{d['artist'] or '—'} — {label}"
        if isinstance(conf, (int, float)):
            title += f"  ·  {conf:.0%}"
        with st.expander(title):
            st.markdown(
                f"<span style='background:{color};color:#fff;padding:2px 10px;"
                f"border-radius:8px;font-weight:600'>{label}</span>",
                unsafe_allow_html=True)
            if v.get("summary"):
                st.markdown(v["summary"])
            diss = v.get("dissent") or {}
            if diss.get("condition"):
                st.caption(f"Dissent / re-hearing: {diss['condition']}")
            for a in sorted(d["args"], key=lambda p: p.get("turn", 0)):
                st.markdown(f"**{a.get('role', '?').title()}**"
                            + (" · _mock_" if a.get("mode") == "mock" else ""))
                claims = a.get("claims") or []
                if not claims:
                    st.caption("(no grounded claims survived)")
                for c in claims:
                    ptrs = ", ".join(c.get("evidence") or []) or "—"
                    st.markdown(f"- {c.get('text', '')}  `{ptrs}`")


# ── Dossiers tab ──────────────────────────────────────────────────────────────

def _render_dossiers(bb) -> None:
    st.subheader("📇 Artist dossiers", divider=True)
    st.caption("The living, diffed per-artist dossier — written only when the "
               "facts move materially.")
    try:
        records = bb.read(kinds=["dossier_update"], limit=600)
    except Exception as exc:  # noqa: BLE001
        st.info(f"Dossiers unavailable: {exc}")
        return
    if not records:
        st.info("No dossiers yet. The curator writes one when an artist's facts "
                "materially change (daily 07:00).")
        return

    by_artist: dict = {}
    for r in records:                       # newest-first (id desc)
        by_artist.setdefault(r.artist_name or "—", []).append(r)
    pick = st.selectbox("Artist", sorted(by_artist))
    revisions = by_artist[pick]
    latest = revisions[0].payload or {}
    facts = latest.get("facts") or {}

    c1, c2, c3 = st.columns(3)
    c1.metric("Heat", facts.get("heat"))
    c2.metric("Last verdict", facts.get("last_verdict") or "—")
    c3.metric("Revision", latest.get("revision"))

    if latest.get("narrative"):
        st.markdown(latest["narrative"])
    else:
        st.caption("No narrative (preview mode — the facts and diff are the "
                   "record; enable the AI for a written paragraph).")

    counts = facts.get("signal_counts") or {}
    if counts:
        st.markdown("**Signals (90d):** "
                    + ", ".join(f"{k}×{v}" for k, v in counts.items()))
    if facts.get("first_seen"):
        st.caption(f"First seen: {str(facts['first_seen'])[:16]}")

    st.markdown("**Revision history**")
    for r in revisions:
        p = r.payload or {}
        st.markdown(f"- rev **{p.get('revision')}** · {str(r.created_at)[:16]}")
        for line in (p.get("diff") or []):
            st.caption(f"　{line}")


# ── entrypoint ────────────────────────────────────────────────────────────────

def render_os_console() -> None:
    st.title("🖥️ Agent OS")
    st.caption("The blackboard — signals, candidates, verdicts and alerts — "
               "plus the kernel's health. Read-only.")
    bb, err = _open_bb()
    if bb is None:
        st.info(f"Blackboard not reachable ({err}). The OS console needs the "
                "kernel's SQLite/Supabase backend; nothing to show yet.")
        return
    try:
        feed_tab, chamber_tab, dossiers_tab, ops_tab = st.tabs(
            ["Feed", "Chamber", "Dossiers", "Ops"])
        with feed_tab:
            _render_feed(bb)
        with chamber_tab:
            _render_chamber(bb)
        with dossiers_tab:
            _render_dossiers(bb)
        with ops_tab:
            _render_ops(bb)
    finally:
        try:
            bb.close()
        except Exception:  # noqa: BLE001
            pass
