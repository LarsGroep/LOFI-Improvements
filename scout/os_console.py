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
        feed_tab, ops_tab = st.tabs(["Feed", "Ops"])
        with feed_tab:
            _render_feed(bb)
        with ops_tab:
            _render_ops(bb)
    finally:
        try:
            bb.close()
        except Exception:  # noqa: BLE001
            pass
