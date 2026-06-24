"""
Per-artist validation panel — "Can they sell tickets at LOFI, how many, do they
fit?" (Phase 3).

A single web-search-grounded Claude call reasons over LOFI's own ticketing
history + genre-matched comparable events. Trust-first: ticket ranges only when
there is real grounding; otherwise it says so. Runs ON DEMAND (a button), never
on page load, because each run costs web searches.

English by stakeholder request. Numbers are grounded in LOFI data; web search
adds reputation/risk context, shown separately so the booker sees provenance.
"""
from __future__ import annotations

import streamlit as st

from agents.core import compliance_status, validate_artist
from scout.context import build_validation_view

_REC = {
    "book_now":  ("✅ Book now", "#1DB954"),
    "monitor":   ("👀 Monitor", "#FF9900"),
    "too_early": ("⏳ Too early", "#3b82f6"),
    "not_a_fit": ("⛔ Not a fit", "#e05252"),
}
_TYPE = {"producer_led": "Producer-led", "dj_led": "DJ-led",
         "hybrid": "Hybrid", "unknown": "Unclassified"}


def _badge(text: str, color: str) -> str:
    return (f"<span style='background:{color};color:#fff;padding:2px 10px;"
            f"border-radius:12px;font-weight:600;font-size:0.85em'>{text}</span>")


def _render_result(res: dict) -> None:
    rec_label, rec_color = _REC.get(res.get("recommendation"), ("—", "#888"))
    atype = _TYPE.get(res.get("artist_type"), res.get("artist_type") or "—")
    lens = (res.get("lens") or "").upper()
    st.markdown(
        f"{_badge(rec_label, rec_color)} &nbsp; **{atype}** &nbsp;·&nbsp; "
        f"lens: {lens or '—'} &nbsp;·&nbsp; confidence: "
        f"{res.get('data_confidence', '—')}", unsafe_allow_html=True)

    # ── LOFI-grounded block (the trustworthy numbers) ────────────────────────
    st.markdown("**Expected draw at LOFI** (grounded in LOFI's own events)")
    est = res.get("ticket_estimate")
    if isinstance(est, dict) and est.get("base") is not None:
        c1, c2, c3 = st.columns(3)
        c1.metric("Conservative", f"{est.get('conservative', '—')}")
        c2.metric("Base", f"{est.get('base', '—')}")
        c3.metric("Bull", f"{est.get('bull', '—')}")
        st.caption(f"Basis: {res.get('ticket_basis', '—')} · sell-through "
                   f"likelihood: **{res.get('sell_through', 'unknown')}**")
    else:
        st.info("Not enough comparable LOFI data for a ticket estimate yet — "
                "see *what's missing* below. (Refusing to guess is the point.)")

    fit = res.get("lofi_fit") or {}
    score = fit.get("score")
    if isinstance(score, (int, float)):
        st.markdown(f"**LOFI fit: {int(score)}/100**")
        st.progress(min(max(int(score), 0), 100) / 100)
    if fit.get("explanation"):
        st.caption(fit["explanation"])

    comps = res.get("comparable_events") or []
    if comps:
        st.markdown("**Comparable LOFI events used**")
        st.dataframe(
            [{"Event": c.get("event"), "Date": c.get("date"),
              "Tickets": c.get("tickets"), "Why comparable": c.get("why")}
             for c in comps],
            hide_index=True, use_container_width=True)

    sig, risk = res.get("key_signals") or [], res.get("key_risks") or []
    if sig or risk:
        a, b = st.columns(2)
        with a:
            if sig:
                st.markdown("**Signals in favour**")
                for s in sig:
                    st.markdown(f"- {s}")
        with b:
            if risk:
                st.markdown("**Risks / caveats**")
                for r in risk:
                    st.markdown(f"- {r}")

    # ── Web-enriched block (provenance kept separate from the numbers) ────────
    web = res.get("web_context") or []
    if web:
        st.markdown("**Web-enriched context** (reputation / intel — not the numbers)")
        for w in web:
            claim, url = w.get("claim") or "", w.get("url") or ""
            if claim and url:
                st.markdown(f"- {claim} — [source]({url})")
            elif url:
                st.markdown(f"- [{url}]({url})")

    missing = res.get("missing_data") or []
    if missing:
        with st.expander("What would sharpen this verdict"):
            for m in missing:
                st.markdown(f"- {m}")


def render_validation_result(res: dict) -> None:
    """Public wrapper so other surfaces (e.g. the Scout page) can render a
    validation verdict without re-implementing the layout."""
    _render_result(res)


def render_validation(artist_id: str, name: str, profile: dict,
                      ml: dict | None, ext: dict | None = None,
                      ra_df=None, pf_data: dict | None = None, vdf=None,
                      nl_score=None) -> None:
    st.divider()
    st.subheader("Validation — can they sell at LOFI?")

    status = compliance_status()
    if status["mode"] == "live":
        st.caption(f"AI live — {status['model']}, region {status['inference_geo']}. "
                   "Numbers grounded in LOFI's ticketing history; web search "
                   "optional (toggle below).")
    else:
        st.caption(f"Preview mode — {status['reason']}. Set LOFI_LLM_ENABLED for a "
                   "real, web-grounded validation.")

    store = st.session_state.setdefault("validation", {})
    note = st.text_input(
        "Optional booker note for this check (real-world result, intel, "
        "correction)", key=f"val_note_{artist_id}",
        placeholder="e.g. sold ~300 at BRET last year · switching agency soon")

    ctrl = st.columns([1, 2])
    use_web = ctrl[1].checkbox(
        "🔍 Use web search", value=True, key=f"val_web_{artist_id}",
        help="Search the web for reputation / residencies / releases — especially "
             "useful for DJ-led artists with thin streaming data. Off = reason on "
             "LOFI + Chartmetric data only (cheaper, nothing leaves).")
    run = ctrl[0].button("Run AI validation", key=f"val_run_{artist_id}",
                         type="primary")
    if run:
        spinner = ("Reasoning over LOFI history + web…" if use_web
                   else "Reasoning over LOFI history…")
        with st.spinner(spinner):
            try:
                view = build_validation_view(
                    artist_id, name, profile, ml, ext=ext, ra_df=ra_df,
                    pf_data=pf_data, vdf=vdf, nl_score=nl_score)
                store[artist_id] = {
                    "result": validate_artist(view, booker_note=note.strip() or None,
                                              use_web=use_web),
                    "note": note.strip()}
                if note.strip():           # log the booker note for the loop
                    try:
                        from scout.feedback import capture_feedback
                        capture_feedback(artist_id, note.strip())
                    except Exception:
                        pass
            except Exception as e:  # noqa: BLE001
                st.error(f"Validation failed: {e}")

    cached = store.get(artist_id)
    if cached:
        _render_result(cached["result"])
    else:
        st.caption("Click **Run AI validation** for a grounded verdict on draw, "
                   "fit, fee logic and timing.")
