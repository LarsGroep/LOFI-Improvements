"""
Deal radar UI — renders the predict/ deal sheet, the watchlist alerts tab and
the model-health tab inside the Scout page.

Only rendering lives here; every number comes from predict/ (framework-free,
tested). The panel is explicit about provenance: each estimate names its
method and sample size, and 'insufficient' renders as an honest empty state,
never a guessed number.
"""
from __future__ import annotations

import streamlit as st

from predict.backtest import HORIZON_DAYS, evaluate, read_log
from predict.deal import deal_sheet_for_candidate
from predict.draw import backtest as draw_backtest
from predict.fees import quote_check
from predict.rank_weights import DEFAULTS, learned_weights
from predict.watchlist import detect_spikes, load_watchlist, save_watchlist

_SLOT_LABEL = {"headliner": "🎤 Headliner", "co_headliner": "🎤 Co-headliner",
               "support": "🎶 Support", "opener": "🌱 Opener",
               "too_big": "🚀 Too big for the room"}
_WINDOW_LABEL = {"book_now": "✅ Book now", "act_fast": "⚡ Act fast",
                 "monitor": "👀 Monitor", "no_rush": "🧊 No rush"}


def _eur(v) -> str:
    return "—" if v is None else f"€{v:,.0f}".replace(",", ".")


def _rng(d: dict, keys=("conservative", "base", "high"), money=False) -> str:
    vals = [d.get(k) for k in keys]
    if any(v is None for v in vals):
        return "—"
    fmt = _eur if money else (lambda v: f"{v:,.0f}".replace(",", "."))
    return f"{fmt(vals[0])} / {fmt(vals[1])} / {fmt(vals[2])}"


def _basis(d: dict) -> str:
    m = d.get("method") or "—"
    bits = []
    if d.get("n_own"):
        bits.append(f"{d['n_own']} own")
    if d.get("n_comparables"):
        bits.append(f"{d['n_comparables']} comparables")
    return m + (f" ({', '.join(bits)})" if bits else "")


# ── deal radar (per-artist panel) ─────────────────────────────────────────────

def render_deal_radar(c: dict, flat_by_id: dict, ml: dict,
                      flat_profiles: list[dict]) -> None:
    """The economics of booking this artist — every number a model's, with
    provenance. Cached per artist in session state; built on first expand."""
    key = f"deal_{c['artist_id']}"
    with st.expander("💶 Deal radar — modelled fee, draw, margin & timing"):
        if key not in st.session_state:
            with st.spinner("Running the deal models…"):
                st.session_state[key] = deal_sheet_for_candidate(
                    c, flat_by_id=flat_by_id, ml=ml, flat_profiles=flat_profiles)
        sheet = st.session_state[key]

        draw, fee = sheet.get("draw_estimate", {}), sheet.get("fee_estimate", {})
        margin, slot = sheet.get("margin_estimate", {}), sheet.get("slot_fit", {})
        cols = st.columns(3)
        cols[0].metric("Draw (cons/base/high)", _rng(draw),
                       help="Empirical quantiles over LOFI's own events — "
                            + _basis(draw))
        cols[1].metric("Fee (low/base/high)",
                       _rng(fee, ("low", "base", "high"), money=True),
                       help="Draw-adjusted comparable gages — " + _basis(fee))
        cols[2].metric("Margin (base)",
                       _eur(margin.get("base")) if margin.get("method") == "modelled" else "—",
                       help="Door revenue − fee at base scenario "
                            f"(net {_eur(margin.get('net_per_head'))}/head)")

        win = sheet.get("booking_window", {})
        line = []
        if slot.get("slot"):
            line.append(f"{_SLOT_LABEL.get(slot['slot'], slot['slot'])} "
                        f"({slot.get('occupancy_base', 0):.0%} of the room)")
        if win.get("verdict"):
            line.append(f"{_WINDOW_LABEL.get(win['verdict'], win['verdict'])} — "
                        f"{win.get('note', '')}")
        if line:
            st.markdown("  \n".join(line))

        for h, hd in (win.get("horizons") or {}).items():
            if hd.get("wait_cost") is not None:
                st.caption(f"In {h}: projected fee {_eur(hd['projected_fee'])} "
                           f"(waiting ≈ {_eur(hd['wait_cost'])})")

        twins = sheet.get("trajectory_twins") or []
        if twins:
            st.markdown("**Trajectory twins** — booked artists who looked like "
                        "this on the way up:")
            for t in twins:
                o = t.get("outcome") or {}
                bits = []
                if o.get("avg_tickets"):
                    bits.append(f"avg {o['avg_tickets']} tickets")
                if o.get("events_played"):
                    bits.append(f"{o['events_played']} events")
                if o.get("last_fee_paid"):
                    bits.append(f"last fee {_eur(o['last_fee_paid'])}")
                st.caption(f"· **{t['name']}** (match {t['similarity']:.0%})"
                           + (" — " + ", ".join(bits) if bits else ""))

        # agent-quote sanity check — the negotiation artefact
        q = st.number_input("Agent quote (€) — check against the model", 0,
                            step=250, key=f"quote_{c['artist_id']}")
        if q:
            chk = quote_check(float(q), fee)
            if chk["verdict"] == "fair":
                st.success(f"Within the modelled fair range "
                           f"({_eur(fee.get('low'))}–{_eur(fee.get('high'))}).")
            elif chk["verdict"] == "no_model":
                st.info("No fee model for this artist (not enough gage data).")
            else:
                st.warning(f"≈ {chk['ratio']}× the modelled base fee — above "
                           f"fair range {_eur(fee.get('low'))}–{_eur(fee.get('high'))}.")


# ── tab: alerts / watchlist ───────────────────────────────────────────────────

def render_alerts_tab(candidates: list[dict]) -> None:
    st.markdown("**Watchlist** — get pinged when an artist spikes, instead of "
                "refreshing dashboards. Wire `LOFI_ALERT_WEBHOOK` + a cron on "
                "`python -m predict.watchlist` for push delivery.")
    names = sorted({c["artist_name"] for c in candidates})
    watch = st.multiselect("Artists to watch", names, default=[
        n for n in load_watchlist() if n in set(names)])
    if st.button("Save watchlist"):
        save_watchlist(watch)
        st.toast("Watchlist saved")

    st.divider()
    scope = st.radio("Scan", ["Watchlist", "Whole pool"], horizontal=True,
                     label_visibility="collapsed")
    alerts = detect_spikes(candidates,
                           watch=watch if scope == "Watchlist" else None)
    if not alerts:
        st.info("No spikes right now." if watch or scope == "Whole pool"
                else "Pick artists to watch, or scan the whole pool.")
        return
    st.caption(f"{len(alerts)} artist(s) tripping thresholds:")
    for a in alerts[:25]:
        genres = ", ".join(a.get("genres") or [])
        st.markdown(f"⚡ **{a['artist_name']}**"
                    + (f" ({genres})" if genres else "")
                    + " — " + "; ".join(a["reasons"]))


# ── tab: model health ─────────────────────────────────────────────────────────

def render_model_health(candidates: list[dict]) -> None:
    st.markdown("**Is the machinery calibrated?** The honest report — models "
                "you can't verify are models you can't lean on.")

    st.subheader("XGBoost 90-day forecast", divider=True)
    current = {c["artist_id"]: c.get("spotify_listeners")
               for c in candidates if c.get("spotify_listeners")}
    rep = evaluate(read_log(), current)
    if not rep.get("n_mature"):
        st.info(rep.get("note", "No mature logged forecasts yet."))
    else:
        k = st.columns(4)
        k[0].metric("Scored forecasts", rep["n_mature"])
        k[1].metric("Direction hit rate", f"{rep['direction_hit_rate_pct']}%")
        k[2].metric(f"Within ±{rep['tolerance_points']:.0f} pts",
                    f"{rep['within_tolerance_pct']}%")
        k[3].metric("Bias", f"{rep['bias_points']:+.1f} pts",
                    help="> 0 = the forecast systematically over-promises")
        st.caption(f"MAE {rep['mae_points']} growth points over "
                   f"{HORIZON_DAYS}-day horizons.")

    st.subheader("Draw model (leave-one-out over the events corpus)",
                 divider=True)
    if st.button("Run draw backtest"):
        from scout.lofi_events import _events
        with st.spinner("Backtesting draw quantiles…"):
            st.session_state["draw_bt"] = draw_backtest(list(_events()))
    bt = st.session_state.get("draw_bt")
    if bt:
        if not bt.get("n_events"):
            st.info("Events corpus not available (NDA zip not present).")
        else:
            k = st.columns(3)
            k[0].metric("Events scored", bt["n_events"])
            k[1].metric("Range coverage", f"{bt['interval_coverage_pct']}%",
                        help=f"Target ≈ {bt['target_coverage_pct']}% — actuals "
                             "inside [conservative, high]")
            k[2].metric("Median error (base)", f"{bt['median_ape_pct']}%")

    st.subheader("Ranking weights", divider=True)
    lw = learned_weights()
    if lw:
        st.success("Using weights **learned from LOFI's booking outcomes** "
                   "(`predict/data/rank_weights.json`):")
        st.json(lw)
    else:
        st.info("Using hand-tuned defaults — fit learned weights with "
                "`python -m predict.rank_weights` once enough booked labels "
                "exist.")
        st.json(DEFAULTS)
