"""
Scout page — find and pressure-test artists for a LOFI booking.

Two ways in, kept deliberately simple so the team isn't faced with a wall of
table:
  - 🔍 Scout   : search-to-select artists, then run AI analysis on just those
                 (one card per artist — no duplicated column). A collapsible
                 "browse the full pool" keeps the deterministic ranked list.
  - ⚖️ Compare : pick 2-3 artists for a head-to-head AI booking comparison.

Rankings are deterministic (no LLM); AI runs ON DEMAND via a button. Drops into
the dashboard nav via `render_scout_page()`; also runs standalone via app.py.
"""
from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from agents.core import compare_artists, compliance_status, generate_rationales
from scout.data import load_flat_profiles, load_ml_features, make_client
from scout.ranking import (
    build_candidates,
    load_predictions,
    load_taxonomy,
    rank_candidates,
)

_SCORE_FIELDS = [
    ("momentum", "Momentum"),
    ("growth", "Groei"),
    ("market_relevance", "Marktpositie"),
    ("future_potential", "Potentieel"),
    ("confidence", "Data"),
]


# ── data loading (cached) ────────────────────────────────────────────────────

@st.cache_resource
def _client():
    return make_client()


@st.cache_data(ttl=3600, show_spinner="Loading artists…")
def _candidates_and_taxonomy():
    client = _client()
    flat = load_flat_profiles(client)
    ml = load_ml_features(client)
    candidates = build_candidates(flat, ml, load_predictions())
    return candidates, load_taxonomy()


# ── shared table builders ─────────────────────────────────────────────────────

def _browse_df(rows: list[dict]) -> pd.DataFrame:
    """Deterministic ranked table — no AI column (so nothing can duplicate)."""
    return pd.DataFrame([{
        "artist_id": c["artist_id"],
        "Artiest": c["artist_name"],
        "Score": c["rank"],
        "Groei": c["growth"],
        "Potentieel": c["future_potential"],
        "Forecast": c.get("forecast_90d"),
        "Genres": ", ".join(c["genres"][:3]),
        "Waarom": c["waarom"],
    } for c in rows])


def _prog(label: str):
    return st.column_config.ProgressColumn(label, min_value=0, max_value=100,
                                           format="%d")


_TABLE_COLS = {
    "Artiest": st.column_config.TextColumn("Artiest", width="medium", pinned=True),
    "Score": _prog("Scout-score"),
    "Groei": _prog("Groei"),
    "Potentieel": _prog("Potentieel"),
    "Forecast": st.column_config.NumberColumn("Forecast 90d", format="%.0f%%"),
    "Genres": st.column_config.TextColumn(width="medium"),
    "Waarom": st.column_config.TextColumn("Waarom", width="large"),
}
_TABLE_ORDER = ["Artiest", "Score", "Groei", "Potentieel", "Forecast",
                "Genres", "Waarom"]


def _detail(c: dict) -> None:
    st.subheader(c["artist_name"])
    top = st.columns(3)
    top[0].metric("Scout-score", f"{c['rank']:.0f}/100")
    fc = c.get("forecast_90d")
    top[1].metric("Forecast 90d", "—" if fc is None else f"+{fc:.0f}%")
    top[2].metric("Spotify-luisteraars",
                  "—" if not c.get("spotify_listeners")
                  else f"{int(c['spotify_listeners']):,}".replace(",", "."))

    st.caption("Waarom op de shortlist: " + c["waarom"])

    chart_df = pd.DataFrame([
        {"Score": label, "Waarde": c[key]} for key, label in _SCORE_FIELDS
    ])
    chart = (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            x=alt.X("Waarde:Q", scale=alt.Scale(domain=[0, 100]), title=None),
            y=alt.Y("Score:N", sort=[l for _, l in _SCORE_FIELDS], title=None),
            color=alt.Color("Waarde:Q",
                            scale=alt.Scale(domain=[0, 45, 70, 100],
                                            range=["#d62728", "#ff7f0e",
                                                   "#ff7f0e", "#2ca02c"]),
                            legend=None),
            tooltip=["Score", "Waarde"],
        )
        .properties(height=160)
    )
    st.altair_chart(chart, use_container_width=True)
    if c["genres"]:
        st.caption("Genres: " + ", ".join(c["genres"]))
    st.caption("Open het volledige profiel via de pagina **Artiest Profiel**.")


def _ai_card(c: dict, r: dict | None) -> None:
    """One AI analysis card per selected artist — rendered once, never in a
    shared column, so it cannot duplicate."""
    with st.container(border=True):
        head = st.columns([4, 1])
        head[0].markdown(f"**{c['artist_name']}**  ·  "
                         f"{', '.join(c['genres'][:3]) or '—'}")
        fit = (r or {}).get("fit_score")
        head[1].metric("Fit", f"{int(fit)}" if isinstance(fit, (int, float))
                       else f"{c['rank']:.0f}")
        st.markdown((r or {}).get("rationale_nl") or c["waarom"])


# ── tab: Scout (search → select → analyse) ───────────────────────────────────

def _render_scout_tab(candidates: list[dict], taxonomy: dict,
                      by_name: dict, names: list[str]) -> None:
    st.markdown("**Search artists to analyse**")
    picks = st.multiselect(
        "Search and add artists", names, key="scout_picks",
        label_visibility="collapsed", placeholder="Type an artist name…",
        help="Search the unbooked pool; add one or more, then run AI analysis.")
    sel = [by_name[n] for n in picks]

    if sel:
        st.dataframe(_browse_df(sel), hide_index=True, use_container_width=True,
                     column_order=_TABLE_ORDER, column_config=_TABLE_COLS)

        status = compliance_status()
        bc = st.columns([1, 3])
        if bc[0].button("Run AI analysis", type="primary", key="scout_run_ai"):
            with st.spinner("Analysing selected artists…"):
                try:
                    st.session_state["scout_ai"] = generate_rationales(sel)
                except Exception as e:  # noqa: BLE001
                    st.error(f"AI analysis failed: {e}")
        if status["mode"] == "live":
            bc[1].caption(f"AI live — {status['model']}, region "
                          f"{status['inference_geo']}.")
        else:
            bc[1].caption(f"Preview mode — {status['reason']}. Showing the "
                          "data-driven rationale; AI fills this in when enabled.")

        ai = st.session_state.get("scout_ai") or {}
        for c in sel:
            _ai_card(c, ai.get(c["artist_id"]))
        st.divider()

    with st.expander("Browse the full ranked pool", expanded=not sel):
        _render_browse(candidates, taxonomy)


def _render_browse(candidates: list[dict], taxonomy: dict) -> None:
    g = taxonomy.get("genres", {})
    genre_opts = sorted(set(g.get("tier_1", []) + g.get("tier_2", [])))
    f = st.columns([2, 1, 1, 1])
    sel_genres = f[0].multiselect("Genre", genre_opts,
                                  help="Empty = all on-feel genres")
    min_conf = f[1].slider("Min. data-score", 0, 100, 30,
                           help="Filter out artists with little data")
    sort_label = f[2].selectbox(
        "Sort by", ["Scout-score", "Groei", "Potentieel", "Momentum", "Forecast"])
    query = f[3].text_input("Search", placeholder="name…")
    core_only = st.checkbox(
        "Core genres only (tech house / house etc.)", value=False,
        help="Show only artists with a core or adjacent genre")

    sort_map = {"Scout-score": "rank", "Groei": "growth",
                "Potentieel": "future_potential", "Momentum": "momentum",
                "Forecast": "forecast"}
    ranked = rank_candidates(
        candidates, taxonomy, genres=sel_genres, min_confidence=float(min_conf),
        query=query, core_only=core_only, sort_by=sort_map[sort_label])
    if not ranked:
        st.info("No artists match these filters.")
        return

    with_fc = [c for c in ranked if c.get("forecast_90d") is not None]
    k = st.columns(4)
    k[0].metric("Candidates", len(ranked))
    k[1].metric("With forecast", len(with_fc))
    k[2].metric("Avg. growth",
                f"{sum(c['growth'] for c in ranked) / len(ranked):.0f}/100")
    k[3].metric("Avg. potential",
                f"{sum(c['future_potential'] for c in ranked) / len(ranked):.0f}/100")

    df = _browse_df(ranked)
    event = st.dataframe(
        df, hide_index=True, use_container_width=True, on_select="rerun",
        selection_mode="single-row", key="scout_browse_table",
        column_order=_TABLE_ORDER, column_config=_TABLE_COLS)

    st.download_button(
        "Download shortlist (CSV)",
        df.drop(columns=["artist_id"]).to_csv(index=False).encode("utf-8"),
        file_name="lofi_scout_shortlist.csv", mime="text/csv")

    rows = event.selection.rows if event and event.selection else []
    if rows:
        st.divider()
        _detail(ranked[rows[0]])
    else:
        st.caption("Select a row for an artist's score breakdown.")


# ── tab: Compare (2-3 artists, head-to-head AI) ──────────────────────────────

def _compare_df(sel: list[dict]) -> pd.DataFrame:
    rows = [("Scout score", "rank"), ("Growth", "growth"),
            ("Potential", "future_potential"), ("Momentum", "momentum"),
            ("Market position", "market_relevance"), ("Data confidence", "confidence")]
    table: dict[str, list] = {"Metric": [label for label, _ in rows]
                              + ["Forecast 90d", "Genres"]}
    for c in sel:
        col = [c.get(key) for _, key in rows]
        fc = c.get("forecast_90d")
        col.append("—" if fc is None else f"{fc:+.0f}%")
        col.append(", ".join(c.get("genres", [])[:3]) or "—")
        table[c["artist_name"]] = col
    return pd.DataFrame(table)


def _render_compare_tab(by_name: dict, names: list[str]) -> None:
    st.markdown("**Compare 2–3 artists head-to-head**")
    picks = st.multiselect(
        "Artists to compare", names, max_selections=3, key="cmp_picks",
        label_visibility="collapsed", placeholder="Pick 2 or 3 artists…")
    if len(picks) < 2:
        st.info("Select at least 2 artists to compare.")
        return

    sel = [by_name[n] for n in picks]
    st.dataframe(_compare_df(sel), hide_index=True, use_container_width=True)

    status = compliance_status()
    bc = st.columns([1, 3])
    go = bc[0].button("Compare with AI", type="primary", key="cmp_run")
    if status["mode"] == "live":
        bc[1].caption(f"AI live — {status['model']}, region "
                      f"{status['inference_geo']}.")
    else:
        bc[1].caption(f"Preview mode — {status['reason']}.")

    if go:
        with st.spinner("Comparing artists…"):
            try:
                st.session_state["scout_compare"] = {
                    "key": tuple(sorted(picks)), "text": compare_artists(sel)}
            except Exception as e:  # noqa: BLE001
                st.error(f"Comparison failed: {e}")

    cmp = st.session_state.get("scout_compare")
    if cmp and cmp.get("key") == tuple(sorted(picks)):
        st.markdown(cmp["text"])
    else:
        st.caption("Click **Compare with AI** for a head-to-head booking verdict.")


# ── main render ──────────────────────────────────────────────────────────────

def render_scout_page() -> None:
    st.title("Scout")
    st.caption(
        "Find and pressure-test artists for a LOFI booking. Search to add "
        "artists and run AI analysis, or compare a few head-to-head. Rankings "
        "are data-driven; AI runs on demand.")

    try:
        candidates, taxonomy = _candidates_and_taxonomy()
    except Exception as e:  # noqa: BLE001 — surface config/connection issues clearly
        st.error(f"Could not load artists: {e}")
        st.info("Check that SUPABASE_URL and SUPABASE_KEY are set.")
        return

    if not candidates:
        st.warning("No unbooked artists found.")
        return

    by_name: dict[str, dict] = {}
    for c in candidates:
        by_name.setdefault(c["artist_name"], c)  # first wins on rare name clashes
    names = sorted(by_name)

    tab_scout, tab_compare = st.tabs(["🔍 Scout", "⚖️ Compare"])
    with tab_scout:
        _render_scout_tab(candidates, taxonomy, by_name, names)
    with tab_compare:
        _render_compare_tab(by_name, names)
