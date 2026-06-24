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

from agents.core import (
    compare_artists,
    compliance_status,
    recommend_booking,
    shortlist_chat_stream,
    validate_artist,
)
from scout.context import build_validation_view
from scout.data import load_flat_profiles, load_ml_features, make_client
from scout.ranking import (
    build_candidates,
    load_predictions,
    load_taxonomy,
    rank_candidates,
)
from scout.validation import render_validation_result

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
    # raw profile by id — needed to run the full validation from the Scout page
    flat_by_id = {p["artist_id"]: p for p in flat if p.get("artist_id")}
    return candidates, load_taxonomy(), flat_by_id, ml


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


# ── broad genre buckets (so the team picks a genre, not a subgenre) ───────────

# Ordered: more specific buckets first. The first keyword a subgenre matches
# wins; "electronic/electronica" is checked before "electro" so it doesn't get
# swallowed. Anything unmatched falls into "Other".
_BROAD_RULES = [
    ("House", ["house"]),
    ("Techno", ["techno", "schranz", "minimal"]),
    ("Disco / Nu-Disco", ["disco", "italo", "boogie"]),
    ("Garage / UKG", ["garage", "ukg", "2-step", "bassline"]),
    ("Trance", ["trance", "psy"]),
    ("Drum & Bass", ["drum and bass", "drum & bass", "dnb", "jungle"]),
    ("Dubstep / Bass", ["dubstep", "bass music", "wonky"]),
    ("Electronic", ["electronic", "electronica", "idm", "leftfield", "left field"]),
    ("Electro", ["electro"]),
    ("Ambient / Downtempo", ["ambient", "downtempo", "lo-fi", "lofi", "balearic",
                             "chill"]),
    ("Hip-Hop / Rap", ["hip hop", "hip-hop", "rap", "trap", "grime"]),
    ("Pop", ["pop"]),
    ("Hard", ["hardstyle", "hardcore", "hard techno", "gabber"]),
]


def _broad_genres(genres) -> list[str]:
    """Map an artist's subgenres to the broad families they belong to."""
    found: list[str] = []
    for g in (genres or []):
        gl = str(g).lower()
        for bucket, kws in _BROAD_RULES:
            if any(k in gl for k in kws):
                if bucket not in found:
                    found.append(bucket)
                break
        else:
            if "Other" not in found:
                found.append("Other")
    return found


# ── tab: Scout (genre → up to 6 → deep ranked analysis) ──────────────────────

def _compare_tab_label() -> str:
    """Tab label with a live counter of artists queued for comparison."""
    n = len(st.session_state.get("cmp_picks", []) or [])
    return f"⚖️ Compare ({n})" if n else "⚖️ Compare"


def _add_to_compare(name: str) -> None:
    cur = list(st.session_state.get("cmp_picks", []))
    if name not in cur:
        st.session_state["cmp_picks"] = (cur + [name])[:3]
        st.toast(f"Added {name} to Compare ⚖️")
    else:
        st.toast(f"{name} is already in Compare")


def _render_reco(reco: dict, flat_by_id: dict, ml: dict) -> None:
    """Render a finished shortlist analysis + an interactive follow-up chat."""
    res = reco["result"]
    st.markdown(res.get("analysis") or "*(no analysis returned)*")

    sources = res.get("sources") or []
    if sources:
        with st.expander("Sources (web)"):
            for s in sources:
                st.markdown(f"- [{s.get('title') or s['url']}]({s['url']})")

    st.markdown("**Ask a follow-up about this analysis**")
    hist = st.session_state.setdefault("scout_reco_chat", [])
    for m in hist:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    q = st.chat_input("e.g. why is #1 ahead of #2? what fee makes sense?",
                      key="scout_reco_chat_in")
    if q:
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            try:
                full = st.write_stream(shortlist_chat_stream(
                    reco["views"], res.get("analysis", ""), list(hist), q))
            except Exception as e:  # noqa: BLE001
                full = f"Something went wrong with the assistant: {e}"
                st.error(full)
        hist.append({"role": "user", "content": q})
        hist.append({"role": "assistant", "content": full})


def _render_scout_tab(candidates: list[dict], taxonomy: dict,
                      by_name: dict, names: list[str],
                      flat_by_id: dict, ml: dict) -> None:
    # index candidates by broad genre (size-ordered, "Other" last)
    broad_index: dict[str, list[dict]] = {}
    for c in candidates:
        for b in _broad_genres(c.get("genres")):
            broad_index.setdefault(b, []).append(c)
    buckets = sorted((b for b in broad_index if b != "Other"),
                     key=lambda b: -len(broad_index[b]))
    if "Other" in broad_index:
        buckets.append("Other")

    st.markdown("**Pick a genre, then choose up to 6 artists for a deep, "
                "LOFI-grounded booking analysis.**")
    genre = st.selectbox(
        "Genre", ["—"] + buckets, index=0, label_visibility="collapsed",
        format_func=lambda b: "Choose a genre…" if b == "—"
        else f"{b} ({len(broad_index[b])})")

    if genre != "—":
        # reset the artist picks when the genre changes (avoid stale options)
        if st.session_state.get("scout_genre_last") != genre:
            st.session_state["scout_genre_last"] = genre
            st.session_state.pop("scout_genre_picks", None)

        pool = sorted(broad_index[genre], key=lambda c: c["rank"], reverse=True)
        seen: set[str] = set()
        pool_names = [c["artist_name"] for c in pool
                      if not (c["artist_name"] in seen or seen.add(c["artist_name"]))]
        st.caption(f"{len(pool_names)} artists in **{genre}** — pick up to 6.")
        picks = st.multiselect(
            "Artists", pool_names, max_selections=6, key="scout_genre_picks",
            label_visibility="collapsed", placeholder="Select up to 6 artists…")
        sel = [by_name[n] for n in picks]

        if sel:
            st.dataframe(_compare_df(sel), hide_index=True,
                         use_container_width=True)
            status = compliance_status()
            bc = st.columns([1.4, 1, 2])
            use_web = bc[1].checkbox(
                "🔍 web search", value=True, key="scout_reco_web",
                help="Use web reputation in the analysis (esp. DJ-led artists). "
                     "Off = LOFI + Chartmetric data only.")
            run = bc[0].button("Run AI booking analysis", type="primary",
                               key="scout_reco_run")
            if status["mode"] == "live":
                bc[2].caption(f"AI live — {status['model']}, region "
                              f"{status['inference_geo']}.")
            else:
                bc[2].caption(f"Preview mode — {status['reason']}.")

            if run:
                with st.spinner("Running a deep booking analysis over the "
                                "shortlist — this can take a bit…"):
                    try:
                        views = [build_validation_view(
                            c["artist_id"], c["artist_name"],
                            flat_by_id.get(c["artist_id"]) or {},
                            ml.get(c["artist_id"])) for c in sel]
                        st.session_state["scout_reco"] = {
                            "key": tuple(sorted(picks)),
                            "result": recommend_booking(views, use_web=use_web),
                            "views": views}
                        st.session_state["scout_reco_chat"] = []  # fresh follow-up
                    except Exception as e:  # noqa: BLE001
                        st.error(f"Analysis failed: {e}")

            reco = st.session_state.get("scout_reco")
            if reco and reco.get("key") == tuple(sorted(picks)):
                st.divider()
                _render_reco(reco, flat_by_id, ml)
        else:
            st.caption("Select at least one artist (up to 6) to analyse.")
    else:
        st.caption("Pick a genre to start. Or open **Browse** below to explore "
                   "the full ranked pool.")

    st.divider()
    with st.expander("Browse the full ranked pool"):
        _render_browse(candidates, taxonomy, flat_by_id, ml)


def _render_browse(candidates: list[dict], taxonomy: dict,
                   flat_by_id: dict, ml: dict) -> None:
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
    if not rows:
        st.caption("Select a row for the score breakdown and one-click actions.")
        return

    c = ranked[rows[0]]
    st.divider()
    _detail(c)

    # ── one-click actions: jump straight into a validation or a comparison ─────
    aid = c["artist_id"]
    act = st.columns([1.2, 1.4, 1.6])
    use_web = act[2].checkbox(
        "🔍 web search", value=True, key=f"browse_web_{aid}",
        help="Use web reputation in the validation (esp. DJ-led artists).")
    validate = act[0].button("🎟️ Validate", key=f"browse_val_{aid}",
                             type="primary",
                             help="Ticket & fit verdict grounded in LOFI history")
    if act[1].button("➕ Add to Compare", key=f"browse_cmp_{aid}"):
        _add_to_compare(c["artist_name"])

    if validate:
        with st.spinner("Validating selected artist…"):
            try:
                view = build_validation_view(
                    aid, c["artist_name"], flat_by_id.get(aid) or {},
                    ml.get(aid))
                st.session_state.setdefault("validation", {})[aid] = {
                    "result": validate_artist(view, use_web=use_web), "note": ""}
            except Exception as e:  # noqa: BLE001
                st.error(f"Validation failed: {e}")

    cached = st.session_state.get("validation", {}).get(aid)
    if cached:
        st.divider()
        render_validation_result(cached["result"])


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
        candidates, taxonomy, flat_by_id, ml = _candidates_and_taxonomy()
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

    tab_scout, tab_compare = st.tabs(["🔍 Scout", _compare_tab_label()])
    with tab_scout:
        _render_scout_tab(candidates, taxonomy, by_name, names, flat_by_id, ml)
    with tab_compare:
        _render_compare_tab(by_name, names)
