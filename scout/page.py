"""
Scout page — deterministic, no-LLM booking shortlist (Phase 1).

A nice, useful interface for the booking team: filter the pool of unbooked,
on-feel artists; scan a ranked table with visual score bars; drill into one
artist; export the shortlist. Built as `render_scout_page()` so it drops into the
main dashboard's nav with a one-line change, and also runs standalone via
`scout/app.py`.
"""
from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from agents.core import compliance_status, generate_rationales
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


# ── score styling (mirrors the dashboard) ────────────────────────────────────

def _score_color(v: float | None) -> str:
    if v is None:
        return "gray"
    if v >= 70:
        return "green"
    if v >= 45:
        return "orange"
    return "red"


def _score_label(v: float | None) -> str:
    if v is None:
        return "Geen data"
    if v >= 75:
        return "Zeer sterk"
    if v >= 60:
        return "Sterk"
    if v >= 45:
        return "Matig"
    if v >= 30:
        return "Zwak"
    return "Zeer zwak"


# ── data loading (cached) ────────────────────────────────────────────────────

@st.cache_resource
def _client():
    return make_client()


@st.cache_data(ttl=3600, show_spinner="Artiesten laden…")
def _candidates_and_taxonomy():
    client = _client()
    flat = load_flat_profiles(client)
    ml = load_ml_features(client)
    candidates = build_candidates(flat, ml, load_predictions())
    return candidates, load_taxonomy()


# ── helpers ──────────────────────────────────────────────────────────────────

def _to_dataframe(ranked: list[dict], ai: dict | None = None) -> pd.DataFrame:
    ai = ai or {}
    return pd.DataFrame([{
        "artist_id": c["artist_id"],
        "Artiest": c["artist_name"],
        "Score": c["rank"],
        "Groei": c["growth"],
        "Potentieel": c["future_potential"],
        "Momentum": c["momentum"],
        "Marktpositie": c["market_relevance"],
        "Data": c["confidence"],
        "Forecast": c.get("forecast_90d"),
        "Genres": ", ".join(c["genres"][:3]),
        "Waarom": ai.get(c["artist_id"], {}).get("rationale_nl") or c["waarom"],
    } for c in ranked])


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


# ── main render ──────────────────────────────────────────────────────────────

def render_scout_page() -> None:
    st.title("Scout")
    st.caption(
        "Opkomende artiesten die nog niet geboekt zijn, gerangschikt op groei en "
        "potentieel. Klik op een rij voor details. (Nog zonder AI — dit is de "
        "datagedreven basis.)"
    )

    try:
        candidates, taxonomy = _candidates_and_taxonomy()
    except Exception as e:  # noqa: BLE001 — surface config/connection issues clearly
        st.error(f"Kon artiesten niet laden: {e}")
        st.info("Controleer of SUPABASE_URL en SUPABASE_KEY zijn ingesteld.")
        return

    if not candidates:
        st.warning("Geen ongeboekte artiesten gevonden.")
        return

    # ── filters ──────────────────────────────────────────────────────────────
    g = taxonomy.get("genres", {})
    genre_opts = sorted(set(g.get("tier_1", []) + g.get("tier_2", [])))
    f = st.columns([2, 1, 1, 1])
    sel_genres = f[0].multiselect("Genre", genre_opts,
                                  help="Leeg = alle on-feel genres")
    min_conf = f[1].slider("Min. data-score", 0, 100, 30,
                           help="Filter artiesten met weinig data weg")
    sort_label = f[2].selectbox(
        "Sorteer op",
        ["Scout-score", "Groei", "Potentieel", "Momentum", "Forecast"],
    )
    query = f[3].text_input("Zoek artiest", placeholder="naam…")
    core_only = st.checkbox(
        "Alleen kerngenres (tech house / house e.d.)", value=False,
        help="Toon alleen artiesten met een kern- of aanverwant genre")

    sort_map = {
        "Scout-score": "rank", "Groei": "growth", "Potentieel": "future_potential",
        "Momentum": "momentum", "Forecast": "forecast",
    }
    ranked = rank_candidates(
        candidates, taxonomy,
        genres=sel_genres, min_confidence=float(min_conf),
        query=query, core_only=core_only, sort_by=sort_map[sort_label],
    )

    if not ranked:
        st.info("Geen artiesten voldoen aan deze filters.")
        return

    # ── KPIs ─────────────────────────────────────────────────────────────────
    with_fc = [c for c in ranked if c.get("forecast_90d") is not None]
    k = st.columns(4)
    k[0].metric("Kandidaten", len(ranked))
    k[1].metric("Met forecast", len(with_fc))
    k[2].metric("Gem. groei",
                f"{sum(c['growth'] for c in ranked) / len(ranked):.0f}/100")
    k[3].metric("Gem. potentieel",
                f"{sum(c['future_potential'] for c in ranked) / len(ranked):.0f}/100")

    # ── AI-toelichting (Claude, of voorbeeldmodus) ───────────────────────────
    status = compliance_status()
    bc = st.columns([1, 3])
    if bc[0].button("Genereer AI-toelichting", type="primary"):
        with st.spinner("Toelichting genereren…"):
            try:
                st.session_state["scout_ai"] = generate_rationales(ranked[:25])
            except Exception as e:  # noqa: BLE001 — surface LLM/config errors
                st.error(f"AI-toelichting mislukt: {e}")
    if status["mode"] == "live":
        bc[1].caption(
            f"AI actief — {status['model']}, regio {status['inference_geo']}.")
    else:
        bc[1].caption(
            f"Voorbeeldmodus — {status['reason']}. De 'Waarom' is nu de "
            "datagedreven toelichting; AI vult dit later aan.")
    ai = st.session_state.get("scout_ai")

    # ── ranked table ─────────────────────────────────────────────────────────
    df = _to_dataframe(ranked, ai)
    prog = lambda label: st.column_config.ProgressColumn(  # noqa: E731
        label, min_value=0, max_value=100, format="%d")
    event = st.dataframe(
        df,
        hide_index=True,
        use_container_width=True,
        on_select="rerun",
        selection_mode="single-row",
        key="scout_table",
        column_order=["Artiest", "Score", "Groei", "Potentieel",
                      "Forecast", "Genres", "Waarom"],
        column_config={
            "Artiest": st.column_config.TextColumn("Artiest", width="medium",
                                                   pinned=True),
            "Score": prog("Scout-score"),
            "Groei": prog("Groei"),
            "Potentieel": prog("Potentieel"),
            "Forecast": st.column_config.NumberColumn("Forecast 90d", format="%.0f%%"),
            "Genres": st.column_config.TextColumn(width="medium"),
            "Waarom": st.column_config.TextColumn("Waarom", width="large"),
        },
    )

    st.download_button(
        "Download shortlist (CSV)",
        df.drop(columns=["artist_id"]).to_csv(index=False).encode("utf-8"),
        file_name="lofi_scout_shortlist.csv",
        mime="text/csv",
    )

    # ── detail panel ─────────────────────────────────────────────────────────
    rows = event.selection.rows if event and event.selection else []
    if rows:
        st.divider()
        _detail(ranked[rows[0]])
    else:
        st.caption("Selecteer een rij voor de score-opbouw van een artiest.")
