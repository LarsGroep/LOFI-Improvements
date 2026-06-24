"""
Per-artist chat dock + booker-in-the-loop intake (Phase 2).

Renders at the bottom of the artist profile. The chat is an artist insight &
booking advisor; the intake lets a booker add real-world / correction / intel
feedback that the AI files and reuses in every future answer.
"""
from __future__ import annotations

import streamlit as st

from agents.core import chat_stream, compliance_status
from scout.context import build_artist_view

# Quick-action chips — the high-value questions, one click for the team.
_CHIPS = [
    ("Good booking?",
     "Is this a good booking for LOFI? Give a reasoned verdict (yes/no/maybe) "
     "with the key signals."),
    ("Realistic fee?",
     "What's a realistic fee for this artist, based on comparable past LOFI "
     "bookings?"),
    ("Why growing?",
     "Explain in plain terms why this artist is growing or cooling off."),
    ("Comparable to…",
     "Which reference artists and past LOFI bookings is this artist comparable "
     "to, and why?"),
    ("Risks",
     "What are the risks or caveats in booking this artist?"),
    ("Team note",
     "Write a short internal note (3-4 sentences) about this artist for the "
     "booking team."),
]


@st.cache_data(ttl=300, show_spinner=False)
def _cached_feedback(artist_id: str):
    from scout.feedback import load_feedback
    return load_feedback(artist_id)


def _feedback_intake(artist_id: str, name: str, feedback_count: int) -> None:
    with st.expander(f"➕ Add booker feedback on {name}"):
        st.caption(
            "Type what you know — a real-world result, a correction, or industry "
            "intel. The AI files it and uses it in every future answer. This is "
            "LOFI's edge over generic data tools.")
        txt = st.text_area(
            "Feedback", key=f"fb_in_{artist_id}", height=80,
            label_visibility="collapsed",
            placeholder="e.g. Sold ~300 tickets at BRET · momentum looks too "
                        "high · agency switch coming")
        if st.button("Save feedback", key=f"fb_save_{artist_id}",
                     type="primary") and txt.strip():
            try:
                from scout.feedback import capture_feedback
                s = capture_feedback(artist_id, txt.strip())
                _cached_feedback.clear()
                st.success(f"Logged as [{s.get('category', 'note')}]: "
                           f"{s.get('summary', '')}")
                st.rerun()
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not save feedback: {e}")
        if feedback_count:
            st.caption(f"{feedback_count} feedback item(s) on record — used by "
                       "the AI.")


def render_artist_chat(artist_id: str, name: str, profile: dict,
                       ml: dict | None, ext: dict | None = None,
                       ra_df=None, pf_data: dict | None = None, vdf=None,
                       nl_score=None) -> None:
    st.divider()
    st.subheader("AI assistant")

    status = compliance_status()
    if status["mode"] == "live":
        st.caption(f"AI live — {status['model']}, region "
                   f"{status['inference_geo']}. Ask about {name}.")
    else:
        st.caption(f"Preview mode — {status['reason']}. Set LOFI_LLM_ENABLED for "
                   "real answers.")

    feedback = _cached_feedback(artist_id)
    view = build_artist_view(artist_id, name, profile, ml, ext=ext,
                             ra_df=ra_df, pf_data=pf_data, vdf=vdf,
                             nl_score=nl_score, booker_feedback=feedback)
    if (not view["booking_history"] and not view["comparables"]
            and not view.get("lofi_record")):
        st.caption("Airtable booking data not connected yet — I can't answer "
                   "fee/ticket questions.")

    _feedback_intake(artist_id, name, len(feedback))

    # per-artist conversation history
    hist = st.session_state.setdefault("artist_chat", {}).setdefault(artist_id, [])
    for m in hist:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    clicked = None
    cols = st.columns(len(_CHIPS))
    for col, (label, question) in zip(cols, _CHIPS):
        if col.button(label, key=f"chip_{artist_id}_{label}"):
            clicked = question

    typed = st.chat_input(f"Ask about {name}…")
    user_msg = typed or clicked
    if not user_msg:
        return

    with st.chat_message("user"):
        st.markdown(user_msg)
    with st.chat_message("assistant"):
        try:
            full = st.write_stream(chat_stream(view, list(hist), user_msg))
        except Exception as e:  # noqa: BLE001
            full = f"Something went wrong with the AI assistant: {e}"
            st.error(full)

    hist.append({"role": "user", "content": user_msg})
    hist.append({"role": "assistant", "content": full})
