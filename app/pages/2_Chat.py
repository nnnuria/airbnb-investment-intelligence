"""Conversational chat — ask the co-pilot follow-ups about the analysis."""

from __future__ import annotations

import streamlit as st

from components.branding import (
    BORDER_SUBTLE,
    DISCLAIMER,
    KPMG_BLUE,
    KPMG_BLUE_TINT,
    TEXT_MUTED,
    TEXT_SECONDARY,
    WHITE,
)
from airbnb_iip.decision.engine import CITY_LABELS
from components.api_client import APIError
from components.api_client import chat as chat_api
from components.styling import apply_page_style, footer_disclaimer, hero

GREETING = (
    "Hi, I'm your investment co-pilot. Ask about the recommendation, projected "
    "revenue, occupancy, comparable listings, regulations, or improvement ideas — "
    "I answer from the same models and agents the decision page uses."
)

apply_page_style("Chat")

hero(
    eyebrow="Co-pilot",
    title="Talk to the analyst",
    lede="Ask follow-ups about projected revenue, the sell-vs-Airbnb call, "
         "regulations, or improvement ideas. Responses are explainable and "
         "cite the same models used in the decision page.",
)

scen = st.session_state.get("scenario")
prop = st.session_state.get("property")


def _assistant_reply(prompt: str) -> dict:
    """Route a message through the API coordinator, with the active analysis
    as context. Returns ``{content, sources}``; falls back to a friendly error
    if the API is unreachable."""
    try:
        res = chat_api(prompt, property=prop, scenario=scen)
        return {"content": res.get("answer", ""), "sources": res.get("sources", [])}
    except APIError as exc:
        return {"content": f"⚠️ {exc}", "sources": []}


# ── Context banner (above messages) ──────────────────────────────────────────
if scen and prop:
    st.markdown(
        f"""
        <div style="padding: 0.8rem 1.1rem; border: 1px solid {BORDER_SUBTLE};
                    border-left: 4px solid {KPMG_BLUE};
                    background: linear-gradient(90deg, {KPMG_BLUE_TINT} 0%, {WHITE} 90%);
                    border-radius: 10px; margin-bottom: 1rem;
                    font-size: 0.92rem; color: {TEXT_SECONDARY};">
          <b>Talking about:</b> {prop.district}, {CITY_LABELS[prop.city]}
          · {prop.size_m2:.0f} m² · recommendation
          <b style="color: {KPMG_BLUE};">{scen.recommendation.upper()}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f"""
        <div style="padding: 0.8rem 1.1rem; border: 1px dashed {BORDER_SUBTLE};
                    background: {WHITE}; border-radius: 10px;
                    margin-bottom: 1rem; color: {TEXT_MUTED}; font-size: 0.92rem;">
          No active analysis. The chat will give general guidance until you
          run one — open <b>New analysis</b> in the sidebar.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Conversation state ───────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = [
        {"role": "assistant", "content": GREETING},
    ]

for msg in st.session_state["chat_history"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            st.caption("Sources: " + " · ".join(msg["sources"]))

# ── Quick-prompt chips ───────────────────────────────────────────────────────
st.markdown(
    f'<div style="color: {TEXT_MUTED}; font-size: 0.8rem; '
    f'margin: 1rem 0 0.4rem 0;">Suggested questions</div>',
    unsafe_allow_html=True,
)
chip_cols = st.columns(4)
suggestions = [
    "Why this recommendation?",
    "How do I compare to the market?",
    "What about regulations?",
    "What if I added AC?",
]
for col, prompt in zip(chip_cols, suggestions):
    with col:
        if st.button(prompt, key=f"chip_{prompt}", use_container_width=True):
            st.session_state["chat_history"].append({"role": "user", "content": prompt})
            with st.spinner("Thinking…"):
                reply = _assistant_reply(prompt)
            st.session_state["chat_history"].append(
                {"role": "assistant", "content": reply["content"], "sources": reply["sources"]},
            )
            st.rerun()

# ── Free-form input ──────────────────────────────────────────────────────────
user_input = st.chat_input("Ask a question — try 'should I sell?'")
if user_input:
    st.session_state["chat_history"].append({"role": "user", "content": user_input})
    with st.spinner("Thinking…"):
        reply = _assistant_reply(user_input)
    st.session_state["chat_history"].append(
        {"role": "assistant", "content": reply["content"], "sources": reply["sources"]},
    )
    st.rerun()

st.write("")
if st.button("Clear conversation", type="secondary"):
    st.session_state["chat_history"] = [
        {"role": "assistant", "content": GREETING},
    ]
    st.rerun()

footer_disclaimer(DISCLAIMER)
