"""Landing page — Airbnb Investment Intelligence.

Run with:

    streamlit run app/Home.py
"""

from __future__ import annotations

import streamlit as st

from components.branding import (
    BORDER_SUBTLE,
    DISCLAIMER,
    KPMG_BLUE,
    KPMG_BLUE_TINT,
    PARTNERSHIP,
    PRODUCT_NAME,
    TEXT_MUTED,
    TEXT_SECONDARY,
    WHITE,
)
from components.storage import load_all
from components.styling import apply_page_style, footer_disclaimer, hero

apply_page_style("Home")

hero(
    eyebrow=PARTNERSHIP,
    title=f"{PRODUCT_NAME}.",
    lede=(
        "A governed, multi-agent decision-support platform for short-term "
        "rental investors. Answer one question — list it on Airbnb or sell — "
        "with data-grounded, explainable recommendations and clear uncertainty."
    ),
)

# ── Primary CTAs ─────────────────────────────────────────────────────────────
cta_left, cta_right = st.columns([1, 1])
with cta_left:
    if st.button("Start a new analysis", use_container_width=True, type="primary"):
        st.switch_page("pages/1_New_Analysis.py")
with cta_right:
    if st.button("Open the chat", use_container_width=True):
        st.switch_page("pages/2_Chat.py")

st.write("")
st.write("")

# ── Three-up feature row ─────────────────────────────────────────────────────
def feature_block(eyebrow: str, title: str, body: str) -> None:
    st.markdown(
        f"""
        <div style="padding: 1.5rem; border: 1px solid {BORDER_SUBTLE};
                    background: {WHITE}; border-radius: 14px; height: 100%;">
          <div style="color: {KPMG_BLUE}; font-size: 0.72rem;
                      font-weight: 700; letter-spacing: 0.08em;
                      text-transform: uppercase; margin-bottom: 0.65rem;">
            {eyebrow}
          </div>
          <div style="font-size: 1.1rem; font-weight: 700; line-height: 1.3;
                      margin-bottom: 0.4rem;">
            {title}
          </div>
          <div style="color: {TEXT_SECONDARY}; line-height: 1.55;
                      font-size: 0.95rem;">
            {body}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


c1, c2, c3 = st.columns(3, gap="medium")
with c1:
    feature_block(
        "Decide",
        "Airbnb or sell?",
        "Compare projected Airbnb net revenue against indicative sale value, "
        "with P10/P50/P90 confidence bands and a clear break-even timeline.",
    )
with c2:
    feature_block(
        "Optimise",
        "Improve revenue with confidence",
        "Ranked, ROI-weighted improvements — amenities, photos, dynamic "
        "pricing — each with cost, expected uplift, and payback period.",
    )
with c3:
    feature_block(
        "Converse",
        "Ask anything",
        "A conversational co-pilot that explains the model's reasoning, "
        "cites regulatory sources, and flags uncertainty rather than hiding it.",
    )

# ── How it works ─────────────────────────────────────────────────────────────
st.write("")
st.write("")
st.markdown("### How it works")
steps_cols = st.columns(4, gap="medium")
steps = [
    ("01", "Enter property details", "City, district, size, capacity, amenities."),
    ("02", "We run the models", "Price, occupancy, finance, comparables — all locally."),
    ("03", "Review the recommendation", "Net income vs sale value, with uncertainty bands."),
    ("04", "Save and revisit", "Compare scenarios side by side; ask the co-pilot follow-ups."),
]
for col, (num, title, body) in zip(steps_cols, steps):
    with col:
        st.markdown(
            f"""
            <div style="padding: 1.25rem 0;">
              <div style="color: {KPMG_BLUE}; font-weight: 700;
                          font-size: 0.85rem; margin-bottom: 0.4rem;">
                {num}
              </div>
              <div style="font-weight: 600; font-size: 1rem;
                          margin-bottom: 0.25rem;">
                {title}
              </div>
              <div style="color: {TEXT_MUTED}; font-size: 0.88rem;
                          line-height: 1.5;">
                {body}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ── Live session summary ─────────────────────────────────────────────────────
st.write("")
st.markdown("### Session summary")

saved = load_all()
current = st.session_state.get("scenario")
property_in_session = st.session_state.get("property")

summary_left, summary_right = st.columns(2, gap="medium")
with summary_left:
    if current and property_in_session:
        st.markdown(
            f"""
            <div style="padding: 1.25rem 1.5rem; border: 1px solid {KPMG_BLUE};
                        background: linear-gradient(180deg, {KPMG_BLUE_TINT} 0%, {WHITE} 100%);
                        border-radius: 14px;">
              <div style="color: {KPMG_BLUE}; font-weight: 700;
                          font-size: 0.72rem; letter-spacing: 0.08em;
                          text-transform: uppercase; margin-bottom: 0.5rem;">
                In progress
              </div>
              <div style="font-weight: 700; font-size: 1.1rem;
                          margin-bottom: 0.35rem;">
                {property_in_session.district}, {property_in_session.city.title()}
              </div>
              <div style="color: {TEXT_SECONDARY}; font-size: 0.92rem;">
                Recommendation: <b>{current.recommendation.title()}</b>
                · break-even {current.breakeven_years:.1f} years
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div style="padding: 1.25rem 1.5rem; border: 1px dashed {BORDER_SUBTLE};
                        background: {WHITE}; border-radius: 14px;
                        color: {TEXT_MUTED};">
              No active analysis. Start one to see results here.
            </div>
            """,
            unsafe_allow_html=True,
        )
with summary_right:
    st.markdown(
        f"""
        <div style="padding: 1.25rem 1.5rem; border: 1px solid {BORDER_SUBTLE};
                    background: {WHITE}; border-radius: 14px;">
          <div style="color: {TEXT_MUTED}; font-weight: 600;
                      font-size: 0.72rem; letter-spacing: 0.08em;
                      text-transform: uppercase; margin-bottom: 0.5rem;">
            Saved
          </div>
          <div style="font-weight: 700; font-size: 1.6rem;
                      margin-bottom: 0.15rem;">
            {len(saved)}
          </div>
          <div style="color: {TEXT_SECONDARY}; font-size: 0.9rem;">
            properties under analysis
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

footer_disclaimer(DISCLAIMER)
