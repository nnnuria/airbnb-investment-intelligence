"""Optimisation — ROI-ranked improvement ideas for an Airbnb-bound property."""

from __future__ import annotations

import streamlit as st

from components.branding import (
    BORDER_SUBTLE,
    DISCLAIMER,
    KPMG_BLUE,
    KPMG_BLUE_DARK,
    KPMG_BLUE_TINT,
    SUCCESS,
    TEXT_MUTED,
    TEXT_SECONDARY,
    WARNING,
    WHITE,
)
from components.engine import CITY_LABELS, suggest_improvements
from components.styling import apply_page_style, footer_disclaimer, hero

apply_page_style("Optimisation")

hero(
    eyebrow="Step 04",
    title="Optimisation",
    lede="If you go the Airbnb route, here are improvements ranked by "
         "payback period. Each one estimates investment, monthly revenue "
         "uplift, and how confident the recommendation is.",
)

scen = st.session_state.get("scenario")
prop = st.session_state.get("property")

if not (scen and prop):
    st.info(
        "Run an analysis first — improvement ideas are tailored to the "
        "specific property and its baseline numbers.",
        icon="◼",
    )
    if st.button("Go to new analysis", type="primary"):
        st.switch_page("pages/1_New_Analysis.py")
    footer_disclaimer(DISCLAIMER)
    st.stop()

st.markdown(
    f"""
    <div style="padding: 0.8rem 1.1rem; border: 1px solid {BORDER_SUBTLE};
                border-left: 4px solid {KPMG_BLUE};
                background: linear-gradient(90deg, {KPMG_BLUE_TINT} 0%, {WHITE} 90%);
                border-radius: 10px; margin-bottom: 1.5rem;
                font-size: 0.92rem; color: {TEXT_SECONDARY};">
      <b>Baseline:</b> {prop.district}, {CITY_LABELS[prop.city]}
      · €{scen.net_revenue_year_eur:,.0f} net / year
      · occupancy {scen.occupancy_rate_annual*100:.0f}%
    </div>
    """,
    unsafe_allow_html=True,
)

improvements = suggest_improvements(prop, scen)

# ── Ranked list of improvements ──────────────────────────────────────────────
for idx, imp in enumerate(improvements, start=1):
    confidence_color = {
        "high": SUCCESS, "medium": KPMG_BLUE, "low": WARNING,
    }[imp.confidence]
    payback_color = SUCCESS if imp.payback_months <= 12 else \
                    KPMG_BLUE if imp.payback_months <= 24 else WARNING
    payback_label = (f"{imp.payback_months:.0f} months" if imp.payback_months < 60
                     else "—")
    annual_uplift = imp.monthly_revenue_uplift_eur * 12

    st.markdown(
        f"""
        <div style="padding: 1.5rem 1.75rem; border: 1px solid {BORDER_SUBTLE};
                    background: {WHITE}; border-radius: 14px; margin-bottom: 1rem;">
          <div style="display: flex; justify-content: space-between;
                      gap: 1.5rem; align-items: flex-start;">
            <div style="flex: 1;">
              <div style="display: flex; gap: 0.6rem; align-items: center;
                          margin-bottom: 0.5rem;">
                <span style="color: {KPMG_BLUE}; font-weight: 700;
                             font-size: 0.85rem;">#{idx:02d}</span>
                <span style="padding: 0.18rem 0.55rem;
                             background: {confidence_color}; color: {WHITE};
                             font-size: 0.68rem; font-weight: 600;
                             letter-spacing: 0.04em; border-radius: 999px;
                             text-transform: uppercase;">
                  {imp.confidence} confidence
                </span>
              </div>
              <div style="font-size: 1.15rem; font-weight: 700;
                          margin-bottom: 0.35rem; color: {KPMG_BLUE_DARK};">
                {imp.name}
              </div>
              <div style="color: {TEXT_SECONDARY}; font-size: 0.92rem;
                          line-height: 1.55;">
                {imp.rationale}
              </div>
            </div>
            <div style="min-width: 220px; text-align: right;">
              <div style="font-size: 0.72rem; color: {TEXT_MUTED};
                          font-weight: 600; letter-spacing: 0.08em;
                          text-transform: uppercase; margin-bottom: 0.2rem;">
                Annual uplift
              </div>
              <div style="font-size: 1.5rem; font-weight: 700; color: {SUCCESS};">
                +€{annual_uplift:,.0f}
              </div>
              <div style="margin-top: 0.6rem; color: {TEXT_SECONDARY};
                          font-size: 0.88rem;">
                Investment <b>€{imp.investment_eur:,.0f}</b>
              </div>
              <div style="margin-top: 0.2rem; color: {payback_color};
                          font-size: 0.92rem; font-weight: 600;">
                Payback {payback_label}
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Totals ───────────────────────────────────────────────────────────────────
total_uplift = sum(i.monthly_revenue_uplift_eur for i in improvements) * 12
total_invest = sum(i.investment_eur for i in improvements)

st.write("")
st.markdown("### If you implemented everything")
t1, t2, t3 = st.columns(3, gap="medium")
with t1:
    st.markdown(f"""
    <div style="padding: 1.5rem; border: 1px solid {KPMG_BLUE};
                background: linear-gradient(180deg, {KPMG_BLUE_TINT} 0%, {WHITE} 80%);
                border-radius: 14px;">
      <div style="color: {KPMG_BLUE_DARK}; font-size: 0.72rem; font-weight: 600;
                  letter-spacing: 0.08em; text-transform: uppercase;
                  margin-bottom: 0.4rem;">Combined annual uplift</div>
      <div style="font-size: 1.85rem; font-weight: 700; color: {KPMG_BLUE_DARK};">
        +€{total_uplift:,.0f}</div>
    </div>""", unsafe_allow_html=True)
with t2:
    st.markdown(f"""
    <div style="padding: 1.5rem; border: 1px solid {BORDER_SUBTLE};
                background: {WHITE}; border-radius: 14px;">
      <div style="color: {TEXT_MUTED}; font-size: 0.72rem; font-weight: 600;
                  letter-spacing: 0.08em; text-transform: uppercase;
                  margin-bottom: 0.4rem;">Total investment</div>
      <div style="font-size: 1.85rem; font-weight: 700;">
        €{total_invest:,.0f}</div>
    </div>""", unsafe_allow_html=True)
with t3:
    combined_payback = (total_invest / total_uplift * 12) if total_uplift > 0 else 0
    st.markdown(f"""
    <div style="padding: 1.5rem; border: 1px solid {BORDER_SUBTLE};
                background: {WHITE}; border-radius: 14px;">
      <div style="color: {TEXT_MUTED}; font-size: 0.72rem; font-weight: 600;
                  letter-spacing: 0.08em; text-transform: uppercase;
                  margin-bottom: 0.4rem;">Combined payback</div>
      <div style="font-size: 1.85rem; font-weight: 700;">
        {combined_payback:.0f} months</div>
    </div>""", unsafe_allow_html=True)

footer_disclaimer(DISCLAIMER)
