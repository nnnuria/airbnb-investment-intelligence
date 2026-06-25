"""Optimisation — ROI-ranked improvement ideas for an Airbnb-bound property.

Powered by ``airbnb_iip.agents.optimisation``: per-amenity uplift (price-model
counterfactual for AC, ``price − price_hat`` residual for the rest), a peer-group
revenue gap, and Apriori-style association rules as the "why".
"""

from __future__ import annotations

import re

import streamlit as st

from components.api_client import APIError
from components.api_client import optimise as api_optimise
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
from airbnb_iip.decision.engine import CITY_LABELS
from components.styling import apply_page_style, footer_disclaimer, hero

apply_page_style("Optimisation")

hero(
    eyebrow="Step 04",
    title="Optimisation",
    lede="If you go the Airbnb route, here are improvements ranked by expected "
         "return. Toggle the ones you'd actually make and enter your own "
         "investment for each — the summary updates to match.",
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

# ── Run the optimisation service ──────────────────────────────────────────────
spec = {
    "city": prop.city,
    "district": prop.district,
    "room_type": prop.room_type,
    "accommodates": prop.accommodates,
    "bedrooms": prop.bedrooms,
    "bathrooms": prop.bathrooms,
    "has_ac": prop.has_ac,
    "has_balcony": prop.has_balcony,
    "has_elevator": prop.has_elevator,
    "has_pool": prop.has_pool,
    "has_parking": prop.has_parking,
    "has_workspace": prop.has_workspace,
}
# Cache the plan per property so the per-row toggles and investment inputs below
# don't re-hit /optimise on every Streamlit rerun (it's deterministic per spec).
_opt_sig = (tuple(sorted(spec.items())), scen.nights_booked_year)
if st.session_state.get("_opt_sig") != _opt_sig:
    try:
        st.session_state["_opt_plan"] = api_optimise(
            spec, projected_annual_nights=scen.nights_booked_year
        )
        st.session_state["_opt_sig"] = _opt_sig
        st.session_state.pop("_opt_error", None)
    except APIError as exc:  # pragma: no cover - defensive UI guard
        st.session_state["_opt_error"] = str(exc)
        st.session_state.pop("_opt_plan", None)

if st.session_state.get("_opt_error"):
    st.warning(
        f"Optimisation data is unavailable right now ({st.session_state['_opt_error']})."
    )
    footer_disclaimer(DISCLAIMER)
    st.stop()

plan = st.session_state["_opt_plan"]
recommendations = plan.recommendations

# ── Peer-group revenue gap (feature-gap framing) ──────────────────────────────
if plan.peer_n:
    st.markdown(
        f"""
        <div style="padding: 1.25rem 1.5rem; border: 1px solid {KPMG_BLUE};
                    background: linear-gradient(180deg, {KPMG_BLUE_TINT} 0%, {WHITE} 85%);
                    border-radius: 14px; margin-bottom: 1.25rem;">
          <div style="color: {KPMG_BLUE_DARK}; font-size: 0.72rem; font-weight: 600;
                      letter-spacing: 0.08em; text-transform: uppercase;
                      margin-bottom: 0.4rem;">
            Revenue gap vs top performers
          </div>
          <div style="font-size: 0.95rem; color: {TEXT_SECONDARY}; line-height: 1.55;">
            Among <b>{plan.peer_n:,}</b> comparable {CITY_LABELS[prop.city]} listings
            ({prop.room_type.lower()}), the typical one earns
            <b>€{plan.peer_median_revenue_eur:,.0f}/yr</b> while the top quartile reaches
            <b>€{plan.peer_target_revenue_eur:,.0f}/yr</b> — a headroom of
            <b style="color: {KPMG_BLUE};">€{plan.gap_to_top_quartile_eur:,.0f}/yr</b>.
            The improvements below are ways to close it.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

if not recommendations:
    st.success(
        "This property already has the high-ROI amenities we'd recommend for "
        "its peer group — there's no obvious low-payback improvement left."
    )
    footer_disclaimer(DISCLAIMER)
    st.stop()

method_label = {"counterfactual": "model counterfactual", "residual": "market residual"}


def _row_key(name: str) -> str:
    """Stable widget key from an improvement name (per property)."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


st.markdown("##### Choose improvements & enter your costs")
st.caption(
    "Toggle the improvements you'd actually make, then enter your own investment "
    "for each. The summary at the bottom reflects only what's switched on."
)

# ── Interactive list of improvements ─────────────────────────────────────────
# (included, investment, annual_uplift) per row → drives the totals below.
row_state: list[tuple[bool, float, float]] = []
for idx, imp in enumerate(recommendations, start=1):
    confidence_color = {
        "high": SUCCESS, "medium": KPMG_BLUE, "low": WARNING,
    }[imp.confidence]
    lift_chip = (
        f'<span style="margin-left: 0.4rem; color: {TEXT_MUTED}; font-size: 0.72rem;">'
        f'lift {imp.lift:.2f}×</span>'
        if imp.lift else ""
    )
    key = _row_key(imp.name)

    with st.container(border=True):
        tog_col, body_col, inv_col = st.columns(
            [0.7, 3, 1.7], gap="medium", vertical_alignment="center",
        )
        with tog_col:
            included = st.toggle(
                "Include this improvement",
                value=True,
                key=f"opt_on_{key}",
                label_visibility="collapsed",
            )
        with body_col:
            name_color = KPMG_BLUE_DARK if included else TEXT_MUTED
            st.markdown(
                f"""
                <div style="display: flex; gap: 0.6rem; align-items: center;
                            margin-bottom: 0.4rem; flex-wrap: wrap;">
                  <span style="color: {KPMG_BLUE}; font-weight: 700;
                               font-size: 0.85rem;">#{idx:02d}</span>
                  <span style="padding: 0.18rem 0.55rem;
                               background: {confidence_color}; color: {WHITE};
                               font-size: 0.68rem; font-weight: 600;
                               letter-spacing: 0.04em; border-radius: 999px;
                               text-transform: uppercase;">
                    {imp.confidence} confidence</span>
                  <span style="color: {TEXT_MUTED}; font-size: 0.72rem;">
                    {method_label.get(imp.method, imp.method)}</span>
                  {lift_chip}
                </div>
                <div style="font-size: 1.15rem; font-weight: 700;
                            margin-bottom: 0.35rem; color: {name_color};">
                  {imp.name}</div>
                <div style="color: {TEXT_SECONDARY}; font-size: 0.92rem;
                            line-height: 1.55;">{imp.rationale}</div>
                """,
                unsafe_allow_html=True,
            )
        with inv_col:
            st.markdown(
                f"""
                <div style="font-size: 0.72rem; color: {TEXT_MUTED};
                            font-weight: 600; letter-spacing: 0.08em;
                            text-transform: uppercase;">Annual uplift</div>
                <div style="font-size: 1.4rem; font-weight: 700; color: {SUCCESS};
                            margin-bottom: 0.35rem;">+€{imp.annual_uplift_eur:,.0f}</div>
                """,
                unsafe_allow_html=True,
            )
            investment = st.number_input(
                "Your investment (€)",
                min_value=0.0, value=0.0, step=50.0,
                key=f"opt_inv_{key}", format="%.0f",
                disabled=not included,
            )
            if included and investment > 0 and imp.annual_uplift_eur > 0:
                pm = investment / imp.annual_uplift_eur * 12
                pcolor = SUCCESS if pm <= 12 else KPMG_BLUE if pm <= 24 else WARNING
                st.markdown(
                    f"<div style='color: {pcolor}; font-size: 0.9rem; "
                    f"font-weight: 600;'>Payback {pm:.0f} months</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.caption("Enter an investment for payback")

    row_state.append((included, investment, imp.annual_uplift_eur))

# ── Totals (only toggled-on rows, with the user's investment figures) ─────────
selected = [(inv, up) for on, inv, up in row_state if on]
total_uplift = sum(up for _, up in selected)
total_invest = sum(inv for inv, _ in selected)
combined_payback_str = (
    f"{total_invest / total_uplift * 12:.0f} months"
    if total_uplift > 0 and total_invest > 0 else "—"
)

st.write("")
st.markdown(f"### Your selection — {len(selected)} of {len(row_state)} improvements")
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
    st.markdown(f"""
    <div style="padding: 1.5rem; border: 1px solid {BORDER_SUBTLE};
                background: {WHITE}; border-radius: 14px;">
      <div style="color: {TEXT_MUTED}; font-size: 0.72rem; font-weight: 600;
                  letter-spacing: 0.08em; text-transform: uppercase;
                  margin-bottom: 0.4rem;">Combined payback</div>
      <div style="font-size: 1.85rem; font-weight: 700;">
        {combined_payback_str}</div>
    </div>""", unsafe_allow_html=True)

st.caption(
    "Uplift method — air conditioning is a price-model feature, so its figure is "
    "a counterfactual (re-predicting with the amenity toggled). Other amenities "
    "use the price − price_hat residual, controlling for location and size. "
    "Lift is from Apriori-style association rules over the city's listings."
)

footer_disclaimer(DISCLAIMER)
