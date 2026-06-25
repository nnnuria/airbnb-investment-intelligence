"""New analysis — property input → Airbnb-vs-sell decision."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from airbnb_iip.agents.comparables import get_comparables_agent
from airbnb_iip.agents.governance import apply_regulatory_guardrails
from airbnb_iip.agents.regulatory import get_regulatory_agent
from components.branding import (
    BORDER_SUBTLE,
    DANGER,
    DISCLAIMER,
    KPMG_BLUE,
    KPMG_BLUE_DARK,
    KPMG_BLUE_TINT,
    SUCCESS,
    TEXT_MUTED,
    WARNING,
    WHITE,
)
from airbnb_iip.decision.engine import (
    CITIES,
    CITY_LABELS,
    ROOM_TYPES,
    Property,
    get_city_districts,
)
from components.api_client import APIError, get_scenario
from components.storage import save_record
from components.styling import (
    apply_page_style,
    card,
    footer_disclaimer,
    hero,
)

apply_page_style("New analysis")

hero(
    eyebrow="Step 01",
    title="New analysis",
    lede="Enter the property details. We'll compare projected Airbnb net "
         "income against indicative sale value and give a clear recommendation.",
)

# ── Amenity configuration ─────────────────────────────────────────────────────

AMENITY_BUNDLES: dict[str, list[str]] = {
    "(none)": [],
    "Essentials": [
        "has_ac", "has_elevator", "has_washer", "has_dryer",
    ],
    "Remote Work Ready": [
        "has_ac", "has_elevator", "has_workspace", "has_self_checkin",
        "has_long_term_ok", "has_dishwasher",
    ],
    "Premium Luxury": [
        "has_ac", "has_elevator", "has_pool", "has_hot_tub", "has_gym",
        "has_view", "has_balcony", "has_dishwasher",
    ],
    "Family Friendly": [
        "has_ac", "has_washer", "has_dryer", "has_crib", "has_pets",
        "has_dishwasher",
    ],
    "Urban Comfort": [
        "has_ac", "has_elevator", "has_parking", "has_dishwasher",
        "has_washer", "has_dryer",
    ],
}

AMENITY_CATEGORIES: list[tuple[str, list[tuple[str, str]]]] = [
    ("Comfort & Building", [
        ("has_ac", "Air conditioning"),
        ("has_elevator", "Elevator"),
        ("has_bathtub", "Bathtub"),
        ("has_hot_tub", "Hot tub / Jacuzzi"),
        ("has_gym", "Gym / Fitness"),
    ]),
    ("Outdoor & Views", [
        ("has_balcony", "Balcony or terrace"),
        ("has_outdoor_space", "Outdoor space / Garden"),
        ("has_view", "View / Skyline"),
        ("has_beach", "Beach access"),
        ("has_pool", "Pool"),
    ]),
    ("Work & Smart Stays", [
        ("has_workspace", "Dedicated workspace"),
        ("has_self_checkin", "Self check-in"),
        ("has_long_term_ok", "Long-term stays OK"),
        ("has_ev_charger", "EV charger"),
        ("has_private_entrance", "Private entrance"),
    ]),
    ("Appliances", [
        ("has_washer", "Washer"),
        ("has_dryer", "Dryer"),
        ("has_dishwasher", "Dishwasher"),
        ("has_cleaning_service", "Cleaning service"),
    ]),
    ("Family & Pets", [
        ("has_crib", "Crib / Baby bed"),
        ("has_pets", "Pets allowed"),
        ("has_parking", "Parking"),
    ]),
]

_ALL_AMENITY_KEYS: list[str] = [k for _, items in AMENITY_CATEGORIES for k, _ in items]

# Default-checked amenities (sensible baseline for a typical urban flat)
_DEFAULT_ON: frozenset[str] = frozenset({"has_ac", "has_elevator"})

# ── Property inputs ───────────────────────────────────────────────────────────
# City is rendered OUTSIDE any form so that changing it immediately reruns
# the page and the district dropdown updates to reflect the new city.

col1, col2, col3 = st.columns([1, 1, 1], gap="large")

with col1:
    st.markdown("**Location**")
    city = st.selectbox("City", CITIES, format_func=lambda c: CITY_LABELS[c])
    districts = get_city_districts(city) or ["Centro"]
    district = st.selectbox("District", districts)
    nickname = st.text_input(
        "Nickname (optional)",
        placeholder="e.g. Salamanca pied-à-terre",
    )

with col2:
    st.markdown("**Property**")
    size_m2 = st.number_input("Size (m²)", min_value=20, max_value=400, value=85, step=5)
    bedrooms = st.number_input("Bedrooms", min_value=0, max_value=8, value=2)
    bathrooms = st.number_input(
        "Bathrooms", min_value=1.0, max_value=6.0, value=1.0, step=0.5,
    )
    accommodates = st.number_input("Accommodates", min_value=1, max_value=16, value=4)

with col3:
    st.markdown("**Listing type**")
    room_type = st.selectbox("Room type", ROOM_TYPES)

# ── Amenities section ─────────────────────────────────────────────────────────

st.write("")
st.markdown("**Amenities**")
st.caption(
    "Select a bundle to pre-fill common configurations, then adjust individually. "
    "All checked amenities affect the analysis: "
    "**AC, dishwasher, and crib** feed directly into the price model; "
    "**pool, balcony, elevator, parking, and workspace** apply data-driven price adjustments; "
    "everything else contributes to the amenity score used to find comparable listings."
)

_AMENITY_KEY_TO_LABEL: dict[str, str] = {
    key: label for _, items in AMENITY_CATEGORIES for key, label in items
}

def _bundle_format(name: str) -> str:
    keys = AMENITY_BUNDLES.get(name, [])
    if not keys:
        return name
    labels = ", ".join(_AMENITY_KEY_TO_LABEL.get(k, k) for k in keys)
    return f"{name} — {labels}"

bun_col, apply_col = st.columns([3, 1], gap="small")
with bun_col:
    selected_bundle = st.selectbox(
        "Bundle",
        list(AMENITY_BUNDLES.keys()),
        format_func=_bundle_format,
        key="bundle_select",
        label_visibility="collapsed",
    )
with apply_col:
    if st.button("Apply bundle", use_container_width=True):
        bundle_set = set(AMENITY_BUNDLES.get(st.session_state["bundle_select"], []))
        for key in _ALL_AMENITY_KEYS:
            st.session_state[f"amenity_{key}"] = key in bundle_set
        st.rerun()

# Render amenity checkboxes in category columns
amenity_vals: dict[str, bool] = {}
cat_cols = st.columns(len(AMENITY_CATEGORIES), gap="small")
for i, (cat_name, items) in enumerate(AMENITY_CATEGORIES):
    with cat_cols[i]:
        st.caption(f"**{cat_name}**")
        for key, label in items:
            default = key in _DEFAULT_ON
            amenity_vals[key] = st.checkbox(
                label,
                value=st.session_state.get(f"amenity_{key}", default),
                key=f"amenity_{key}",
            )

st.write("")
extra_text = st.text_area(
    "Additional amenities (one per line)",
    placeholder="e.g.\nRooftop terrace\nSauna\nBilliard table",
    height=90,
    help="Free-text amenities not covered above. Each line counts as one extra amenity "
         "toward the total amenity score used in comparable-listing search.",
)
extra_amenities = [a.strip() for a in extra_text.splitlines() if a.strip()]

st.write("")

# ── Advanced financial assumptions ────────────────────────────────────────────
st.write("")
st.markdown("**Financial assumptions**")
st.caption(
    "Defaults are set to representative Spanish market values. "
    "Adjust to reflect your property's actual costs and your investment horizon."
)
adv_col1, adv_col2, adv_col3 = st.columns(3, gap="large")
with adv_col1:
    st.markdown("**Property costs**")
    cadastral_value_input = st.number_input(
        "Cadastral value (€) — for IBI",
        min_value=0,
        max_value=2_000_000,
        value=0,
        step=5_000,
        help="Valor catastral — appears on your IBI bill or Agencia Tributaria online. "
             "Leave at 0 and the agent will estimate it from the modelled sale price.",
    )
    ibi_eur_override_input = st.number_input(
        "IBI manual override (€/year, optional)",
        min_value=0,
        max_value=10_000,
        value=0,
        step=50,
        help="Override the computed IBI with a known amount. "
             "Takes priority over the cadastral value input above.",
    )
    setup_cost_input = st.number_input(
        "Airbnb setup cost (€)",
        min_value=0,
        max_value=50_000,
        value=1_500,
        step=500,
        help="One-time upfront cost to prepare the property for Airbnb: "
             "professional photography, deep clean, linen and supplies, minor repairs. "
             "Minimum recommended: €1,500.",
    )
    purchase_price_input = st.number_input(
        "Original purchase price (€) — for CGT",
        min_value=0,
        max_value=5_000_000,
        value=0,
        step=10_000,
        help="Used to calculate capital gains tax on sale. "
             "Leave at 0 to use a model default (70% of current estimated value).",
    )
with adv_col2:
    st.markdown("**Growth assumptions**")
    noi_growth_pct = st.slider(
        "Annual rent growth (%)",
        min_value=0.0,
        max_value=8.0,
        value=2.5,
        step=0.5,
        format="%.1f%%",
        help="Expected annual increase in Airbnb revenue (CPI/HPI blend). "
             "Spain 10-yr avg: CPI ~2%, rental index ~3%. Default: 2.5%.",
    )
    appreciation_pct = st.slider(
        "Annual property appreciation (%)",
        min_value=0.0,
        max_value=8.0,
        value=3.0,
        step=0.5,
        format="%.1f%%",
        help="Expected annual property price growth, applied to the terminal "
             "sale value at the end of the holding period.",
    )
    holding_years_input = st.slider(
        "Holding period (years)",
        min_value=5,
        max_value=30,
        value=10,
        step=1,
        help="How many years you plan to hold before selling. "
             "The terminal sale value is discounted back to today at the end of this period.",
    )
    discount_rate_pct = st.slider(
        "Discount rate (opportunity cost, %)",
        min_value=3.0,
        max_value=15.0,
        value=7.0,
        step=0.5,
        format="%.1f%%",
        help="Your personal hurdle rate — the return you'd expect from an alternative "
             "investment of the same capital. 7% is a reasonable default for Spanish property.",
    )
with adv_col3:
    st.markdown("**Tax options**")
    include_tax = st.toggle(
        "Include income tax in analysis",
        value=True,
        help="When on, IRPF progressive brackets (19–47%) are applied to STR "
             "rental profit. Turn off to see the pre-tax comparison alongside.",
    )
    st.caption(
        "Both pre-tax and post-tax NPV figures are always shown in the results. "
        "This toggle determines which drives the primary recommendation."
    )

st.write("")
submitted = st.button("Run analysis", type="primary")

# ── Compute on submit ─────────────────────────────────────────────────────────
if submitted:
    prop = Property(
        city=city,
        district=district,
        size_m2=size_m2,
        bedrooms=int(bedrooms),
        bathrooms=float(bathrooms),
        accommodates=int(accommodates),
        room_type=room_type,
        has_ac=amenity_vals["has_ac"],
        has_elevator=amenity_vals["has_elevator"],
        has_balcony=amenity_vals["has_balcony"],
        has_pool=amenity_vals["has_pool"],
        has_parking=amenity_vals["has_parking"],
        has_workspace=amenity_vals["has_workspace"],
        has_gym=amenity_vals["has_gym"],
        has_hot_tub=amenity_vals["has_hot_tub"],
        has_beach=amenity_vals["has_beach"],
        has_view=amenity_vals["has_view"],
        has_washer=amenity_vals["has_washer"],
        has_dishwasher=amenity_vals["has_dishwasher"],
        has_self_checkin=amenity_vals["has_self_checkin"],
        has_pets=amenity_vals["has_pets"],
        has_crib=amenity_vals["has_crib"],
        has_private_entrance=amenity_vals["has_private_entrance"],
        has_bathtub=amenity_vals["has_bathtub"],
        has_dryer=amenity_vals["has_dryer"],
        has_ev_charger=amenity_vals["has_ev_charger"],
        has_outdoor_space=amenity_vals["has_outdoor_space"],
        has_long_term_ok=amenity_vals["has_long_term_ok"],
        has_cleaning_service=amenity_vals["has_cleaning_service"],
        extra_amenities=extra_amenities,
        nickname=nickname or None,
    )
    st.session_state["property"] = prop
    # The decision engine now runs behind the FastAPI /scenario endpoint; the
    # app is a thin client of it (one source of truth, no in-process model load).
    try:
        st.session_state["scenario"] = get_scenario(
            prop,
            managed=st.session_state.get("managed_toggle", True),
            ibi_eur=float(ibi_eur_override_input) if ibi_eur_override_input > 0 else None,
            cadastral_value=float(cadastral_value_input) if cadastral_value_input > 0 else None,
            setup_cost_eur=float(setup_cost_input),
            noi_growth_rate=noi_growth_pct / 100.0,
            property_appreciation_rate=appreciation_pct / 100.0,
            include_income_tax=include_tax,
            purchase_price=float(purchase_price_input) if purchase_price_input > 0 else None,
            holding_years=int(holding_years_input),
            discount_rate=discount_rate_pct / 100.0,
        )
        st.session_state.pop("api_error", None)
    except APIError as exc:
        st.session_state["scenario"] = None
        st.session_state["regulatory"] = None
        st.session_state["api_error"] = str(exc)

    # Regulatory risk — only if the core analysis succeeded. Computed once per
    # "Run analysis" click (not on every rerun) since it makes a real Gemini
    # call; cached in session_state like the scenario itself so later reruns
    # (e.g. clicking "Save") don't re-trigger it.
    if st.session_state.get("scenario") is not None:
        try:
            reg_raw = get_regulatory_agent().get_risk_flag(prop.city, prop.district)
            st.session_state["regulatory"] = apply_regulatory_guardrails(reg_raw)
        except Exception as exc:
            st.session_state["regulatory"] = {"error": str(exc)}

def _recompute_scenario() -> None:
    """Re-run the scenario when the management-mode switch flips, reusing the
    stored property so the user doesn't have to click 'Run analysis' again."""
    stored = st.session_state.get("property")
    if stored is None:
        return
    try:
        st.session_state["scenario"] = get_scenario(
            stored, managed=st.session_state.get("managed_toggle", True)
        )
        st.session_state.pop("api_error", None)
    except APIError as exc:
        st.session_state["scenario"] = None
        st.session_state["api_error"] = str(exc)


prop = st.session_state.get("property")
scen = st.session_state.get("scenario")
reg = st.session_state.get("regulatory")
api_error = st.session_state.get("api_error")

# ── API unreachable banner ───────────────────────────────────────────────────
if api_error:
    st.error(
        f"**Analysis unavailable.** {api_error}",
        icon="🚫",
    )

# ── Results ──────────────────────────────────────────────────────────────────
if scen and prop:
    st.write("")
    st.write("")

    label_map = {
        "airbnb": ("LIST ON AIRBNB", SUCCESS),
        "sell":   ("SELL", WARNING),
        "marginal": ("MARGINAL — INSPECT ASSUMPTIONS", TEXT_MUTED),
    }
    rec_text, rec_color = label_map[scen.recommendation]

    st.markdown(
        f"""
        <div style="display: flex; align-items: baseline; gap: 0.85rem; margin-bottom: 0.35rem;">
          <span style="display: inline-block; padding: 0.4rem 0.85rem;
                       background: {rec_color}; color: {WHITE};
                       font-weight: 600; font-size: 0.8rem;
                       letter-spacing: 0.04em; border-radius: 999px;">
            {rec_text}
          </span>
          <span style="color: {TEXT_MUTED}; font-size: 0.88rem;">
            Confidence ~{scen.confidence*100:.0f}%
          </span>
        </div>
        <div style="font-size: 1.65rem; font-weight: 700;
                    line-height: 1.3; margin-bottom: 1rem;">
          {prop.district}, {CITY_LABELS[prop.city]}
          <span style="font-weight: 500; color: {TEXT_MUTED};
                       font-size: 1rem;"> · {prop.size_m2:.0f} m² · {prop.bedrooms} bed</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Management mode switch ───────────────────────────────────────────────
    # Toggles the 20% management fee across revenue, costs, payback and the
    # recommendation. Flipping it recomputes the scenario for the same property.
    _, sw_toggle = st.columns([3, 1.15], gap="medium")
    with sw_toggle:
        st.toggle(
            "Professionally managed",
            value=True,
            key="managed_toggle",
            on_change=_recompute_scenario,
            help=(
                "On — a property manager runs the listing (−20% of gross revenue "
                "in management fees). Off — you self-manage and keep that fee."
            ),
        )

    # ── Regulatory risk banner (shown ABOVE KPIs when risk is HIGH/UNKNOWN) ──
    if reg and "error" not in reg:
        risk_flag = reg.get("risk_flag", "UNKNOWN")
        if risk_flag in ("HIGH", "UNKNOWN"):
            sources_txt = "; ".join(reg.get("sources", [])) or "see official municipal sources"
            st.warning(
                f"**⚠️ Regulatory risk: {risk_flag}** — STR licensing in "
                f"{prop.district}, {CITY_LABELS[prop.city]} may be restricted or "
                f"unavailable. This has been factored into the recommendation "
                f"(confidence capped at Marginal). "
                f"Verify with official sources before proceeding.\n\n"
                f"Sources: {sources_txt}"
            )

    # ── Governance banner ────────────────────────────────────────────────────
    gov = scen.governance or {}
    violations = gov.get("violations", [])
    if gov.get("human_review_required") or violations:
        lines = [v["message"] for v in violations]
        if gov.get("human_review_required") and not violations:
            lines.append(
                "Recommendation is marginal — review the assumptions before "
                "treating this as a clear signal."
            )
        banner = st.error if any(v["severity"] == "blocked" for v in violations) else st.warning
        banner(
            "**Governance review flagged this analysis:**\n\n"
            + "\n".join(f"- {line}" for line in lines)
        )

    # KPI row
    k1, k2, k3, k4, k5 = st.columns(5, gap="medium")
    with k1:
        card(
            "Gross annual revenue",
            f"€{scen.gross_revenue_year_eur:,.0f}",
            f"€{scen.predicted_nightly_eur:,.0f}/night · {scen.nights_booked_year} nights",
        )
    with k2:
        card(
            "Net profit",
            f"€{scen.net_revenue_year_eur:,.0f}",
            f"P10–P90: €{scen.net_revenue_p10_eur:,.0f} – €{scen.net_revenue_p90_eur:,.0f}",
            featured=True,
        )
    with k3:
        card(
            "Indicative sale value",
            f"€{scen.sale_price_eur:,.0f}",
            f"€{scen.sale_price_per_m2_eur:,.0f}/m²",
        )
    with k4:
        card(
            "Payback period",
            f"{scen.breakeven_years:.1f} yrs"
            if scen.breakeven_years < 50 else "50+ yrs",
            "net profit to recover sale value",
        )
    with k5:
        card(
            "Projected occupancy",
            f"{scen.occupancy_rate_annual*100:.0f}%",
            f"{scen.nights_booked_year} nights/year",
        )

    st.write("")

    # ── NPV & IRR comparison row ─────────────────────────────────────────────
    st.markdown("##### Investment comparison (10-year horizon)")
    irr_col, npv_ab_col, npv_ab_pre_col, npv_s_col, npv_s_pre_col = st.columns(5, gap="medium")
    with irr_col:
        irr_display = (
            f"{scen.irr_airbnb_pct * 100:.1f}%"
            if scen.irr_airbnb_pct is not None
            else "—"
        )
        card(
            "IRR (Airbnb, 10yr)",
            irr_display,
            "Internal rate of return on setup investment",
            featured=True,
        )
    with npv_ab_col:
        card(
            "NPV Airbnb (after-tax)",
            f"€{scen.npv_airbnb_p50_eur:,.0f}",
            "P50 · with income tax & CGT",
        )
    with npv_ab_pre_col:
        _adv = getattr(scen, "npv_advantage_eur", scen.npv_airbnb_p50_eur - scen.npv_sell_eur)
        _adv_sign = "+" if _adv >= 0 else ""
        card(
            "Airbnb advantage (P50)",
            f"{_adv_sign}€{_adv:,.0f}",
            "vs selling today · after tax",
            featured=_adv > 0,
        )
    with npv_s_col:
        card(
            "Sell net (after CGT)",
            f"€{scen.npv_sell_eur:,.0f}",
            "After agent fees + capital gains tax",
        )
    with npv_s_pre_col:
        card(
            "Sell gross (pre-tax)",
            f"€{scen.npv_sell_pretax_eur:,.0f}",
            "Before CGT — no reinvestment assumed",
        )

    _ibi_method_label = {"cadastral": "exact cadastral", "manual": "manual override", "estimated": "estimated"}.get(
        getattr(scen, "ibi_method", "estimated"), "estimated"
    )
    _hy = getattr(scen, "holding_years", 10)
    _dr = getattr(scen, "discount_rate", 0.07)
    st.caption(
        f"IBI applied: **€{scen.ibi_eur_used:,.0f}/yr** ({_ibi_method_label}) · "
        f"Basuras: **€{getattr(scen, 'basuras_eur_used', 0):,.0f}/yr** · "
        f"Setup cost: **€{scen.setup_cost_eur_used:,.0f}** · "
        f"Discount rate: **{_dr*100:.1f}%** · Holding period: **{_hy} years**"
    )
    _ibi_explanation = getattr(scen, "ibi_explanation", "")
    if _ibi_explanation:
        st.caption(f"IBI derivation: {_ibi_explanation}")

    st.write("")
    st.write("")

    # ── Charts row ───────────────────────────────────────────────────────────
    chart_left, chart_right = st.columns([1.1, 1], gap="large")

    with chart_left:
        st.markdown("##### Net profit uncertainty")
        st.caption(
            "Indicative P10 / median / P90 net profit after all costs. Uncertainty is "
            "dominated by occupancy variability."
        )
        _bars = [scen.net_revenue_p10_eur,
                 scen.net_revenue_year_eur,
                 scen.net_revenue_p90_eur]
        # Headroom above the tallest bar (and below 0 if any band is negative)
        # so the "outside" value labels render fully instead of being clipped.
        _ymax, _ymin = max(_bars), min(0, min(_bars))
        _headroom = (_ymax - _ymin) * 0.18 or 1.0
        fig = go.Figure(
            go.Bar(
                x=["P10", "Median", "P90"],
                y=_bars,
                marker_color=[KPMG_BLUE_TINT, KPMG_BLUE, KPMG_BLUE_DARK],
                text=[f"€{v:,.0f}" for v in _bars],
                textposition="outside",
                cliponaxis=False,
            )
        )
        fig.update_layout(
            height=320,
            margin=dict(l=10, r=10, t=40, b=20),
            plot_bgcolor=WHITE, paper_bgcolor=WHITE,
            yaxis=dict(showgrid=True, gridcolor=BORDER_SUBTLE, zeroline=False,
                       range=[_ymin, _ymax + _headroom]),
            xaxis=dict(showgrid=False),
            showlegend=False,
            uniformtext=dict(minsize=10, mode="show"),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with chart_right:
        st.markdown("##### Seasonality")
        st.caption("Monthly demand index (1.0 = annual baseline).")
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        fig2 = go.Figure(
            go.Scatter(
                x=months, y=scen.monthly_seasonality,
                mode="lines+markers",
                line=dict(color=KPMG_BLUE, width=3),
                marker=dict(size=8, color=KPMG_BLUE_DARK),
                fill="tozeroy", fillcolor="rgba(0,51,141,0.08)",
            )
        )
        fig2.add_hline(y=1.0, line_dash="dot", line_color=TEXT_MUTED,
                       annotation_text="Baseline", annotation_position="top right")
        fig2.update_layout(
            height=320,
            margin=dict(l=10, r=10, t=20, b=20),
            plot_bgcolor=WHITE, paper_bgcolor=WHITE,
            yaxis=dict(showgrid=True, gridcolor=BORDER_SUBTLE,
                       zeroline=False, range=[0.7, 1.45]),
            xaxis=dict(showgrid=False),
            showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    st.write("")

    # ── Breakdown row ────────────────────────────────────────────────────────
    bd_left, bd_right = st.columns([1, 1], gap="large")

    with bd_left:
        st.markdown("##### Cost breakdown (annual)")
        st.caption("Costs deducted from gross revenue to arrive at net profit.")
        # Combine costs + net, sort descending so the largest bar leads.
        cost_items = list(scen.cost_breakdown) + [
            ("Net profit", scen.net_revenue_year_eur),
        ]
        cost_items.sort(key=lambda x: x[1], reverse=True)
        cost_labels = [name for name, _ in cost_items]
        cost_values = [v for _, v in cost_items]
        cost_colors = [
            KPMG_BLUE_DARK if name == "Net profit" else
            "#5577B0" if v >= 3000 else
            "#A8B9DD" if v >= 1500 else
            KPMG_BLUE_TINT
            for name, v in cost_items
        ]
        fig3 = go.Figure(
            go.Bar(
                x=cost_labels, y=cost_values,
                marker_color=cost_colors,
                text=[f"€{v:,.0f}" for v in cost_values],
                textposition="outside",
                cliponaxis=False,
            )
        )
        fig3.update_layout(
            height=360, margin=dict(l=10, r=10, t=40, b=30),
            plot_bgcolor=WHITE, paper_bgcolor=WHITE,
            yaxis=dict(showgrid=True, gridcolor=BORDER_SUBTLE, zeroline=False,
                       tickformat=",.0f"),
            xaxis=dict(showgrid=False, tickangle=-25),
            showlegend=False,
            uniformtext=dict(minsize=10, mode="show"),
        )
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

    with bd_right:
        st.markdown("##### Why this recommendation")
        st.caption("Top feature contributions. Indicative — full SHAP in the model card.")
        # Sort by absolute magnitude descending.
        drivers_sorted = sorted(scen.feature_drivers, key=lambda x: -abs(x[1]))
        driver_names = [n for n, _ in drivers_sorted]
        driver_vals = [v * 100 for _, v in drivers_sorted]
        marker_colors = [SUCCESS if v >= 0 else WARNING for v in driver_vals]
        fig4 = go.Figure(
            go.Bar(
                x=driver_names, y=driver_vals,
                marker_color=marker_colors,
                text=[f"{'+' if v >= 0 else ''}{v:.1f}%" for v in driver_vals],
                textposition="outside",
                cliponaxis=False,
            )
        )
        fig4.update_layout(
            height=360, margin=dict(l=10, r=10, t=40, b=30),
            plot_bgcolor=WHITE, paper_bgcolor=WHITE,
            yaxis=dict(showgrid=True, gridcolor=BORDER_SUBTLE,
                       zeroline=True, zerolinecolor=TEXT_MUTED,
                       title="Contribution (%)"),
            xaxis=dict(showgrid=False, tickangle=-25),
            showlegend=False,
            uniformtext=dict(minsize=10, mode="show"),
        )
        st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar": False})

    st.write("")
    st.write("")

    # ── Market comparables ───────────────────────────────────────────────────
    try:
        comp_result = get_comparables_agent().find_comparables(
            {
                "city": prop.city,
                "district": prop.district,
                "room_type": prop.room_type,
                "accommodates": prop.accommodates,
                "bedrooms": prop.bedrooms,
                "bathrooms_number": prop.bathrooms,
                "amenity_count": prop.amenity_count,
            },
            k=5,
        )
        comps = comp_result["comparables"]
        bench = comp_result["benchmark"]
    except Exception:
        comps, bench, comp_result = [], {}, {}

    with st.expander("Market comparables — 5 similar listings", expanded=False):
        if not comps:
            st.info("No comparable listings found for this district and room type.")
        else:
            price_b = bench.get("price", {})
            rev_b = bench.get("estimated_revenue_l365d", {})
            rat_b = bench.get("review_scores_rating", {})

            b1, b2, b3 = st.columns(3, gap="medium")
            with b1:
                card(
                    "Median nightly price",
                    f"€{price_b.get('median', 0):,.0f}",
                    f"P25–P75: €{price_b.get('p25', 0):,.0f} – €{price_b.get('p75', 0):,.0f}",
                )
            with b2:
                card(
                    "Median annual revenue",
                    f"€{rev_b.get('median', 0):,.0f}",
                    f"P25–P75: €{rev_b.get('p25', 0):,.0f} – €{rev_b.get('p75', 0):,.0f}",
                )
            with b3:
                median_rat = rat_b.get("median")
                card(
                    "Median rating",
                    f"{median_rat:.2f} / 5" if median_rat else "—",
                    (
                        f"P25–P75: {rat_b.get('p25', 0):.2f} – {rat_b.get('p75', 0):.2f}"
                        if median_rat else "No ratings data"
                    ),
                )

            st.write("")

            rows = []
            for c in comps:
                raw_name = c.get("name") or "—"
                name = raw_name[:48] + "…" if len(raw_name) > 48 else raw_name
                rows.append({
                    "Name": name,
                    "Neighbourhood": c.get("neighbourhood_cleansed") or "—",
                    "Beds": int(c["bedrooms"]) if c.get("bedrooms") is not None else None,
                    "Nightly (€)": c.get("price"),
                    "Revenue / yr (€)": c.get("estimated_revenue_l365d"),
                    "Rating": c.get("review_scores_rating"),
                })

            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Nightly (€)": st.column_config.NumberColumn(format="€%.0f"),
                    "Revenue / yr (€)": st.column_config.NumberColumn(format="€%.0f"),
                    "Rating": st.column_config.NumberColumn(format="%.2f ⭐"),
                },
            )

            filters_note = ", ".join(
                f"{k.replace('_', ' ')}: {v}"
                for k, v in comp_result.get("filters_applied", {}).items()
                if k != "city"
            )
            st.caption(
                f"KNN on beds, capacity, bathrooms, and amenity count · "
                f"Filters: {filters_note or 'city only'}"
            )

    st.write("")
    st.write("")

    # ── Detailed cost breakdown table ────────────────────────────────────────
    with st.expander("Full cost breakdown (itemised)", expanded=False):
        if scen.cost_breakdown:
            gross = scen.gross_revenue_year_eur or 1
            total_costs = sum(v for _, v in scen.cost_breakdown)
            rows_data = [
                {
                    "Cost item": label,
                    "€ / year": f"€{value:,.0f}",
                    "Share of gross": f"{value / gross * 100:.1f}%",
                }
                for label, value in scen.cost_breakdown
            ]
            rows_data.append({
                "Cost item": "**Total costs**",
                "€ / year": f"**€{total_costs:,.0f}**",
                "Share of gross": f"**{total_costs / gross * 100:.1f}%**",
            })
            st.markdown(
                "\n".join(
                    ["| Cost item | €/year | Share of gross |", "|-----------|--------|----------------|"]
                    + [f"| {r['Cost item']} | {r['€ / year']} | {r['Share of gross']} |" for r in rows_data]
                )
            )
            st.caption(
                f"Gross revenue: €{gross:,.0f}/yr · "
                f"Net profit: €{scen.net_revenue_year_eur:,.0f}/yr · "
                f"Income tax included: {'Yes' if any('tax' in l.lower() for l, _ in scen.cost_breakdown) else 'No'}"
            )
        else:
            st.info("Cost breakdown not available for this analysis.")

    st.write("")

    # ── Regulatory risk ──────────────────────────────────────────────────────
    _reg_expanded = bool(reg and "error" not in reg and reg.get("risk_flag") in ("HIGH", "UNKNOWN"))
    with st.expander("Regulatory risk — STR licensing for this district", expanded=_reg_expanded):
        if not reg:
            st.info("No regulatory check available for this analysis.")
        elif "error" in reg:
            if "RESOURCE_EXHAUSTED" in reg["error"] or "429" in reg["error"]:
                friendly = (
                    "Regulatory check unavailable right now — the Gemini API's free-tier "
                    "daily quota has been reached. Try again later."
                )
            else:
                friendly = "Regulatory check unavailable right now (unexpected error)."
            st.warning(
                f"{friendly} Treat this property's regulatory status as unverified — "
                "check official municipal/regional sources manually."
            )
        else:
            risk_colors = {
                "HIGH": DANGER, "MEDIUM": WARNING, "LOW": SUCCESS, "UNKNOWN": TEXT_MUTED,
            }
            risk_color = risk_colors.get(reg["risk_flag"], TEXT_MUTED)
            st.markdown(
                f"""
                <span style="display: inline-block; padding: 0.3rem 0.7rem;
                             background: {risk_color}; color: {WHITE};
                             font-weight: 600; font-size: 0.78rem;
                             letter-spacing: 0.03em; border-radius: 999px;">
                  RISK: {reg['risk_flag']}
                </span>
                """,
                unsafe_allow_html=True,
            )
            st.write("")
            st.write(reg["reason"])
            if reg.get("sources"):
                st.caption("Sources: " + "; ".join(reg["sources"]))
            if reg.get("governance", {}).get("human_review_required"):
                st.warning(
                    "This regulatory result needs human review before relying on it "
                    "(risk could not be clearly determined from the official documents)."
                )
            # The LLM is prompted to end every answer with the disclaimer already;
            # only show it again here if that didn't happen.
            if reg["disclaimer"] not in reg["reason"]:
                st.caption(reg["disclaimer"])

    st.write("")
    st.write("")

    # ── Actions ──────────────────────────────────────────────────────────────
    action_l, action_m, action_r = st.columns([1, 1, 1])
    with action_l:
        if st.button("Save this analysis", type="primary", use_container_width=True):
            rec = save_record(prop, scen, nickname=prop.nickname or prop.district)
            st.success(f"Saved as **{rec['nickname']}**.")
    with action_m:
        if st.button("Ask follow-up questions", use_container_width=True):
            st.switch_page("pages/2_Chat.py")
    with action_r:
        if st.button("See improvement ideas", use_container_width=True):
            st.switch_page("pages/4_Optimisation.py")

    st.write("")
else:
    st.info(
        "Fill in the property details above and click **Run analysis**.",
        icon="◼",
    )

footer_disclaimer(DISCLAIMER)
