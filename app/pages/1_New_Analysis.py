"""New analysis — property input → Airbnb-vs-sell decision."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from airbnb_iip.agents.comparables import get_comparables_agent
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
from components.engine import (
    CITIES,
    CITY_LABELS,
    ROOM_TYPES,
    Property,
    compute_scenario,
    get_city_districts,
)
from components.storage import save_record
from components.styling import (
    apply_page_style,
    card,
    footer_disclaimer,
    hero,
    recommendation_pill,
)

apply_page_style("New analysis")

hero(
    eyebrow="Step 01",
    title="New analysis",
    lede="Enter the property details. We'll compare projected Airbnb net "
         "income against indicative sale value and give a clear recommendation.",
)

# ── Form ─────────────────────────────────────────────────────────────────────
with st.form("analysis_form", border=False):
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
        st.markdown("**Amenities**")
        has_ac        = st.checkbox("Air conditioning", value=True)
        has_balcony   = st.checkbox("Balcony or terrace", value=False)
        has_elevator  = st.checkbox("Elevator", value=True)
        has_pool      = st.checkbox("Pool", value=False)
        has_parking   = st.checkbox("Parking", value=False)
        has_workspace = st.checkbox("Dedicated workspace", value=False)

    st.write("")
    submitted = st.form_submit_button(
        "Run analysis", use_container_width=False, type="primary",
    )

# ── Compute on submit (or recompute on rerun if values are in session) ───────
if submitted:
    prop = Property(
        city=city,
        district=district,
        size_m2=size_m2,
        bedrooms=int(bedrooms),
        bathrooms=float(bathrooms),
        accommodates=int(accommodates),
        room_type=room_type,
        has_ac=has_ac,
        has_balcony=has_balcony,
        has_elevator=has_elevator,
        has_pool=has_pool,
        has_parking=has_parking,
        has_workspace=has_workspace,
        nickname=nickname or None,
    )
    st.session_state["property"] = prop
    st.session_state["scenario"] = compute_scenario(prop)

prop = st.session_state.get("property")
scen = st.session_state.get("scenario")

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

    # KPI row
    k1, k2, k3, k4 = st.columns(4, gap="medium")
    with k1:
        card(
            "Net annual revenue",
            f"€{scen.net_revenue_year_eur:,.0f}",
            f"P10–P90: €{scen.net_revenue_p10_eur:,.0f} – €{scen.net_revenue_p90_eur:,.0f}",
            featured=True,
        )
    with k2:
        card(
            "Indicative sale value",
            f"€{scen.sale_price_eur:,.0f}",
            f"€{scen.sale_price_per_m2_eur:,.0f}/m²",
        )
    with k3:
        card(
            "Payback period",
            f"{scen.breakeven_years:.1f} yrs"
            if scen.breakeven_years < 50 else "50+ yrs",
            "net income to recover sale value",
        )
    with k4:
        card(
            "Projected occupancy",
            f"{scen.occupancy_rate_annual*100:.0f}%",
            f"{scen.nights_booked_year} nights · €{scen.predicted_nightly_eur:,.0f}/night",
        )

    st.write("")
    st.write("")

    # ── Charts row ───────────────────────────────────────────────────────────
    chart_left, chart_right = st.columns([1.1, 1], gap="large")

    with chart_left:
        st.markdown("##### Net revenue uncertainty")
        st.caption(
            "Indicative P10 / median / P90 net revenue. Uncertainty is "
            "dominated by occupancy variability."
        )
        fig = go.Figure(
            go.Bar(
                x=["P10", "Median", "P90"],
                y=[scen.net_revenue_p10_eur,
                   scen.net_revenue_year_eur,
                   scen.net_revenue_p90_eur],
                marker_color=[KPMG_BLUE_TINT, KPMG_BLUE, KPMG_BLUE_DARK],
                text=[f"€{v:,.0f}" for v in (
                    scen.net_revenue_p10_eur,
                    scen.net_revenue_year_eur,
                    scen.net_revenue_p90_eur,
                )],
                textposition="outside",
            )
        )
        fig.update_layout(
            height=320,
            margin=dict(l=10, r=10, t=20, b=20),
            plot_bgcolor=WHITE, paper_bgcolor=WHITE,
            yaxis=dict(showgrid=True, gridcolor=BORDER_SUBTLE, zeroline=False),
            xaxis=dict(showgrid=False),
            showlegend=False,
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
        st.caption("Costs deducted from gross to arrive at net revenue.")
        # Combine costs + net, sort descending so the largest bar leads.
        cost_items = list(scen.cost_breakdown) + [
            ("Net to owner", scen.net_revenue_year_eur),
        ]
        cost_items.sort(key=lambda x: x[1], reverse=True)
        cost_labels = [name for name, _ in cost_items]
        cost_values = [v for _, v in cost_items]
        cost_colors = [
            KPMG_BLUE_DARK if name == "Net to owner" else
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
