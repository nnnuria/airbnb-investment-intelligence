"""Saved properties — list, compare, delete."""

from __future__ import annotations

from datetime import datetime

import plotly.graph_objects as go
import streamlit as st

from components.branding import (
    BORDER_SUBTLE,
    DISCLAIMER,
    KPMG_BLUE,
    KPMG_BLUE_DARK,
    KPMG_BLUE_TINT,
    TEXT_MUTED,
    TEXT_SECONDARY,
    WHITE,
)
from components.storage import clear_all, delete_record, load_all
from components.styling import apply_page_style, footer_disclaimer, hero

apply_page_style("Saved properties")

hero(
    eyebrow="Library",
    title="Saved properties",
    lede="Every analysis you save lives here. Compare scenarios side by "
         "side and revisit recommendations later.",
)

records = load_all()

# ── Empty state ──────────────────────────────────────────────────────────────
if not records:
    st.markdown(
        f"""
        <div style="padding: 3rem 2rem; text-align: center;
                    border: 1px dashed {BORDER_SUBTLE}; border-radius: 14px;
                    background: {WHITE}; color: {TEXT_MUTED};">
          <div style="font-weight: 700; color: {KPMG_BLUE}; font-size: 1rem;
                      margin-bottom: 0.5rem;">No saved analyses yet</div>
          <div style="font-size: 0.95rem; color: {TEXT_SECONDARY};">
            Run a new analysis and click <b>Save this analysis</b> to keep it here.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Start a new analysis", type="primary"):
        st.switch_page("pages/1_New_Analysis.py")
    footer_disclaimer(DISCLAIMER)
    st.stop()

# ── Summary ──────────────────────────────────────────────────────────────────
s1, s2, s3 = st.columns(3, gap="medium")
totals = {
    "airbnb": sum(1 for r in records if r["scenario"]["recommendation"] == "airbnb"),
    "sell":   sum(1 for r in records if r["scenario"]["recommendation"] == "sell"),
    "marg":   sum(1 for r in records if r["scenario"]["recommendation"] == "marginal"),
}
with s1:
    st.markdown(_summary_block := f"""
    <div style="padding: 1.25rem 1.5rem; border: 1px solid {BORDER_SUBTLE};
                background: {WHITE}; border-radius: 14px;">
      <div style="color: {TEXT_MUTED}; font-weight: 600; font-size: 0.72rem;
                  letter-spacing: 0.08em; text-transform: uppercase;
                  margin-bottom: 0.4rem;">Total saved</div>
      <div style="font-weight: 700; font-size: 1.75rem;">{len(records)}</div>
    </div>""", unsafe_allow_html=True)
with s2:
    st.markdown(f"""
    <div style="padding: 1.25rem 1.5rem; border: 1px solid {KPMG_BLUE};
                background: {KPMG_BLUE_TINT}; border-radius: 14px;">
      <div style="color: {KPMG_BLUE_DARK}; font-weight: 600; font-size: 0.72rem;
                  letter-spacing: 0.08em; text-transform: uppercase;
                  margin-bottom: 0.4rem;">Airbnb recommended</div>
      <div style="font-weight: 700; font-size: 1.75rem; color: {KPMG_BLUE_DARK};">
        {totals["airbnb"]}</div>
    </div>""", unsafe_allow_html=True)
with s3:
    st.markdown(f"""
    <div style="padding: 1.25rem 1.5rem; border: 1px solid {BORDER_SUBTLE};
                background: {WHITE}; border-radius: 14px;">
      <div style="color: {TEXT_MUTED}; font-weight: 600; font-size: 0.72rem;
                  letter-spacing: 0.08em; text-transform: uppercase;
                  margin-bottom: 0.4rem;">Sell recommended</div>
      <div style="font-weight: 700; font-size: 1.75rem;">{totals["sell"]}</div>
    </div>""", unsafe_allow_html=True)

st.write("")
st.write("")

tab_list, tab_compare = st.tabs(["List", "Compare"])

# ── List view ────────────────────────────────────────────────────────────────
with tab_list:
    for rec in sorted(records, key=lambda r: r["saved_at"], reverse=True):
        prop, scen = rec["property"], rec["scenario"]
        rec_color = {
            "airbnb": KPMG_BLUE, "sell": "#A47102", "marginal": TEXT_MUTED,
        }.get(scen["recommendation"], TEXT_MUTED)
        saved_dt = datetime.fromisoformat(rec["saved_at"]).strftime("%b %d, %Y")

        with st.container(border=False):
            st.markdown(
                f"""
                <div style="padding: 1.25rem 1.5rem; border: 1px solid {BORDER_SUBTLE};
                            background: {WHITE}; border-radius: 14px; margin-bottom: 0.85rem;">
                  <div style="display: flex; justify-content: space-between; gap: 1rem;
                              align-items: flex-start;">
                    <div>
                      <div style="display: inline-block; padding: 0.2rem 0.6rem;
                                  background: {rec_color}; color: {WHITE};
                                  font-size: 0.7rem; font-weight: 600;
                                  letter-spacing: 0.04em; border-radius: 999px;
                                  margin-bottom: 0.45rem;">
                        {scen["recommendation"].upper()}
                      </div>
                      <div style="font-size: 1.1rem; font-weight: 700;
                                  margin-bottom: 0.15rem;">
                        {rec.get("nickname") or prop["district"]}
                      </div>
                      <div style="color: {TEXT_MUTED}; font-size: 0.85rem;">
                        {prop["district"]}, {prop["city"].title()}
                        · {prop["size_m2"]:.0f} m² · {prop["bedrooms"]} bed
                        · saved {saved_dt}
                      </div>
                    </div>
                    <div style="text-align: right; min-width: 240px;">
                      <div style="font-size: 0.72rem; color: {TEXT_MUTED};
                                  font-weight: 600; letter-spacing: 0.08em;
                                  text-transform: uppercase; margin-bottom: 0.2rem;">
                        Net revenue / Sale
                      </div>
                      <div style="font-size: 1.1rem; font-weight: 700;">
                        €{scen["net_revenue_year_eur"]:,.0f}
                        <span style="color: {TEXT_MUTED}; font-weight: 500;">/yr</span>
                      </div>
                      <div style="color: {TEXT_SECONDARY}; font-size: 0.88rem;">
                        Sale €{scen["sale_price_eur"]:,.0f}
                        · break-even {scen["breakeven_years"]:.1f} yrs
                      </div>
                    </div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            cols = st.columns([1, 1, 6])
            with cols[0]:
                if st.button("Reopen", key=f"reopen_{rec['id']}", type="secondary",
                              use_container_width=True):
                    from airbnb_iip.decision.engine import Property, Scenario
                    st.session_state["property"] = Property(**rec["property"])
                    st.session_state["scenario"] = Scenario(**rec["scenario"])
                    st.switch_page("pages/1_New_Analysis.py")
            with cols[1]:
                if st.button("Delete", key=f"del_{rec['id']}", type="secondary",
                              use_container_width=True):
                    delete_record(rec["id"])
                    st.rerun()

    st.write("")
    if st.button("Clear all saved analyses", type="secondary"):
        clear_all()
        st.rerun()

# ── Compare view ─────────────────────────────────────────────────────────────
with tab_compare:
    if len(records) < 2:
        st.info("Save at least two analyses to enable side-by-side comparison.",
                icon="◼")
    else:
        labels = {
            r["id"]: f"{r.get('nickname') or r['property']['district']} "
                     f"· {r['property']['city'].title()}"
            for r in records
        }
        chosen = st.multiselect(
            "Pick analyses to compare",
            options=list(labels.keys()),
            format_func=lambda x: labels[x],
            default=list(labels.keys())[:3],
        )
        if chosen:
            chosen_records = [r for r in records if r["id"] in chosen]
            fig = go.Figure()
            names = [labels[r["id"]] for r in chosen_records]
            fig.add_trace(go.Bar(
                name="Net annual revenue", x=names,
                y=[r["scenario"]["net_revenue_year_eur"] for r in chosen_records],
                marker_color=KPMG_BLUE,
            ))
            fig.add_trace(go.Bar(
                name="Sale value (÷ 15)", x=names,
                y=[r["scenario"]["sale_price_eur"] / 15 for r in chosen_records],
                marker_color="#5DADE2",
            ))
            fig.update_layout(
                barmode="group", height=420,
                margin=dict(l=10, r=10, t=10, b=20),
                plot_bgcolor=WHITE, paper_bgcolor=WHITE,
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor=BORDER_SUBTLE,
                           zeroline=False, title="EUR"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                            xanchor="right", x=1),
            )
            st.plotly_chart(fig, use_container_width=True,
                            config={"displayModeBar": False})
            st.caption(
                "Sale value is scaled by 1/15 so both bars share a common "
                "y-axis. Break-even years are shown below."
            )

            # Summary table
            import pandas as pd
            rows = []
            for r in chosen_records:
                p, s = r["property"], r["scenario"]
                rows.append({
                    "Property": labels[r["id"]],
                    "Recommendation": s["recommendation"].title(),
                    "Net / yr": f"€{s['net_revenue_year_eur']:,.0f}",
                    "Sale value": f"€{s['sale_price_eur']:,.0f}",
                    "Break-even (yrs)": f"{s['breakeven_years']:.1f}",
                    "Occupancy": f"{s['occupancy_rate_annual']*100:.0f}%",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True,
                         hide_index=True)

footer_disclaimer(DISCLAIMER)
