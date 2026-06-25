"""Market Analysis report — the Market Analyst agent's narrated brief.

Runs on the *active analysis* the user already produced on the New Analysis page
(``st.session_state["property"]`` / ``["scenario"]``) — it never recomputes the
scenario, so the report's numbers match what the dashboard showed. It layers peer
positioning (comparables median / P25-P75) on top and asks the agent for an
LLM-narrated market brief.

Source of truth: POST /market_report → airbnb_iip.agents.market_analyst.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.api_client import APIError, market_report as api_market_report
from components.branding import (
    DANGER,
    DISCLAIMER,
    SUCCESS,
    TEXT_MUTED,
)
from components.styling import (
    apply_page_style,
    card,
    footer_disclaimer,
    hero,
    recommendation_pill,
)

apply_page_style("Market analysis")

hero(
    eyebrow="Market intelligence",
    title="Market analysis",
    lede="Where this property sits against comparable listings nearby, and the "
         "Airbnb-vs-sell call — narrated from the live numbers in your last analysis.",
)


# ── Guard: need an active analysis ────────────────────────────────────────────
prop = st.session_state.get("property")
scen = st.session_state.get("scenario")

if not prop or not scen:
    st.info(
        "No active analysis yet. Open **New analysis**, run a property, and the "
        "market report will build on those exact numbers."
    )
    st.stop()


def _fingerprint(scenario) -> str:
    """Stable hash of the active scenario so the report (and its LLM call) is
    only regenerated when the underlying analysis actually changes."""
    payload = asdict(scenario) if is_dataclass(scenario) else scenario
    return hashlib.md5(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()


fp = _fingerprint(scen)
header_col, refresh_col = st.columns([4, 1], gap="small")
with refresh_col:
    if st.button("Regenerate", type="secondary", use_container_width=True):
        st.session_state.pop("market_report_fp", None)

if st.session_state.get("market_report_fp") != fp:
    with st.spinner("Building the market analysis…"):
        try:
            st.session_state["market_report"] = api_market_report(prop, scen)
            st.session_state["market_report_fp"] = fp
            st.session_state.pop("market_report_error", None)
        except APIError as exc:
            st.session_state["market_report_error"] = str(exc)

if st.session_state.get("market_report_error"):
    st.error(st.session_state["market_report_error"])
    st.stop()

report = st.session_state.get("market_report")
if not report:
    st.stop()


# ── Recommendation + headline KPIs ────────────────────────────────────────────
rec = str(report.get("recommendation", "")).upper()
p_airbnb = (report.get("confidence") or {}).get("p_airbnb_gt_sell")
recommendation_pill(f"Recommendation: {rec}")
if p_airbnb is not None:
    st.caption(f"Confidence P(Airbnb beats Sell) = {p_airbnb}")

subj = report.get("subject", {})
occ = subj.get("occupancy_rate_annual")
k1, k2, k3, k4 = st.columns(4, gap="medium")
with k1:
    card("Predicted nightly", f"€{subj.get('nightly_eur', 0):,.0f}", "From the LightGBM price model", featured=True)
with k2:
    card(
        "Occupancy",
        f"{occ * 100:.0f}%" if isinstance(occ, (int, float)) else "—",
        f"{subj.get('nights_booked_year', '—')} nights/year",
    )
with k3:
    card(
        "Net revenue / yr",
        f"€{subj.get('annual_net_eur', 0):,.0f}",
        f"P10–P90: €{subj.get('p10_eur', 0):,.0f}–€{subj.get('p90_eur', 0):,.0f}",
    )
with k4:
    card(
        "NPV Airbnb vs sell",
        f"€{subj.get('npv_airbnb_p50_eur', 0):,.0f}",
        f"Sell proceeds €{subj.get('npv_sell_eur', 0):,.0f} · break-even {subj.get('break_even_years', '—')} yr",
    )

st.write("")


# ── The narrative brief ───────────────────────────────────────────────────────
st.markdown("##### Analyst brief")
st.markdown(report.get("narrative", "_No narrative available._"))

st.write("")


# ── Market positioning ────────────────────────────────────────────────────────
st.markdown("##### Market positioning")
pos = report.get("positioning", {})
peer_n = pos.get("peer_n", 0)
nightly, revenue = pos.get("nightly", {}), pos.get("revenue", {})

if not peer_n:
    st.info(
        "No comparable peer set was available for this district and room type, so "
        "positioning rests on the model estimate alone."
    )
else:
    st.caption(f"Benchmarked against {peer_n} comparable listings nearby.")

    def _pos_delta(block: dict) -> str:
        median = block.get("median")
        pos_label = block.get("position", "—")
        median_txt = f"peer median €{median:,.0f}" if isinstance(median, (int, float)) else "no peer median"
        return f"{median_txt} · {pos_label}"

    pc1, pc2, pc3 = st.columns(3, gap="medium")
    with pc1:
        card("Nightly vs market", f"€{nightly.get('subject', 0):,.0f}", _pos_delta(nightly))
    with pc2:
        card("Annual revenue vs market", f"€{revenue.get('subject', 0):,.0f}", _pos_delta(revenue))
    with pc3:
        rating = pos.get("review_rating_median")
        card("Peer guest rating", f"{rating:.2f}" if isinstance(rating, (int, float)) else "—", "Median review score of peers")

    # Diverging bar: % vs peer median (scale-free, so nightly and revenue compare
    # on one axis despite very different absolute magnitudes).
    labels, values = [], []
    for name, block in (("Nightly rate", nightly), ("Annual revenue", revenue)):
        v = block.get("pct_vs_median")
        if isinstance(v, (int, float)):
            labels.append(name)
            values.append(v)
    if values:
        colors = [SUCCESS if v >= 0 else DANGER for v in values]
        fig = go.Figure(
            go.Bar(
                x=values,
                y=labels,
                orientation="h",
                marker_color=colors,
                text=[f"{v:+.0f}%" for v in values],
                textposition="outside",
                hovertemplate="%{y}: %{x:+.1f}% vs peer median<extra></extra>",
            )
        )
        fig.add_vline(x=0, line_width=1, line_color=TEXT_MUTED)
        fig.update_layout(
            title="Subject vs peer median (%)",
            height=240,
            margin=dict(l=10, r=30, t=40, b=10),
            xaxis_title="% above / below peer median",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

st.write("")


# ── Price drivers ─────────────────────────────────────────────────────────────
drivers = report.get("drivers", [])
if drivers:
    st.markdown("##### What drives the nightly rate")
    st.markdown(
        "\n".join(
            f"- **{d['feature']}** — {d['direction']} the nightly rate"
            for d in drivers[:5]
        )
    )
    st.write("")


# ── Comparable listings ───────────────────────────────────────────────────────
comps = report.get("comparables", [])
if comps:
    st.markdown("##### Comparable listings")
    df = pd.DataFrame(comps)
    cols = [
        c for c in (
            "name", "neighbourhood_cleansed", "room_type", "accommodates",
            "bedrooms", "price", "estimated_revenue_l365d", "review_scores_rating",
        ) if c in df.columns
    ]
    df = df[cols].rename(columns={
        "name": "Listing",
        "neighbourhood_cleansed": "Neighbourhood",
        "room_type": "Room type",
        "accommodates": "Sleeps",
        "bedrooms": "Beds",
        "price": "Nightly €",
        "estimated_revenue_l365d": "Annual revenue €",
        "review_scores_rating": "Rating",
    })
    st.dataframe(df, use_container_width=True, hide_index=True)

footer_disclaimer(DISCLAIMER)
