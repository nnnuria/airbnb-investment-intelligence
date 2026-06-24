"""Sentiment dashboard — per-listing guest-review sentiment.

Trimmed to only what is comparable, actionable, and accurate:
  * per-listing sentiment distribution (the optimisation/comparables signal)
  * poorly-reviewed share + negative-review rate (comparable across cities)
  * honest validation (sentiment vs star rating — a weak signal)

Reads the real outputs:
  * Data/processed/sentiment_summary.csv          — per-city aggregates
  * Data/processed/<city>_sentiment.parquet       — per-listing sentiment

Pipeline (src/airbnb_iip/models/nlp.py): multilingual lexicon weak-labels
(VADER for EN + ES/FR/DE/IT/PT word-lists) → TF-IDF + Multinomial Naïve Bayes.
Deliberately omitted: cross-city positive-rate comparison (confounded by the
language/scoring mix) and NB accuracy (circular — agreement with weak labels).
"""

from __future__ import annotations

import ast
import html
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.branding import (
    BORDER_SUBTLE,
    DANGER,
    DISCLAIMER,
    KPMG_BLUE,
    KPMG_BLUE_DARK,
    SUCCESS,
    TEXT_MUTED,
    TEXT_SECONDARY,
    WHITE,
)
from components.styling import apply_page_style, card, footer_disclaimer, hero

apply_page_style("Sentiment")

hero(
    eyebrow="Optimisation signal",
    title="Guest sentiment",
    lede="Per-listing review sentiment as a listing-quality signal for the optimisation "
         "& comparables flow. The left tail — listings with poor reviews — is the actionable part.",
)

PROC = Path(__file__).resolve().parents[2] / "Data" / "processed"
SUMMARY = PROC / "sentiment_summary.csv"
DISPLAY = {"madrid": "Madrid", "barcelona": "Barcelona", "malaga": "Málaga"}
LANG = {"en": "English", "es": "Spanish", "fr": "French", "de": "German",
        "it": "Italian", "pt": "Portuguese", "nl": "Dutch", "unknown": "Unknown"}
LANG_PALETTE = [KPMG_BLUE, "#5DADE2", "#1ABC9C", "#F39C12", "#9B59B6", "#34495E", "#95A5A6"]


@st.cache_data
def load():
    s = pd.read_csv(SUMMARY)
    s["lang_split"] = s["lang_split"].map(ast.literal_eval)
    s["label_split"] = s["label_split"].map(ast.literal_eval)
    pq = {c: pd.read_parquet(PROC / f"{c}_sentiment.parquet")
          for c in s["city"] if (PROC / f"{c}_sentiment.parquet").exists()}
    ep = PROC / "sentiment_examples.json"
    ex = json.loads(ep.read_text(encoding="utf-8")) if ep.exists() else {}
    return s, pq, ex


if not SUMMARY.exists():
    st.error(f"`{SUMMARY}` not found. Run `scripts/run_sentiment.py` to regenerate it.")
    st.stop()

summary, parquets, examples = load()
keys = list(summary["city"])


def review_card(r, color):
    st.markdown(
        f"""<div style="border:1px solid {BORDER_SUBTLE};border-left:4px solid {color};
        background:{WHITE};border-radius:10px;padding:0.7rem 0.9rem;margin-bottom:0.6rem;">
        <div style="font-size:0.68rem;color:{color};font-weight:700;margin-bottom:0.25rem;">
        score {r['score']:+.2f}</div>
        <div style="color:{TEXT_SECONDARY};font-size:0.86rem;line-height:1.5;">
        &ldquo;{html.escape(r['text'])}&rdquo;</div></div>""",
        unsafe_allow_html=True,
    )


def hist(pq, label):
    fig = go.Figure(go.Histogram(
        x=pq["mean_sentiment"], nbinsx=40, marker_color=KPMG_BLUE,
        marker_line_color=WHITE, marker_line_width=0.5,
    ))
    fig.add_vline(x=float(pq["mean_sentiment"].mean()), line_dash="dot", line_color=KPMG_BLUE_DARK,
                  annotation_text=f"mean {pq['mean_sentiment'].mean():.2f}", annotation_position="top")
    fig.update_layout(
        height=320, margin=dict(l=10, r=10, t=20, b=20), plot_bgcolor=WHITE, paper_bgcolor=WHITE,
        yaxis=dict(showgrid=True, gridcolor=BORDER_SUBTLE, zeroline=False, title="listings"),
        xaxis=dict(showgrid=False, title=f"mean sentiment per listing — {label}", range=[-1, 1]),
        showlegend=False,
    )
    return fig


choice = st.radio("View", ["All cities"] + [DISPLAY[k] for k in keys],
                  horizontal=True, label_visibility="collapsed")
st.write("")

# ── PER-CITY ──────────────────────────────────────────────────────────────────
if choice != "All cities":
    key = next(k for k in keys if DISPLAY[k] == choice)
    row = summary[summary["city"] == key].iloc[0]
    lm = row["lang_split"]
    pq = parquets[key]
    neg_tail = float((pq["mean_sentiment"] < 0).mean() * 100)

    st.markdown(f"### {DISPLAY[key]}")
    k1, k2, k3 = st.columns(3, gap="medium")
    with k1:
        card("Listings scored", f"{len(pq):,}", "in the 40k-review sample", featured=True)
    with k2:
        card("Poorly-reviewed", f"{neg_tail:.1f}%", "listings with mean sentiment < 0")
    with k3:
        card("Corr. vs rating", f"{row['corr_sentiment_vs_rating']:.2f}", "validation (weak signal)")

    st.write("")
    st.markdown("##### Per-listing sentiment distribution")
    st.caption(
        f"Mean sentiment per listing (−1…+1) across {len(pq):,} {DISPLAY[key]} listings. "
        "The **left tail is the actionable part** — listings whose reviews flag real problems."
    )
    st.plotly_chart(hist(pq, DISPLAY[key]), use_container_width=True, config={"displayModeBar": False})

    st.write("")
    st.markdown("##### Review languages")
    st.caption("Reviews are genuinely multilingual — which is why an English-only model wasn't viable.")
    items = sorted(lm.items(), key=lambda x: -x[1])
    fig = go.Figure(go.Bar(
        x=[v * 100 for _, v in items], y=[LANG.get(k, k) for k, _ in items],
        orientation="h", marker_color=KPMG_BLUE,
        text=[f"{v * 100:.0f}%" for _, v in items], textposition="outside",
    ))
    fig.update_layout(
        height=240, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor=WHITE, paper_bgcolor=WHITE,
        xaxis=dict(showgrid=True, gridcolor=BORDER_SUBTLE, title="% of reviews", range=[0, 70]),
        yaxis=dict(autorange="reversed"), showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    ex = examples.get(key, {})
    if ex.get("positive") or ex.get("negative"):
        st.write("")
        st.markdown("##### Example reviews")
        st.caption("Illustrative English-language reviews scored by the model — not a statistical sample.")
        pc, nc = st.columns(2, gap="large")
        with pc:
            st.markdown(
                f"<div style='color:{SUCCESS};font-weight:700;font-size:0.75rem;"
                f"text-transform:uppercase;letter-spacing:.05em;margin-bottom:0.4rem;'>Positive</div>",
                unsafe_allow_html=True)
            for r in ex.get("positive", []):
                review_card(r, SUCCESS)
        with nc:
            st.markdown(
                f"<div style='color:{DANGER};font-weight:700;font-size:0.75rem;"
                f"text-transform:uppercase;letter-spacing:.05em;margin-bottom:0.4rem;'>Negative</div>",
                unsafe_allow_html=True)
            for r in ex.get("negative", []):
                review_card(r, DANGER)

# ── ALL CITIES ────────────────────────────────────────────────────────────────
else:
    rows = {r["city"]: r for _, r in summary.iterrows()}
    total_listings = sum(len(parquets[k]) for k in keys if k in parquets)
    corr_lo = min(rows[k]["corr_sentiment_vs_rating"] for k in keys)
    corr_hi = max(rows[k]["corr_sentiment_vs_rating"] for k in keys)
    neg_lo = min(rows[k]["label_split"].get("negative", 0) for k in keys) * 100
    neg_hi = max(rows[k]["label_split"].get("negative", 0) for k in keys) * 100

    k1, k2, k3, k4 = st.columns(4, gap="medium")
    with k1:
        card("Reviews scored", f"{int(summary['reviews_scored'].sum()):,}", "40k per city", featured=True)
    with k2:
        card("Listings scored", f"{total_listings:,}", "with sentiment in sample")
    with k3:
        card("Negative reviews", f"{neg_lo:.1f}–{neg_hi:.1f}%", "≈ same across cities")
    with k4:
        card("Corr. vs rating", f"{corr_lo:.2f}–{corr_hi:.2f}", "validation (weak signal)")

    st.write("")
    st.markdown("##### Per-listing sentiment distribution by city")
    st.caption("The comparable view — where each city's listings sit on the sentiment scale.")
    cols = st.columns(len(keys), gap="medium")
    for col, k in zip(cols, keys):
        with col:
            if k in parquets:
                st.plotly_chart(hist(parquets[k], DISPLAY[k]),
                                use_container_width=True, config={"displayModeBar": False})

# ── How it's used + methodology (both views) ─────────────────────────────────
st.write("")
st.markdown(
    f"""
    <div style="padding:1rem 1.25rem;border:1px solid {BORDER_SUBTLE};border-left:4px solid {KPMG_BLUE};
                background:{WHITE};border-radius:12px;color:{TEXT_SECONDARY};font-size:0.95rem;">
      <b>How this is used:</b> a <b>per-listing quality signal</b>, not part of the Airbnb-vs-sell
      decision. It feeds the <b>comparables &amp; optimisation flow</b> — listings in the negative
      tail are flagged for guest-experience improvements.
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")
with st.expander("Methodology & honest caveats"):
    st.markdown(
        """
- **Model:** TF-IDF (1–2 grams) + **Multinomial Naïve Bayes** (`src/airbnb_iip/models/nlp.py`),
  trained on **multilingual lexicon weak-labels** (VADER for EN + ES/FR/DE/IT/PT word-lists).
- **Per listing:** `mean_sentiment` (−1…+1), `pct_positive`, `pct_negative`, `n_reviews` —
  aggregated over the **~40k-review sample per city** (listing counts = sample coverage).
- **Validation is weak by design:** the honest metric is correlation with star ratings
  (**0.22–0.29**), so sentiment is treated as a *soft* signal, one input among many.
- **Deliberately not shown:** (1) cross-city *positive*-rate comparison — confounded, because
  English reviews are scored by VADER while other languages fall back to neutral; (2) NB accuracy
  (~0.95) — it only measures agreement with the weak labels, not ground truth.
        """
    )

footer_disclaimer(DISCLAIMER)
