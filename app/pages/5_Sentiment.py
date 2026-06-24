"""Sentiment dashboard — guest-review sentiment, per city, on real data.

Reads the actual sentiment outputs:
  * Data/processed/sentiment_summary.csv          — per-city aggregates
  * Data/processed/<city>_sentiment.parquet       — per-listing sentiment

Pipeline (src/airbnb_iip/models/nlp.py): multilingual lexicon weak-labels
(VADER for EN + ES/FR/DE/IT/PT word-lists) → TF-IDF + Multinomial Naïve Bayes,
aggregated per listing, validated against star ratings.
"""

from __future__ import annotations

import ast
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
    KPMG_BLUE_TINT,
    SUCCESS,
    TEXT_MUTED,
    TEXT_SECONDARY,
    WHITE,
)
from components.styling import apply_page_style, card, footer_disclaimer, hero

apply_page_style("Sentiment")

hero(
    eyebrow="Insights",
    title="Guest sentiment",
    lede="Review sentiment scored per listing (TF-IDF + Naïve Bayes on multilingual "
         "reviews). Pick a city for its breakdown and per-listing distribution, or "
         "compare all three.",
)

PROC = Path(__file__).resolve().parents[2] / "Data" / "processed"
SUMMARY = PROC / "sentiment_summary.csv"

DISPLAY = {"madrid": "Madrid", "barcelona": "Barcelona", "malaga": "Málaga"}
LANG = {"en": "English", "es": "Spanish", "fr": "French", "de": "German",
        "it": "Italian", "pt": "Portuguese", "nl": "Dutch", "unknown": "Unknown"}
SENT_COLORS = {"Positive": SUCCESS, "Neutral": TEXT_MUTED, "Negative": DANGER}
LANG_PALETTE = [KPMG_BLUE, "#5DADE2", "#1ABC9C", "#F39C12", "#9B59B6", "#34495E", "#95A5A6"]


@st.cache_data
def load():
    s = pd.read_csv(SUMMARY)
    s["lang_split"] = s["lang_split"].map(ast.literal_eval)
    s["label_split"] = s["label_split"].map(ast.literal_eval)
    pq = {}
    for c in s["city"]:
        p = PROC / f"{c}_sentiment.parquet"
        if p.exists():
            pq[c] = pd.read_parquet(p)
    return s, pq


if not SUMMARY.exists():
    st.error(f"`{SUMMARY}` not found. Run `scripts/run_sentiment.py` to regenerate it.")
    st.stop()

summary, parquets = load()
keys = list(summary["city"])


def donut(labels, values, colors, center):
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.62, marker=dict(colors=colors),
        sort=False, direction="clockwise", textinfo="label+percent", textposition="outside",
    ))
    fig.update_layout(
        height=320, margin=dict(l=10, r=10, t=20, b=10), paper_bgcolor=WHITE, showlegend=False,
        annotations=[dict(text=center, x=0.5, y=0.5, font_size=14,
                          font_color=KPMG_BLUE_DARK, showarrow=False)],
    )
    return fig


choice = st.radio(
    "View", ["All cities"] + [DISPLAY[k] for k in keys],
    horizontal=True, label_visibility="collapsed",
)
st.write("")

# ══════════════════════════════════════════════════════════════════════════════
# PER-CITY
# ══════════════════════════════════════════════════════════════════════════════
if choice != "All cities":
    key = next(k for k in keys if DISPLAY[k] == choice)
    row = summary[summary["city"] == key].iloc[0]
    ls = row["label_split"]
    lm = row["lang_split"]
    pq = parquets.get(key)

    st.markdown(f"### {DISPLAY[key]}")
    k1, k2, k3, k4 = st.columns(4, gap="medium")
    with k1:
        card("Positive", f"{ls.get('positive', 0) * 100:.1f}%",
             f"{int(row['reviews_scored']):,} reviews scored", featured=True)
    with k2:
        card("Negative", f"{ls.get('negative', 0) * 100:.1f}%", "≈ same across cities")
    with k3:
        card("Listings in sample", f"{len(pq):,}" if pq is not None else "—",
             "covered by the 40k-review sample")
    with k4:
        card("Corr. vs rating", f"{row['corr_sentiment_vs_rating']:.2f}",
             f"NB acc {row['nb_accuracy']:.2f}")

    st.write("")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("##### Sentiment split")
        st.caption("Share of scored reviews by label.")
        st.plotly_chart(
            donut(["Positive", "Neutral", "Negative"],
                  [ls.get("positive", 0), ls.get("neutral", 0), ls.get("negative", 0)],
                  [SENT_COLORS["Positive"], SENT_COLORS["Neutral"], SENT_COLORS["Negative"]],
                  f"{ls.get('positive', 0) * 100:.0f}%<br>positive"),
            use_container_width=True, config={"displayModeBar": False})
    with c2:
        st.markdown("##### Review languages")
        st.caption("Detected language of reviews (top 6).")
        items = sorted(lm.items(), key=lambda x: -x[1])
        st.plotly_chart(
            donut([LANG.get(k, k) for k, _ in items], [v for _, v in items],
                  LANG_PALETTE[:len(items)], "language<br>mix"),
            use_container_width=True, config={"displayModeBar": False})

    if pq is not None:
        st.write("")
        st.markdown("##### Per-listing sentiment distribution")
        st.caption(
            f"Mean sentiment per listing (in [−1, +1]) across the {len(pq):,} {DISPLAY[key]} "
            "listings covered by the 40k-review sample — averaged over each listing's sampled "
            "reviews. Most cluster positive; the left tail flags listings with poor reviews."
        )
        fig = go.Figure(go.Histogram(
            x=pq["mean_sentiment"], nbinsx=40, marker_color=KPMG_BLUE,
            marker_line_color=WHITE, marker_line_width=0.5,
        ))
        fig.add_vline(x=float(pq["mean_sentiment"].mean()), line_dash="dot",
                      line_color=KPMG_BLUE_DARK,
                      annotation_text=f"mean {pq['mean_sentiment'].mean():.2f}",
                      annotation_position="top")
        fig.update_layout(
            height=320, margin=dict(l=10, r=10, t=20, b=20),
            plot_bgcolor=WHITE, paper_bgcolor=WHITE,
            yaxis=dict(showgrid=True, gridcolor=BORDER_SUBTLE, zeroline=False, title="listings"),
            xaxis=dict(showgrid=False, title="mean sentiment per listing", range=[-1, 1]),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ══════════════════════════════════════════════════════════════════════════════
# ALL CITIES
# ══════════════════════════════════════════════════════════════════════════════
else:
    rows = {r["city"]: r for _, r in summary.iterrows()}
    names = [DISPLAY[k] for k in keys]
    happiest = max(keys, key=lambda k: rows[k]["label_split"].get("positive", 0))
    avg_pos = sum(rows[k]["label_split"].get("positive", 0) for k in keys) / len(keys) * 100
    corr_lo = min(rows[k]["corr_sentiment_vs_rating"] for k in keys)
    corr_hi = max(rows[k]["corr_sentiment_vs_rating"] for k in keys)

    k1, k2, k3, k4 = st.columns(4, gap="medium")
    with k1:
        card("Avg. positive", f"{avg_pos:.0f}%", "across the three cities", featured=True)
    with k2:
        card("Happiest city", DISPLAY[happiest],
             f"{rows[happiest]['label_split'].get('positive', 0) * 100:.0f}% positive")
    with k3:
        card("Reviews scored", f"{int(summary['reviews_scored'].sum()):,}", "40k per city")
    with k4:
        card("Corr. vs rating", f"{corr_lo:.2f}–{corr_hi:.2f}", "honest validation")

    st.write("")
    left, right = st.columns([1.05, 1], gap="large")
    with left:
        st.markdown("##### Sentiment breakdown by city")
        st.caption("Share of reviews positive / neutral / negative.")
        fig = go.Figure()
        for label, lk in [("Positive", "positive"), ("Neutral", "neutral"), ("Negative", "negative")]:
            fig.add_trace(go.Bar(
                name=label, x=names,
                y=[rows[k]["label_split"].get(lk, 0) * 100 for k in keys],
                marker_color=SENT_COLORS[label],
                text=[f"{rows[k]['label_split'].get(lk, 0) * 100:.1f}%" for k in keys],
                textposition="inside",
            ))
        fig.update_layout(
            barmode="stack", height=340, margin=dict(l=10, r=10, t=10, b=20),
            plot_bgcolor=WHITE, paper_bgcolor=WHITE,
            yaxis=dict(showgrid=True, gridcolor=BORDER_SUBTLE, zeroline=False,
                       title="% of reviews", range=[0, 100]),
            xaxis=dict(showgrid=False),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    with right:
        st.markdown("##### Sentiment vs. star rating")
        st.caption("Correlation of per-listing mean sentiment with review_scores_rating.")
        fig2 = go.Figure(go.Bar(
            x=names, y=[rows[k]["corr_sentiment_vs_rating"] for k in keys],
            marker_color=KPMG_BLUE,
            text=[f"{rows[k]['corr_sentiment_vs_rating']:.2f}" for k in keys],
            textposition="outside",
        ))
        fig2.update_layout(
            height=340, margin=dict(l=10, r=10, t=20, b=20),
            plot_bgcolor=WHITE, paper_bgcolor=WHITE,
            yaxis=dict(showgrid=True, gridcolor=BORDER_SUBTLE, zeroline=False,
                       title="Pearson r", range=[0, 0.4]),
            xaxis=dict(showgrid=False), showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    st.write("")
    st.markdown("##### Model performance (per city)")
    perf = pd.DataFrame({
        "City": names,
        "Reviews scored": [f"{int(rows[k]['reviews_scored']):,}" for k in keys],
        "NB accuracy": [f"{rows[k]['nb_accuracy']:.3f}" for k in keys],
        "NB F1 (macro)": [f"{rows[k]['nb_f1_macro']:.3f}" for k in keys],
        "Corr vs rating": [f"{rows[k]['corr_sentiment_vs_rating']:.3f}" for k in keys],
    })
    st.dataframe(perf, use_container_width=True, hide_index=True)

# ── Methodology (both views) ──────────────────────────────────────────────────
st.write("")
with st.expander("Methodology & honest caveat"):
    st.markdown(
        """
- **Model:** TF-IDF (1–2 grams) + **Multinomial Naïve Bayes** (`src/airbnb_iip/models/nlp.py`).
- **Labels:** reviews have no sentiment labels, so weak labels are generated by a **multilingual
  lexicon scorer** — VADER for English, curated ES/FR/DE/IT/PT word-lists for the rest — then the
  NB model is trained on those labels.
- **Per listing:** `mean_sentiment` (mean score in [−1, +1]), `pct_positive`, `pct_negative`,
  `n_reviews` — aggregated in `<city>_sentiment.parquet` over the **~40k-review sample per city**
  (so listing counts reflect sample coverage, not all reviewed listings).
- **⚠️ Caveat:** NB accuracy (~0.94–0.95) measures agreement with the **weak lexicon labels**, not
  ground truth. The honest validation metric is the **correlation with real star ratings
  (0.22–0.29)**.
- **Source:** ~40k reviews scored per city; aggregates in `Data/processed/sentiment_summary.csv`.
        """
    )

footer_disclaimer(DISCLAIMER)
