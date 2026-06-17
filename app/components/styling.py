"""Page-level styling. Inject CSS once per page.

Style philosophy: light, restrained, lots of whitespace, KPMG blue used as
an accent rather than a fill. Body type is Inter (with system fallback).
"""

from __future__ import annotations

import streamlit as st

from .branding import (
    BORDER_SUBTLE,
    KPMG_BLUE,
    KPMG_BLUE_DARK,
    KPMG_BLUE_TINT,
    SURFACE,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    WHITE,
)

CUSTOM_CSS = f"""
<style>
  /* ── Typography ──────────────────────────────────────────────────────── */
  @import url("https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap");

  html, body, [class*="css"] {{
    font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    color: {TEXT_PRIMARY};
  }}

  h1, h2, h3, h4 {{
    font-weight: 700;
    letter-spacing: -0.015em;
    color: {TEXT_PRIMARY};
  }}

  h1 {{ font-size: 2.25rem; line-height: 1.15; }}
  h2 {{ font-size: 1.625rem; line-height: 1.25; }}
  h3 {{ font-size: 1.25rem; line-height: 1.3; }}

  /* ── Layout ──────────────────────────────────────────────────────────── */
  .block-container {{
    padding-top: 2.5rem;
    padding-bottom: 4rem;
    max-width: 1180px;
  }}

  section[data-testid="stSidebar"] {{
    background: {SURFACE};
    border-right: 1px solid {BORDER_SUBTLE};
  }}

  section[data-testid="stSidebar"] .block-container {{
    padding-top: 2rem;
  }}

  /* ── Sidebar headline ────────────────────────────────────────────────── */
  .sidebar-brand {{
    padding: 0.5rem 0 1.25rem 0;
    border-bottom: 1px solid {BORDER_SUBTLE};
    margin-bottom: 1rem;
  }}
  .sidebar-brand .mark {{
    font-weight: 800;
    color: {KPMG_BLUE};
    letter-spacing: 0.01em;
    font-size: 0.95rem;
  }}
  .sidebar-brand .subtitle {{
    color: {TEXT_MUTED};
    font-size: 0.75rem;
    margin-top: 0.15rem;
  }}

  /* ── Top hero ────────────────────────────────────────────────────────── */
  .hero {{
    padding: 2.25rem 0 1rem 0;
    border-bottom: 1px solid {BORDER_SUBTLE};
    margin-bottom: 2rem;
  }}
  .hero .eyebrow {{
    color: {KPMG_BLUE};
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.75rem;
  }}
  .hero .title {{
    font-size: 2.5rem;
    font-weight: 700;
    line-height: 1.15;
    color: {TEXT_PRIMARY};
    margin: 0;
  }}
  .hero .lede {{
    color: {TEXT_SECONDARY};
    font-size: 1.05rem;
    max-width: 720px;
    margin-top: 0.85rem;
    line-height: 1.55;
  }}

  /* ── Cards ───────────────────────────────────────────────────────────── */
  .card {{
    background: {WHITE};
    border: 1px solid {BORDER_SUBTLE};
    border-radius: 14px;
    padding: 1.25rem 1.25rem 1.1rem;
    height: 100%;
    min-width: 0;
  }}
  .card .label {{
    color: {TEXT_MUTED};
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin: 0 0 0.5rem 0;
    line-height: 1.25;
  }}
  .card .value {{
    font-size: clamp(1.1rem, 1.9vw, 1.55rem);
    font-weight: 700;
    color: {TEXT_PRIMARY};
    line-height: 1.15;
    margin: 0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    letter-spacing: -0.01em;
  }}
  .card .delta {{
    color: {TEXT_SECONDARY};
    font-size: 0.8rem;
    margin-top: 0.45rem;
    line-height: 1.4;
  }}

  .card.featured {{
    border-color: {KPMG_BLUE};
    background: linear-gradient(180deg, {KPMG_BLUE_TINT} 0%, {WHITE} 80%);
  }}
  .card.featured .label {{
    color: {KPMG_BLUE_DARK};
  }}

  /* ── Recommendation pill ─────────────────────────────────────────────── */
  .rec-pill {{
    display: inline-block;
    padding: 0.4rem 0.85rem;
    background: {KPMG_BLUE};
    color: {WHITE};
    font-weight: 600;
    font-size: 0.85rem;
    letter-spacing: 0.02em;
    border-radius: 999px;
    margin-bottom: 0.75rem;
  }}

  /* ── Buttons ─────────────────────────────────────────────────────────── */
  .stButton > button {{
    background: {KPMG_BLUE};
    color: {WHITE};
    border: 1px solid {KPMG_BLUE};
    border-radius: 10px;
    padding: 0.55rem 1.25rem;
    font-weight: 600;
    transition: background 120ms ease, transform 80ms ease;
  }}
  .stButton > button:hover {{
    background: {KPMG_BLUE_DARK};
    border-color: {KPMG_BLUE_DARK};
    color: {WHITE};
  }}
  .stButton > button:active {{
    transform: translateY(1px);
  }}
  .stButton > button:focus:not(:active) {{
    border-color: {KPMG_BLUE_DARK};
    color: {WHITE};
    box-shadow: 0 0 0 3px {KPMG_BLUE_TINT};
  }}

  /* Secondary / ghost buttons */
  button[kind="secondary"] {{
    background: {WHITE} !important;
    color: {KPMG_BLUE} !important;
    border: 1px solid {BORDER_SUBTLE} !important;
  }}
  button[kind="secondary"]:hover {{
    background: {KPMG_BLUE_TINT} !important;
    border-color: {KPMG_BLUE} !important;
  }}

  /* ── Inputs ──────────────────────────────────────────────────────────── */
  div[data-baseweb="input"] > div,
  div[data-baseweb="select"] > div,
  textarea {{
    border-radius: 10px !important;
    border-color: {BORDER_SUBTLE} !important;
  }}

  /* ── Footer ──────────────────────────────────────────────────────────── */
  .footer-note {{
    color: {TEXT_MUTED};
    font-size: 0.78rem;
    border-top: 1px solid {BORDER_SUBTLE};
    padding-top: 1rem;
    margin-top: 3rem;
    line-height: 1.5;
  }}

  /* ── Streamlit chrome cleanup ────────────────────────────────────────── */
  #MainMenu {{visibility: hidden;}}
  footer {{visibility: hidden;}}
  header {{visibility: hidden;}}

  /* Tabs */
  div[data-baseweb="tab-list"] {{
    border-bottom: 1px solid {BORDER_SUBTLE};
    gap: 0.5rem;
  }}
  button[data-baseweb="tab"] {{
    color: {TEXT_SECONDARY} !important;
    font-weight: 500;
  }}
  button[data-baseweb="tab"][aria-selected="true"] {{
    color: {KPMG_BLUE} !important;
    font-weight: 600;
  }}
</style>
"""


def apply_page_style(page_title: str | None = None) -> None:
    """Set page config + inject the custom CSS. Call once at the top of every page."""
    st.set_page_config(
        page_title=f"{page_title} — Airbnb Investment Intelligence"
        if page_title
        else "Airbnb Investment Intelligence",
        page_icon="◼",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    _sidebar_brand()


def _sidebar_brand() -> None:
    st.sidebar.markdown(
        """
        <div class="sidebar-brand">
          <div class="mark">AIRBNB-IIP</div>
          <div class="subtitle">IE × KPMG · Capstone 2026</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def hero(eyebrow: str, title: str, lede: str | None = None) -> None:
    """Top-of-page hero block — eyebrow → title → optional lede."""
    lede_html = f'<div class="lede">{lede}</div>' if lede else ""
    st.markdown(
        f"""
        <div class="hero">
          <div class="eyebrow">{eyebrow}</div>
          <div class="title">{title}</div>
          {lede_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def footer_disclaimer(text: str) -> None:
    st.markdown(
        f'<div class="footer-note">{text}</div>',
        unsafe_allow_html=True,
    )


def card(
    label: str,
    value: str,
    delta: str | None = None,
    featured: bool = False,
) -> None:
    """A KPI card. Featured = KPMG-blue accent."""
    delta_html = f'<div class="delta">{delta}</div>' if delta else ""
    klass = "card featured" if featured else "card"
    st.markdown(
        f"""
        <div class="{klass}">
          <div class="label">{label}</div>
          <div class="value">{value}</div>
          {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def recommendation_pill(text: str) -> None:
    st.markdown(f'<span class="rec-pill">{text}</span>', unsafe_allow_html=True)
