"""Brand constants — colours, typography, product names.

Centralised so swapping the corporate palette later means editing one file.
The default palette is KPMG's Pantone 280 corporate blue.
"""

from __future__ import annotations

# ── Corporate palette (KPMG Pantone 280) ─────────────────────────────────────
KPMG_BLUE = "#00338D"
KPMG_BLUE_DARK = "#00205B"
KPMG_BLUE_LIGHT = "#1E49E2"
KPMG_BLUE_TINT = "#E6EBF4"

# ── Neutral surface palette (Anthropic-style, generous whitespace) ───────────
WHITE = "#FFFFFF"
SURFACE = "#F7F8FA"
SURFACE_RAISED = "#FFFFFF"
BORDER_SUBTLE = "#E5E7EB"
BORDER_STRONG = "#D1D5DB"

# ── Text ─────────────────────────────────────────────────────────────────────
TEXT_PRIMARY = "#1A1A1A"
TEXT_SECONDARY = "#4B5563"
TEXT_MUTED = "#6B7280"
TEXT_INVERTED = "#FFFFFF"

# ── Semantic colours ─────────────────────────────────────────────────────────
SUCCESS = "#0F9D58"
WARNING = "#F4B400"
DANGER = "#D93025"
INFO = KPMG_BLUE

# ── Plotly palette ───────────────────────────────────────────────────────────
PLOTLY_PALETTE = [
    KPMG_BLUE,
    "#5DADE2",
    "#1ABC9C",
    "#F39C12",
    "#9B59B6",
    "#34495E",
]

# ── Product strings ──────────────────────────────────────────────────────────
PRODUCT_NAME = "Airbnb Investment Intelligence"
PRODUCT_SHORT = "Airbnb-IIP"
PARTNERSHIP = "IE × KPMG Lighthouse · Capstone 2026"
DISCLAIMER = (
    "Indicative analysis only. Not financial, legal, or investment advice. "
    "Verify regulatory information against current official municipal sources."
)
