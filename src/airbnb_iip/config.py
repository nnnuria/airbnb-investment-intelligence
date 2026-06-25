# src/airbnb_iip/config.py
PATHS = {
    "raw": "data/raw",
    "processed": "data/processed",
    "external": "data/external",
}
CITIES = ["madrid", "barcelona", "malaga"]

# Occupancy is now a learned LightGBM model trained on Inside Airbnb calendar
# data (airbnb_iip.models.occupancy) — the San Francisco review-based estimator
# and its review_rate / max_occupancy assumptions have been retired. The only
# surviving knob is avg_length_of_stay, a finance assumption used to turn booked
# nights into cleaning turnovers (airbnb_iip.finance.costs.cleaning_cost_annual).
OCCUPANCY = {"avg_length_of_stay": 3}
FINANCE = {
    "cleaning_fee_per_stay_eur": 45,
    "management_pct": 0.20,
    "platform_fee_pct": 0.03,
    # 21% effective rate for a Spanish resident STR owner (midpoint of IRPF
    # after deductions); sourced from COST_ASSUMPTIONS_SOURCES.md §2.
    "income_tax_pct": 0.21,
    "maintenance_pct_of_value": 0.0125,  # 1.25% of property value, per year
    "accounting_fee_eur": 1000,
    "str_insurance_eur": 700,
    "agent_commission_pct": 0.045,  # real-estate agent fee on sale, mid of 3-6%
    "notary_registry_pct": 0.01,
}

# Barcelona-only pass-through tourist tax (Madrid/Málaga: 0). EUR per guest per night.
TOURIST_TAX_EUR_PER_GUEST_NIGHT = {"barcelona": 4.40, "madrid": 0.0, "malaga": 0.0}

# Spanish CGT brackets on sale (2024), as (upper_bound, rate). The last bound is
# math.inf so it always matches. Applied progressively, like income tax.
CGT_BRACKETS = [
    (6_000, 0.19),
    (50_000, 0.21),
    (200_000, 0.23),
    (float("inf"), 0.26),
]

# Regulatory shock probability (Bernoulli, per year) used as a Monte Carlo input —
# see docs/INVESTMENT_DECISION_FRAMEWORK.md Section 8.
REGULATORY_SHOCK_PROB = {"madrid": 0.08, "barcelona": 0.20, "malaga": 0.05}

RANDOM_STATE = 42

# ── Financial analysis enhancements ──────────────────────────────────────────

# IBI (Impuesto sobre Bienes Inmuebles) — 2025 municipal rates.
# Source: COST_ASSUMPTIONS_SOURCES.md §1 (Hacienda consolidated PDF +
# Ajuntament OF 1.1 / Gestrisam). Formula: IBI_€ = rate × cadastral_value.
IBI_RATE_BY_CITY: dict[str, float] = {
    "madrid":    0.00428,   # 0.428% general urban (2025; 0.414% from 2026)
    "barcelona": 0.00660,   # 0.66% urban — Ajuntament OF 1.1 IBI 2025
    "malaga":    0.00451,   # 0.451% general urban — Gestrisam (frozen since 2014)
}
IBI_RATE_DEFAULT = 0.004   # legal lower bound ~0.4%; used for unknown cities

# Typical cadastral-to-market-value ratio per city.
# Used by IbiAgent to estimate cadastral value when it is not supplied.
# Cadastral values in Spain are assessed at 30–60% of market value; exact
# figure depends on when the municipality last did a collective reassessment.
CADASTRAL_TO_MARKET_RATIO: dict[str, float] = {
    "madrid":    0.35,   # relatively recent reassessment
    "barcelona": 0.50,   # older assessments
    "malaga":    0.40,   # conservative mid-range
}
CADASTRAL_TO_MARKET_RATIO_DEFAULT = 0.40

# Waste tax (basuras / TGR). Source: COST_ASSUMPTIONS_SOURCES.md §1.
# Madrid: new TGR effective 01-Sep-2025 (~€141 avg/household).
# Barcelona: >€140 per year (Bankinter). Málaga: national avg placeholder.
BASURAS_EUR_BY_CITY: dict[str, float] = {
    "madrid":    141.0,
    "barcelona": 140.0,
    "malaga":     78.0,
}
BASURAS_EUR_DEFAULT = 100.0

# Minimum one-time setup cost to list a property on Airbnb (professional
# photography, deep clean, linen/supplies top-up, minor repairs). Users may
# override upward for heavier refurbishments.
AIRBNB_SETUP_COST_EUR_MIN = 1_500.0

# Annual NOI growth rate (CPI/HPI blend). Spain 10-yr avg CPI ~2%, housing
# rental index ~3% — 2.5% is a conservative central estimate.
NOI_GROWTH_RATE_DEFAULT = 0.025

# Annual property price appreciation. Conservative estimate for Madrid/BCN/Málaga
# based on 10-yr historical averages; users can adjust for sensitivity analysis.
PROPERTY_APPRECIATION_RATE_DEFAULT = 0.03

# Spanish IRPF progressive brackets applicable to STR rental income (2024).
# Format: (upper_income_bound_eur, marginal_rate). These apply only to the
# portion of income from the STR activity, not total personal income (which
# requires knowing all other sources — deliberately not modelled here).
# Flat 19% default is used when the user opts out of progressive brackets.
IRPF_BRACKETS: list[tuple[float, float]] = [
    (12_450, 0.19),
    (20_200, 0.24),
    (35_200, 0.30),
    (60_000, 0.37),
    (300_000, 0.45),
    (float("inf"), 0.47),
]
