# src/airbnb_iip/config.py
PATHS = {
    "raw": "data/raw",
    "processed": "data/processed",
    "external": "data/external",
}
CITIES = ["madrid", "barcelona", "malaga"]

OCCUPANCY = {"review_rate": 0.50, "avg_length_of_stay": 3, "max_occupancy": 0.70}
FINANCE = {
    "cleaning_fee_per_stay_eur": 45,
    "management_pct": 0.20,
    "platform_fee_pct": 0.03,
    "income_tax_pct": 0.19,
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
