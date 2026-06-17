"""Demo decision engine — mock model responses for the KPMG presentation.

Wired against the real occupancy estimator (:mod:`airbnb_iip.data.occupancy`)
because task 2 is merged. Price prediction and sale-price lookup are mocked
with district premiums until tasks 4/5 land — same function shape, so the
mock can be swapped for a FastAPI call without touching the UI.

Per SPRINT_STATUS line 78: never run live LLM/API calls during the demo.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Literal

from airbnb_iip.config import FINANCE, OCCUPANCY
from airbnb_iip.data.occupancy import estimate_occupancy

# ── Demo lookup tables ───────────────────────────────────────────────────────
# Indicative district price premiums vs city baseline. Real values come from
# the price ML model (task 4) and external sale-price data (task 1).
CITY_BASE_PRICE_PER_NIGHT_EUR: dict[str, float] = {
    "madrid":    115.0,
    "barcelona": 135.0,
    "malaga":    105.0,
}

CITY_BASE_SALE_EUR_PER_M2: dict[str, float] = {
    "madrid":    5400.0,
    "barcelona": 5800.0,
    "malaga":    3300.0,
}

# Multiplier on the city baseline. Hand-picked for plausibility.
DISTRICT_PREMIUM: dict[tuple[str, str], float] = {
    # Madrid
    ("madrid", "Centro"):              1.45,
    ("madrid", "Salamanca"):           1.55,
    ("madrid", "Chamberí"):            1.40,
    ("madrid", "Chamartín"):           1.35,
    ("madrid", "Retiro"):              1.42,
    ("madrid", "Arganzuela"):          1.18,
    ("madrid", "Moncloa-Aravaca"):     1.30,
    ("madrid", "Tetuán"):              1.10,
    ("madrid", "Latina"):              0.92,
    ("madrid", "Carabanchel"):         0.85,
    ("madrid", "Puente de Vallecas"):  0.82,
    # Barcelona
    ("barcelona", "Eixample"):           1.40,
    ("barcelona", "Ciutat Vella"):       1.35,
    ("barcelona", "Sarrià-Sant Gervasi"):1.50,
    ("barcelona", "Gràcia"):             1.25,
    ("barcelona", "Sant Martí"):         1.10,
    ("barcelona", "Sants-Montjuïc"):     1.05,
    ("barcelona", "Les Corts"):          1.30,
    ("barcelona", "Horta-Guinardó"):     0.95,
    ("barcelona", "Nou Barris"):         0.80,
    # Málaga
    ("malaga", "Centro"):                1.40,
    ("malaga", "Este"):                  1.30,
    ("malaga", "Ciudad Jardín"):         0.95,
    ("malaga", "Carretera de Cádiz"):    1.10,
    ("malaga", "Teatinos-Universidad"):  1.05,
    ("malaga", "Puerto de la Torre"):    0.92,
}

CITIES = ("madrid", "barcelona", "malaga")
CITY_LABELS = {"madrid": "Madrid", "barcelona": "Barcelona", "malaga": "Málaga"}
ROOM_TYPES = ("Entire home/apt", "Private room", "Hotel room", "Shared room")


# ── Property + scenario types ────────────────────────────────────────────────

@dataclass
class Property:
    """User-entered property. The single input the decision engine reads."""
    city: str
    district: str
    size_m2: float
    bedrooms: int
    bathrooms: float
    accommodates: int
    room_type: str
    has_balcony: bool = False
    has_ac: bool = False
    has_elevator: bool = False
    has_pool: bool = False
    has_parking: bool = False
    has_workspace: bool = False
    nickname: str | None = None
    notes: str | None = None


@dataclass
class Scenario:
    """Output of the decision engine."""
    recommendation: Literal["airbnb", "sell", "marginal"]
    confidence: float                       # 0..1

    predicted_nightly_eur: float
    occupancy_rate_annual: float            # 0..1
    nights_booked_year: int

    gross_revenue_year_eur: float
    net_revenue_year_eur: float
    net_revenue_p10_eur: float
    net_revenue_p90_eur: float

    sale_price_eur: float
    sale_price_per_m2_eur: float
    breakeven_years: float

    monthly_seasonality: list[float] = field(default_factory=list)
    cost_breakdown: list[tuple[str, float]] = field(default_factory=list)
    feature_drivers: list[tuple[str, float]] = field(default_factory=list)


# ── The decision engine ──────────────────────────────────────────────────────

def estimate_nightly_price(prop: Property) -> float:
    """Mock nightly-rate estimate."""
    city = prop.city.lower()
    base = CITY_BASE_PRICE_PER_NIGHT_EUR.get(city, 100.0)
    premium = DISTRICT_PREMIUM.get((city, prop.district), 1.0)

    # Size + capacity effects.
    size_mult = 1.0 + max(0, prop.size_m2 - 60) / 240
    cap_mult = 0.7 + 0.1 * prop.accommodates
    room_mult = {
        "Entire home/apt": 1.00,
        "Private room": 0.55,
        "Hotel room": 1.10,
        "Shared room": 0.35,
    }.get(prop.room_type, 1.0)

    amenities = sum(
        m for flag, m in (
            (prop.has_balcony, 0.04),
            (prop.has_ac, 0.06),
            (prop.has_elevator, 0.03),
            (prop.has_pool, 0.10),
            (prop.has_parking, 0.05),
            (prop.has_workspace, 0.03),
        )
        if flag
    )

    nightly = base * premium * size_mult * cap_mult * room_mult * (1 + amenities)
    return round(nightly, 2)


def reviews_per_month_for_demo(prop: Property) -> float:
    """Indicative reviews/month from district desirability + amenities."""
    city = prop.city.lower()
    premium = DISTRICT_PREMIUM.get((city, prop.district), 1.0)
    base = 1.8 if premium >= 1.3 else 1.2 if premium >= 1.0 else 0.8
    amenity_lift = 0.15 * sum(
        flag for flag in (
            prop.has_balcony, prop.has_ac, prop.has_elevator, prop.has_workspace,
        )
    )
    return round(base + amenity_lift, 2)


def estimate_sale_value(prop: Property) -> tuple[float, float]:
    """Returns ``(total_eur, eur_per_m2)``."""
    city = prop.city.lower()
    base_per_m2 = CITY_BASE_SALE_EUR_PER_M2.get(city, 3500.0)
    premium = DISTRICT_PREMIUM.get((city, prop.district), 1.0)
    per_m2 = base_per_m2 * premium
    total = per_m2 * prop.size_m2
    return round(total, -3), round(per_m2, 0)


def compute_scenario(prop: Property) -> Scenario:
    """Full Airbnb-vs-sell scenario for one property — the demo's centrepiece."""
    nightly = estimate_nightly_price(prop)
    rpm = reviews_per_month_for_demo(prop)

    occ = estimate_occupancy(rpm)
    monthly_nights = occ["nights_booked_per_month"]
    annual_nights = round(monthly_nights * 12)
    occupancy_rate = monthly_nights / 30

    gross_year = nightly * annual_nights

    cleaning_fee_total = FINANCE["cleaning_fee_per_stay_eur"] * (
        annual_nights / OCCUPANCY["avg_length_of_stay"]
    )
    platform_fee = gross_year * FINANCE["platform_fee_pct"]
    mgmt_fee = gross_year * FINANCE["management_pct"]
    pre_tax = gross_year - platform_fee - mgmt_fee - cleaning_fee_total
    tax = max(pre_tax, 0) * FINANCE["income_tax_pct"]
    net_year = pre_tax - tax

    # Indicative P10/P90 band — uncertainty is dominated by occupancy.
    net_p10 = net_year * 0.78
    net_p90 = net_year * 1.18

    sale_value, sale_per_m2 = estimate_sale_value(prop)
    breakeven = sale_value / net_year if net_year > 0 else math.inf

    if breakeven <= 12:
        rec = "airbnb"
        confidence = max(0.55, min(0.92, 1 - (breakeven / 24)))
    elif breakeven >= 22:
        rec = "sell"
        confidence = max(0.55, min(0.92, (breakeven - 12) / 24))
    else:
        rec = "marginal"
        confidence = 0.55

    seasonality = _demo_seasonality(prop.city.lower())
    costs = [
        ("Platform fee (3%)", round(platform_fee, 0)),
        ("Management (20%)", round(mgmt_fee, 0)),
        ("Cleaning fees", round(cleaning_fee_total, 0)),
        ("Income tax (19%)", round(tax, 0)),
    ]
    drivers = _demo_drivers(prop)

    return Scenario(
        recommendation=rec,
        confidence=confidence,
        predicted_nightly_eur=nightly,
        occupancy_rate_annual=round(occupancy_rate, 3),
        nights_booked_year=annual_nights,
        gross_revenue_year_eur=round(gross_year, 0),
        net_revenue_year_eur=round(net_year, 0),
        net_revenue_p10_eur=round(net_p10, 0),
        net_revenue_p90_eur=round(net_p90, 0),
        sale_price_eur=sale_value,
        sale_price_per_m2_eur=sale_per_m2,
        breakeven_years=round(breakeven, 1) if breakeven != math.inf else 99.0,
        monthly_seasonality=seasonality,
        cost_breakdown=costs,
        feature_drivers=drivers,
    )


# ── Optimisation — improvement ideas ─────────────────────────────────────────

@dataclass
class Improvement:
    name: str
    investment_eur: float
    monthly_revenue_uplift_eur: float
    payback_months: float
    confidence: Literal["high", "medium", "low"]
    rationale: str


def suggest_improvements(prop: Property, scen: Scenario) -> list[Improvement]:
    """Indicative improvements ranked by ROI."""
    items: list[Improvement] = []
    rev_month = scen.net_revenue_year_eur / 12

    if not prop.has_ac:
        uplift = rev_month * 0.06
        items.append(Improvement(
            "Install air conditioning",
            investment_eur=1800,
            monthly_revenue_uplift_eur=round(uplift, 0),
            payback_months=round(1800 / uplift, 1) if uplift > 0 else 99,
            confidence="high",
            rationale="AC is among the top three amenities by booking lift in "
                      "warm-climate markets; comparable listings without AC "
                      "lose 5–8% of bookings in peak months.",
        ))

    if not prop.has_workspace:
        uplift = rev_month * 0.04
        items.append(Improvement(
            "Add a dedicated workspace",
            investment_eur=350,
            monthly_revenue_uplift_eur=round(uplift, 0),
            payback_months=round(350 / uplift, 1) if uplift > 0 else 99,
            confidence="high",
            rationale="Workspace amenity unlocks the long-stay segment "
                      "(28+ nights) and tends to lift weekday occupancy.",
        ))

    if not prop.has_balcony:
        uplift = rev_month * 0.03
        items.append(Improvement(
            "Highlight outdoor space in listing",
            investment_eur=120,
            monthly_revenue_uplift_eur=round(uplift, 0),
            payback_months=round(120 / uplift, 1) if uplift > 0 else 99,
            confidence="medium",
            rationale="If you have any patio, balcony, or terrace already, "
                      "ensuring it shows in photos is a cheap revenue lever.",
        ))

    items.append(Improvement(
        "Professional photography refresh",
        investment_eur=250,
        monthly_revenue_uplift_eur=round(rev_month * 0.05, 0),
        payback_months=round(250 / max(rev_month * 0.05, 1), 1),
        confidence="high",
        rationale="Improving listing photos is the single highest-ROI "
                  "change short of a renovation.",
    ))

    items.append(Improvement(
        "Adopt dynamic pricing (PriceLabs / Wheelhouse)",
        investment_eur=20 * 12,
        monthly_revenue_uplift_eur=round(rev_month * 0.07, 0),
        payback_months=round((20 * 12) / max(rev_month * 0.07, 1), 1),
        confidence="medium",
        rationale="Algorithmic pricing typically lifts revenue 5–10% by "
                  "capturing event spikes and softening shoulder months.",
    ))

    items.sort(key=lambda x: x.payback_months)
    return items


# ── Chat — canned conversational responses ───────────────────────────────────

CHAT_INTROS = [
    "Hi, I'm your investment co-pilot. Tell me about a property — city, "
    "district, size, bedrooms — or ask anything about an existing analysis.",
    "Hello. I can compare Airbnb vs sale for any property in Madrid, "
    "Barcelona, or Málaga. What would you like to explore?",
]


def chat_reply(user_msg: str, scen: Scenario | None) -> str:
    """Pattern-matched, deterministic answers — no live LLM (per SPRINT_STATUS)."""
    msg = user_msg.lower().strip()

    if scen is None:
        if any(k in msg for k in ("start", "begin", "analyse", "analyze", "property")):
            return (
                "Open **New analysis** in the sidebar to enter property details. "
                "Once analysed, return here and I'll answer questions about it."
            )
        return random.choice(CHAT_INTROS)

    if any(k in msg for k in ("sell", "sale", "should i sell")):
        return (
            f"Indicative sale value is **€{scen.sale_price_eur:,.0f}** "
            f"(€{scen.sale_price_per_m2_eur:,.0f}/m²). Break-even against "
            f"projected Airbnb net income is **{scen.breakeven_years:.1f} years**. "
            + (
                "With break-even under 12 years, the Airbnb path tends to be the "
                "stronger long-term play unless capital recycling is a priority."
                if scen.breakeven_years <= 12 else
                "With break-even above 20 years, a sale is usually preferred — the "
                "capital can be redeployed into higher-yield instruments."
                if scen.breakeven_years >= 20 else
                "This sits in the marginal zone (12–20 years). Other factors — "
                "tax position, capital needs, regulation risk — typically tip the call."
            )
        )

    if any(k in msg for k in ("revenue", "income", "earn", "make")):
        return (
            f"Estimated net annual revenue is "
            f"**€{scen.net_revenue_year_eur:,.0f}** "
            f"(P10–P90 band: €{scen.net_revenue_p10_eur:,.0f}–"
            f"€{scen.net_revenue_p90_eur:,.0f}). "
            f"Gross is €{scen.gross_revenue_year_eur:,.0f}; the gap is costs — "
            "platform fee, management, cleaning, and tax."
        )

    if any(k in msg for k in ("occupancy", "booked", "vacancy")):
        return (
            f"Projected occupancy is **{scen.occupancy_rate_annual * 100:.0f}%** "
            f"({scen.nights_booked_year} nights/year). This uses the San Francisco "
            f"model — inferred from review activity, capped at our "
            f"70% realistic ceiling."
        )

    if any(k in msg for k in ("rate", "price", "nightly", "night")):
        return (
            f"Projected nightly rate is "
            f"**€{scen.predicted_nightly_eur:,.0f}/night**, set by district "
            "premium, size, capacity, and amenity mix."
        )

    if any(k in msg for k in ("regulation", "license", "licence", "legal", "permit")):
        return (
            "Each city has its own short-term-rental regime. Madrid restricts "
            "STR in many residential blocks; Barcelona requires an HUT licence; "
            "Málaga enforces tourist registration with the Junta de Andalucía. "
            "Always verify against current official municipal sources before "
            "listing. (Full regulatory corpus available in the agent layer.)"
        )

    if any(k in msg for k in ("improve", "optimise", "optimize", "uplift", "amenity")):
        return (
            "See **Optimisation** in the sidebar — I rank improvements by ROI. "
            "Top moves usually are: professional photos, AC if absent, dedicated "
            "workspace, and dynamic pricing."
        )

    if any(k in msg for k in ("why", "explain", "how")):
        return (
            f"The recommendation is **{scen.recommendation.upper()}** "
            f"(confidence ~{scen.confidence*100:.0f}%). The strongest drivers are: "
            + ", ".join(f"{name}" for name, _ in scen.feature_drivers[:3])
            + ". Break-even sits at "
            f"{scen.breakeven_years:.1f} years."
        )

    return (
        "I can answer questions about projected revenue, occupancy, the "
        "sale-vs-Airbnb comparison, costs, regulations, or recommended "
        "improvements. Try \"why this recommendation?\" or "
        "\"what about regulations?\"."
    )


# ── Internals ────────────────────────────────────────────────────────────────

def _demo_seasonality(city: str) -> list[float]:
    """Plausible monthly multipliers per city. Mean = 1.0."""
    patterns = {
        "madrid":    [0.85, 0.88, 0.95, 1.05, 1.10, 1.05, 1.00, 0.92, 1.08, 1.10, 0.95, 1.07],
        "barcelona": [0.82, 0.85, 0.95, 1.10, 1.20, 1.18, 1.15, 1.08, 1.10, 1.05, 0.85, 0.85],
        "malaga":    [0.85, 0.88, 1.00, 1.10, 1.18, 1.25, 1.30, 1.30, 1.15, 1.00, 0.85, 0.92],
    }
    return patterns.get(city, [1.0] * 12)


def _demo_drivers(prop: Property) -> list[tuple[str, float]]:
    """Pretend SHAP values — feature → directional contribution to net revenue."""
    drivers: list[tuple[str, float]] = [
        ("District premium", DISTRICT_PREMIUM.get((prop.city.lower(), prop.district), 1.0) - 1),
        ("Property size", max(0, prop.size_m2 - 60) / 240),
        ("Capacity", (prop.accommodates - 2) * 0.05),
    ]
    if prop.has_pool:
        drivers.append(("Pool", 0.10))
    if prop.has_ac:
        drivers.append(("Air conditioning", 0.06))
    if prop.has_balcony:
        drivers.append(("Balcony/terrace", 0.04))
    if prop.has_workspace:
        drivers.append(("Workspace", 0.03))
    drivers.sort(key=lambda x: -abs(x[1]))
    return drivers[:6]


__all__ = [
    "CITIES", "CITY_LABELS", "ROOM_TYPES", "DISTRICT_PREMIUM",
    "Property", "Scenario", "Improvement",
    "compute_scenario", "estimate_nightly_price", "estimate_sale_value",
    "suggest_improvements", "chat_reply",
]
