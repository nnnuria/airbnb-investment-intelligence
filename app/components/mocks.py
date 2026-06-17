"""Demo decision engine — real finance + occupancy, mocked price + sale inputs.

Wired against:

* :mod:`airbnb_iip.data.occupancy` — real San Francisco occupancy estimator
  (task 2, merged).
* :mod:`airbnb_iip.finance.costs` and :mod:`airbnb_iip.finance.scenarios`
  — real NOI + NPV + break-even + Monte Carlo (task 5, merged). All defaults
  are taken from the finance module; this file does not override discount
  rate, sigmas, or cost rates.

Still mocked, because the upstream models haven't shipped a UI-friendly
inference API yet:

* :func:`estimate_nightly_price` — district premium × capacity × amenities.
  Replace when the price model (task 4) exposes a per-property endpoint.
* :func:`estimate_sale_value` — district €/m² lookup table. Replace when
  the sell model exposes a per-property endpoint.
* :func:`reviews_per_month_for_demo` — feeds the occupancy estimator.

Per SPRINT_STATUS line 78: never run live LLM/API calls during the demo.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from airbnb_iip.config import FINANCE, OCCUPANCY
from airbnb_iip.data.occupancy import estimate_occupancy
from airbnb_iip.finance.costs import (
    annual_gross_revenue,
    cleaning_cost_annual,
    compute_noi,
    tourist_tax_annual,
)
from airbnb_iip.finance.scenarios import (
    break_even_horizon,
    monte_carlo,
    npv_sell,
)

# Holding horizon for the NPV comparison. A UI-side choice — the finance
# module's API takes ``years`` as a required parameter; 10 years is the
# typical investor decision window.
HOLDING_YEARS = 10

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
    """Full Airbnb-vs-sell scenario for one property — the demo's centrepiece.

    Wires the real finance module end-to-end:

    1. Mocked price + occupancy → point estimates.
    2. :func:`airbnb_iip.finance.costs.compute_noi` → full Spanish STR
       cost stack (platform fee, cleaning, tourist tax, insurance,
       maintenance, management, accounting, income tax).
    3. :func:`airbnb_iip.finance.scenarios.npv_sell` → net sale proceeds
       after agent fee, notary, CGT.
    4. :func:`break_even_horizon` → smallest ``T`` where
       ``NPV_airbnb(T) > NPV_sell``. The finance API expects callers to
       pass a ``terminal_value_fn``; we credit the eventual sale at year
       ``T`` (zero appreciation — conservative).
    5. :func:`monte_carlo` → resamples price + occupancy uncertainty into
       NPV bands and ``P(Airbnb > Sell)``, which drives the recommendation
       and confidence.
    6. Local resample for *annual NOI* P10/P50/P90 bands, since the
       finance MC returns NPV bands but the UI's KPI card displays NOI.

    All finance-module assumptions (discount rate, MC sigmas, cost rates)
    come from their built-in defaults — this function does not override
    them.
    """
    nightly = estimate_nightly_price(prop)
    rpm = reviews_per_month_for_demo(prop)

    occ = estimate_occupancy(rpm)
    monthly_nights = occ["nights_booked_per_month"]
    annual_nights = round(monthly_nights * 12)
    occupancy_rate = monthly_nights / 30

    sale_value, sale_per_m2 = estimate_sale_value(prop)

    # ── Real NOI via the finance module ──────────────────────────────────────
    def _make_cost_kwargs(occ_nights: float) -> dict:
        """Cost-kwargs factory — cleaning + tourist tax scale with occupancy."""
        return dict(
            property_value=sale_value,
            cleaning_cost_eur=cleaning_cost_annual(
                occ_nights,
                avg_length_of_stay=OCCUPANCY["avg_length_of_stay"],
            ),
            tourist_tax_eur=tourist_tax_annual(occ_nights, city=prop.city),
            management_fee_rate=FINANCE["management_pct"],
        )

    gross = annual_gross_revenue(nightly, annual_nights)
    cost_kwargs = _make_cost_kwargs(annual_nights)
    noi = compute_noi(gross, **cost_kwargs)
    net_year = noi["noi"]

    # ── Real sale-side: NPV_sell with realistic owner assumptions ────────────
    # The form doesn't collect purchase price; assume the typical Spanish
    # owner bought 10 yrs ago at 70% of today's value with 4% acquisition
    # costs and 2% in documented improvements. CGT is computed progressively
    # by the finance module from the resulting gain.
    sell = npv_sell(
        sale_price=sale_value,
        purchase_price=sale_value * 0.70,
        purchase_costs=sale_value * 0.04,
        documented_improvements=sale_value * 0.02,
    )
    sell_net = sell["net_proceeds"]

    # ── Real break-even horizon ──────────────────────────────────────────────
    # Per the finance API docstring, callers should supply a terminal value
    # for a realistic comparison. Use a flat (no-appreciation) terminal
    # equal to today's net sale proceeds.
    be_t = break_even_horizon(
        noi_annual=net_year,
        npv_sell_value=sell_net,
        terminal_value_fn=lambda _t: sell_net,
        max_years=30,
    )
    breakeven_years = float(be_t) if be_t is not None else 99.0

    # ── Annual NOI P10/P50/P90 (local MC; UI card displays NOI, not NPV) ─────
    p10, p50, p90 = _resample_annual_noi(
        nightly, annual_nights, _make_cost_kwargs, n=1500,
    )

    # ── Recommendation: NPV-based via the finance Monte Carlo ────────────────
    mc = monte_carlo(
        price_hat=nightly,
        occ_hat=annual_nights,
        years=HOLDING_YEARS,
        npv_sell_value=sell_net,
        cost_kwargs=cost_kwargs,
        terminal_value_net=sell_net,
        random_state=42,
        n_simulations=1500,
    )
    p_airbnb = mc["p_airbnb_gt_sell"]
    if p_airbnb >= 0.60:
        rec, confidence = "airbnb", p_airbnb
    elif p_airbnb <= 0.40:
        rec, confidence = "sell", 1.0 - p_airbnb
    else:
        rec, confidence = "marginal", max(p_airbnb, 1.0 - p_airbnb)

    # ── Cost breakdown for the UI: surface every non-zero NOI line ───────────
    cost_lines = [
        ("Platform fee", noi["platform_fee"]),
        ("Cleaning", noi["cleaning_cost"]),
        ("Tourist tax", noi["tourist_tax"]),
        ("Management", noi["management_fee"]),
        ("Maintenance", noi["maintenance"]),
        ("Insurance", noi["insurance"]),
        ("Accounting", noi["accounting_fee"]),
        ("Income tax", noi["tax"]),
    ]
    costs = [(label, round(v, 0)) for label, v in cost_lines if v > 0]

    seasonality = _demo_seasonality(prop.city.lower())
    drivers = _demo_drivers(prop)

    return Scenario(
        recommendation=rec,
        confidence=confidence,
        predicted_nightly_eur=nightly,
        occupancy_rate_annual=round(occupancy_rate, 3),
        nights_booked_year=annual_nights,
        gross_revenue_year_eur=round(gross, 0),
        net_revenue_year_eur=round(net_year, 0),
        net_revenue_p10_eur=round(p10, 0),
        net_revenue_p90_eur=round(p90, 0),
        sale_price_eur=sale_value,
        sale_price_per_m2_eur=sale_per_m2,
        breakeven_years=round(breakeven_years, 1),
        monthly_seasonality=seasonality,
        cost_breakdown=costs,
        feature_drivers=drivers,
    )


def _resample_annual_noi(
    price_hat: float,
    occ_hat: float,
    cost_kwargs_factory,
    *,
    n: int = 1500,
    price_sigma_pct: float = 0.225,
    occ_sigma_pct: float = 0.20,
    random_state: int = 42,
) -> tuple[float, float, float]:
    """Resample annual NOI to get its P10/P50/P90 bands.

    The finance module's :func:`monte_carlo` returns *NPV* bands (discounted
    over many years). The "Net annual revenue" KPI card displays *annual
    NOI*, so we run a small local Monte Carlo using the same noise model
    as the finance MC but compute NOI directly. The ``cost_kwargs_factory``
    is called per draw with the sampled occupancy so cleaning and tourist
    tax scale correctly.
    """
    rng = np.random.default_rng(random_state)
    price_draws = rng.normal(price_hat, price_hat * price_sigma_pct, n).clip(min=0)
    occ_draws = rng.normal(occ_hat, occ_hat * occ_sigma_pct, n).clip(min=0, max=365)

    nois = np.empty(n)
    for i in range(n):
        gross = annual_gross_revenue(float(price_draws[i]), float(occ_draws[i]))
        ck = cost_kwargs_factory(float(occ_draws[i]))
        nois[i] = compute_noi(gross, **ck)["noi"]
    p10, p50, p90 = np.percentile(nois, [10, 50, 90])
    return float(p10), float(p50), float(p90)


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
