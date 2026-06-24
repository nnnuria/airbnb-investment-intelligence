"""Decision engine — real ML models + real Airbnb / Idealista data.

All price, sale, and occupancy values come from trained LightGBM models and
data-driven lookups. Finance calculations use the full Spanish STR cost stack
from airbnb_iip.finance.
"""

from __future__ import annotations

import logging
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)

from airbnb_iip.agents.governance import apply_financial_guardrails
from airbnb_iip.config import FINANCE, OCCUPANCY
from airbnb_iip.data.occupancy import estimate_occupancy
from airbnb_iip.models.price import get_city_price_predictor
from airbnb_iip.models.sale import get_sale_predictor
from airbnb_iip.finance.costs import (
    annual_gross_revenue,
    cleaning_cost_annual,
    compute_noi,
    tourist_tax_annual,
)
from airbnb_iip.finance.scenarios import break_even_horizon, monte_carlo, npv_sell

HOLDING_YEARS = 10

CITIES = ("madrid", "barcelona", "malaga")
CITY_LABELS = {"madrid": "Madrid", "barcelona": "Barcelona", "malaga": "Málaga"}
ROOM_TYPES = ("Entire home/apt", "Private room", "Hotel room", "Shared room")

# Airbnb neighbourhood_group_cleansed → Idealista district used by the sale model.
# Only entries that differ between the two naming conventions are listed.
_DISTRICT_SALE_MAP: dict[str, dict[str, str]] = {
    "madrid": {
        "Salamanca": "Barrio de Salamanca",
        "Moncloa-Aravaca": "Moncloa",
        "Fuencarral - El Pardo": "Fuencarral",
        "San Blas - Canillejas": "San Blas",
    },
    "barcelona": {
        "Horta-Guinardó": "Horta Guinardó",
    },
    "malaga": {},
}

# District lists used to populate the UI dropdown. Derived from the Airbnb data
# at runtime; this is only the static fallback if the parquet is unavailable.
_FALLBACK_DISTRICTS: dict[str, list[str]] = {
    "madrid": [
        "Arganzuela", "Barajas", "Carabanchel", "Centro", "Chamartín",
        "Chamberí", "Ciudad Lineal", "Fuencarral - El Pardo", "Hortaleza",
        "Latina", "Moncloa-Aravaca", "Moratalaz", "Puente de Vallecas",
        "Retiro", "Salamanca", "San Blas - Canillejas", "Tetuán",
        "Usera", "Villa de Vallecas", "Villaverde",
    ],
    "barcelona": [
        "Ciutat Vella", "Eixample", "Gràcia", "Horta-Guinardó",
        "Les Corts", "Nou Barris", "Sant Andreu", "Sant Martí",
        "Sants-Montjuïc", "Sarrià-Sant Gervasi",
    ],
    "malaga": [
        "Bailén - Miraflores", "Carretera de Cádiz", "Centro",
        "Ciudad Jardín", "Cruz de Humilladero", "Este",
        "Puerto de la Torre", "Teatinos",
    ],
}

# ── Listings data paths ───────────────────────────────────────────────────────

_ABT_PATH = (
    Path(__file__).resolve().parents[2]
    / "Data" / "processed" / "listings_all_cities.parquet"
)
_PRICE_HAT_PATH = (
    Path(__file__).resolve().parents[2]
    / "Data" / "processed" / "listings_with_price_hat.parquet"
)
_MIN_COMPARABLES = 5

# Amenities not in the by-city LightGBM model's RFE-selected feature set.
# Their marginal effect is captured from data-driven residuals at runtime.
_NON_MODEL_AMENITY_COLS: tuple[str, ...] = (
    "has_balcony", "has_elevator", "has_pool", "has_parking", "has_workspace",
)

# All 22 amenity flags (matches AMENITY_PATTERNS in src/airbnb_iip/features/amenities.py).
_ALL_AMENITY_FIELDS: tuple[str, ...] = (
    "has_pool", "has_gym", "has_parking", "has_hot_tub", "has_beach",
    "has_view", "has_ac", "has_elevator", "has_washer", "has_dishwasher",
    "has_workspace", "has_self_checkin", "has_pets", "has_crib",
    "has_private_entrance", "has_balcony", "has_bathtub", "has_dryer",
    "has_ev_charger", "has_outdoor_space", "has_long_term_ok",
    "has_cleaning_service",
)


# ── Property + Scenario dataclasses ──────────────────────────────────────────

@dataclass
class Property:
    """User-entered property spec — the single input the decision engine reads."""
    city: str
    district: str
    size_m2: float
    bedrooms: int
    bathrooms: float
    accommodates: int
    room_type: str
    # Amenities — all 22 from AMENITY_PATTERNS (model inputs, residual adjustments, count)
    has_ac: bool = False
    has_elevator: bool = False
    has_balcony: bool = False
    has_pool: bool = False
    has_parking: bool = False
    has_workspace: bool = False
    has_gym: bool = False
    has_hot_tub: bool = False
    has_beach: bool = False
    has_view: bool = False
    has_washer: bool = False
    has_dishwasher: bool = False
    has_self_checkin: bool = False
    has_pets: bool = False
    has_crib: bool = False
    has_private_entrance: bool = False
    has_bathtub: bool = False
    has_dryer: bool = False
    has_ev_charger: bool = False
    has_outdoor_space: bool = False
    has_long_term_ok: bool = False
    has_cleaning_service: bool = False
    extra_amenities: list = field(default_factory=list)
    nickname: str | None = None
    notes: str | None = None

    @property
    def amenity_count(self) -> int:
        """Total amenity count: flag-based + free-text extras."""
        return sum(
            1 for f in _ALL_AMENITY_FIELDS if getattr(self, f)
        ) + len(self.extra_amenities)


@dataclass
class Scenario:
    """Full output of the decision engine."""
    recommendation: Literal["airbnb", "sell", "marginal"]
    confidence: float

    predicted_nightly_eur: float
    occupancy_rate_annual: float
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
    governance: dict = field(default_factory=dict)


# ── Listings lookup (occupancy + price context) ───────────────────────────────

def _norm(s: str) -> str:
    """Lowercase, strip diacritics, normalise hyphens and whitespace."""
    s = unicodedata.normalize("NFD", str(s))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.lower().replace(" - ", "-").strip()


@lru_cache(maxsize=1)
def _load_listings():
    """Load listings columns needed for lookups. Cached process-wide."""
    import pandas as pd

    cols = [
        "city", "neighbourhood_group_cleansed", "neighbourhood_cleansed",
        "room_type", "reviews_per_month", "number_of_reviews_ltm",
        "latitude", "longitude",
    ]
    df = pd.read_parquet(_ABT_PATH, columns=cols)
    # Keep only well-operated listings for occupancy benchmarking.
    df = df[df["reviews_per_month"] >= 1.5]
    df["_district"] = df["neighbourhood_group_cleansed"].fillna(
        df["neighbourhood_cleansed"]
    )
    df["_city_norm"] = df["city"].apply(_norm)
    df["_district_norm"] = df["_district"].apply(
        lambda x: _norm(x) if x == x else ""
    )
    return df


def get_city_districts(city: str) -> list[str]:
    """Return sorted district list for the given city, derived from real data."""
    if not _ABT_PATH.exists():
        return sorted(_FALLBACK_DISTRICTS.get(city.lower(), ["Centro"]))
    try:
        df = _load_listings()
        city_norm = _norm(city)
        subset = df[df["_city_norm"] == city_norm]
        raw = subset["_district"].dropna().unique().tolist()
        districts = sorted(set(raw)) if raw else []
        return districts or sorted(_FALLBACK_DISTRICTS.get(city.lower(), ["Centro"]))
    except Exception:
        return sorted(_FALLBACK_DISTRICTS.get(city.lower(), ["Centro"]))


def reviews_per_month_from_data(prop: Property) -> float:
    """Median reviews/month from well-operated comparable listings.

    Lookup cascade (requires ≥ 5 listings at each level):
      1. city + district + room_type
      2. city + district
      3. city
    """
    if not _ABT_PATH.exists():
        return 1.5

    df = _load_listings()
    city_norm = _norm(prop.city)
    district_norm = _norm(prop.district)

    city_mask = df["_city_norm"] == city_norm
    district_mask = df["_district_norm"] == district_norm
    room_mask = df["room_type"] == prop.room_type

    for mask in (
        city_mask & district_mask & room_mask,
        city_mask & district_mask,
        city_mask,
    ):
        subset = df.loc[mask, "reviews_per_month"].dropna()
        if len(subset) >= _MIN_COMPARABLES:
            return round(float(subset.median()), 2)

    return 1.5


@lru_cache(maxsize=1)
def _load_amenity_residuals() -> dict[str, dict[str, float]]:
    """Compute per-city residual EUR uplift for amenities not in the price model.

    For each amenity A and city C:
        delta = median(price − price_hat | A=1, C) − median(price − price_hat | A=0, C)

    This measures what the amenity adds to the listing price BEYOND what the
    LightGBM model already captures through neighbourhood encoding, capacity,
    and other selected features. Returns {amenity: {city_norm: eur_delta}}.
    Falls back to an empty dict if the parquet is not available.
    """
    import pandas as pd

    if not _PRICE_HAT_PATH.exists():
        logger.warning("Price-hat parquet not found at %s — amenity residuals unavailable", _PRICE_HAT_PATH)
        return {}
    try:
        needed = ["city", "price", "price_hat"] + list(_NON_MODEL_AMENITY_COLS)
        df = pd.read_parquet(_PRICE_HAT_PATH, columns=needed)
        df = df[(df["price"] > 20) & (df["price"] < 2000)].copy()
        df["_residual"] = df["price"] - df["price_hat"]
        df["_city_norm"] = df["city"].apply(_norm)

        result: dict[str, dict[str, float]] = {}
        for amenity in _NON_MODEL_AMENITY_COLS:
            if amenity not in df.columns:
                continue
            city_map: dict[str, float] = {}
            for city_norm in df["_city_norm"].unique():
                sub = df[df["_city_norm"] == city_norm]
                med_with = sub[sub[amenity] == 1]["_residual"].median()
                med_without = sub[sub[amenity] == 0]["_residual"].median()
                if pd.notna(med_with) and pd.notna(med_without):
                    city_map[city_norm] = float(med_with - med_without)
            result[amenity] = city_map
        return result
    except Exception as exc:
        logger.warning("Failed to load amenity residuals: %s", exc)
        return {}


def _amenity_residual_adjustment(prop: Property) -> tuple[float, list[tuple[str, float]]]:
    """Return total EUR adjustment and per-amenity breakdown for non-model amenities.

    Only adds a correction when the amenity is present (flag=True); absence is
    the reference category, so no subtraction is needed.
    """
    residuals = _load_amenity_residuals()
    city_norm = _norm(prop.city)
    total = 0.0
    breakdown: list[tuple[str, float]] = []
    amenity_labels = {
        "has_balcony": "Balcony/terrace",
        "has_elevator": "Elevator",
        "has_pool": "Pool",
        "has_parking": "Parking",
        "has_workspace": "Workspace",
    }
    for amenity in _NON_MODEL_AMENITY_COLS:
        if not getattr(prop, amenity, False):
            continue
        delta = residuals.get(amenity, {}).get(city_norm, 0.0)
        if delta != 0.0:
            total += delta
            breakdown.append((amenity_labels.get(amenity, amenity), delta))
    return total, breakdown


# ── Nightly price (LightGBM price model) ─────────────────────────────────────

# Human-readable labels for the top price model features shown in the UI.
_PRICE_FEATURE_LABELS: dict[str, str] = {
    "neighbourhood_target_enc": "Neighbourhood",
    "accommodates": "Guest capacity",
    "bedrooms": "Bedrooms",
    "bathrooms_number": "Bathrooms",
    "latitude": "Location",
    "longitude": "Location",
    "has_ac": "Air conditioning",
    "has_balcony": "Balcony/terrace",
    "has_elevator": "Elevator",
    "has_pool": "Pool",
    "has_parking": "Parking",
    "has_workspace": "Workspace",
    "review_scores_rating": "Rating",
    "review_scores_cleanliness": "Cleanliness score",
    "review_scores_location": "Location score",
    "reviews_per_month": "Booking frequency",
    "minimum_nights": "Min-nights policy",
    "host_tenure_years": "Host experience",
    "instant_bookable": "Instant bookable",
    "competitive_density_500m": "Local competition",
}


def _district_context(prop: Property) -> dict:
    """Median lat/lon and modal neighbourhood_cleansed for a district."""
    if not _ABT_PATH.exists():
        return {}
    try:
        df = _load_listings()
        city_mask = df["_city_norm"] == _norm(prop.city)
        district_mask = df["_district_norm"] == _norm(prop.district)
        subset = df[city_mask & district_mask]
        if len(subset) < _MIN_COMPARABLES:
            subset = df[city_mask]
        ctx: dict = {}
        if len(subset) > 0:
            ctx["latitude"] = float(subset["latitude"].median())
            ctx["longitude"] = float(subset["longitude"].median())
            mode = subset["neighbourhood_cleansed"].dropna().mode()
            if len(mode) > 0:
                ctx["neighbourhood_cleansed"] = str(mode.iloc[0])
        return ctx
    except Exception:
        return {}


def predict_nightly_price(prop: Property) -> float:
    """Predict nightly rate (EUR) using the trained LightGBM price model.

    has_ac is a selected feature and is handled directly by the model.
    has_balcony / has_elevator / has_pool / has_parking / has_workspace are not
    in the 29 selected features; their marginal EUR contributions are computed
    from data-driven residuals (price − price_hat) in the listings parquet and
    added here.
    """
    ctx = _district_context(prop)
    spec = {
        "accommodates": prop.accommodates,
        "bedrooms": prop.bedrooms,
        "bathrooms_number": prop.bathrooms,
        "has_ac": int(prop.has_ac),
        "has_crib": int(prop.has_crib),
        "has_dishwasher": int(prop.has_dishwasher),
        "instant_bookable": 1,
        **ctx,
    }
    base = get_city_price_predictor().predict({**spec, "city": prop.city})
    adjustment, _ = _amenity_residual_adjustment(prop)
    return round(max(base + adjustment, 0.0), 2)


def _shap_drivers(prop: Property) -> list[tuple[str, float]]:
    """SHAP contributions from the price model, augmented with residual amenity effects.

    Returns list of (label, fraction) where fraction is the contribution as a
    share of total absolute SHAP magnitude.  Non-model amenity residuals are
    expressed as EUR-delta / predicted-price so they share the same scale.
    """
    import pandas as pd
    import warnings

    ctx = _district_context(prop)
    spec = {
        "accommodates": prop.accommodates,
        "bedrooms": prop.bedrooms,
        "bathrooms_number": prop.bathrooms,
        "has_ac": int(prop.has_ac),
        "has_crib": int(prop.has_crib),
        "has_dishwasher": int(prop.has_dishwasher),
        "instant_bookable": 1,
        **ctx,
    }

    # SHAP from the city-specific LightGBM model
    shap_contributions: dict[str, float] = {}
    total_abs = 1.0
    try:
        predictor = get_city_price_predictor()
        city_spec = {**spec, "city": prop.city}
        shap_entries = predictor.explain(city_spec, top_n=len(predictor.features))
        total_abs = sum(abs(e["shap_value"]) for e in shap_entries) or 1.0
        for entry in shap_entries:
            label = _PRICE_FEATURE_LABELS.get(entry["feature"])
            if label is None:
                continue
            shap_contributions[label] = (
                shap_contributions.get(label, 0.0) + entry["shap_value"] / total_abs
            )
    except Exception as exc:
        logger.warning("SHAP computation failed: %s", exc)

    # Residual amenity contributions for non-model amenities.
    # Express as eur_delta / base_price so they are a fraction of the nightly
    # rate — comparable to the SHAP fractions that sum to 1.0 across features.
    _, amenity_breakdown = _amenity_residual_adjustment(prop)
    if amenity_breakdown:
        base_price = float(get_city_price_predictor().predict({
            "city": prop.city,
            "accommodates": prop.accommodates,
            "bedrooms": prop.bedrooms,
            "bathrooms_number": prop.bathrooms,
            "has_ac": int(prop.has_ac),
            "has_crib": int(prop.has_crib),
            "has_dishwasher": int(prop.has_dishwasher),
            "instant_bookable": 1,
            **ctx,
        })) or 1.0
        for label, eur_delta in amenity_breakdown:
            shap_contributions[label] = eur_delta / base_price

    drivers = sorted(shap_contributions.items(), key=lambda x: -abs(x[1]))
    return drivers[:6]


# ── Sale price (LightGBM sale model) ─────────────────────────────────────────

def predict_sale_value(prop: Property) -> tuple[float, float]:
    """Predict sale price (EUR) using the trained Idealista sale model.

    Returns ``(total_eur, eur_per_m2)``.
    """
    city = prop.city.lower()
    idealista_district = _DISTRICT_SALE_MAP.get(city, {}).get(
        prop.district, prop.district
    )
    total = get_sale_predictor().predict({
        "city": city,
        "district": idealista_district,
        "size_m2": prop.size_m2,
        "rooms": prop.bedrooms,
        "bathrooms": prop.bathrooms,
        "has_lift": int(prop.has_elevator),
    })
    per_m2 = total / prop.size_m2 if prop.size_m2 > 0 else 0.0
    return round(total, -3), round(per_m2, 0)


# ── Main scenario engine ──────────────────────────────────────────────────────

def compute_scenario(prop: Property) -> Scenario:
    """Full Airbnb-vs-sell scenario for one property.

    1. LightGBM nightly price model.
    2. Data-driven occupancy via the SF model on real comparable listings.
    3. Full Spanish STR cost stack (airbnb_iip.finance.costs).
    4. LightGBM sale price model.
    5. Break-even horizon and Monte Carlo recommendation.
    6. Real SHAP feature contributions for the explanation chart.
    """
    import warnings
    warnings.filterwarnings("ignore")

    nightly = predict_nightly_price(prop)
    rpm = reviews_per_month_from_data(prop)

    occ = estimate_occupancy(rpm)
    monthly_nights = occ["nights_booked_per_month"]
    annual_nights = round(monthly_nights * 12)
    occupancy_rate = monthly_nights / 30

    sale_value, sale_per_m2 = predict_sale_value(prop)

    def _make_cost_kwargs(occ_nights: float) -> dict:
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

    sell = npv_sell(
        sale_price=sale_value,
        purchase_price=sale_value * 0.70,
        purchase_costs=sale_value * 0.04,
        documented_improvements=sale_value * 0.02,
    )
    sell_net = sell["net_proceeds"]

    # Simple payback: how many years of net Airbnb income equals today's net
    # sale proceeds?  This is the metric investors intuitively understand.
    # The NPV-based break_even_horizon produces T=1 when NOI yield > discount
    # rate (mathematically correct but misleading in the UI), so we use the
    # simpler sell_net / NOI ratio here. The recommendation is still driven by
    # the Monte Carlo NPV comparison below.
    if net_year > 0:
        breakeven_years = round(sell_net / net_year, 1)
    else:
        breakeven_years = 99.0

    p10, _p50, p90 = _resample_annual_noi(
        nightly, annual_nights, _make_cost_kwargs, n=1500,
    )

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

    drivers = _shap_drivers(prop)
    seasonality = _city_seasonality(prop.city.lower())

    # Governance: bounds-check the financial outputs (negative revenue,
    # implausible yield, inverted P10/P90, marginal recommendation) before
    # they reach the UI. npv_sell_eur uses sell_net (post-CGT/fees net
    # proceeds), matching the denominator the gross-yield guardrail expects.
    governance_report = apply_financial_guardrails({
        "recommendation": rec,
        "nightly_price_eur": nightly,
        "annual_gross_eur": gross,
        "annual_net_eur": net_year,
        "npv_sell_eur": sell_net,
        "p10_eur": p10,
        "p90_eur": p90,
    })["governance"]

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
        governance=governance_report,
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
    """P10/P50/P90 annual NOI via local Monte Carlo."""
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


# ── Seasonality ───────────────────────────────────────────────────────────────

def _city_seasonality(city: str) -> list[float]:
    """Monthly demand multipliers calibrated to each city's STR pattern (mean=1.0)."""
    patterns = {
        "madrid": [
            0.85, 0.88, 0.95, 1.05, 1.10, 1.05,
            1.00, 0.92, 1.08, 1.10, 0.95, 1.07,
        ],
        "barcelona": [
            0.82, 0.85, 0.95, 1.10, 1.20, 1.18,
            1.15, 1.08, 1.10, 1.05, 0.85, 0.85,
        ],
        "malaga": [
            0.85, 0.88, 1.00, 1.10, 1.18, 1.25,
            1.30, 1.30, 1.15, 1.00, 0.85, 0.92,
        ],
    }
    return patterns.get(city, [1.0] * 12)


# ── Optimisation ──────────────────────────────────────────────────────────────

@dataclass
class Improvement:
    name: str
    investment_eur: float
    monthly_revenue_uplift_eur: float
    payback_months: float
    confidence: Literal["high", "medium", "low"]
    rationale: str


def suggest_improvements(prop: Property, scen: Scenario) -> list[Improvement]:
    """Improvement ideas ranked by payback period."""
    items: list[Improvement] = []
    rev_month = scen.net_revenue_year_eur / 12

    if not prop.has_ac:
        uplift = rev_month * 0.06
        items.append(Improvement(
            "Install air conditioning", 1800,
            round(uplift, 0),
            round(1800 / uplift, 1) if uplift > 0 else 99,
            "high",
            "AC is among the top three amenities by booking lift in warm-climate "
            "markets; comparable listings without AC lose 5–8% of bookings in peak months.",
        ))
    if not prop.has_workspace:
        uplift = rev_month * 0.04
        items.append(Improvement(
            "Add a dedicated workspace", 350,
            round(uplift, 0),
            round(350 / uplift, 1) if uplift > 0 else 99,
            "high",
            "Workspace amenity unlocks the long-stay segment (28+ nights) and "
            "tends to lift weekday occupancy.",
        ))
    if not prop.has_balcony:
        uplift = rev_month * 0.03
        items.append(Improvement(
            "Highlight outdoor space in listing", 120,
            round(uplift, 0),
            round(120 / uplift, 1) if uplift > 0 else 99,
            "medium",
            "If you have any patio, balcony, or terrace, ensuring it shows in "
            "photos is a cheap revenue lever.",
        ))
    items.append(Improvement(
        "Professional photography refresh", 250,
        round(rev_month * 0.05, 0),
        round(250 / max(rev_month * 0.05, 1), 1),
        "high",
        "Improving listing photos is the single highest-ROI change short of a renovation.",
    ))
    items.append(Improvement(
        "Adopt dynamic pricing (PriceLabs / Wheelhouse)", 20 * 12,
        round(rev_month * 0.07, 0),
        round((20 * 12) / max(rev_month * 0.07, 1), 1),
        "medium",
        "Algorithmic pricing typically lifts revenue 5–10% by capturing event "
        "spikes and softening shoulder months.",
    ))
    items.sort(key=lambda x: x.payback_months)
    return items


# ── Chat ──────────────────────────────────────────────────────────────────────

CHAT_INTROS = [
    "Hi, I'm your investment co-pilot. Tell me about a property — city, "
    "district, size, bedrooms — or ask anything about an existing analysis.",
    "Hello. I can compare Airbnb vs sale for any property in Madrid, "
    "Barcelona, or Málaga. What would you like to explore?",
]


def chat_reply(user_msg: str, scen: Scenario | None) -> str:
    """Pattern-matched answers — used until the LLM agent layer is wired in."""
    import random
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
                "With break-even above 20 years, a sale is usually preferred."
                if scen.breakeven_years >= 20 else
                "This sits in the marginal zone (12–20 years). Other factors — "
                "tax position, capital needs, regulation risk — typically tip the call."
            )
        )
    if any(k in msg for k in ("revenue", "income", "earn", "make")):
        return (
            f"Estimated net annual revenue is **€{scen.net_revenue_year_eur:,.0f}** "
            f"(P10–P90: €{scen.net_revenue_p10_eur:,.0f}–€{scen.net_revenue_p90_eur:,.0f}). "
            f"Gross is €{scen.gross_revenue_year_eur:,.0f}; the gap is costs — "
            "platform fee, management, cleaning, and tax."
        )
    if any(k in msg for k in ("occupancy", "booked", "vacancy")):
        return (
            f"Projected occupancy is **{scen.occupancy_rate_annual * 100:.0f}%** "
            f"({scen.nights_booked_year} nights/year). This uses the San Francisco "
            "model — inferred from review activity of comparable well-operated listings."
        )
    if any(k in msg for k in ("rate", "price", "nightly", "night")):
        return (
            f"Projected nightly rate is **€{scen.predicted_nightly_eur:,.0f}/night**, "
            "estimated by a LightGBM model trained on real Airbnb listings across "
            "Madrid, Barcelona, and Málaga."
        )
    if any(k in msg for k in ("regulation", "license", "licence", "legal", "permit")):
        return (
            "Each city has its own short-term-rental regime. Madrid restricts STR "
            "in many residential blocks; Barcelona requires an HUT licence; Málaga "
            "enforces tourist registration with the Junta de Andalucía. Always verify "
            "against current official municipal sources before listing."
        )
    if any(k in msg for k in ("improve", "optimise", "optimize", "uplift", "amenity")):
        return (
            "See **Optimisation** in the sidebar — improvements are ranked by ROI. "
            "Top moves: professional photos, AC if absent, dedicated workspace, "
            "and dynamic pricing."
        )
    if any(k in msg for k in ("why", "explain", "how")):
        return (
            f"The recommendation is **{scen.recommendation.upper()}** "
            f"(confidence ~{scen.confidence*100:.0f}%). The strongest price drivers "
            "come from the LightGBM model's SHAP values: "
            + ", ".join(name for name, _ in scen.feature_drivers[:3])
            + f". Break-even sits at {scen.breakeven_years:.1f} years."
        )
    return (
        "I can answer questions about projected revenue, occupancy, the "
        "sale-vs-Airbnb comparison, costs, regulations, or recommended "
        "improvements. Try \"why this recommendation?\" or \"what about regulations?\"."
    )


__all__ = [
    "CITIES", "CITY_LABELS", "ROOM_TYPES",
    "Property", "Scenario", "Improvement",
    "get_city_districts",
    "compute_scenario", "predict_nightly_price", "predict_sale_value",
    "reviews_per_month_from_data", "suggest_improvements", "chat_reply",
]
