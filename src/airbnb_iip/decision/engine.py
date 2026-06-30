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
from airbnb_iip.config import (
    AIRBNB_SETUP_COST_EUR_MIN,
    BASURAS_EUR_BY_CITY,
    BASURAS_EUR_DEFAULT,
    DISCOUNT_RATE_DEFAULT,
    FINANCE,
    HOLDING_YEARS_DEFAULT,
    IRPF_BRACKETS,
    NOI_GROWTH_RATE_DEFAULT,
    OCCUPANCY,
    PROPERTY_APPRECIATION_RATE_DEFAULT,
    REGULATORY_SHOCK_PROB_BY_CITY,
    REGULATORY_SHOCK_PROB_DEFAULT,
)
from airbnb_iip.data.occupancy import DAYS_PER_YEAR
from airbnb_iip.models.occupancy import get_occupancy_predictor
from airbnb_iip.models.price import get_city_price_predictor
from airbnb_iip.models.sale import get_sale_predictor
from airbnb_iip.finance.costs import (
    annual_gross_revenue,
    cleaning_cost_annual,
    compute_noi,
    tourist_tax_annual,
)
from airbnb_iip.finance.scenarios import (
    break_even_horizon,
    irr_airbnb,
    monte_carlo,
    npv_airbnb,
    npv_sell,
)

HOLDING_YEARS = HOLDING_YEARS_DEFAULT  # kept for any external references

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

# engine.py lives at src/airbnb_iip/decision/ → parents[3] is the repo root,
# matching how models/price.py and models/sale.py resolve Data/.
_ABT_PATH = (
    Path(__file__).resolve().parents[3]
    / "Data" / "processed" / "listings_all_cities.parquet"
)
_PRICE_HAT_PATH = (
    Path(__file__).resolve().parents[3]
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

    # NPV view of the same decision (exposed for the agent narrative / chat).
    # Defaulted so older saved records deserialise without these keys.
    npv_airbnb_p50_eur: float = 0.0
    npv_sell_eur: float = 0.0
    p_airbnb_gt_sell: float = 0.0

    # Extended financial metrics
    irr_airbnb_pct: float | None = None          # IRR as a fraction (e.g. 0.082 = 8.2%)
    npv_airbnb_pretax_p50_eur: float = 0.0       # NPV ignoring income tax
    npv_sell_pretax_eur: float = 0.0             # sell proceeds before CGT
    ibi_eur_used: float = 0.0                    # IBI actually applied
    setup_cost_eur_used: float = 0.0             # setup cost actually applied

    # IbiAgent provenance
    ibi_method: str = "estimated"                # "cadastral" | "estimated" | "manual"
    ibi_explanation: str = ""                    # one-sentence derivation from IbiAgent
    basuras_eur_used: float = 0.0               # waste-tax amount applied

    # Investment horizon & discount rate used (echoed for UI display / sensitivity)
    npv_advantage_eur: float = 0.0              # npv_airbnb_p50_eur − npv_sell_eur
    holding_years: int = 10
    discount_rate: float = 0.07

    # Annual regulatory-shock probability folded into the Monte Carlo for this
    # city (0 if none modelled). Surfaced so the UI/agent can explain why a
    # high-shock city (Barcelona) leans toward selling.
    regulatory_shock_prob: float = 0.0

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


# ── Occupancy (LightGBM calendar model) ───────────────────────────────────────

def _occupancy_spec(prop: Property, nightly_price: float) -> dict:
    """Build the OccupancyPredictor input from a Property + its nightly price.

    Mirrors :func:`predict_nightly_price`: location (lat/lon + modal
    neighbourhood) comes from the district context, capacity/room-type/amenities
    from the property itself, and ``price`` is the model-predicted nightly rate
    (a demand signal). Features the spec doesn't carry (e.g. minimum_nights,
    property_type) are imputed with training medians inside the predictor.
    """
    ctx = _district_context(prop)  # latitude, longitude, neighbourhood_cleansed
    return {
        "city": prop.city,
        "room_type": prop.room_type,
        "accommodates": prop.accommodates,
        "bedrooms": prop.bedrooms,
        "bathrooms_number": prop.bathrooms,
        "price": nightly_price,
        "instant_bookable": 1,
        **{f: int(getattr(prop, f, False)) for f in _ALL_AMENITY_FIELDS},
        **ctx,
    }


def predict_occupancy(prop: Property, *, nightly_price: float | None = None) -> float:
    """Predict annual occupancy rate (fraction in [0, 1]) via the calendar model.

    Uses :class:`airbnb_iip.models.occupancy.OccupancyPredictor` — a LightGBM
    model trained on Inside Airbnb calendar availability (``available == 'f'`` ⇒
    booked). This replaces the retired San Francisco review-based estimator.
    Pass ``nightly_price`` to reuse an already-computed price (avoids predicting
    it twice); omit it and the nightly price is predicted internally.
    """
    if nightly_price is None:
        nightly_price = predict_nightly_price(prop)
    return get_occupancy_predictor().predict(_occupancy_spec(prop, nightly_price))


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
        # Text-mined amenity flags the sale model now learns from. Mapped from
        # the property's Airbnb-style toggles where one exists; the three with no
        # UI equivalent (terrace / storage / heating) stay at the model's
        # "not advertised" baseline (0). has_garden ← outdoor_space, since the
        # Airbnb flag bundles garden/backyard/courtyard.
        "has_pool": int(prop.has_pool),
        "has_parking": int(prop.has_parking),
        "has_ac": int(prop.has_ac),
        "has_balcony": int(prop.has_balcony),
        "has_garden": int(prop.has_outdoor_space),
    })
    per_m2 = total / prop.size_m2 if prop.size_m2 > 0 else 0.0
    return round(total, -3), round(per_m2, 0)


# ── Main scenario engine ──────────────────────────────────────────────────────

def compute_scenario(
    prop: Property,
    *,
    managed: bool = True,
    ibi_eur: float | None = None,
    cadastral_value: float | None = None,
    basuras_eur: float | None = None,
    setup_cost_eur: float = AIRBNB_SETUP_COST_EUR_MIN,
    noi_growth_rate: float = NOI_GROWTH_RATE_DEFAULT,
    property_appreciation_rate: float = PROPERTY_APPRECIATION_RATE_DEFAULT,
    include_income_tax: bool = True,
    purchase_price: float | None = None,
    holding_years: int = HOLDING_YEARS_DEFAULT,
    discount_rate: float = DISCOUNT_RATE_DEFAULT,
) -> Scenario:
    """Full Airbnb-vs-sell scenario for one property.

    1. LightGBM nightly price model.
    2. LightGBM occupancy model trained on Inside Airbnb calendar availability.
    3. Full Spanish STR cost stack (airbnb_iip.finance.costs).
    4. LightGBM sale price model.
    5. Break-even horizon and Monte Carlo recommendation.
    6. Real SHAP feature contributions for the explanation chart.

    ``managed`` toggles professional management: when ``False`` (self-managed)
    the 20% management fee is dropped from every downstream figure — net
    revenue, P10/P90, payback, NPV, the recommendation and the cost breakdown.

    ``ibi_eur``: annual IBI property tax. Defaults to city-specific estimate
    from config if not provided. ``setup_cost_eur``: one-time Airbnb setup
    cost (photography, staging, supplies). ``noi_growth_rate``: annual CPI/HPI
    growth applied to NOI across the holding period. ``property_appreciation_rate``:
    annual property price growth applied to the terminal sale value.
    ``include_income_tax``: when False, income tax is excluded from NOI so the
    caller can show the pre-tax comparison alongside the after-tax figure.
    ``purchase_price``: used for CGT calculation on sale; defaults to 70% of
    current sale value if not supplied.
    """
    import warnings
    warnings.filterwarnings("ignore")

    nightly = predict_nightly_price(prop)

    # Occupancy: LightGBM model trained on Inside Airbnb calendar availability
    # (available == 'f' ⇒ booked). Replaces the retired SF review-based estimate.
    occupancy_rate = predict_occupancy(prop, nightly_price=nightly)
    annual_nights = int(round(occupancy_rate * DAYS_PER_YEAR))

    sale_value, sale_per_m2 = predict_sale_value(prop)

    management_rate = FINANCE["management_pct"] if managed else 0.0

    city_key = prop.city.strip().lower()

    # IbiAgent: resolve IBI using the sourced municipal rate × cadastral value.
    # Falls back to estimating cadastral value from market price when not given.
    from airbnb_iip.agents.property_tax import IbiAgent
    _ibi_agent = IbiAgent(use_llm=False)  # deterministic during scenario; narration is separate
    ibi_estimate = _ibi_agent.resolve(
        city_key,
        market_value=sale_value,
        cadastral_value=cadastral_value,
        ibi_eur_override=ibi_eur,
    )
    ibi_resolved = ibi_estimate.ibi_eur

    # Basuras (waste tax): city default unless the caller supplied an override
    basuras_resolved = (
        basuras_eur if basuras_eur is not None
        else BASURAS_EUR_BY_CITY.get(city_key, BASURAS_EUR_DEFAULT)
    )

    def _make_cost_kwargs(occ_nights: float, *, with_tax: bool = True) -> dict:
        return dict(
            property_value=sale_value,
            cleaning_cost_eur=cleaning_cost_annual(
                occ_nights,
                avg_length_of_stay=OCCUPANCY["avg_length_of_stay"],
            ),
            tourist_tax_eur=tourist_tax_annual(occ_nights, city=prop.city),
            management_fee_rate=management_rate,
            ibi_eur=ibi_resolved,
            basuras_eur=basuras_resolved,
            irpf_brackets=IRPF_BRACKETS if with_tax else None,
            include_income_tax=with_tax and include_income_tax,
        )

    gross = annual_gross_revenue(nightly, annual_nights)
    cost_kwargs = _make_cost_kwargs(annual_nights)
    noi = compute_noi(gross, **cost_kwargs)
    net_year = noi["noi"]

    # Pre-tax cost kwargs (no income tax) for the pre-tax NPV comparison
    cost_kwargs_pretax = _make_cost_kwargs(annual_nights, with_tax=False)

    # CGT basis — use provided purchase price or estimate as 70% of current value
    purchase_price_basis = purchase_price if purchase_price is not None else sale_value * 0.70
    sell = npv_sell(
        sale_price=sale_value,
        purchase_price=purchase_price_basis,
        purchase_costs=sale_value * 0.04,
        documented_improvements=sale_value * 0.02,
    )
    sell_net = sell["net_proceeds"]
    # Pre-tax sell: sale price minus transaction costs only, no CGT
    sell_pretax = sale_value * (1 - FINANCE["agent_commission_pct"] - FINANCE["notary_registry_pct"])

    # Appreciated terminal value at end of holding period
    terminal_price = sale_value * (1 + property_appreciation_rate) ** holding_years
    terminal_sell = npv_sell(
        sale_price=terminal_price,
        purchase_price=purchase_price_basis,
        purchase_costs=sale_value * 0.04,
        documented_improvements=sale_value * 0.02,
    )
    terminal_value_net = terminal_sell["net_proceeds"]

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

    # Per-city annual regulatory-shock probability (licence loss / moratorium /
    # saturation cap). Truncates the Airbnb NOI stream in the Monte Carlo from
    # the shock year onward — Barcelona's 0.20 is the dominant driver of its
    # sell-leaning recommendations (framework doc §7.2).
    shock_prob = REGULATORY_SHOCK_PROB_BY_CITY.get(
        city_key, REGULATORY_SHOCK_PROB_DEFAULT
    )

    mc = monte_carlo(
        price_hat=nightly,
        occ_hat=annual_nights,
        years=holding_years,
        npv_sell_value=sell_net,
        cost_kwargs=cost_kwargs,
        terminal_value_net=terminal_value_net,
        setup_cost=setup_cost_eur,
        noi_growth_rate=noi_growth_rate,
        discount_rate=discount_rate,
        shock_prob=shock_prob,
        random_state=42,
        n_simulations=1500,
    )

    # Pre-tax Monte Carlo (no income tax in NOI)
    mc_pretax = monte_carlo(
        price_hat=nightly,
        occ_hat=annual_nights,
        years=holding_years,
        npv_sell_value=sell_pretax,
        cost_kwargs=cost_kwargs_pretax,
        terminal_value_net=terminal_value_net,
        setup_cost=setup_cost_eur,
        noi_growth_rate=noi_growth_rate,
        discount_rate=discount_rate,
        shock_prob=shock_prob,
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

    # IRR on the after-tax NOI stream
    noi_series = [
        net_year * (1 + noi_growth_rate) ** t
        for t in range(holding_years)
    ]
    # IRR uses the forgone sale proceeds as the "investment" (opportunity cost
    # of not selling today) plus the Airbnb setup outlay. This gives the effective
    # annualised return on the capital that would otherwise have been received
    # from selling — the most meaningful metric for the hold-vs-sell decision.
    irr_value = irr_airbnb(
        noi_series,
        setup_cost=sell_net + setup_cost_eur,
        terminal_value_net=terminal_value_net,
    )

    cost_lines = [
        ("Platform fee", noi["platform_fee"]),
        ("Cleaning", noi["cleaning_cost"]),
        ("Tourist tax", noi["tourist_tax"]),
        ("IBI (property tax)", noi["ibi"]),
        ("Waste tax (basuras)", noi["basuras"]),
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
        npv_airbnb_p50_eur=round(mc["p50"], 0),
        npv_sell_eur=round(sell_net, 0),
        p_airbnb_gt_sell=round(p_airbnb, 3),
        irr_airbnb_pct=round(irr_value, 4) if irr_value is not None else None,
        npv_airbnb_pretax_p50_eur=round(mc_pretax["p50"], 0),
        npv_sell_pretax_eur=round(sell_pretax, 0),
        ibi_eur_used=round(ibi_resolved, 0),
        ibi_method=ibi_estimate.method,
        ibi_explanation=ibi_estimate.explanation,
        basuras_eur_used=round(basuras_resolved, 0),
        setup_cost_eur_used=round(setup_cost_eur, 0),
        npv_advantage_eur=round(mc["p50"] - sell_net, 0),
        holding_years=holding_years,
        discount_rate=discount_rate,
        regulatory_shock_prob=shock_prob,
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
# The revenue-optimisation flow now lives in airbnb_iip.agents.optimisation
# (per-amenity counterfactual / residual uplift + Apriori rules + peer-gap), and
# is consumed directly by pages/4_Optimisation.py and the /optimise API endpoint.


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
            f"({scen.nights_booked_year} nights/year). This comes from a LightGBM model "
            "trained on Inside Airbnb calendar availability (booked nights) for comparable "
            "listings — driven by location, price, capacity, amenities and min-nights policy."
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
    "Property", "Scenario",
    "get_city_districts",
    "compute_scenario", "predict_nightly_price", "predict_occupancy",
    "predict_sale_value", "reviews_per_month_from_data", "chat_reply",
]
