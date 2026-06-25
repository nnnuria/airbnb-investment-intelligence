"""Pydantic request/response models — the API contract.

These schemas are the contract the Streamlit UI and the agent layer code
against. The two finance endpoints (``/estimate_revenue``, ``/airbnb_vs_sell``)
and ``/optimise`` return **flagged stubs** today (``_stub: true``) that already
match their final response shape, so downstream work can integrate now and the
real services swap in without schema churn.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

CITY_PATTERN = r"(?i)^(madrid|barcelona|m[aá]laga)$"


# ── /predict_price ─────────────────────────────────────────────────────────────

class PredictPriceRequest(BaseModel):
    """Property spec for nightly-price prediction.

    Field names mirror the model's own columns. Every field is optional —
    missing ones are imputed with training medians. ``extra="allow"`` lets you
    pass any of the model's 29 features directly if you have them.
    """

    model_config = ConfigDict(extra="allow")

    city: Optional[str] = Field(default=None, examples=["Madrid"])
    property_type_std: str = Field(default="Entire place", examples=["Entire place"])
    neighbourhood_cleansed: Optional[str] = Field(default=None, examples=["Salamanca"])

    accommodates: int = Field(default=2, ge=1, le=20)
    bedrooms: Optional[float] = Field(default=None, ge=0)
    beds: Optional[float] = Field(default=None, ge=0)
    bathrooms_number: Optional[float] = Field(default=None, ge=0)

    latitude: Optional[float] = None
    longitude: Optional[float] = None
    minimum_nights: Optional[int] = Field(default=None, ge=1)

    reviews_per_month: Optional[float] = Field(default=None, ge=0)
    number_of_reviews: Optional[int] = Field(default=None, ge=0)
    review_scores_rating: Optional[float] = None
    review_scores_cleanliness: Optional[float] = None
    review_scores_location: Optional[float] = None
    review_scores_value: Optional[float] = None
    review_scores_accuracy: Optional[float] = None

    host_response_time: Optional[str] = Field(default=None, examples=["within an hour"])
    host_response_rate: Optional[float] = Field(default=None, ge=0, le=1)
    host_acceptance_rate: Optional[float] = Field(default=None, ge=0, le=1)
    host_tenure_years: Optional[float] = Field(default=None, ge=0)
    host_identity_verified: Optional[bool] = None

    instant_bookable: Optional[bool] = None
    competitive_density_500m: Optional[float] = Field(default=None, ge=0)
    description_length: Optional[int] = Field(default=None, ge=0)

    has_ac: Optional[bool] = None
    has_dishwasher: Optional[bool] = None
    has_crib: Optional[bool] = None


class PredictPriceResponse(BaseModel):
    price_per_night: float = Field(examples=[148.0])
    currency: str = "EUR"
    city: Optional[str] = None
    model: str = "LightGBM"


# ── /explain_price ───────────────────────────────────────────────────────────

class FeatureContribution(BaseModel):
    feature: str = Field(examples=["neighbourhood_target_enc"])
    shap_value: float = Field(
        examples=[0.1066],
        description="SHAP value in the model's own output space (log1p(price)).",
    )
    direction: str = Field(examples=["increases"], description='"increases" | "decreases"')


class ExplainPriceResponse(BaseModel):
    price_per_night: float = Field(examples=[148.0])
    drivers: list[FeatureContribution]
    model: str = "LightGBM"


# ── /estimate_occupancy ─────────────────────────────────────────────────────────

class EstimateOccupancyRequest(BaseModel):
    """Inputs for the San Francisco occupancy estimator.

    Assumption overrides default to ``config.yaml`` values when omitted.
    """

    reviews_per_month: float = Field(ge=0, examples=[1.5])
    review_rate: Optional[float] = Field(default=None, gt=0, le=1)
    avg_length_of_stay: Optional[float] = Field(default=None, gt=0)
    max_occupancy: Optional[float] = Field(default=None, gt=0, le=1)


class EstimateOccupancyResponse(BaseModel):
    occupancy_rate: float = Field(examples=[0.30], description="Fraction in [0, max_occupancy]")
    nights_booked_per_month: float
    estimated_nights_per_year: int
    assumptions: dict


# ── /estimate_revenue (stub) ─────────────────────────────────────────────────────

class EstimateRevenueRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    price_per_night: Optional[float] = Field(default=None, ge=0)
    occupancy_rate: Optional[float] = Field(default=None, ge=0, le=1)
    city: Optional[str] = None


class EstimateRevenueResponse(BaseModel):
    annual_gross_eur: float
    annual_net_eur: float
    p10_eur: float
    p50_eur: float
    p90_eur: float
    stub: bool = Field(default=True, alias="_stub")
    note: str = "Stub — wire to airbnb_iip.finance.costs when merged."


# ── /airbnb_vs_sell (stub) ───────────────────────────────────────────────────────

class AirbnbVsSellRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    city: Optional[str] = None
    neighbourhood_cleansed: Optional[str] = None
    sq_m: Optional[float] = Field(default=None, gt=0)
    holding_years: int = Field(default=10, ge=1, le=30)
    discount_rate: float = Field(default=0.07, gt=0, lt=1)


class AirbnbVsSellResponse(BaseModel):
    recommendation: str = Field(examples=["Airbnb"], description='"Airbnb" | "Sell"')
    npv_airbnb_p50_eur: float
    npv_sell_eur: float
    break_even_years: Optional[int]
    confidence: dict
    disclaimer: str = "Indicative only — not financial advice."
    stub: bool = Field(default=True, alias="_stub")
    note: str = "Stub — wire to airbnb_iip.finance.scenarios + sale-value service."


# ── /optimise (live) ─────────────────────────────────────────────────────────────

class OptimiseRequest(BaseModel):
    """Property spec for the revenue-optimisation plan.

    ``extra="allow"`` carries the current amenity flags (``has_ac`` …, truthy =
    already installed) and an optional ``projected_annual_nights`` override.
    """

    model_config = ConfigDict(extra="allow")

    city: Optional[str] = None
    district: Optional[str] = None
    neighbourhood_cleansed: Optional[str] = None
    room_type: Optional[str] = None
    accommodates: Optional[int] = None
    bedrooms: Optional[int] = None
    bathrooms: Optional[float] = None


class OptimiseAction(BaseModel):
    action: str
    annual_uplift_eur: float
    investment_eur: float
    payback_months: float
    confidence: str            # high | medium | low
    method: str                # counterfactual | residual
    lift: Optional[float] = None
    rationale: str


class OptimiseResponse(BaseModel):
    actions: list[OptimiseAction]
    city: str
    district: str
    room_type: str
    peer_n: int
    peer_median_revenue_eur: float
    peer_target_revenue_eur: float
    gap_to_top_quartile_eur: float
    projected_annual_nights: int
    disclaimer: str = "Indicative only — uplift estimates are modelled, not guaranteed."
    stub: bool = Field(default=False, alias="_stub")
    note: str = "Live — airbnb_iip.agents.optimisation (counterfactual + residual + Apriori)."


# ── /scenario ────────────────────────────────────────────────────────────────────
# Full Airbnb-vs-sell decision in one call. Mirrors the engine's Property /
# Scenario dataclasses (airbnb_iip.decision.engine) field-for-field so the
# Streamlit UI and agents share one source of truth instead of a second engine.

class ScenarioRequest(BaseModel):
    """Property spec — the single input the decision engine reads.

    Field names match ``airbnb_iip.decision.engine.Property``. ``extra="allow"``
    carries the full amenity set (Property has 22 ``has_*`` flags + free-text
    ``extra_amenities``) without this schema having to track every one; the
    router filters to Property's actual fields before constructing it.
    """

    model_config = ConfigDict(extra="allow")

    city: str = Field(examples=["madrid"])
    district: str = Field(examples=["Salamanca"])
    size_m2: float = Field(gt=0, examples=[85])
    bedrooms: int = Field(ge=0, examples=[2])
    bathrooms: float = Field(ge=0, examples=[1.0])
    accommodates: int = Field(ge=1, examples=[4])
    room_type: str = Field(default="Entire home/apt", examples=["Entire home/apt"])

    # Decision option (not a Property field; the router pulls it out separately).
    # True = professionally managed; False = self-managed (drops the mgmt fee).
    managed: bool = Field(default=True, examples=[True])

    has_balcony: bool = False
    has_ac: bool = False
    has_elevator: bool = False
    has_pool: bool = False
    has_parking: bool = False
    has_workspace: bool = False

    nickname: Optional[str] = None
    notes: Optional[str] = None


class ScenarioResponse(BaseModel):
    """Full decision-engine output. Mirrors ``engine.Scenario`` field-for-field
    so the client can rebuild the dataclass with ``Scenario(**response)``."""

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

    npv_airbnb_p50_eur: float = 0.0
    npv_sell_eur: float = 0.0
    p_airbnb_gt_sell: float = 0.0

    monthly_seasonality: list[float] = Field(default_factory=list)
    # (label, eur) cost lines and (label, fraction) SHAP drivers; tuples
    # serialise to JSON arrays and the client converts them back to tuples.
    cost_breakdown: list[tuple[str, float]] = Field(default_factory=list)
    feature_drivers: list[tuple[str, float]] = Field(default_factory=list)
    governance: dict = Field(default_factory=dict)


# ── /comparables ─────────────────────────────────────────────────────────────────

class ComparablesRequest(BaseModel):
    """Property spec for the KNN comparables agent. ``extra="allow"`` so any of
    the agent's numeric features (accommodates/bedrooms/…) can be passed."""

    model_config = ConfigDict(extra="allow")

    city: str = Field(examples=["madrid"])
    district: Optional[str] = Field(default=None, examples=["Salamanca"])
    neighbourhood_cleansed: Optional[str] = None
    segment: Optional[str] = None
    room_type: Optional[str] = Field(default=None, examples=["Entire home/apt"])
    k: int = Field(default=5, ge=1, le=25)


class ComparablesResponse(BaseModel):
    comparables: list[dict] = Field(default_factory=list)
    filters_applied: dict = Field(default_factory=dict)
    benchmark: dict = Field(default_factory=dict)


# ── /regulatory_risk ─────────────────────────────────────────────────────────────

class RegulatoryRequest(BaseModel):
    city: str = Field(examples=["barcelona"])
    neighbourhood: Optional[str] = Field(default=None, examples=["Eixample"])
    question: Optional[str] = Field(
        default=None,
        description="Free-form regulatory question. If omitted, a default "
                    "'can I start a new STR here?' risk check is run.",
    )


class RegulatoryResponse(BaseModel):
    risk_flag: str = Field(examples=["HIGH"], description='"HIGH" | "MEDIUM" | "LOW" | "UNKNOWN"')
    reason: str
    sources: list[str] = Field(default_factory=list)
    city: Optional[str] = None
    governance: dict = Field(default_factory=dict)
    disclaimer: str = "Indicative only — not legal advice."


# ── /chat ────────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(examples=["Why this recommendation?"])
    property: Optional[dict] = Field(
        default=None, description="Current Property-shaped context (city, district, …)."
    )
    scenario: Optional[dict] = Field(
        default=None, description="Current /scenario response for the active analysis."
    )
    use_llm: bool = Field(
        default=True,
        description="False forces the deterministic (LLM-free) answer path.",
    )


class ChatResponse(BaseModel):
    answer: str
    intent: str = Field(examples=["decision"])
    sources: list = Field(default_factory=list)
    meta: dict = Field(default_factory=dict)
    disclaimer: str = "Indicative only — not financial advice."


# ── health ─────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
    price_model_loaded: bool
