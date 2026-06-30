"""Pydantic request/response models — the API contract.

These schemas are the contract the React SPA and the agent layer code against.
All eleven endpoints are live: ``/estimate_revenue`` and ``/airbnb_vs_sell`` are
backed by ``airbnb_iip.finance`` + the ML models, and ``/optimise`` by
``airbnb_iip.agents.optimisation``.
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
    """Property spec for the calendar-based LightGBM occupancy model.

    Field names mirror the engine's ``Property``; ``extra="allow"`` carries the
    amenity flags (``has_ac`` …). Occupancy is **predicted from the calendar
    model**, not derived from review velocity. ``price_per_night`` is optional —
    the nightly-price model fills it in when omitted.
    """

    model_config = ConfigDict(extra="allow")

    city: str = Field(examples=["madrid"])
    district: Optional[str] = Field(default=None, examples=["Salamanca"])
    neighbourhood_cleansed: Optional[str] = None
    room_type: str = Field(default="Entire home/apt", examples=["Entire home/apt"])
    accommodates: int = Field(default=2, ge=1, le=20)
    bedrooms: int = Field(default=1, ge=0)
    bathrooms: float = Field(default=1.0, ge=0)
    size_m2: float = Field(default=70.0, gt=0)
    price_per_night: Optional[float] = Field(
        default=None, ge=0,
        description="Nightly rate (demand signal). Predicted by the price model when omitted.",
    )


class EstimateOccupancyResponse(BaseModel):
    occupancy_rate: float = Field(examples=[0.42], description="Annual occupancy fraction in [0, 1]")
    estimated_nights_per_year: int
    predicted_nightly_eur: float
    model: str = "LightGBM (calendar)"
    basis: str = "Inside Airbnb calendar availability — available == 'f' counted as a booked night"


# ── /estimate_revenue ─────────────────────────────────────────────────────────

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


# ── /airbnb_vs_sell ───────────────────────────────────────────────────────────

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


# ── /optimise ─────────────────────────────────────────────────────────────────

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

    # Financial assumption overrides (all optional; engine uses city defaults when omitted)
    ibi_eur: Optional[float] = Field(default=None, ge=0, description="Annual IBI property tax (€). Overrides IbiAgent estimate when provided.")
    cadastral_value: Optional[float] = Field(default=None, ge=0, description="Cadastral value (valor catastral) for exact IBI computation. IbiAgent estimates it from market price when omitted.")
    basuras_eur: Optional[float] = Field(default=None, ge=0, description="Annual waste tax (basuras/TGR). Defaults to city value when omitted.")
    setup_cost_eur: float = Field(default=1500.0, ge=0, description="One-time Airbnb setup cost (photos, staging, supplies).")
    noi_growth_rate: float = Field(default=0.025, ge=0, le=0.15, description="Annual NOI growth rate (CPI/HPI blend).")
    property_appreciation_rate: float = Field(default=0.03, ge=0, le=0.20, description="Annual property price appreciation rate.")
    include_income_tax: bool = Field(default=True, description="Whether to deduct income tax from NOI. False shows pre-tax comparison.")
    purchase_price: Optional[float] = Field(default=None, ge=0, description="Original purchase price for CGT calculation. Defaults to 70% of current value.")
    holding_years: int = Field(default=10, ge=5, le=30, description="Investment horizon in years (5–30).")
    discount_rate: float = Field(default=0.07, ge=0.03, le=0.20, description="Opportunity cost of capital / discount rate.")


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

    # Extended financial metrics
    irr_airbnb_pct: Optional[float] = None
    npv_airbnb_pretax_p50_eur: float = 0.0
    npv_sell_pretax_eur: float = 0.0
    ibi_eur_used: float = 0.0
    ibi_method: str = "estimated"
    ibi_explanation: str = ""
    basuras_eur_used: float = 0.0
    setup_cost_eur_used: float = 0.0
    npv_advantage_eur: float = 0.0
    holding_years: int = 10
    discount_rate: float = 0.07
    regulatory_shock_prob: float = 0.0

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


# ── /market_report ───────────────────────────────────────────────────────────────

class MarketReportRequest(BaseModel):
    """Build a market-analysis report from an *already-computed* scenario.

    The report reuses the active analysis (so its numbers can't drift from what
    the UI shows) and adds peer positioning from the comparables agent; it never
    recomputes the scenario.
    """

    property: Optional[dict] = Field(
        default=None, description="Property-shaped context (city, district, room_type, …) — used for comparable filters."
    )
    scenario: dict = Field(description="The active /scenario response to build the report on.")
    use_llm: bool = Field(
        default=True, description="False forces the deterministic (LLM-free) narrative."
    )


class MarketReportResponse(BaseModel):
    city: str
    neighbourhood: str
    recommendation: str
    confidence: dict = Field(default_factory=dict)
    subject: dict = Field(default_factory=dict, description="The subject property's own headline numbers.")
    positioning: dict = Field(default_factory=dict, description="Subject vs peer median/P25-P75 for nightly + revenue.")
    drivers: list = Field(default_factory=list)
    comparables: list = Field(default_factory=list)
    narrative: str
    disclaimer: str = "Indicative only — not financial advice."


# ── /saved ─────────────────────────────────────────────────────────────────────

class SavedCreateRequest(BaseModel):
    """A property + its computed scenario to persist. Both are stored as-is."""

    property: dict
    scenario: dict
    nickname: Optional[str] = None


class SavedRecord(BaseModel):
    id: str
    saved_at: str
    nickname: Optional[str] = None
    property: dict = Field(default_factory=dict)
    scenario: dict = Field(default_factory=dict)


# ── /sentiment ─────────────────────────────────────────────────────────────────

class SentimentExample(BaseModel):
    text: str
    score: float


class SentimentBucket(BaseModel):
    label: str  # left edge of the mean_sentiment bin, e.g. "-0.2"
    count: int


class SentimentResponse(BaseModel):
    city: str
    reviews_scored: int
    label_split: dict[str, float] = Field(default_factory=dict, description="positive/neutral/negative shares (donut).")
    lang_split: dict[str, float] = Field(default_factory=dict, description="Top review-language shares.")
    metrics: dict[str, float] = Field(default_factory=dict, description="nb_accuracy / nb_f1_macro / corr_sentiment_vs_rating.")
    examples: dict[str, list[SentimentExample]] = Field(default_factory=dict)
    listing_histogram: list[SentimentBucket] = Field(default_factory=list)
    n_listings: int = 0
    disclaimer: str = ""


# ── health ─────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
    price_model_loaded: bool
