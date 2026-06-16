"""Pydantic request/response models — the API contract.

These schemas are the contract the Streamlit UI and the agent layer code
against. The two finance endpoints (``/estimate_revenue``, ``/airbnb_vs_sell``)
and ``/optimise`` return **flagged stubs** today (``_stub: true``) that already
match their final response shape, so downstream work can integrate now and the
real services swap in without schema churn.
"""

from __future__ import annotations

from typing import Optional

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


# ── /optimise (stub) ─────────────────────────────────────────────────────────────

class OptimiseRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    city: Optional[str] = None
    neighbourhood_cleansed: Optional[str] = None


class OptimiseAction(BaseModel):
    action: str
    estimated_uplift_eur: float
    estimated_cost_eur: float
    category: str


class OptimiseResponse(BaseModel):
    actions: list[OptimiseAction]
    stub: bool = Field(default=True, alias="_stub")
    note: str = "Stub — wire to the optimisation flow (Apriori + feature gap)."


# ── health ─────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
    price_model_loaded: bool
