"""Live endpoints: /predict_price and /estimate_occupancy.

Both serve real LightGBM models: the nightly-price model and the calendar-based
occupancy model (``airbnb_iip.models.occupancy``, target = Inside Airbnb
calendar availability, ``available == 'f'`` ⇒ booked).
"""

from __future__ import annotations

from dataclasses import fields

from fastapi import APIRouter, Depends

from airbnb_iip.data.occupancy import DAYS_PER_YEAR
from airbnb_iip.decision.engine import Property, predict_occupancy, predict_nightly_price
from airbnb_iip.models.price import PricePredictor

from api.dependencies import price_predictor
from api.schemas import (
    EstimateOccupancyRequest,
    EstimateOccupancyResponse,
    ExplainPriceResponse,
    PredictPriceRequest,
    PredictPriceResponse,
)

router = APIRouter(tags=["airbnb"])

_PROPERTY_FIELDS = {f.name for f in fields(Property)}


@router.post("/predict_price", response_model=PredictPriceResponse)
def predict_price(
    req: PredictPriceRequest,
    predictor: PricePredictor = Depends(price_predictor),
) -> PredictPriceResponse:
    """Predict the Airbnb nightly price (EUR) for a property."""
    spec = req.model_dump(exclude_none=True)
    price = predictor.predict(spec)
    return PredictPriceResponse(
        price_per_night=round(price, 2),
        city=req.city,
        model=predictor.best_model,
    )


@router.post("/explain_price", response_model=ExplainPriceResponse)
def explain_price(
    req: PredictPriceRequest,
    predictor: PricePredictor = Depends(price_predictor),
) -> ExplainPriceResponse:
    """SHAP feature attribution for the nightly-price prediction — the agent layer's "why"."""
    spec = req.model_dump(exclude_none=True)
    price = predictor.predict(spec)
    drivers = predictor.explain(spec)
    return ExplainPriceResponse(
        price_per_night=round(price, 2),
        drivers=drivers,
        model=predictor.best_model,
    )


@router.post("/estimate_occupancy", response_model=EstimateOccupancyResponse)
def estimate_occupancy_endpoint(
    req: EstimateOccupancyRequest,
) -> EstimateOccupancyResponse:
    """Predict annual occupancy with the calendar-based LightGBM model.

    Wraps ``engine.predict_occupancy`` so this endpoint and ``/scenario`` share
    one occupancy source. The nightly price is predicted internally (or taken
    from ``price_per_night`` when supplied) as a demand signal for the model.
    """
    spec = req.model_dump(exclude_none=True)
    spec["district"] = req.district or req.neighbourhood_cleansed or "Centro"
    prop = Property(**{k: v for k, v in spec.items() if k in _PROPERTY_FIELDS})

    nightly = req.price_per_night
    if nightly is None:
        nightly = predict_nightly_price(prop)
    occ_rate = predict_occupancy(prop, nightly_price=nightly)

    return EstimateOccupancyResponse(
        occupancy_rate=round(occ_rate, 4),
        estimated_nights_per_year=round(occ_rate * DAYS_PER_YEAR),
        predicted_nightly_eur=round(float(nightly), 2),
    )
