"""Live endpoints: /predict_price and /estimate_occupancy.

Both depend only on already-merged code (the LightGBM price model and
``airbnb_iip.data.occupancy``), so they return real predictions today.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from airbnb_iip.config import OCCUPANCY
from airbnb_iip.data.occupancy import DAYS_PER_YEAR, estimate_occupancy
from airbnb_iip.models.price import PricePredictor

from api.dependencies import price_predictor
from api.schemas import (
    EstimateOccupancyRequest,
    EstimateOccupancyResponse,
    PredictPriceRequest,
    PredictPriceResponse,
)

router = APIRouter(tags=["airbnb"])


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


@router.post("/estimate_occupancy", response_model=EstimateOccupancyResponse)
def estimate_occupancy_endpoint(
    req: EstimateOccupancyRequest,
) -> EstimateOccupancyResponse:
    """Estimate occupancy from review velocity (San Francisco model)."""
    review_rate = req.review_rate or OCCUPANCY["review_rate"]
    avg_los = req.avg_length_of_stay or OCCUPANCY["avg_length_of_stay"]
    max_occ = req.max_occupancy or OCCUPANCY["max_occupancy"]

    result = estimate_occupancy(
        req.reviews_per_month,
        review_rate=review_rate,
        avg_length_of_stay=avg_los,
        max_occupancy=max_occ,
    )
    return EstimateOccupancyResponse(
        occupancy_rate=round(result["occupancy_rate"], 4),
        nights_booked_per_month=round(result["nights_booked_per_month"], 2),
        estimated_nights_per_year=round(result["occupancy_rate"] * DAYS_PER_YEAR),
        assumptions={
            "review_rate": review_rate,
            "avg_length_of_stay": avg_los,
            "max_occupancy": max_occ,
        },
    )
