"""Precomputed review-sentiment per city → GET /sentiment.

Serves the artifacts produced by ``scripts/run_sentiment.py`` (donut label
split, language split, sample quotes, per-listing distribution). Read-only and
cached; no scoring at request time.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from airbnb_iip.sentiment import get_city_sentiment
from api.schemas import SentimentResponse

router = APIRouter(tags=["sentiment"])


@router.get("/sentiment", response_model=SentimentResponse)
def sentiment(city: str = Query(..., examples=["madrid"])) -> dict:
    """Guest-review sentiment for ``city`` (madrid | barcelona | malaga)."""
    try:
        return get_city_sentiment(city)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail="Sentiment data is not available. Run scripts/run_sentiment.py first.",
        ) from exc
