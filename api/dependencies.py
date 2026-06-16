"""Shared FastAPI dependencies — load-once service singletons."""

from __future__ import annotations

from airbnb_iip.models.price import PricePredictor, get_price_predictor


def price_predictor() -> PricePredictor:
    """FastAPI dependency: the process-wide PricePredictor singleton."""
    return get_price_predictor()
