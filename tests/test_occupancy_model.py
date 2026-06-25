"""Tests for the calendar-based LightGBM occupancy model serving layer.

These run against the trained artefacts in ``models/`` when present; if they
are absent (fresh checkout before training) the predictor falls back to a flat
rate and the artefact-dependent assertions are skipped.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from airbnb_iip.models.occupancy import (
    OccupancyPredictor,
    get_occupancy_predictor,
)

ROOT = Path(__file__).resolve().parents[1]
ARTEFACTS = ROOT / "models" / "occupancy_feature_cols.json"
_HAS_MODEL = ARTEFACTS.exists()

requires_model = pytest.mark.skipif(
    not _HAS_MODEL, reason="occupancy artefacts not trained (run scripts/train_occupancy_model.py)"
)

_BASE = dict(
    city="Madrid", neighbourhood_cleansed="Embajadores", room_type="Entire home/apt",
    property_type_std="Entire place", accommodates=4, bedrooms=2, bathrooms_number=1,
    beds=2, latitude=40.41, longitude=-3.70, price=120, minimum_nights=2,
    instant_bookable=1, has_ac=1,
)


def test_singleton_is_cached():
    assert get_occupancy_predictor() is get_occupancy_predictor()


@requires_model
def test_predict_returns_fraction_in_range():
    p = get_occupancy_predictor()
    assert p.available
    rate = p.predict(_BASE)
    assert 0.0 <= rate <= 1.0
    assert p.clip_lo <= rate <= p.clip_hi


@requires_model
def test_predict_nights_consistent_with_rate():
    p = get_occupancy_predictor()
    rate = p.predict(_BASE)
    assert p.predict_nights(_BASE) == round(rate * 365)


@requires_model
def test_higher_price_lowers_occupancy():
    # Demand elasticity: a much higher nightly price should not increase occupancy.
    p = get_occupancy_predictor()
    cheap = p.predict({**_BASE, "price": 90})
    pricey = p.predict({**_BASE, "price": 400})
    assert pricey <= cheap


@requires_model
def test_long_minimum_nights_lowers_occupancy():
    p = get_occupancy_predictor()
    short = p.predict({**_BASE, "minimum_nights": 2})
    long_stay = p.predict({**_BASE, "minimum_nights": 30})
    assert long_stay <= short


@requires_model
def test_minimal_spec_still_predicts():
    # City only — everything else imputed with training medians.
    rate = get_occupancy_predictor().predict({"city": "Barcelona"})
    assert 0.0 < rate < 1.0


@requires_model
def test_accent_insensitive_city_routing():
    # "malaga" must route to the "Málaga" city model without error.
    rate = get_occupancy_predictor().predict({**_BASE, "city": "malaga"})
    assert 0.0 < rate < 1.0


@requires_model
def test_explain_returns_ranked_drivers():
    drivers = get_occupancy_predictor().explain(_BASE, top_n=5)
    assert 1 <= len(drivers) <= 5
    mags = [abs(d["shap_value"]) for d in drivers]
    assert mags == sorted(mags, reverse=True)
    for d in drivers:
        assert d["direction"] in ("increases", "decreases")


def test_missing_artefacts_fall_back(tmp_path):
    # An empty model dir → predictor reports unavailable and returns a flat rate.
    p = OccupancyPredictor(model_dir=tmp_path)
    assert p.available is False
    assert 0.0 < p.predict(_BASE) < 1.0
