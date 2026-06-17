"""Smoke tests for the SalePredictor serving layer.

These load the committed artefacts (sale_best_model.pkl, sale_feature_cols.json,
sale_district_defaults.json) — no raw data needed, so they run in CI.
"""

from __future__ import annotations

import pytest

from airbnb_iip.models.sale import (
    DEFAULT_MODEL,
    SalePredictor,
    estimate_sale_value,
    get_sale_predictor,
)

pytestmark = pytest.mark.skipif(
    not DEFAULT_MODEL.exists(),
    reason="sale_best_model.pkl not present (run notebooks/sell_price_model.ipynb)",
)


def test_predict_plausible_range():
    p = SalePredictor()
    val = p.predict({"city": "madrid", "district": "Chamberí", "size_m2": 90,
                     "rooms": 3, "bathrooms": 2, "property_type": "flat", "status": "good"})
    assert 100_000 < val < 3_000_000


def test_district_imputation_restores_location_signal():
    """Sparse (city, district, size) must still rank a prime district above a
    cheap one — the whole point of district-level imputation. (Magnitude gaps
    compress for large flats since €/m² varies less by district at size, so we
    assert ordering, which is the real requirement.)"""
    p = SalePredictor()
    for city, prime_d, cheap_d in [
        ("madrid", "Barrio de Salamanca", "Villaverde"),
        ("barcelona", "Sarrià-Sant Gervasi", "Nou Barris"),
        ("malaga", "Este", "Ciudad Jardín"),
    ]:
        prime = p.predict({"city": city, "district": prime_d, "size_m2": 90})
        cheap = p.predict({"city": city, "district": cheap_d, "size_m2": 90})
        assert prime > cheap, f"{city}: {prime_d} (€{prime:,.0f}) !> {cheap_d} (€{cheap:,.0f})"


def test_bigger_flat_costs_more():
    p = SalePredictor()
    small = p.predict({"city": "barcelona", "district": "Eixample", "size_m2": 60})
    big = p.predict({"city": "barcelona", "district": "Eixample", "size_m2": 120})
    assert big > small


def test_sparse_city_only_still_predicts():
    assert estimate_sale_value("malaga") > 0


def test_unknown_district_falls_back_to_city():
    # An unseen district shouldn't crash — it falls back to city defaults.
    assert estimate_sale_value("madrid", district="NotARealDistrict", size_m2=80) > 0


def test_estimate_sale_value_matches_predictor():
    direct = get_sale_predictor().predict(
        {"city": "malaga", "district": "Este", "size_m2": 100}
    )
    via = estimate_sale_value("malaga", "Este", 100)
    assert direct == via
