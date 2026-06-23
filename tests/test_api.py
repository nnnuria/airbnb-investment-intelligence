"""Smoke tests for the FastAPI layer.

Live endpoints are checked against the real model/estimator; stubs are checked
for shape + the ``_stub`` flag so a future wiring change is caught if it
accidentally changes the response contract.
"""

from __future__ import annotations

import pytest

from airbnb_iip.data.occupancy import estimate_occupancy

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from api.main import app  # noqa: E402

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_price_returns_plausible_value():
    r = client.post(
        "/predict_price",
        json={
            "city": "Madrid",
            "property_type_std": "Entire place",
            "accommodates": 4,
            "bedrooms": 2,
            "bathrooms_number": 1,
            "neighbourhood_cleansed": "Salamanca",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["currency"] == "EUR"
    # A 2-bed entire place in central Madrid should land in a sane nightly band.
    assert 20 < body["price_per_night"] < 2000


def test_predict_price_minimal_spec_still_works():
    r = client.post("/predict_price", json={"city": "Malaga"})
    assert r.status_code == 200
    assert r.json()["price_per_night"] > 0


def test_explain_price_returns_ranked_drivers():
    r = client.post(
        "/explain_price",
        json={
            "city": "Madrid",
            "property_type_std": "Entire place",
            "accommodates": 4,
            "bedrooms": 2,
            "bathrooms_number": 1,
            "neighbourhood_cleansed": "Salamanca",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["price_per_night"] > 0
    assert 1 <= len(body["drivers"]) <= 6
    shap_values = [abs(d["shap_value"]) for d in body["drivers"]]
    assert shap_values == sorted(shap_values, reverse=True)
    for d in body["drivers"]:
        assert d["direction"] in ("increases", "decreases")
        assert (d["shap_value"] > 0) == (d["direction"] == "increases")


def test_estimate_occupancy_matches_function():
    reviews = 1.5
    r = client.post("/estimate_occupancy", json={"reviews_per_month": reviews})
    assert r.status_code == 200
    expected = estimate_occupancy(reviews)
    assert r.json()["occupancy_rate"] == pytest.approx(
        round(expected["occupancy_rate"], 4)
    )


def test_estimate_occupancy_caps_at_ceiling():
    r = client.post("/estimate_occupancy", json={"reviews_per_month": 1000})
    assert r.status_code == 200
    assert r.json()["occupancy_rate"] <= 0.70


def test_stub_endpoints_are_flagged():
    # /optimise is still a stub (waits on the optimisation flow); the finance
    # endpoints below are now live.
    r = client.post("/optimise", json={"city": "Madrid"})
    assert r.status_code == 200
    assert r.json().get("_stub") is True


def test_estimate_revenue_is_live_and_coherent():
    r = client.post(
        "/estimate_revenue",
        json={"price_per_night": 120, "occupancy_rate": 0.5, "city": "Madrid"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["_stub"] is False
    # Gross = 120 * 0.5 * 365; net is after costs + tax, so strictly lower.
    assert body["annual_gross_eur"] == pytest.approx(120 * 0.5 * 365, rel=1e-6)
    assert 0 < body["annual_net_eur"] < body["annual_gross_eur"]
    assert body["p10_eur"] <= body["p50_eur"] <= body["p90_eur"]


def test_airbnb_vs_sell_is_live_and_coherent():
    r = client.post(
        "/airbnb_vs_sell",
        json={"city": "Madrid", "neighbourhood_cleansed": "Salamanca", "sq_m": 80},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["_stub"] is False
    assert body["recommendation"] in ("Airbnb", "Sell")
    assert body["npv_sell_eur"] > 0
    assert body["break_even_years"] is None or isinstance(body["break_even_years"], int)
    conf = body["confidence"]
    assert set(conf) >= {"p_airbnb_gt_sell", "p10_eur", "p90_eur"}
    assert 0.0 <= conf["p_airbnb_gt_sell"] <= 1.0
    assert conf["p10_eur"] <= conf["p90_eur"]
