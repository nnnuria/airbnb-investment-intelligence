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
    for path, payload in [
        ("/estimate_revenue", {"price_per_night": 120, "occupancy_rate": 0.5}),
        ("/airbnb_vs_sell", {"city": "Madrid", "sq_m": 80}),
        ("/optimise", {"city": "Madrid"}),
    ]:
        r = client.post(path, json=payload)
        assert r.status_code == 200, path
        assert r.json().get("_stub") is True, path
