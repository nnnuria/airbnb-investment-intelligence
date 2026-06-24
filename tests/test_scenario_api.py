"""Tests for POST /scenario — the single decision engine behind HTTP.

Guards the core invariant of the orchestration work: the API path and the
in-process engine must produce identical numbers (one source of truth, no
drift between the Streamlit app and the agents).
"""

from __future__ import annotations

from dataclasses import asdict

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from api.main import app  # noqa: E402
from airbnb_iip.decision.engine import Property, compute_scenario  # noqa: E402

SAMPLE = {
    "city": "madrid", "district": "Salamanca", "size_m2": 85, "bedrooms": 2,
    "bathrooms": 1.0, "accommodates": 4, "room_type": "Entire home/apt",
    "has_ac": True, "has_elevator": True,
}


def test_scenario_returns_full_dashboard_payload():
    with TestClient(app) as client:
        resp = client.post("/scenario", json=SAMPLE)
    assert resp.status_code == 200
    data = resp.json()
    for key in (
        "recommendation", "confidence", "predicted_nightly_eur",
        "occupancy_rate_annual", "net_revenue_year_eur", "sale_price_eur",
        "breakeven_years", "npv_airbnb_p50_eur", "npv_sell_eur",
        "p_airbnb_gt_sell", "monthly_seasonality", "cost_breakdown",
        "feature_drivers", "governance",
    ):
        assert key in data, f"missing {key}"
    assert len(data["monthly_seasonality"]) == 12
    assert data["recommendation"] in ("airbnb", "sell", "marginal")


def test_scenario_matches_in_process_engine():
    prop = Property(
        city="barcelona", district="Eixample", size_m2=95, bedrooms=3,
        bathrooms=2.0, accommodates=5, room_type="Entire home/apt",
        has_ac=True, has_elevator=True, has_balcony=True,
    )
    direct = asdict(compute_scenario(prop))
    with TestClient(app) as client:
        via_api = client.post("/scenario", json=asdict(prop)).json()

    for key in (
        "recommendation", "predicted_nightly_eur", "occupancy_rate_annual",
        "net_revenue_year_eur", "sale_price_eur", "breakeven_years",
        "npv_airbnb_p50_eur", "npv_sell_eur",
    ):
        assert via_api[key] == direct[key], f"drift on {key}"
