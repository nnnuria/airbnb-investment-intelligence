"""Live endpoint: /scenario — the full Airbnb-vs-sell decision in one call.

This wraps ``airbnb_iip.decision.engine.compute_scenario`` so the Streamlit UI
and the agent layer consume the same engine over HTTP instead of importing it
directly. One engine, one source of truth — no more drift between the app's
in-process path and the API.

The response carries everything the dashboard renders: recommendation, nightly
price, occupancy, gross/net revenue with P10/P90 bands, sale value, break-even,
seasonality, the cost breakdown, the SHAP price drivers, and the governance
report.
"""

from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter

from airbnb_iip.decision.engine import Property, compute_scenario

from api.schemas import ScenarioRequest, ScenarioResponse

router = APIRouter(tags=["decision"])


@router.post("/scenario", response_model=ScenarioResponse)
def scenario(req: ScenarioRequest) -> ScenarioResponse:
    """Run the full decision engine for one property."""
    prop = Property(**req.model_dump())
    result = compute_scenario(prop)
    return ScenarioResponse(**asdict(result))
