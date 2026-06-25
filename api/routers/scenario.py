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

from dataclasses import asdict, fields

from fastapi import APIRouter

from airbnb_iip.decision.engine import Property, compute_scenario

from api.schemas import ScenarioRequest, ScenarioResponse

router = APIRouter(tags=["decision"])

# Build Property only from fields it actually declares, so the request schema's
# extra="allow" amenity flags pass through but unknown keys can't break it.
_PROPERTY_FIELDS = {f.name for f in fields(Property)}


@router.post("/scenario", response_model=ScenarioResponse)
def scenario(req: ScenarioRequest) -> ScenarioResponse:
    """Run the full decision engine for one property."""
    spec = {k: v for k, v in req.model_dump().items() if k in _PROPERTY_FIELDS}
    result = compute_scenario(
        Property(**spec),
        managed=req.managed,
        ibi_eur=req.ibi_eur,
        cadastral_value=req.cadastral_value,
        basuras_eur=req.basuras_eur,
        setup_cost_eur=req.setup_cost_eur,
        noi_growth_rate=req.noi_growth_rate,
        property_appreciation_rate=req.property_appreciation_rate,
        include_income_tax=req.include_income_tax,
        purchase_price=req.purchase_price,
        holding_years=req.holding_years,
        discount_rate=req.discount_rate,
    )
    return ScenarioResponse(**asdict(result))
