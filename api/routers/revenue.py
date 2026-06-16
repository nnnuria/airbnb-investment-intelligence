"""Stub endpoints: /estimate_revenue and /airbnb_vs_sell.

These wrap the finance engine (``airbnb_iip.finance``), which is still being
built. They return flagged placeholders that already match the final response
schema, so the UI/agents can integrate now. To go live, replace the body of
each handler with the real service call at the marked TODO — the response
shape does not change.
"""

from __future__ import annotations

from fastapi import APIRouter

from api.schemas import (
    AirbnbVsSellRequest,
    AirbnbVsSellResponse,
    EstimateRevenueRequest,
    EstimateRevenueResponse,
)

router = APIRouter(tags=["finance (stub)"])


@router.post("/estimate_revenue", response_model=EstimateRevenueResponse)
def estimate_revenue(req: EstimateRevenueRequest) -> EstimateRevenueResponse:
    """Annual Airbnb gross/net revenue with P10/P50/P90 bands. (STUB)"""
    # TODO: from airbnb_iip.finance.costs import annual_gross_revenue, compute_noi
    price = req.price_per_night or 130.0
    occ = req.occupancy_rate or 0.55
    gross = price * occ * 365
    net = gross * 0.62  # placeholder net margin after costs+tax
    return EstimateRevenueResponse(
        annual_gross_eur=round(gross, 2),
        annual_net_eur=round(net, 2),
        p10_eur=round(net * 0.8, 2),
        p50_eur=round(net, 2),
        p90_eur=round(net * 1.2, 2),
    )


@router.post("/airbnb_vs_sell", response_model=AirbnbVsSellResponse)
def airbnb_vs_sell(req: AirbnbVsSellRequest) -> AirbnbVsSellResponse:
    """Compare keep-as-Airbnb vs sell on an NPV basis. (STUB)"""
    # TODO: from airbnb_iip.finance.scenarios import npv_airbnb, npv_sell, recommend
    #       sale_value from the Idealista €/m² sale-value service.
    return AirbnbVsSellResponse(
        recommendation="Airbnb",
        npv_airbnb_p50_eur=420_000.0,
        npv_sell_eur=380_000.0,
        break_even_years=7,
        confidence={"p_airbnb_gt_sell": 0.68, "p10_eur": 350_000.0, "p90_eur": 510_000.0},
    )
