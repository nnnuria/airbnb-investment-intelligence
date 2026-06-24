"""Live endpoint: /optimise.

Wraps the optimisation flow (per-property amenity uplift + feature-gap
benchmark + Apriori-style association rules) implemented in
``airbnb_iip.agents.optimisation``. Returns a ranked, cost-aware improvement
plan for an Airbnb-bound property.
"""

from __future__ import annotations

from fastapi import APIRouter

from airbnb_iip.agents.optimisation import get_optimisation_service
from api.schemas import OptimiseAction, OptimiseRequest, OptimiseResponse

router = APIRouter(tags=["optimise"])


@router.post("/optimise", response_model=OptimiseResponse)
def optimise(req: OptimiseRequest) -> OptimiseResponse:
    """Ranked, cost-aware improvement recommendations (live)."""
    spec = req.model_dump(exclude_none=True)
    nights = spec.pop("projected_annual_nights", None)

    plan = get_optimisation_service().recommend(
        spec, projected_annual_nights=nights,
    )

    return OptimiseResponse(
        actions=[
            OptimiseAction(
                action=r.name,
                annual_uplift_eur=r.annual_uplift_eur,
                investment_eur=r.investment_eur,
                payback_months=r.payback_months,
                confidence=r.confidence,
                method=r.method,
                lift=r.lift,
                rationale=r.rationale,
            )
            for r in plan.recommendations
        ],
        city=plan.city,
        district=plan.district,
        room_type=plan.room_type,
        peer_n=plan.peer_n,
        peer_median_revenue_eur=plan.peer_median_revenue_eur,
        peer_target_revenue_eur=plan.peer_target_revenue_eur,
        gap_to_top_quartile_eur=plan.gap_to_top_quartile_eur,
        projected_annual_nights=plan.projected_annual_nights,
        disclaimer=plan.disclaimer,
    )
