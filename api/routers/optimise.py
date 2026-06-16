"""Stub endpoint: /optimise.

Wraps the optimisation flow (feature-gap analysis + Apriori amenity rules),
which is not yet built. Returns a flagged placeholder matching the final
response schema. Replace the body at the TODO when the flow lands.
"""

from __future__ import annotations

from fastapi import APIRouter

from api.schemas import OptimiseAction, OptimiseRequest, OptimiseResponse

router = APIRouter(tags=["optimise (stub)"])


@router.post("/optimise", response_model=OptimiseResponse)
def optimise(req: OptimiseRequest) -> OptimiseResponse:
    """Ranked, cost-aware improvement recommendations. (STUB)"""
    # TODO: wire to the optimisation flow (Phase 7b) once built.
    return OptimiseResponse(
        actions=[
            OptimiseAction(
                action="Add dedicated workspace",
                estimated_uplift_eur=1200.0,
                estimated_cost_eur=300.0,
                category="amenity",
            ),
            OptimiseAction(
                action="Install air conditioning",
                estimated_uplift_eur=2400.0,
                estimated_cost_eur=1500.0,
                category="renovation",
            ),
        ],
    )
