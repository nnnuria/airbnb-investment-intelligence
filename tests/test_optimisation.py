"""Tests for the optimisation agent (per-amenity uplift + gap + Apriori rules)."""
import warnings

import pytest

from airbnb_iip.agents.optimisation import (
    OptimisationPlan,
    get_optimisation_service,
    optimise,
)

warnings.filterwarnings("ignore")


def _spec(**overrides):
    base = {
        "city": "madrid",
        "district": "Salamanca",
        "room_type": "Entire home/apt",
        "accommodates": 4,
        "bedrooms": 2,
        "bathrooms": 1.0,
    }
    base.update(overrides)
    return base


# ── core plan shape ────────────────────────────────────────────────────────────

def test_recommend_returns_ranked_positive_plan():
    plan = get_optimisation_service().recommend(_spec(has_elevator=True))
    assert isinstance(plan, OptimisationPlan)
    assert plan.peer_n > 0
    assert plan.projected_annual_nights > 0
    assert plan.recommendations, "expected at least one recommendation"

    paybacks = [r.payback_months for r in plan.recommendations]
    assert paybacks == sorted(paybacks)  # ranked by payback ascending
    for r in plan.recommendations:
        assert r.annual_uplift_eur > 0
        assert r.investment_eur > 0
        assert r.payback_months <= 120  # max-payback guard
        assert r.confidence in ("high", "medium", "low")
        assert r.method in ("counterfactual", "residual")


def test_peer_gap_is_non_negative_and_target_above_median():
    plan = get_optimisation_service().recommend(_spec())
    assert plan.peer_target_revenue_eur >= plan.peer_median_revenue_eur
    assert plan.gap_to_top_quartile_eur >= 0


# ── already-owned amenities are excluded ───────────────────────────────────────

def test_owned_amenity_is_not_recommended():
    plan = get_optimisation_service().recommend(_spec(has_ac=True))
    assert all(r.amenity != "has_ac" for r in plan.recommendations)


def test_ac_uses_model_counterfactual_when_absent():
    plan = get_optimisation_service().recommend(_spec(has_ac=False))
    ac = next((r for r in plan.recommendations if r.amenity == "has_ac"), None)
    assert ac is not None, "AC should be recommended when absent"
    assert ac.method == "counterfactual"
    assert ac.confidence == "high"          # genuine model feature
    assert ac.annual_uplift_eur > 0


# ── uplift scales with occupancy ───────────────────────────────────────────────

def test_more_nights_means_more_uplift_for_ac():
    svc = get_optimisation_service()
    low = svc.recommend(_spec(has_ac=False), projected_annual_nights=100)
    high = svc.recommend(_spec(has_ac=False), projected_annual_nights=300)
    ac_low = next(r for r in low.recommendations if r.amenity == "has_ac")
    ac_high = next(r for r in high.recommendations if r.amenity == "has_ac")
    assert ac_high.annual_uplift_eur > ac_low.annual_uplift_eur


# ── validation + dict wrapper ──────────────────────────────────────────────────

def test_missing_city_raises():
    with pytest.raises(ValueError, match="city"):
        get_optimisation_service().recommend({"district": "Centro"})


def test_optimise_dict_wrapper_keys():
    out = optimise(_spec())
    assert set(out) >= {
        "city", "district", "room_type", "peer_n",
        "peer_median_revenue_eur", "peer_target_revenue_eur",
        "gap_to_top_quartile_eur", "projected_annual_nights",
        "recommendations", "disclaimer",
    }
    assert isinstance(out["recommendations"], list)
    if out["recommendations"]:
        r = out["recommendations"][0]
        assert {"name", "annual_uplift_eur", "payback_months", "method"} <= set(r)
