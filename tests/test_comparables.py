"""Tests for the Comparables agent.

Runs against the real segmented listings parquet (no mocking) since the
data is small, committed-adjacent, and the point of these tests is to
catch real data-shape regressions (e.g. the city-accent mismatch found
during development: the data stores "Málaga", callers pass "malaga").
"""

from __future__ import annotations

from pathlib import Path

import pytest

from airbnb_iip.agents.comparables import ComparablesAgent, get_comparables_agent

DATA_PATH = Path(__file__).resolve().parents[1] / "Data" / "processed" / "listings_segmented.parquet"
pytestmark = pytest.mark.skipif(not DATA_PATH.exists(), reason="segmented listings parquet not present")


@pytest.fixture(scope="module")
def agent() -> ComparablesAgent:
    return ComparablesAgent()


def test_city_only_returns_comparables(agent: ComparablesAgent):
    result = agent.find_comparables({"city": "madrid"}, k=5)
    assert result["filters_applied"] == {"city": "madrid"}
    assert len(result["comparables"]) == 5


def test_city_match_is_accent_insensitive(agent: ComparablesAgent):
    """Data stores "Málaga"; callers use the ASCII slug "malaga" — must still match."""
    result = agent.find_comparables({"city": "malaga"}, k=3)
    assert len(result["comparables"]) == 3
    assert all(c["city"].lower().replace("á", "a") == "malaga" for c in result["comparables"])


def test_district_filters_to_broader_neighbourhood_group(agent: ComparablesAgent):
    result = agent.find_comparables({"city": "madrid", "district": "Salamanca"}, k=5)
    assert result["filters_applied"].get("neighbourhood_group_cleansed") == "Salamanca"
    assert len(result["comparables"]) > 0


def test_overconstrained_filters_relax_gracefully(agent: ComparablesAgent):
    """A combination unlikely to have 5+ exact matches should still return comps,
    not an empty list — filters relax one at a time until enough candidates remain."""
    result = agent.find_comparables(
        {
            "city": "malaga",
            "district": "Centro",
            "neighbourhood_cleansed": "La Trinidad-Perchel",
            "segment": "Tourist-core studios",
            "room_type": "Shared room",
        },
        k=5,
    )
    assert len(result["comparables"]) > 0
    # Some filter must have been relaxed away from the full over-narrow set.
    assert len(result["filters_applied"]) < 5


def test_comparables_are_ranked_by_ascending_distance(agent: ComparablesAgent):
    result = agent.find_comparables(
        {"city": "madrid", "accommodates": 4, "bedrooms": 2, "bathrooms_number": 1}, k=5,
    )
    distances = [c["distance"] for c in result["comparables"]]
    assert distances == sorted(distances)


def test_benchmark_stats_are_ordered(agent: ComparablesAgent):
    result = agent.find_comparables({"city": "barcelona"}, k=10)
    price_stats = result["benchmark"]["price"]
    assert price_stats["p25"] <= price_stats["median"] <= price_stats["p75"]


def test_missing_city_raises():
    agent = get_comparables_agent()
    with pytest.raises(ValueError, match="city"):
        agent.find_comparables({}, k=5)


def test_unknown_city_returns_empty_not_error():
    agent = get_comparables_agent()
    result = agent.find_comparables({"city": "atlantis"}, k=5)
    assert result["comparables"] == []
    assert result["benchmark"] == {}


def test_singleton_returns_same_instance():
    assert get_comparables_agent() is get_comparables_agent()
