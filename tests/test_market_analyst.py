"""Tests for the Market Analyst agent.

``gather_analysis`` and the deterministic fallback narrative are LLM-free,
so they run in CI with no Gemini key and no live server: the agent's
httpx client is FastAPI's own ``TestClient`` (an ``httpx.Client`` subclass)
bound directly to the app — no real network or uvicorn process needed.
"""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from api.main import app  # noqa: E402

from airbnb_iip.agents.market_analyst import MarketAnalystAgent  # noqa: E402


@pytest.fixture
def agent() -> MarketAnalystAgent:
    return MarketAnalystAgent(client=TestClient(app), use_llm=False)


SAMPLE_PROPERTY = {
    "city": "madrid",
    "neighbourhood_cleansed": "Salamanca",
    "sq_m": 80,
    "accommodates": 4,
    "bedrooms": 2,
    "bathrooms_number": 1,
    "property_type_std": "Entire place",
}


def test_gather_analysis_assembles_all_fields(agent: MarketAnalystAgent):
    analysis = agent.gather_analysis(SAMPLE_PROPERTY)

    assert analysis["city"] == "madrid"
    assert analysis["nightly_price_eur"] > 0
    assert 1 <= len(analysis["drivers"]) <= 6
    # /scenario uses the engine's own vocabulary (lower-case, plus "marginal").
    assert analysis["recommendation"] in ("airbnb", "sell", "marginal")
    assert analysis["npv_sell_eur"] > 0
    assert analysis["p10_eur"] <= analysis["p90_eur"]
    assert "Indicative only" in analysis["disclaimer"]


def test_gather_analysis_is_deterministic(agent: MarketAnalystAgent):
    first = agent.gather_analysis(SAMPLE_PROPERTY)
    second = agent.gather_analysis(SAMPLE_PROPERTY)
    assert first["nightly_price_eur"] == second["nightly_price_eur"]
    assert first["recommendation"] == second["recommendation"]


def test_analyze_returns_fallback_narrative_with_disclaimer(agent: MarketAnalystAgent):
    result = agent.analyze(SAMPLE_PROPERTY)
    assert "narrative" in result
    assert "Indicative only — not financial advice" in result["narrative"]
    assert result["recommendation"] in result["narrative"]


def test_analyze_minimal_spec_still_works(agent: MarketAnalystAgent):
    result = agent.analyze({"city": "malaga"})
    assert result["nightly_price_eur"] > 0
    assert "narrative" in result


def test_use_llm_without_api_key_raises(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        MarketAnalystAgent(use_llm=True)
