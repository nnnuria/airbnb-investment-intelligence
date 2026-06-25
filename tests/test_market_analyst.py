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


# ── answer() — the consolidated chat 'decision' narration ─────────────────────

def test_answer_targets_the_question(agent: MarketAnalystAgent):
    facts = agent.gather_analysis(SAMPLE_PROPERTY)
    occ = agent.answer("what is the occupancy here?", facts)
    assert "occupancy" in occ.lower()
    rev = agent.answer("how much revenue will it make?", facts)
    assert "revenue" in rev.lower()


def test_answer_market_positions_against_peers(agent: MarketAnalystAgent):
    facts = agent.gather_analysis(SAMPLE_PROPERTY)
    ans = agent.answer_market("how do I compare to the market?", facts, _SYNTHETIC_COMPS)
    assert "comparable listings near" in ans.lower()
    assert "peer median" in ans.lower()
    assert "Indicative only" in ans


def test_answer_market_without_peers_falls_back(agent: MarketAnalystAgent):
    facts = agent.gather_analysis(SAMPLE_PROPERTY)
    ans = agent.answer_market("how do I compare?", facts, {})
    assert "model" in ans.lower()
    assert "Indicative only" in ans


# ── market_positioning() — pure, no I/O ───────────────────────────────────────

_SYNTHETIC_COMPS = {
    "comparables": [{}, {}, {}],
    "benchmark": {
        "price": {"median": 120, "p25": 100, "p75": 140},
        "estimated_revenue_l365d": {"median": 20000, "p25": 15000, "p75": 25000},
        "review_scores_rating": {"median": 4.7},
    },
}


def test_market_positioning_classifies_against_band():
    from airbnb_iip.agents.market_analyst import market_positioning

    analysis = {"nightly_price_eur": 200, "annual_gross_eur": 30000}
    pos = market_positioning(analysis, _SYNTHETIC_COMPS)

    assert pos["peer_n"] == 3
    assert pos["nightly"]["position"] == "above market"   # 200 > p75 140
    assert pos["revenue"]["position"] == "above market"   # 30000 > p75 25000
    assert pos["nightly"]["pct_vs_median"] == round((200 / 120 - 1) * 100, 1)
    assert pos["review_rating_median"] == 4.7


def test_market_positioning_in_line_and_below():
    from airbnb_iip.agents.market_analyst import market_positioning

    pos = market_positioning({"nightly_price_eur": 120, "annual_gross_eur": 12000}, _SYNTHETIC_COMPS)
    assert pos["nightly"]["position"] == "in line with market"  # 120 within 100–140
    assert pos["revenue"]["position"] == "below market"         # 12000 < p25 15000


def test_market_positioning_handles_no_peers():
    from airbnb_iip.agents.market_analyst import market_positioning

    pos = market_positioning({"nightly_price_eur": 150, "annual_gross_eur": 20000}, {})
    assert pos["peer_n"] == 0
    assert pos["nightly"]["position"] == "unknown"
    assert pos["nightly"]["pct_vs_median"] is None


# ── market_report() — full report, LLM-free ───────────────────────────────────

def test_market_report_shape_and_narrative(agent: MarketAnalystAgent):
    scenario = agent.gather_analysis(SAMPLE_PROPERTY)["scenario"]
    report = agent.market_report(
        {"city": "madrid", "district": "Salamanca"}, scenario, comparables=_SYNTHETIC_COMPS,
    )

    assert report["recommendation"] in ("airbnb", "sell", "marginal")
    assert report["subject"]["nightly_eur"] > 0
    assert report["positioning"]["peer_n"] == 3
    assert len(report["comparables"]) == 3
    assert "Indicative only — not financial advice" in report["narrative"]
    # Two-part deterministic brief mentions both positioning and the call.
    assert "comparable listings" in report["narrative"].lower()


def test_market_report_without_peers_degrades_gracefully(agent: MarketAnalystAgent):
    scenario = agent.gather_analysis(SAMPLE_PROPERTY)["scenario"]
    report = agent.market_report({"city": "madrid", "district": "Salamanca"}, scenario, comparables={})
    assert report["positioning"]["peer_n"] == 0
    assert "Indicative only" in report["narrative"]


# ── /market_report endpoint ───────────────────────────────────────────────────

def test_market_report_endpoint():
    client = TestClient(app)
    scen = client.post("/scenario", json={
        "city": "madrid", "district": "Salamanca", "size_m2": 80,
        "bedrooms": 2, "bathrooms": 1.0, "accommodates": 4,
        "room_type": "Entire home/apt",
    }).json()

    resp = client.post("/market_report", json={
        "property": {"city": "madrid", "district": "Salamanca", "room_type": "Entire home/apt",
                     "accommodates": 4, "bedrooms": 2},
        "scenario": scen,
        "use_llm": False,
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["recommendation"] in ("airbnb", "sell", "marginal")
    assert "positioning" in body
    assert "Indicative only" in body["narrative"]
