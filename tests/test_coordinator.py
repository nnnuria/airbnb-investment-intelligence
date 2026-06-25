"""Tests for the chat coordinator (LangGraph router) and its demo cache.

All deterministic: ``use_llm=False`` everywhere, so no Gemini key and no live
network are needed. Covers intent routing, each LLM-free branch, the
no-context guidance, and the DEMO_MODE cache-serving behaviour.
"""

from __future__ import annotations

from dataclasses import asdict

import pytest

pytest.importorskip("langgraph")

from airbnb_iip.agents.coordinator import classify, run_chat  # noqa: E402
from airbnb_iip.decision.engine import Property, compute_scenario  # noqa: E402


@pytest.fixture(scope="module")
def ctx():
    prop = Property(
        city="madrid", district="Salamanca", size_m2=85, bedrooms=2,
        bathrooms=1.0, accommodates=4, room_type="Entire home/apt", has_ac=True,
    )
    return asdict(prop), asdict(compute_scenario(prop))


@pytest.mark.parametrize("message,intent", [
    ("Why this recommendation?", "decision"),
    ("Should I sell?", "decision"),
    ("Is it legal to rent in Barcelona?", "regulatory"),
    ("Do I need a licence?", "regulatory"),
    ("Show me comparable listings", "comparables"),
    ("How do I compare to the market?", "market"),
    ("Where does my nightly rate sit vs the market?", "market"),
    ("How competitive is my listing?", "market"),
    ("What amenities should I add to improve it?", "optimisation"),
    ("hello there", "general"),
])
def test_classify_routes(message, intent):
    assert classify(message) == intent


def test_market_answer_cites_market_analysis(ctx):
    prop, scen = ctx
    res = run_chat("how do I compare to the market?", property=prop, scenario=scen, use_llm=False)
    assert res["intent"] == "market"
    assert any("Market Analysis" in s for s in res["sources"])
    assert "not financial advice" in res["answer"].lower()


def test_market_without_context_gives_guidance():
    res = run_chat("how do I compare to the market?", use_llm=False)
    assert res["intent"] == "market"
    assert "new analysis" in res["answer"].lower()


def test_decision_answer_is_grounded(ctx):
    prop, scen = ctx
    res = run_chat("why this recommendation?", property=prop, scenario=scen, use_llm=False)
    assert res["intent"] == "decision"
    assert scen["recommendation"].upper() in res["answer"]
    assert "not financial advice" in res["answer"].lower()


def test_decision_without_context_gives_guidance():
    res = run_chat("why?", use_llm=False)
    assert "new analysis" in res["answer"].lower()


def test_optimisation_lists_payback(ctx):
    prop, scen = ctx
    res = run_chat("what should I improve?", property=prop, scenario=scen, use_llm=False)
    assert res["intent"] == "optimisation"
    assert "payback" in res["answer"].lower()


def test_demo_mode_serves_cache(tmp_path, monkeypatch, ctx):
    prop, scen = ctx
    import airbnb_iip.agents.chat_cache as cc

    monkeypatch.setattr(cc, "CACHE_PATH", tmp_path / "chat_cache.json")
    monkeypatch.setattr(cc, "_cache", None)
    monkeypatch.delenv("DEMO_MODE", raising=False)

    key = cc.cache_key("seed question", prop)
    cc.set(key, {
        "answer": "FROZEN", "intent": "decision",
        "sources": [], "meta": {}, "disclaimer": "x",
    })

    monkeypatch.setenv("DEMO_MODE", "1")
    # use_llm=True would normally call Gemini; the cache hit must short-circuit it.
    res = run_chat("seed question", property=prop, scenario=scen, use_llm=True)
    assert res["answer"] == "FROZEN"


def test_demo_mode_miss_is_deterministic(tmp_path, monkeypatch, ctx):
    prop, scen = ctx
    import airbnb_iip.agents.chat_cache as cc

    monkeypatch.setattr(cc, "CACHE_PATH", tmp_path / "empty.json")
    monkeypatch.setattr(cc, "_cache", None)
    monkeypatch.setenv("DEMO_MODE", "1")

    # No cache entry + DEMO_MODE → must answer deterministically, never live.
    res = run_chat("what is the occupancy here?", property=prop, scenario=scen, use_llm=True)
    assert res["intent"] == "decision"
    assert "occupancy" in res["answer"].lower()
