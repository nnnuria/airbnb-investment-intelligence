"""Chat coordinator — a LangGraph router over the agent layer.

One entry point (:func:`run_chat`) takes a user message plus the current
analysis context (property + scenario) and routes it to the right specialist:

    route ─┬─ financial     → Financial Analyst (IRR, tax, IBI, cost breakdown)
           ├─ decision      → Market Analyst (narrates the live scenario)
           ├─ regulatory    → Regulatory RAG agent (Gemini + FAISS)
           ├─ comparables   → Comparables agent (KNN over the ABT)
           ├─ market        → Market Analyst positioning (subject vs peer comps)
           ├─ optimisation  → improvement ideas (agents.optimisation: Apriori + gap)
           └─ general       → capability guidance
                              ↓
                           govern (disclaimer enforcement) → END

Routing is deterministic keyword classification — no LLM call needed to pick a
branch, so it is fast, free, and demo-safe. The LLM (Gemini) is only used to
*narrate* a decision answer, and only when ``use_llm=True`` and a key is set;
otherwise every branch falls back to a deterministic answer built from the real
numbers. That means the whole graph runs (and is testable) with no Gemini key.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Optional, TypedDict

from dotenv import load_dotenv

from airbnb_iip.agents.market_analyst import DISCLAIMER, scenario_to_analysis

load_dotenv()


# ── State ─────────────────────────────────────────────────────────────────────

class ChatState(TypedDict, total=False):
    message: str
    property: dict           # Property-shaped dict (city, district, size_m2, …)
    scenario: dict           # /scenario response (or asdict(Scenario))
    use_llm: bool
    intent: str
    answer: str
    sources: list
    meta: dict


INTENTS = ("financial", "decision", "regulatory", "comparables", "market", "optimisation", "general")


def classify(message: str) -> str:
    """Keyword intent classifier. Order matters — the more specific topics are
    checked before the broad 'decision' bucket (which owns generic words like
    'price' and 'why'). 'financial' is checked before 'decision' so that
    detailed financial questions (IRR, tax, IBI, cost breakdown) get the
    specialist agent rather than the general market narrative. 'market'
    (positioning vs peers) sits after 'comparables' so an explicit ask for the
    comp *listings* still routes there, while 'how do I compare?' style
    positioning questions get the market-analysis answer."""
    m = (message or "").lower()
    if any(k in m for k in ("regulat", "licen", "legal", "permit", "law", "allowed to rent")):
        return "regulatory"
    if any(k in m for k in ("comparable", "similar listing", "benchmark", "comps", "competitor", "other listings")):
        return "comparables"
    if any(k in m for k in (
        "market", "positioning", "position vs", "compare", "comparison", "compares",
        "competitive", "stack up", "stacks up", "peer", "versus", "vs the",
    )):
        return "market"
    if any(k in m for k in ("improve", "optimis", "optimiz", "amenity", "amenities", "uplift", "renovat", "upgrade", "add ")):
        return "optimisation"
    if any(k in m for k in (
        "irr", "internal rate", "return on invest",
        "ibi", "property tax", "impuesto",
        "income tax", "irpf", "tax impact", "after.tax", "pre.tax", "pre tax", "post tax",
        "cost breakdown", "break down cost", "itemis", "itemiz",
        "setup cost", "setup investment",
        "npv", "net present value",
    )):
        return "financial"
    if any(k in m for k in (
        "sell", "sale", "revenue", "income", "earn", "occupancy", "booked",
        "price", "nightly", "rate", "why", "recommend", "invest",
        "break", "yield", "profit",
        "cost", "fee", "tax", "expens", "deduct", "noi", "net",
    )):
        return "decision"
    return "general"


# ── LLM (lazy, optional) ──────────────────────────────────────────────────────

def _has_key() -> bool:
    return bool(os.getenv("GEMINI_API_KEY"))


# ── Nodes ─────────────────────────────────────────────────────────────────────

def _route_node(state: ChatState) -> dict:
    return {"intent": classify(state.get("message", ""))}


def _decision_answer_facts(state: ChatState) -> Optional[dict]:
    """Build the analysis facts dict from the scenario, or None if absent."""
    scenario = state.get("scenario")
    if not scenario:
        return None
    prop = state.get("property") or {}
    city = (prop.get("city") or scenario.get("city") or "madrid")
    neighbourhood = prop.get("district") or "city-wide"
    return scenario_to_analysis(scenario, city=city, neighbourhood=neighbourhood)


def _market_node(state: ChatState) -> dict:
    facts = _decision_answer_facts(state)
    if facts is None:
        return {
            "answer": (
                "I don't have an analysis loaded yet. Open **New analysis**, run a "
                "property, and I'll explain the recommendation, revenue, occupancy, "
                "and the sell-vs-Airbnb call with the real numbers."
            ),
            "sources": [],
        }

    # Delegate to the Market Analyst agent — the single source of truth for the
    # decision narrative (LLM and deterministic). Constructed per-call like
    # _financial_node; use_llm is gated on a key so the no-key path never builds
    # a Gemini client, and the agent's httpx client stays lazy (unused here).
    from airbnb_iip.agents.market_analyst import MarketAnalystAgent

    agent = MarketAnalystAgent(use_llm=bool(state.get("use_llm") and _has_key()))
    return {
        "answer": agent.answer(state.get("message", ""), facts),
        "sources": ["LightGBM price model (SHAP drivers)", "finance engine (NPV + Monte-Carlo)"],
    }


def _market_positioning_node(state: ChatState) -> dict:
    """Answer market-positioning questions ('how do I compare to the market?')
    from the Market Analysis: peer comparables benchmark + the live scenario."""
    facts = _decision_answer_facts(state)
    if facts is None:
        return {
            "answer": (
                "I don't have an analysis loaded yet. Run a property on **New "
                "analysis** and I'll show how it sits against comparable listings "
                "nearby — nightly rate and revenue versus the local market."
            ),
            "sources": [],
        }

    prop = state.get("property") or {}
    try:
        from airbnb_iip.agents.comparables import get_comparables_agent

        comparables = get_comparables_agent().find_comparables(prop, k=5)
    except Exception:
        # No peer set (missing data, etc.) — the answer degrades to the model
        # estimate rather than failing.
        comparables = {"comparables": [], "benchmark": {}}

    from airbnb_iip.agents.market_analyst import MarketAnalystAgent, market_positioning

    agent = MarketAnalystAgent(use_llm=bool(state.get("use_llm") and _has_key()))
    return {
        "answer": agent.answer_market(state.get("message", ""), facts, comparables),
        "sources": ["Market Analysis — peer positioning (comparables benchmark) + LightGBM price model"],
        "meta": {"positioning": market_positioning(facts, comparables)},
    }


def _financial_node(state: ChatState) -> dict:
    """Route to FinancialAnalystAgent for IRR, tax, IBI, and cost breakdown questions."""
    scenario = state.get("scenario")
    if not scenario:
        return {
            "answer": (
                "I don't have an analysis loaded yet. Run a property analysis first, "
                "then ask me about IRR, tax impact, IBI, or the cost breakdown."
            ),
            "sources": [],
        }
    try:
        from airbnb_iip.agents.financial_analyst import FinancialAnalystAgent
        agent = FinancialAnalystAgent(use_llm=bool(state.get("use_llm") and _has_key()))
        answer = agent.answer(state.get("message", ""), scenario)
    except Exception as exc:
        answer = (
            f"Couldn't compute the financial breakdown ({type(exc).__name__}). "
            "Please try again or ask a different question."
        )
    return {
        "answer": answer,
        "sources": ["finance engine (IRR + NPV + IRPF brackets)"],
    }


def _regulatory_node(state: ChatState) -> dict:
    prop = state.get("property") or {}
    city = prop.get("city") or _city_from_message(state.get("message", ""))
    try:
        from airbnb_iip.agents.regulatory import get_regulatory_agent

        res = get_regulatory_agent().query(state["message"], city=city)
    except Exception as exc:
        return {
            "answer": (
                "I can't reach the regulatory knowledge base right now "
                f"({type(exc).__name__}). Treat STR rules for this property as "
                "unverified and check official municipal/regional sources."
            ),
            "sources": [],
        }

    meta = {"risk_flag": res.get("risk_flag")}
    try:
        from airbnb_iip.agents.governance import apply_regulatory_guardrails

        guarded = apply_regulatory_guardrails({
            "risk_flag": res.get("risk_flag"),
            "reason": res.get("answer"),
            "sources": res.get("sources", []),
            "city": res.get("city"),
            "disclaimer": res.get("disclaimer"),
        })
        meta["governance"] = guarded.get("governance")
    except Exception:
        pass
    return {"answer": res["answer"], "sources": res.get("sources", []), "meta": meta}


def _comparables_node(state: ChatState) -> dict:
    prop = state.get("property")
    if not prop or not prop.get("city"):
        return {
            "answer": "Tell me the property (city, district, size, bedrooms) and "
                      "I'll pull comparable listings as a benchmark.",
            "sources": [],
        }
    try:
        from airbnb_iip.agents.comparables import get_comparables_agent

        res = get_comparables_agent().find_comparables(prop, k=5)
    except Exception as exc:
        return {"answer": f"Comparables are unavailable right now ({type(exc).__name__}).", "sources": []}

    comps, bench = res["comparables"], res["benchmark"]
    if not comps:
        return {"answer": "No comparable listings found for that district and room type.", "sources": []}

    price = bench.get("price", {})
    rev = bench.get("estimated_revenue_l365d", {})
    where = prop.get("district") or str(prop.get("city")).title()
    answer = (
        f"Across {len(comps)} similar listings near {where}, the median nightly price is "
        f"€{price.get('median', 0):,.0f} (P25-P75 €{price.get('p25', 0):,.0f}-"
        f"€{price.get('p75', 0):,.0f}) and median annual revenue is "
        f"€{rev.get('median', 0):,.0f}. Use that as a market benchmark for your projection."
    )
    return {
        "answer": answer,
        "sources": [f"KNN over segmented ABT · filters: {res.get('filters_applied', {})}"],
        "meta": {"comparables": comps, "benchmark": bench},
    }


def _optimisation_node(state: ChatState) -> dict:
    prop, scenario = state.get("property"), state.get("scenario")
    if not (prop and scenario):
        return {
            "answer": "Run an analysis first and I'll rank improvement ideas by payback "
                      "period (AC, workspace, photography, dynamic pricing, …).",
            "sources": [],
        }
    try:
        from airbnb_iip.agents.optimisation import get_optimisation_service

        plan = get_optimisation_service().recommend(
            prop, projected_annual_nights=scenario.get("nights_booked_year"),
        )
        items = plan.recommendations[:3]
    except Exception as exc:
        return {"answer": f"Couldn't compute improvements ({type(exc).__name__}).", "sources": []}

    if not items:
        return {
            "answer": "This property already has the high-ROI amenities we'd recommend "
                      "for its peer group — there's no obvious low-payback improvement left.",
            "sources": ["airbnb_iip.agents.optimisation"],
        }
    lines = [
        f"- **{i.name}** — €{i.investment_eur:,.0f} upfront, +€{i.annual_uplift_eur:,.0f}/yr, "
        f"payback {i.payback_months:.0f} months ({i.confidence} confidence)"
        for i in items
    ]
    return {
        "answer": "Top improvement ideas by payback:\n\n" + "\n".join(lines),
        "sources": ["airbnb_iip.agents.optimisation (counterfactual + residual + Apriori)"],
        "meta": {
            "improvements": [vars(i) for i in items],
            "peer_gap_eur": plan.gap_to_top_quartile_eur,
        },
    }


def _general_node(state: ChatState) -> dict:
    return {
        "answer": (
            "I can explain the Airbnb-vs-sell recommendation, projected revenue and "
            "occupancy, comparable listings, regulatory risk, and improvement ideas. "
            "Try \"why this recommendation?\", \"what about regulations?\", or "
            "\"show me comparables\"."
        ),
        "sources": [],
    }


def _govern_node(state: ChatState) -> dict:
    answer = state.get("answer", "")
    low = answer.lower()
    if "not financial advice" not in low and "not legal advice" not in low:
        answer = answer.rstrip() + f"\n\n⚠️ {DISCLAIMER}"
    return {"answer": answer}


# ── helpers ───────────────────────────────────────────────────────────────────

_CITY_SLUGS = ("madrid", "barcelona", "malaga", "málaga")


def _city_from_message(message: str) -> Optional[str]:
    m = (message or "").lower()
    for c in _CITY_SLUGS:
        if c in m:
            return "malaga" if c == "málaga" else c
    return None


@lru_cache(maxsize=1)
def _engine_field_sets():
    from dataclasses import fields

    from airbnb_iip.decision.engine import Property, Scenario

    return {f.name for f in fields(Property)}, {f.name for f in fields(Scenario)}


def _property_from_dict(prop: dict):
    from airbnb_iip.decision.engine import Property

    pf, _ = _engine_field_sets()
    return Property(**{k: v for k, v in prop.items() if k in pf})


def _scenario_from_dict(scenario: dict):
    from airbnb_iip.decision.engine import Scenario

    _, sf = _engine_field_sets()
    clean = {k: v for k, v in scenario.items() if k in sf}
    clean["cost_breakdown"] = [tuple(x) for x in clean.get("cost_breakdown", [])]
    clean["feature_drivers"] = [tuple(x) for x in clean.get("feature_drivers", [])]
    return Scenario(**clean)


# ── Graph ─────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _graph():
    from langgraph.graph import END, StateGraph

    g = StateGraph(ChatState)
    g.add_node("route", _route_node)
    g.add_node("financial", _financial_node)
    g.add_node("decision", _market_node)
    g.add_node("regulatory", _regulatory_node)
    g.add_node("comparables", _comparables_node)
    g.add_node("market", _market_positioning_node)
    g.add_node("optimisation", _optimisation_node)
    g.add_node("general", _general_node)
    g.add_node("govern", _govern_node)

    g.set_entry_point("route")
    g.add_conditional_edges("route", lambda s: s["intent"], {i: i for i in INTENTS})
    for node in INTENTS:
        g.add_edge(node, "govern")
    g.add_edge("govern", END)
    return g.compile()


def run_chat(
    message: str,
    *,
    property: Optional[dict] = None,
    scenario: Optional[dict] = None,
    use_llm: bool = True,
) -> dict[str, Any]:
    """Route ``message`` through the coordinator and return the answer + metadata.

    ``property`` / ``scenario`` are the current analysis context (the same
    shapes the UI holds in session). ``use_llm=False`` forces the deterministic
    path — used by tests and the demo-safe mode.

    Caching (see :mod:`airbnb_iip.agents.chat_cache`):
    * DEMO_MODE on  → serve from cache; on a miss, answer deterministically
      (never a live LLM call).
    * DEMO_MODE off → compute live; write live LLM answers through to the cache.
    """
    from airbnb_iip.agents import chat_cache

    demo = chat_cache.demo_mode()
    key = chat_cache.cache_key(message, property)

    if demo:
        cached = chat_cache.get(key)
        if cached is not None:
            return cached
        use_llm = False  # demo miss: deterministic only, no live LLM

    state: ChatState = {
        "message": message,
        "property": property or {},
        "scenario": scenario,
        "use_llm": use_llm,
    }
    result = _graph().invoke(state)
    payload = {
        "answer": result.get("answer", ""),
        "intent": result.get("intent", classify(message)),
        "sources": result.get("sources", []),
        "meta": result.get("meta", {}),
        "disclaimer": DISCLAIMER,
    }

    # Write live LLM answers through to the cache so they're frozen for the demo.
    if not demo and use_llm and _has_key():
        chat_cache.set(key, payload)
    return payload


__all__ = ["run_chat", "classify", "ChatState", "INTENTS"]
