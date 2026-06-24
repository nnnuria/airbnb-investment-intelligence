"""Chat coordinator — a LangGraph router over the agent layer.

One entry point (:func:`run_chat`) takes a user message plus the current
analysis context (property + scenario) and routes it to the right specialist:

    route ─┬─ decision      → Market Analyst (narrates the live scenario)
           ├─ regulatory    → Regulatory RAG agent (Gemini + FAISS)
           ├─ comparables   → Comparables agent (KNN over the ABT)
           ├─ optimisation  → improvement ideas (engine.suggest_improvements)
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


INTENTS = ("decision", "regulatory", "comparables", "optimisation", "general")


def classify(message: str) -> str:
    """Keyword intent classifier. Order matters — the more specific topics are
    checked before the broad 'decision' bucket (which owns generic words like
    'price' and 'why')."""
    m = (message or "").lower()
    if any(k in m for k in ("regulat", "licen", "legal", "permit", "law", "allowed to rent")):
        return "regulatory"
    if any(k in m for k in ("comparable", "similar listing", "benchmark", "comps", "competitor", "other listings")):
        return "comparables"
    if any(k in m for k in ("improve", "optimis", "optimiz", "amenity", "amenities", "uplift", "renovat", "upgrade", "add ")):
        return "optimisation"
    if any(k in m for k in (
        "sell", "sale", "revenue", "income", "earn", "occupancy", "booked",
        "price", "nightly", "rate", "why", "recommend", "invest", "npv",
        "break", "yield", "profit",
        "cost", "fee", "tax", "expens", "deduct", "noi", "net",
    )):
        return "decision"
    return "general"


# ── LLM (lazy, optional) ──────────────────────────────────────────────────────

def _has_key() -> bool:
    return bool(os.getenv("GEMINI_API_KEY"))


@lru_cache(maxsize=1)
def _chat_llm():
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.3,
        max_output_tokens=400,
        thinking_budget=0,  # see market_analyst.py — keeps the visible answer intact
    )


_CHAT_PROMPT = """You are the investment co-pilot for an Airbnb-vs-sell platform.
Answer the owner's question in 2-4 sentences, conversational and specific, using
ONLY the numbers below. Do not invent figures. End with exactly:
"⚠️ Indicative only — not financial advice."

Question: {question}

Facts for {neighbourhood}, {city}:
- Recommendation: {recommendation} (confidence P(Airbnb beats Sell) = {p_airbnb})
- Predicted nightly rate: €{nightly}/night
- Occupancy: {occ_pct}% ({nights} nights/year)
- Net annual revenue: €{net} (P10-P90: €{p10}-€{p90}); gross €{gross}
- Indicative sale value: €{sale} (€{per_m2}/m²)
- Break-even vs. selling today: {break_even} years
- NPV if kept as Airbnb (P50): €{npv_airbnb}; net sale proceeds: €{npv_sell}
- Annual cost breakdown (gross minus these = net): {costs}
- Top price drivers: {drivers}

Answer:"""


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

    if state.get("use_llm") and _has_key():
        answer = _llm_decision_answer(state["message"], facts)
    else:
        answer = _deterministic_decision_answer(state["message"], facts)
    return {
        "answer": answer,
        "sources": ["LightGBM price model (SHAP drivers)", "finance engine (NPV + Monte-Carlo)"],
    }


def _llm_decision_answer(question: str, facts: dict) -> str:
    s = facts["scenario"]
    drivers = ", ".join(d["feature"] for d in facts["drivers"][:3])
    costs = "; ".join(
        f"{label} €{value:,.0f}" for label, value in s.get("cost_breakdown", [])
    ) or "n/a"
    prompt = _CHAT_PROMPT.format(
        question=question,
        neighbourhood=facts["neighbourhood"],
        city=str(facts["city"]).title(),
        recommendation=facts["recommendation"],
        p_airbnb=facts["confidence"].get("p_airbnb_gt_sell"),
        nightly=f"{facts['nightly_price_eur']:.0f}",
        occ_pct=f"{s['occupancy_rate_annual'] * 100:.0f}",
        nights=s["nights_booked_year"],
        net=f"{facts['annual_net_eur']:,.0f}",
        p10=f"{facts['p10_eur']:,.0f}",
        p90=f"{facts['p90_eur']:,.0f}",
        gross=f"{facts['annual_gross_eur']:,.0f}",
        sale=f"{s['sale_price_eur']:,.0f}",
        per_m2=f"{s['sale_price_per_m2_eur']:,.0f}",
        break_even=facts["break_even_years"],
        npv_airbnb=f"{facts['npv_airbnb_p50_eur']:,.0f}",
        npv_sell=f"{facts['npv_sell_eur']:,.0f}",
        costs=costs,
        drivers=drivers,
    )
    try:
        resp = _chat_llm().invoke(prompt)
        text = resp.content if hasattr(resp, "content") else str(resp)
        return text.strip()
    except Exception:
        # Quota/network/etc. — never fail the chat; drop to the deterministic answer.
        return _deterministic_decision_answer(question, facts)


def _deterministic_decision_answer(question: str, facts: dict) -> str:
    """LLM-free targeted answer built from the real scenario numbers."""
    m = (question or "").lower()
    s = facts["scenario"]
    if any(k in m for k in ("cost", "fee", "tax", "expens", "deduct", "noi")):
        lines = "; ".join(
            f"{label} €{value:,.0f}" for label, value in s.get("cost_breakdown", [])
        )
        return (
            f"Gross revenue is €{facts['annual_gross_eur']:,.0f}; after costs the net to "
            f"the owner is €{facts['annual_net_eur']:,.0f}. The annual cost lines are: "
            f"{lines or 'n/a'}. These come from the finance engine (platform fee, "
            f"cleaning, tourist tax, management, maintenance, insurance, accounting and "
            f"income tax) applied to your property's revenue and value."
        )
    if any(k in m for k in ("sell", "sale", "should i")):
        return (
            f"Indicative sale value is €{s['sale_price_eur']:,.0f} "
            f"(€{s['sale_price_per_m2_eur']:,.0f}/m²); break-even against projected "
            f"Airbnb net income is {facts['break_even_years']} years. The model's call "
            f"is **{facts['recommendation'].upper()}** "
            f"(P(Airbnb beats Sell) = {facts['confidence'].get('p_airbnb_gt_sell')})."
        )
    if any(k in m for k in ("occupancy", "booked", "nights")):
        return (
            f"Projected occupancy is {s['occupancy_rate_annual'] * 100:.0f}% "
            f"({s['nights_booked_year']} nights/year), inferred from the review velocity "
            f"of comparable well-operated listings."
        )
    if any(k in m for k in ("price", "nightly", "rate")):
        return (
            f"Projected nightly rate is €{facts['nightly_price_eur']:.0f}/night, from the "
            f"LightGBM price model. Top drivers: "
            f"{', '.join(d['feature'] for d in facts['drivers'][:3])}."
        )
    if any(k in m for k in ("revenue", "income", "earn", "profit", "yield")):
        return (
            f"Estimated net annual revenue is €{facts['annual_net_eur']:,.0f} "
            f"(P10-P90: €{facts['p10_eur']:,.0f}-€{facts['p90_eur']:,.0f}); gross is "
            f"€{facts['annual_gross_eur']:,.0f}, the difference being platform, "
            f"management, cleaning, tax and other costs."
        )
    # "why" / generic decision
    return (
        f"The recommendation is **{facts['recommendation'].upper()}** "
        f"(P(Airbnb beats Sell) = {facts['confidence'].get('p_airbnb_gt_sell')}). "
        f"Nightly rate €{facts['nightly_price_eur']:.0f}, net revenue "
        f"€{facts['annual_net_eur']:,.0f}/yr, break-even {facts['break_even_years']} years. "
        f"Strongest price drivers: {', '.join(d['feature'] for d in facts['drivers'][:3])}."
    )


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
        from airbnb_iip.decision.engine import suggest_improvements

        items = suggest_improvements(_property_from_dict(prop), _scenario_from_dict(scenario))[:3]
    except Exception as exc:
        return {"answer": f"Couldn't compute improvements ({type(exc).__name__}).", "sources": []}

    lines = [
        f"- **{i.name}** — €{i.investment_eur:,.0f} upfront, ~€{i.monthly_revenue_uplift_eur:,.0f}/mo "
        f"uplift, payback {i.payback_months:.1f} months"
        for i in items
    ]
    return {
        "answer": "Top improvement ideas by payback:\n\n" + "\n".join(lines),
        "sources": ["engine.suggest_improvements (ROI-ranked)"],
        "meta": {"improvements": [i.__dict__ for i in items]},
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
    g.add_node("decision", _market_node)
    g.add_node("regulatory", _regulatory_node)
    g.add_node("comparables", _comparables_node)
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
