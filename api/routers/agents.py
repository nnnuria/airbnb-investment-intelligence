"""Agent-layer endpoints: /comparables, /regulatory_risk, /chat.

These put the remaining agents behind the same HTTP surface as the decision
engine, so the UI and the coordinator reach everything one way: over the API.

* /comparables    — KNN over the segmented ABT (lightweight).
* /regulatory_risk — Gemini + FAISS RAG over the STR corpus, guardrail-wrapped.
* /chat           — the LangGraph coordinator that routes a message to the
                    right agent given the current analysis context.
"""

from __future__ import annotations

import os

from fastapi import APIRouter, HTTPException

from airbnb_iip.agents.comparables import get_comparables_agent
from airbnb_iip.agents.coordinator import run_chat
from airbnb_iip.agents.governance import apply_regulatory_guardrails

from api.schemas import (
    ChatRequest,
    ChatResponse,
    ComparablesRequest,
    ComparablesResponse,
    MarketReportRequest,
    MarketReportResponse,
    RegulatoryRequest,
    RegulatoryResponse,
)

router = APIRouter(tags=["agents"])


@router.post("/comparables", response_model=ComparablesResponse)
def comparables(req: ComparablesRequest) -> ComparablesResponse:
    """Return up to ``k`` comparable listings + benchmark stats."""
    spec = req.model_dump(exclude_none=True)
    k = spec.pop("k", 5)
    result = get_comparables_agent().find_comparables(spec, k=k)
    return ComparablesResponse(**result)


@router.post("/regulatory_risk", response_model=RegulatoryResponse)
def regulatory_risk(req: RegulatoryRequest) -> RegulatoryResponse:
    """STR regulatory risk for a city/neighbourhood (or a free-form question)."""
    # Imported lazily: pulls the embeddings model + FAISS index, so it should
    # only load when this endpoint is actually called, not at app import.
    from airbnb_iip.agents.regulatory import get_regulatory_agent

    try:
        agent = get_regulatory_agent()
        if req.question:
            res = agent.query(req.question, city=req.city)
            reason = res["answer"]
        else:
            res = agent.get_risk_flag(req.city, req.neighbourhood or "")
            reason = res["reason"]
    except Exception as exc:  # quota, missing index, etc.
        raise HTTPException(
            status_code=503,
            detail=f"Regulatory service unavailable: {type(exc).__name__}: {exc}",
        ) from exc

    guarded = apply_regulatory_guardrails({
        "risk_flag": res.get("risk_flag"),
        "reason": reason,
        "sources": res.get("sources", []),
        "city": res.get("city", req.city),
        "disclaimer": res.get("disclaimer"),
    })
    return RegulatoryResponse(
        risk_flag=res.get("risk_flag", "UNKNOWN"),
        reason=reason,
        sources=res.get("sources", []),
        city=res.get("city", req.city),
        governance=guarded.get("governance", {}),
        disclaimer=res.get("disclaimer", "Indicative only — not legal advice."),
    )


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """Route a chat message through the LangGraph coordinator."""
    result = run_chat(
        req.message,
        property=req.property,
        scenario=req.scenario,
        use_llm=req.use_llm,
    )
    return ChatResponse(**result)


@router.post("/market_report", response_model=MarketReportResponse)
def market_report(req: MarketReportRequest) -> MarketReportResponse:
    """Build a market-analysis report on the active scenario.

    Orchestrates the I/O (fetch comparables for peer positioning) and hands the
    pre-computed scenario to the Market Analyst agent, which owns the synthesis
    + narrative. The scenario is never recomputed, so the report's numbers match
    the analysis the UI already showed.
    """
    from airbnb_iip.agents.market_analyst import MarketAnalystAgent

    prop = req.property or {}
    try:
        comparables = get_comparables_agent().find_comparables(prop, k=5)
    except Exception:
        # No peer set (e.g. missing data) — the report degrades to model-only
        # positioning rather than failing the whole request.
        comparables = {"comparables": [], "benchmark": {}, "filters_applied": {}}

    # Gate the LLM on a key so the no-key path never tries to build a Gemini
    # client (the agent raises if use_llm=True without GEMINI_API_KEY).
    use_llm = req.use_llm and bool(os.getenv("GEMINI_API_KEY"))
    agent = MarketAnalystAgent(use_llm=use_llm)
    report = agent.market_report(prop, req.scenario, comparables=comparables)
    return MarketReportResponse(**report)
