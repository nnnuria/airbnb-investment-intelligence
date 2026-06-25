"""
Market Analyst Agent — Airbnb-vs-sell recommendation, narrated with SHAP.
src/airbnb_iip/agents/market_analyst.py

Per docs/structure.md's runtime flow (Streamlit -> Coordinator -> agent ->
FastAPI endpoint -> model), this agent is an HTTP client of the existing
live endpoints — it does not import the price/sale/finance modules
directly. That keeps model loading + serving in one place (the API
process) and lets this agent run anywhere the API is reachable.

Flow:
    property spec -> POST /scenario (the full decision engine, one call)
                  -> gather_analysis() (pure, no LLM)
                  -> narrate() (Gemini) -> analysis dict with a narrative

Calling the single /scenario endpoint (rather than stitching /predict_price +
/explain_price + /airbnb_vs_sell + /estimate_revenue) means this agent shares
the exact numbers the Streamlit dashboard shows — same occupancy, same SHAP
drivers — so the narrative can never drift from the UI.
"""

from __future__ import annotations

import os
from typing import Any, Mapping

import httpx
from dotenv import load_dotenv

# Load .env so GEMINI_API_KEY / API_BASE_URL are available
load_dotenv()

DEFAULT_API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

DISCLAIMER = "Indicative only — not financial advice."


def scenario_to_analysis(
    scenario: Mapping[str, Any], *, city: str, neighbourhood: str,
) -> dict[str, Any]:
    """Map a ``/scenario`` response into the analysis dict ``narrate()`` consumes.

    Shared by this agent's :meth:`MarketAnalystAgent.gather_analysis` and the
    chat coordinator's market node, so the two can never describe the same
    numbers differently. ``scenario`` is the JSON body returned by ``/scenario``
    (or ``dataclasses.asdict(Scenario)``); both carry the same keys.
    """
    drivers = [
        {"feature": label, "direction": "increases" if frac >= 0 else "decreases"}
        for label, frac in scenario["feature_drivers"]
    ]
    return {
        "city": city,
        "neighbourhood": neighbourhood,
        "nightly_price_eur": scenario["predicted_nightly_eur"],
        "drivers": drivers,
        "recommendation": scenario["recommendation"],
        "npv_airbnb_p50_eur": scenario["npv_airbnb_p50_eur"],
        "npv_sell_eur": scenario["npv_sell_eur"],
        "break_even_years": scenario["breakeven_years"],
        "confidence": {"p_airbnb_gt_sell": scenario["p_airbnb_gt_sell"]},
        "annual_gross_eur": scenario["gross_revenue_year_eur"],
        "annual_net_eur": scenario["net_revenue_year_eur"],
        "p10_eur": scenario["net_revenue_p10_eur"],
        "p90_eur": scenario["net_revenue_p90_eur"],
        "disclaimer": DISCLAIMER,
        # Full engine output, for the coordinator / chat layer downstream.
        "scenario": scenario,
    }

def _position_label(subject: Any, p25: Any, p75: Any) -> str:
    """Where the subject value sits relative to the peer interquartile band."""
    if subject is None or p25 is None or p75 is None:
        return "unknown"
    if subject > p75:
        return "above market"
    if subject < p25:
        return "below market"
    return "in line with market"


def market_positioning(
    analysis: Mapping[str, Any], comparables: Mapping[str, Any],
) -> dict[str, Any]:
    """Position the subject property's nightly rate + annual revenue against the
    comparable-listings benchmark.

    ``analysis`` is a :func:`scenario_to_analysis` result; ``comparables`` is a
    :meth:`ComparablesAgent.find_comparables` result (``comparables`` list +
    ``benchmark`` stats). Pure — no I/O — so the report is unit-testable.
    """
    bench = (comparables or {}).get("benchmark", {}) or {}
    comps = (comparables or {}).get("comparables", []) or []

    def _block(subject: Any, stats: Mapping[str, Any]) -> dict[str, Any]:
        stats = stats or {}
        median, p25, p75 = stats.get("median"), stats.get("p25"), stats.get("p75")
        pct = ((subject / median - 1) * 100) if (subject and median) else None
        return {
            "subject": subject,
            "median": median,
            "p25": p25,
            "p75": p75,
            "position": _position_label(subject, p25, p75),
            "pct_vs_median": round(pct, 1) if pct is not None else None,
        }

    return {
        "peer_n": len(comps),
        # Benchmark 'price' is nightly; 'estimated_revenue_l365d' is annual gross
        # revenue — compare each against the subject's matching figure.
        "nightly": _block(analysis.get("nightly_price_eur"), bench.get("price", {})),
        "revenue": _block(analysis.get("annual_gross_eur"), bench.get("estimated_revenue_l365d", {})),
        "review_rating_median": (bench.get("review_scores_rating", {}) or {}).get("median"),
    }


_MARKET_QA_PROMPT = """You are a market analyst for an Airbnb-vs-sell platform.
Answer the owner's specific question about how their property sits in the local
market, in 2-4 sentences, conversational and specific, using ONLY the numbers
below. Do not invent figures. End with exactly:
"⚠️ Indicative only — not financial advice."

Question: {question}

Property: {neighbourhood}, {city} — benchmarked against {peer_n} comparable listings
- Nightly rate: €{nightly}/night vs peer median €{nightly_median} ({nightly_position}, {nightly_pct} vs median)
- Annual revenue: €{gross} vs peer median €{revenue_median} ({revenue_position}, {revenue_pct} vs median)
- Peer median guest rating: {rating_median}
- Recommendation: {recommendation} (P(Airbnb beats Sell) = {p_airbnb})

Answer:"""


MARKET_REPORT_PROMPT = """You are a market analyst for an Airbnb-vs-sell platform,
writing a concise market analysis (two short paragraphs, 6-9 sentences total) for a
non-expert property owner. Use ONLY the numbers below — do not invent figures.

Structure:
- Paragraph 1 (market positioning): where this property's nightly rate and annual
  revenue sit versus comparable listings nearby, and what that implies.
- Paragraph 2 (the call): the Airbnb-vs-sell recommendation, the projected
  economics (net revenue with its P10-P90 band, NPV vs selling, break-even), and
  at least two price drivers that explain the nightly rate.
End with exactly: "⚠️ Indicative only — not financial advice."

Property: {neighbourhood}, {city}
Peers compared: {peer_n} similar listings

Market positioning:
- Nightly rate: €{nightly}/night vs peer median €{nightly_median} ({nightly_position}, {nightly_pct} vs median)
- Annual revenue: €{gross} vs peer median €{revenue_median} ({revenue_position}, {revenue_pct} vs median)
- Peer median guest rating: {rating_median}

The call:
- Recommendation: {recommendation} (P(Airbnb beats Sell) = {p_airbnb})
- Occupancy: {occ_pct}% ({nights} nights/year)
- Net annual revenue: €{net} (P10-P90: €{p10}-€{p90})
- NPV if kept as Airbnb (P50): €{npv_airbnb}; net sale proceeds: €{npv_sell}; break-even {break_even} years
- Top price drivers: {drivers}

Market analysis:"""


SYSTEM_PROMPT = """You are a market analyst for the Airbnb Investment
Intelligence Platform. Write a short brief (4-6 sentences) explaining an
Airbnb-vs-sell recommendation for a property, in English, for a non-expert
owner. Use ONLY the numbers given below — do not invent figures.

Rules:
- Lead with the recommendation and the headline numbers.
- Reference at least two of the top price drivers below to explain *why*
  the nightly price is what it is.
- Mention the uncertainty band (P10-P90), not just the point estimate.
- Do not present this as financial advice.
- End every answer with exactly:
  "⚠️ Indicative only — not financial advice."

Property: {city}, {neighbourhood}

Numbers:
- Recommendation: {recommendation}
- Predicted nightly rate: €{nightly_price_eur}/night
- Top price drivers (SHAP, most influential first): {drivers}
- Annual gross revenue: €{annual_gross_eur} | Annual net revenue: €{annual_net_eur}
  (P10-P90 band: €{p10_eur}-€{p90_eur})
- NPV if kept as Airbnb (10yr, P50): €{npv_airbnb_p50_eur}
- Net proceeds if sold today: €{npv_sell_eur}
- Break-even horizon: {break_even_years} years
- Confidence Airbnb beats selling: {p_airbnb_gt_sell}

Answer:"""


# Question-targeted prompt for the chat 'decision' intent — answers the owner's
# *specific* follow-up (cost, occupancy, why, …) rather than emitting the generic
# brief SYSTEM_PROMPT produces. Used by MarketAnalystAgent.answer().
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


class MarketAnalystAgent:
    """Calls the revenue + Airbnb-vs-sell scenario endpoints and narrates
    the result with SHAP-driven feature attributions.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_API_BASE_URL,
        client: httpx.Client | None = None,
        use_llm: bool = True,
    ):
        # Lazy client: the chat coordinator constructs this agent purely to
        # narrate a decision answer (the scenario is already computed and passed
        # in), so we must not open an httpx pool we never use. The client is
        # built on the first /scenario call — see the ``client`` property.
        self._base_url = base_url
        self._client = client
        self.use_llm = use_llm
        self._llm = None
        if use_llm:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError(
                    "GEMINI_API_KEY not found. "
                    "Add it to your .env file: GEMINI_API_KEY=your-key-here"
                )
            from langchain_google_genai import ChatGoogleGenerativeAI

            self._llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=api_key,
                temperature=0.2,
                max_output_tokens=400,
                # Without this, gemini-2.5-flash spends max_output_tokens on
                # internal reasoning and the visible answer gets cut off
                # mid-sentence (finish_reason="MAX_TOKENS") even for short
                # prompts. Confirmed via a raw API smoke test.
                thinking_budget=0,
            )

    # ── Gather (pure — no LLM) ───────────────────────────────────────────────

    def gather_analysis(self, property_spec: Mapping[str, Any]) -> dict[str, Any]:
        """Call ``POST /scenario`` and assemble a structured analysis.

        ``property_spec`` accepts the engine's ``Property`` fields (``city``,
        ``district``, ``size_m2``, ``bedrooms``, ``bathrooms``, ``accommodates``,
        ``room_type``, ``has_*``). Common ``/airbnb_vs_sell`` aliases
        (``neighbourhood_cleansed``→``district``, ``sq_m``→``size_m2``,
        ``bathrooms_number``→``bathrooms``) are accepted too. Deterministic and
        LLM-free, so it is fully unit-testable without a live Gemini key.
        """
        spec = dict(property_spec)

        payload = {
            "city": spec.get("city", "madrid"),
            "district": spec.get("district") or spec.get("neighbourhood_cleansed") or "Centro",
            "size_m2": spec.get("size_m2") or spec.get("sq_m") or 80,
            "bedrooms": spec.get("bedrooms", 2),
            "bathrooms": spec.get("bathrooms") or spec.get("bathrooms_number") or 1,
            "accommodates": spec.get("accommodates", 2),
            "room_type": spec.get("room_type", "Entire home/apt"),
        }
        for amenity in (
            "has_ac", "has_balcony", "has_elevator",
            "has_pool", "has_parking", "has_workspace",
        ):
            if amenity in spec:
                payload[amenity] = spec[amenity]

        s = self._post("/scenario", payload)
        return scenario_to_analysis(
            s, city=payload["city"], neighbourhood=payload["district"],
        )

    # ── Narrate (Gemini) ─────────────────────────────────────────────────────

    def narrate(self, analysis: Mapping[str, Any]) -> str:
        """Turn a :meth:`gather_analysis` result into an English brief."""
        if not self.use_llm:
            return self._fallback_narrative(analysis)

        drivers = ", ".join(
            f"{d['feature']} ({d['direction']})" for d in analysis["drivers"][:3]
        )
        prompt = SYSTEM_PROMPT.format(
            city=analysis["city"].title(),
            neighbourhood=analysis["neighbourhood"],
            recommendation=analysis["recommendation"],
            nightly_price_eur=analysis["nightly_price_eur"],
            drivers=drivers,
            annual_gross_eur=analysis["annual_gross_eur"],
            annual_net_eur=analysis["annual_net_eur"],
            p10_eur=analysis["p10_eur"],
            p90_eur=analysis["p90_eur"],
            npv_airbnb_p50_eur=analysis["npv_airbnb_p50_eur"],
            npv_sell_eur=analysis["npv_sell_eur"],
            break_even_years=analysis["break_even_years"],
            p_airbnb_gt_sell=analysis["confidence"].get("p_airbnb_gt_sell"),
        )
        response = self._llm.invoke(prompt)
        text = response.content if hasattr(response, "content") else str(response)
        if "Indicative only" not in text:
            text = text.rstrip() + f"\n\n⚠️ {DISCLAIMER}"
        return text

    def _fallback_narrative(self, analysis: Mapping[str, Any]) -> str:
        """Deterministic, LLM-free brief — used when ``use_llm=False``
        (tests, or a demo-safe "never call a live LLM" mode)."""
        top_driver = analysis["drivers"][0] if analysis["drivers"] else None
        driver_note = (
            f" The strongest price driver is {top_driver['feature']} "
            f"({top_driver['direction']} the nightly rate)."
            if top_driver
            else ""
        )
        return (
            f"Recommendation: {analysis['recommendation']}. Predicted nightly rate "
            f"€{analysis['nightly_price_eur']:.0f} in {analysis['neighbourhood']}, "
            f"{analysis['city'].title()}.{driver_note} Annual net revenue "
            f"€{analysis['annual_net_eur']:.0f} (P10-P90: €{analysis['p10_eur']:.0f}-"
            f"€{analysis['p90_eur']:.0f}); break-even vs. selling today is "
            f"{analysis['break_even_years']} years.\n\n⚠️ {DISCLAIMER}"
        )

    # ── Answer (question-targeted, for the chat 'decision' intent) ───────────

    def answer(self, question: str, analysis: Mapping[str, Any]) -> str:
        """Answer an owner's *specific* decision question about a pre-computed
        analysis.

        ``analysis`` is a :func:`scenario_to_analysis` result — the chat
        coordinator's market node builds it from the live scenario and passes it
        straight through, so this agent is the single source of truth for the
        ``decision`` intent's narration (LLM and deterministic alike). Mirrors
        :meth:`FinancialAnalystAgent.answer`.
        """
        if self.use_llm:
            return self._llm_decision_answer(question, analysis)
        return self._deterministic_decision_answer(question, analysis)

    def _llm_decision_answer(self, question: str, facts: Mapping[str, Any]) -> str:
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
            resp = self._llm.invoke(prompt)
            text = resp.content if hasattr(resp, "content") else str(resp)
            return text.strip()
        except Exception:
            # Quota/network/etc. — never fail the chat; drop to deterministic.
            return self._deterministic_decision_answer(question, facts)

    @staticmethod
    def _deterministic_decision_answer(question: str, facts: Mapping[str, Any]) -> str:
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

    # ── Market positioning Q&A (for the chat 'market' intent) ────────────────

    def answer_market(
        self,
        question: str,
        analysis: Mapping[str, Any],
        comparables: Mapping[str, Any] | None = None,
    ) -> str:
        """Answer a market-positioning question — how the subject sits versus
        comparable listings nearby. ``analysis`` is a :func:`scenario_to_analysis`
        result; ``comparables`` a :meth:`ComparablesAgent.find_comparables` result.
        """
        positioning = market_positioning(analysis, comparables or {})
        if self.use_llm:
            return self._llm_market_answer(question, analysis, positioning)
        return self._deterministic_market_answer(question, analysis, positioning)

    def _llm_market_answer(
        self, question: str, facts: Mapping[str, Any], positioning: Mapping[str, Any],
    ) -> str:
        n = positioning.get("nightly", {})
        r = positioning.get("revenue", {})

        def _eur(v: Any) -> str:
            return f"{v:,.0f}" if isinstance(v, (int, float)) else "n/a"

        def _pct(v: Any) -> str:
            return f"{v:+.0f}%" if isinstance(v, (int, float)) else "n/a"

        prompt = _MARKET_QA_PROMPT.format(
            question=question,
            neighbourhood=facts["neighbourhood"],
            city=str(facts["city"]).title(),
            peer_n=positioning.get("peer_n", 0),
            nightly=f"{facts['nightly_price_eur']:.0f}",
            nightly_median=_eur(n.get("median")),
            nightly_position=n.get("position", "n/a"),
            nightly_pct=_pct(n.get("pct_vs_median")),
            gross=_eur(facts["annual_gross_eur"]),
            revenue_median=_eur(r.get("median")),
            revenue_position=r.get("position", "n/a"),
            revenue_pct=_pct(r.get("pct_vs_median")),
            rating_median=positioning.get("review_rating_median") or "n/a",
            recommendation=facts["recommendation"],
            p_airbnb=facts["confidence"].get("p_airbnb_gt_sell"),
        )
        try:
            resp = self._llm.invoke(prompt)
            text = resp.content if hasattr(resp, "content") else str(resp)
            text = text.strip()
            if "Indicative only" not in text:
                text = text.rstrip() + f"\n\n⚠️ {DISCLAIMER}"
            return text
        except Exception:
            return self._deterministic_market_answer(question, facts, positioning)

    @staticmethod
    def _deterministic_market_answer(
        question: str, facts: Mapping[str, Any], positioning: Mapping[str, Any],
    ) -> str:
        """LLM-free positioning answer built from the comparables benchmark."""
        n = positioning.get("nightly", {})
        r = positioning.get("revenue", {})
        peer_n = positioning.get("peer_n", 0)

        if not peer_n or not n.get("median"):
            return (
                f"I don't have a comparable peer set for {facts['neighbourhood']}, "
                f"{facts['city'].title()} right now, so I can't position it against "
                f"the market. The model's own estimate is "
                f"€{facts['nightly_price_eur']:.0f}/night and "
                f"€{facts['annual_gross_eur']:,.0f} gross revenue/year.\n\n⚠️ {DISCLAIMER}"
            )

        def _pct(v: Any) -> str:
            return f"{v:+.0f}%" if isinstance(v, (int, float)) else "n/a"

        rating = positioning.get("review_rating_median")
        rating_clause = (
            f" Peer listings carry a median guest rating of {rating:.2f}."
            if isinstance(rating, (int, float)) else ""
        )
        return (
            f"Against {peer_n} comparable listings near {facts['neighbourhood']}, "
            f"{facts['city'].title()}, this property's nightly rate of "
            f"€{facts['nightly_price_eur']:.0f} is {n['position']} "
            f"(peer median €{n['median']:,.0f}, {_pct(n.get('pct_vs_median'))} vs median), "
            f"and its annual revenue of €{facts['annual_gross_eur']:,.0f} is "
            f"{r.get('position', 'n/a')} (peer median €{(r.get('median') or 0):,.0f})."
            f"{rating_clause}\n\n⚠️ {DISCLAIMER}"
        )

    # ── Market report (positioning + narrative) ──────────────────────────────

    def market_report(
        self,
        property: Mapping[str, Any],
        scenario: Mapping[str, Any],
        comparables: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build a market-analysis report from an *already-computed* scenario.

        Reuses ``scenario`` as-is (no ``/scenario`` recompute → the report can
        never drift from the numbers the UI showed) and layers peer positioning
        from ``comparables`` (a :meth:`ComparablesAgent.find_comparables` result)
        on top. Returns a structured report dict + an LLM narrative (deterministic
        fallback when ``use_llm`` is off or the call fails).
        """
        prop = dict(property or {})
        city = prop.get("city") or scenario.get("city") or "madrid"
        neighbourhood = prop.get("district") or "city-wide"
        facts = scenario_to_analysis(scenario, city=city, neighbourhood=neighbourhood)
        positioning = market_positioning(facts, comparables or {})

        report = {
            "city": city,
            "neighbourhood": neighbourhood,
            "recommendation": facts["recommendation"],
            "confidence": facts["confidence"],
            "subject": {
                "nightly_eur": facts["nightly_price_eur"],
                "annual_gross_eur": facts["annual_gross_eur"],
                "annual_net_eur": facts["annual_net_eur"],
                "p10_eur": facts["p10_eur"],
                "p90_eur": facts["p90_eur"],
                "sale_price_eur": scenario.get("sale_price_eur"),
                "sale_price_per_m2_eur": scenario.get("sale_price_per_m2_eur"),
                "occupancy_rate_annual": scenario.get("occupancy_rate_annual"),
                "nights_booked_year": scenario.get("nights_booked_year"),
                "npv_airbnb_p50_eur": facts["npv_airbnb_p50_eur"],
                "npv_sell_eur": facts["npv_sell_eur"],
                "break_even_years": facts["break_even_years"],
            },
            "positioning": positioning,
            "drivers": facts["drivers"],
            "comparables": (comparables or {}).get("comparables", []),
            "disclaimer": DISCLAIMER,
        }
        report["narrative"] = self._report_narrative(facts, positioning)
        return report

    def _report_narrative(
        self, facts: Mapping[str, Any], positioning: Mapping[str, Any],
    ) -> str:
        if not self.use_llm:
            return self._fallback_report_narrative(facts, positioning)

        s = facts["scenario"]
        n = positioning.get("nightly", {})
        r = positioning.get("revenue", {})

        def _eur(v: Any) -> str:
            return f"{v:,.0f}" if isinstance(v, (int, float)) else "n/a"

        def _pct(v: Any) -> str:
            return f"{v:+.0f}%" if isinstance(v, (int, float)) else "n/a"

        prompt = MARKET_REPORT_PROMPT.format(
            neighbourhood=facts["neighbourhood"],
            city=str(facts["city"]).title(),
            peer_n=positioning.get("peer_n", 0),
            nightly=f"{facts['nightly_price_eur']:.0f}",
            nightly_median=_eur(n.get("median")),
            nightly_position=n.get("position", "n/a"),
            nightly_pct=_pct(n.get("pct_vs_median")),
            gross=_eur(facts["annual_gross_eur"]),
            revenue_median=_eur(r.get("median")),
            revenue_position=r.get("position", "n/a"),
            revenue_pct=_pct(r.get("pct_vs_median")),
            rating_median=positioning.get("review_rating_median") or "n/a",
            recommendation=facts["recommendation"],
            p_airbnb=facts["confidence"].get("p_airbnb_gt_sell"),
            occ_pct=f"{s.get('occupancy_rate_annual', 0) * 100:.0f}",
            nights=s.get("nights_booked_year", "n/a"),
            net=_eur(facts["annual_net_eur"]),
            p10=_eur(facts["p10_eur"]),
            p90=_eur(facts["p90_eur"]),
            npv_airbnb=_eur(facts["npv_airbnb_p50_eur"]),
            npv_sell=_eur(facts["npv_sell_eur"]),
            break_even=facts["break_even_years"],
            drivers=", ".join(d["feature"] for d in facts["drivers"][:3]),
        )
        try:
            resp = self._llm.invoke(prompt)
            text = resp.content if hasattr(resp, "content") else str(resp)
            text = text.strip()
            if "Indicative only" not in text:
                text = text.rstrip() + f"\n\n⚠️ {DISCLAIMER}"
            return text
        except Exception:
            return self._fallback_report_narrative(facts, positioning)

    def _fallback_report_narrative(
        self, facts: Mapping[str, Any], positioning: Mapping[str, Any],
    ) -> str:
        """Deterministic two-part brief — used when ``use_llm=False`` or the LLM
        call fails. Same structure as the prompted narrative."""
        n = positioning.get("nightly", {})
        r = positioning.get("revenue", {})
        peer_n = positioning.get("peer_n", 0)
        drivers = ", ".join(d["feature"] for d in facts["drivers"][:2]) or "n/a"

        if peer_n and n.get("median"):
            positioning_text = (
                f"Against {peer_n} comparable listings near {facts['neighbourhood']}, "
                f"{facts['city'].title()}, this property's projected nightly rate of "
                f"€{facts['nightly_price_eur']:.0f} is {n.get('position', 'unknown')} "
                f"(peer median €{n['median']:,.0f}), and its annual revenue of "
                f"€{facts['annual_gross_eur']:,.0f} is {r.get('position', 'unknown')} "
                f"(peer median €{(r.get('median') or 0):,.0f}). "
            )
        else:
            positioning_text = (
                f"No comparable peer set was available for {facts['neighbourhood']}, "
                f"{facts['city'].title()}, so positioning rests on the model estimate alone. "
            )

        call_text = (
            f"The recommendation is {facts['recommendation'].upper()} "
            f"(P(Airbnb beats Sell) = {facts['confidence'].get('p_airbnb_gt_sell')}): projected "
            f"net revenue €{facts['annual_net_eur']:,.0f}/yr (P10-P90 "
            f"€{facts['p10_eur']:,.0f}-€{facts['p90_eur']:,.0f}), NPV as Airbnb "
            f"€{facts['npv_airbnb_p50_eur']:,.0f} versus net sale proceeds "
            f"€{facts['npv_sell_eur']:,.0f}, break-even {facts['break_even_years']} years. "
            f"The nightly rate is driven mainly by {drivers}."
        )
        return f"{positioning_text}{call_text}\n\n⚠️ {DISCLAIMER}"

    # ── Combined ─────────────────────────────────────────────────────────────

    def analyze(self, property_spec: Mapping[str, Any]) -> dict[str, Any]:
        """Full pipeline: gather the numbers, then narrate them."""
        analysis = self.gather_analysis(property_spec)
        analysis["narrative"] = self.narrate(analysis)
        return analysis

    # ── internals ────────────────────────────────────────────────────────────

    @property
    def client(self) -> httpx.Client:
        """The httpx client, created on first use (see ``__init__``)."""
        if self._client is None:
            self._client = httpx.Client(base_url=self._base_url, timeout=15.0)
        return self._client

    def _post(self, path: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        clean = {k: v for k, v in payload.items() if v is not None}
        resp = self.client.post(path, json=clean)
        resp.raise_for_status()
        return resp.json()


# ── Quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running Market Analyst against the local API (start it with "
          "`uvicorn api.main:app --reload`) ...\n")
    agent = MarketAnalystAgent()
    result = agent.analyze(
        {
            "city": "madrid",
            "neighbourhood_cleansed": "Salamanca",
            "sq_m": 80,
            "accommodates": 4,
            "bedrooms": 2,
            "bathrooms_number": 1,
            "property_type_std": "Entire place",
        }
    )
    print(result["narrative"])
