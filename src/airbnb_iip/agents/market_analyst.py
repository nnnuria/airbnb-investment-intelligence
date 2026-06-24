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
        self.client = client or httpx.Client(base_url=base_url, timeout=15.0)
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

    # ── Combined ─────────────────────────────────────────────────────────────

    def analyze(self, property_spec: Mapping[str, Any]) -> dict[str, Any]:
        """Full pipeline: gather the numbers, then narrate them."""
        analysis = self.gather_analysis(property_spec)
        analysis["narrative"] = self.narrate(analysis)
        return analysis

    # ── internals ────────────────────────────────────────────────────────────

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
