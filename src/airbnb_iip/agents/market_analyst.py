"""
Market Analyst Agent — Airbnb-vs-sell recommendation, narrated with SHAP.
src/airbnb_iip/agents/market_analyst.py

Per docs/structure.md's runtime flow (Streamlit -> Coordinator -> agent ->
FastAPI endpoint -> model), this agent is an HTTP client of the existing
live endpoints — it does not import the price/sale/finance modules
directly. That keeps model loading + serving in one place (the API
process) and lets this agent run anywhere the API is reachable.

Flow:
    property spec -> /predict_price + /explain_price + /airbnb_vs_sell
                     + /estimate_revenue -> gather_analysis() (pure, no LLM)
                  -> narrate() (Gemini) -> analysis dict with a narrative
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
        """Call the FastAPI endpoints and assemble a structured analysis.

        ``property_spec`` accepts the same fields as ``/airbnb_vs_sell``
        (``city``, ``neighbourhood_cleansed``, ``sq_m``, plus any
        ``/predict_price`` fields like ``accommodates``/``bedrooms``).
        Deterministic and LLM-free, so it is fully unit-testable without a
        live Gemini key.
        """
        spec = dict(property_spec)
        city = spec.get("city", "madrid")

        price_resp = self._post("/predict_price", spec)
        explain_resp = self._post("/explain_price", spec)
        vs_sell_resp = self._post(
            "/airbnb_vs_sell",
            {
                "city": city,
                "neighbourhood_cleansed": spec.get("neighbourhood_cleansed"),
                "sq_m": spec.get("sq_m"),
                "holding_years": spec.get("holding_years", 10),
                "discount_rate": spec.get("discount_rate", 0.07),
                **{k: v for k, v in spec.items() if k not in ("city", "sq_m")},
            },
        )
        revenue_resp = self._post(
            "/estimate_revenue",
            {
                "price_per_night": price_resp["price_per_night"],
                "occupancy_rate": spec.get("occupancy_rate", 0.55),
                "city": city,
            },
        )

        return {
            "city": city,
            "neighbourhood": spec.get("neighbourhood_cleansed", "city-wide"),
            "nightly_price_eur": price_resp["price_per_night"],
            "drivers": explain_resp["drivers"],
            "recommendation": vs_sell_resp["recommendation"],
            "npv_airbnb_p50_eur": vs_sell_resp["npv_airbnb_p50_eur"],
            "npv_sell_eur": vs_sell_resp["npv_sell_eur"],
            "break_even_years": vs_sell_resp["break_even_years"],
            "confidence": vs_sell_resp["confidence"],
            "annual_gross_eur": revenue_resp["annual_gross_eur"],
            "annual_net_eur": revenue_resp["annual_net_eur"],
            "p10_eur": revenue_resp["p10_eur"],
            "p90_eur": revenue_resp["p90_eur"],
            "disclaimer": vs_sell_resp.get("disclaimer", DISCLAIMER),
        }

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
