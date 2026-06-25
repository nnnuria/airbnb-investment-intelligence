"""Property Tax Agent (IbiAgent) — resolves IBI and narrates the derivation.

IBI (Impuesto sobre Bienes Inmuebles) is the annual Spanish municipal property
tax. Its formula is:

    IBI_€ = municipal_rate[city] × cadastral_value

When the owner's cadastral value is unknown, the agent estimates it from the
modelled market sale price using a city-specific cadastral-to-market ratio.

The agent has two responsibilities:

1. **resolve()** — compute ``ibi_eur`` and return a structured ``IbiEstimate``
   with the method, confidence, and a one-sentence explanation. Called from
   ``compute_scenario()`` so the estimate flows into the NOI calculation.

2. **narrate()** — expand the estimate into a paragraph suitable for chat or
   the UI tooltip. Uses Gemini 2.5 Flash when ``use_llm=True``; deterministic
   fallback otherwise.

Sources embedded in this module:
- Madrid 0.428% (2025): idealista / Hacienda consolidated PDF
- Barcelona 0.66%: Ajuntament OF 1.1 IBI 2025
- Málaga 0.451%: Gestrisam / Ayto. Málaga (frozen since 2014)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

from airbnb_iip.config import (
    CADASTRAL_TO_MARKET_RATIO,
    CADASTRAL_TO_MARKET_RATIO_DEFAULT,
    IBI_RATE_BY_CITY,
    IBI_RATE_DEFAULT,
)

DISCLAIMER = "Indicative only — not financial or tax advice. Verify against your IBI bill."

_NARRATE_PROMPT = """You are a Spanish property tax specialist for an investment platform.
Explain the IBI (Impuesto sobre Bienes Inmuebles) figure to a non-expert property owner
in 3-4 clear English sentences. Use ONLY the numbers provided. End every answer with
exactly: "⚠️ Indicative only — not financial or tax advice. Verify against your IBI bill."

IBI details:
- City: {city}
- Method: {method}
- IBI amount: €{ibi_eur:.0f}/year
- Municipal rate: {rate_pct:.3f}%
- Cadastral value used: €{cadastral_value_used:,.0f}
- Confidence: {confidence}
- One-line derivation: {explanation}

Answer:"""


@dataclass
class IbiEstimate:
    """Structured result from IbiAgent.resolve()."""

    ibi_eur: float
    cadastral_value_used: float
    method: Literal["cadastral", "estimated", "manual"]
    confidence: Literal["high", "medium", "low"]
    explanation: str
    rate_pct: float
    city: str


class IbiAgent:
    """Resolves IBI and narrates the derivation.

    Stateless — create a new instance per call or reuse across the process.
    Does not make HTTP calls; all computation is local.
    """

    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm
        self._llm = None
        if use_llm:
            api_key = os.getenv("GEMINI_API_KEY")
            if api_key:
                from langchain_google_genai import ChatGoogleGenerativeAI
                self._llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash",
                    google_api_key=api_key,
                    temperature=0.1,
                    max_output_tokens=300,
                    thinking_budget=0,
                )

    # ── Resolve ───────────────────────────────────────────────────────────────

    def resolve(
        self,
        city: str,
        *,
        market_value: float,
        cadastral_value: float | None = None,
        ibi_eur_override: float | None = None,
    ) -> IbiEstimate:
        """Compute IBI and return a structured IbiEstimate.

        Priority order:
        1. ``ibi_eur_override`` — owner provided a manual figure
        2. ``cadastral_value`` — exact: rate × cadastral_value
        3. Fallback — estimated: rate × (market_value × ratio)
        """
        city_key = city.strip().lower()
        rate = IBI_RATE_BY_CITY.get(city_key, IBI_RATE_DEFAULT)
        rate_pct = rate * 100

        if ibi_eur_override is not None:
            return IbiEstimate(
                ibi_eur=ibi_eur_override,
                cadastral_value_used=ibi_eur_override / rate if rate > 0 else 0.0,
                method="manual",
                confidence="high",
                explanation=(
                    f"Using your manually specified IBI of €{ibi_eur_override:,.0f}/yr."
                ),
                rate_pct=rate_pct,
                city=city_key,
            )

        if cadastral_value is not None and cadastral_value > 0:
            ibi_eur = rate * cadastral_value
            return IbiEstimate(
                ibi_eur=ibi_eur,
                cadastral_value_used=cadastral_value,
                method="cadastral",
                confidence="high",
                explanation=(
                    f"IBI = {rate_pct:.3f}% × cadastral value "
                    f"€{cadastral_value:,.0f} = €{ibi_eur:,.0f}/yr."
                ),
                rate_pct=rate_pct,
                city=city_key,
            )

        # Fallback: estimate cadastral value from market price
        ratio = CADASTRAL_TO_MARKET_RATIO.get(city_key, CADASTRAL_TO_MARKET_RATIO_DEFAULT)
        estimated_cadastral = market_value * ratio
        ibi_eur = rate * estimated_cadastral
        city_label = {"madrid": "Madrid", "barcelona": "Barcelona", "malaga": "Málaga"}.get(city_key, city_key.title())
        return IbiEstimate(
            ibi_eur=ibi_eur,
            cadastral_value_used=estimated_cadastral,
            method="estimated",
            confidence="medium",
            explanation=(
                f"IBI estimated at €{ibi_eur:,.0f}/yr: {city_label} 2025 rate "
                f"{rate_pct:.3f}% × estimated cadastral value "
                f"€{estimated_cadastral:,.0f} (≈{ratio*100:.0f}% of modelled "
                f"market price €{market_value:,.0f}). "
                f"Check your IBI bill for the exact cadastral value."
            ),
            rate_pct=rate_pct,
            city=city_key,
        )

    # ── Narrate ───────────────────────────────────────────────────────────────

    def narrate(self, estimate: IbiEstimate) -> str:
        """Return a paragraph explaining the IBI estimate."""
        if self.use_llm and self._llm is not None:
            return self._llm_narrate(estimate)
        return self._fallback_narrate(estimate)

    def _llm_narrate(self, estimate: IbiEstimate) -> str:
        prompt = _NARRATE_PROMPT.format(
            city=estimate.city.title(),
            method=estimate.method,
            ibi_eur=estimate.ibi_eur,
            rate_pct=estimate.rate_pct,
            cadastral_value_used=estimate.cadastral_value_used,
            confidence=estimate.confidence,
            explanation=estimate.explanation,
        )
        try:
            resp = self._llm.invoke(prompt)
            text = resp.content if hasattr(resp, "content") else str(resp)
            if "Indicative only" not in text:
                text = text.rstrip() + f"\n\n⚠️ {DISCLAIMER}"
            return text
        except Exception:
            return self._fallback_narrate(estimate)

    def _fallback_narrate(self, estimate: IbiEstimate) -> str:
        """Deterministic narration — no LLM dependency."""
        city_label = {
            "madrid": "Madrid", "barcelona": "Barcelona", "malaga": "Málaga"
        }.get(estimate.city, estimate.city.title())

        if estimate.method == "manual":
            return (
                f"IBI (Impuesto sobre Bienes Inmuebles) is the annual Spanish property tax. "
                f"You have specified €{estimate.ibi_eur:,.0f}/yr directly, which will be "
                f"used as-is in the analysis.\n\n⚠️ {DISCLAIMER}"
            )

        if estimate.method == "cadastral":
            return (
                f"IBI is €{estimate.ibi_eur:,.0f}/yr: {city_label}'s 2025 municipal rate "
                f"of {estimate.rate_pct:.3f}% applied to your cadastral value "
                f"(valor catastral) of €{estimate.cadastral_value_used:,.0f}. "
                f"The cadastral value is set by the municipal assessment and appears on "
                f"your IBI bill — it is typically lower than the market price.\n\n⚠️ {DISCLAIMER}"
            )

        # method == "estimated"
        ratio_pct = estimate.cadastral_value_used / (estimate.ibi_eur / (estimate.rate_pct / 100)) * 100 if estimate.rate_pct > 0 else 0
        return (
            f"IBI is estimated at €{estimate.ibi_eur:,.0f}/yr for {city_label}. "
            f"Since your cadastral value (valor catastral) was not provided, it was "
            f"approximated as {CADASTRAL_TO_MARKET_RATIO.get(estimate.city, CADASTRAL_TO_MARKET_RATIO_DEFAULT)*100:.0f}% "
            f"of the modelled market price — giving an estimated cadastral value of "
            f"€{estimate.cadastral_value_used:,.0f}. Applying {city_label}'s 2025 "
            f"municipal rate of {estimate.rate_pct:.3f}% yields €{estimate.ibi_eur:,.0f}/yr. "
            f"For the exact figure, check your IBI bill or the Agencia Tributaria "
            f"portal — the actual amount may differ.\n\n⚠️ {DISCLAIMER}"
        )
