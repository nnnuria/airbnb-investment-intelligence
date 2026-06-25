"""Financial Analyst Agent — deep financial Q&A: IRR, NPV, tax impact, cost breakdown.

Handles chat intents like "what is the IRR?", "show cost breakdown", "how much does
IBI cost me?", "explain the tax impact", "compare pre- and post-tax NPV".

Follows the same pattern as MarketAnalystAgent: pure data-gathering functions +
optional Gemini narration, with a deterministic fallback for demo/test mode.
"""

from __future__ import annotations

import os
from typing import Any, Mapping

DISCLAIMER = "Indicative only — not financial or tax advice."

SYSTEM_PROMPT = """You are a financial analyst for an Airbnb investment platform,
helping a Spanish property owner understand the detailed financial analysis behind
an Airbnb-vs-sell recommendation. Answer in clear English for a non-expert owner.
Use ONLY the numbers provided — do not invent figures.

Rules:
- Address the user's specific question directly and concisely (4-8 sentences).
- If IRR is available, explain what it means in plain terms.
- When showing pre-tax vs. post-tax NPV, explain the difference clearly.
- If cost breakdown is requested, present it as a markdown table.
- Reference IBI and setup cost if they are material (>5% of gross revenue).
- End every answer with exactly: "⚠️ Indicative only — not financial or tax advice."

Property context:
- City: {city}
- Annual gross revenue: €{gross_revenue_year_eur}
- Annual net revenue (after-tax): €{net_revenue_year_eur}
- NPV Airbnb (10yr, P50, after-tax): €{npv_airbnb_p50_eur}
- NPV Airbnb (10yr, P50, pre-tax): €{npv_airbnb_pretax_p50_eur}
- NPV Sell (after-tax / post-CGT): €{npv_sell_eur}
- NPV Sell (pre-tax / no CGT): €{npv_sell_pretax_eur}
- IRR (Airbnb, 10yr): {irr_str}
- IBI applied: €{ibi_eur_used}/year
- Setup cost: €{setup_cost_eur_used}
- Cost breakdown: {cost_breakdown_str}

User question: {question}

Answer:"""


def build_financial_context(scenario: Mapping[str, Any]) -> dict[str, Any]:
    """Extract financial fields from a scenario dict for the prompt."""
    irr = scenario.get("irr_airbnb_pct")
    irr_str = f"{irr * 100:.1f}%" if irr is not None else "not available"

    cost_breakdown = scenario.get("cost_breakdown", [])
    cost_breakdown_str = ", ".join(
        f"{label} €{value:,.0f}" for label, value in cost_breakdown
    ) if cost_breakdown else "not available"

    return {
        "city": scenario.get("city", "unknown"),
        "gross_revenue_year_eur": f"{scenario.get('gross_revenue_year_eur', 0):,.0f}",
        "net_revenue_year_eur": f"{scenario.get('net_revenue_year_eur', 0):,.0f}",
        "npv_airbnb_p50_eur": f"{scenario.get('npv_airbnb_p50_eur', 0):,.0f}",
        "npv_airbnb_pretax_p50_eur": f"{scenario.get('npv_airbnb_pretax_p50_eur', 0):,.0f}",
        "npv_sell_eur": f"{scenario.get('npv_sell_eur', 0):,.0f}",
        "npv_sell_pretax_eur": f"{scenario.get('npv_sell_pretax_eur', 0):,.0f}",
        "irr_str": irr_str,
        "ibi_eur_used": f"{scenario.get('ibi_eur_used', 0):,.0f}",
        "ibi_method": scenario.get("ibi_method", "estimated"),
        "ibi_explanation": scenario.get("ibi_explanation", ""),
        "basuras_eur_used": f"{scenario.get('basuras_eur_used', 0):,.0f}",
        "setup_cost_eur_used": f"{scenario.get('setup_cost_eur_used', 0):,.0f}",
        "cost_breakdown_str": cost_breakdown_str,
    }


def cost_breakdown_table(scenario: Mapping[str, Any]) -> str:
    """Render the cost breakdown as a markdown table."""
    rows = scenario.get("cost_breakdown", [])
    if not rows:
        return "Cost breakdown not available."
    total = sum(v for _, v in rows)
    lines = ["| Cost item | Amount (€/yr) | Share of gross |",
             "|-----------|--------------|----------------|"]
    gross = scenario.get("gross_revenue_year_eur", 1) or 1
    for label, value in rows:
        pct = value / gross * 100
        lines.append(f"| {label} | €{value:,.0f} | {pct:.1f}% |")
    lines.append(f"| **Total costs** | **€{total:,.0f}** | **{total/gross*100:.1f}%** |")
    return "\n".join(lines)


class FinancialAnalystAgent:
    """Narrates IRR, NPV (pre/post tax), IBI impact, and cost breakdown."""

    def __init__(self, use_llm: bool = True):
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
                max_output_tokens=500,
                thinking_budget=0,
            )

    def answer(self, question: str, scenario: Mapping[str, Any]) -> str:
        """Answer a financial question about the given scenario."""
        # Always handle cost breakdown questions deterministically
        q_lower = question.lower()
        if any(kw in q_lower for kw in ("cost breakdown", "break down", "cost table", "itemis", "itemiz")):
            table = cost_breakdown_table(scenario)
            return (
                f"Here is the annual cost breakdown for your Airbnb:\n\n{table}\n\n"
                f"⚠️ {DISCLAIMER}"
            )

        if not self.use_llm:
            return self._fallback_answer(question, scenario)

        ctx = build_financial_context(scenario)
        prompt = SYSTEM_PROMPT.format(question=question, **ctx)
        response = self._llm.invoke(prompt)
        text = response.content if hasattr(response, "content") else str(response)
        if "Indicative only" not in text:
            text = text.rstrip() + f"\n\n⚠️ {DISCLAIMER}"
        return text

    def _fallback_answer(self, question: str, scenario: Mapping[str, Any]) -> str:
        """Deterministic answer — used in demo/test mode (no LLM)."""
        ctx = build_financial_context(scenario)
        q_lower = question.lower()

        if "irr" in q_lower:
            return (
                f"The Internal Rate of Return (IRR) for holding this property as an Airbnb "
                f"over 10 years is **{ctx['irr_str']}**. This represents the annualised "
                f"return on your upfront setup investment of €{ctx['setup_cost_eur_used']}, "
                f"accounting for all NOI cash flows and the terminal sale proceeds. "
                f"Compare this to your opportunity cost of capital (typically 6–9%) to "
                f"judge whether Airbnb is worthwhile.\n\n⚠️ {DISCLAIMER}"
            )

        if "tax" in q_lower or "irpf" in q_lower or "income tax" in q_lower:
            return (
                f"Income tax (IRPF) reduces your annual net revenue from "
                f"€{ctx['npv_airbnb_pretax_p50_eur']} (pre-tax NPV) to "
                f"€{ctx['npv_airbnb_p50_eur']} (post-tax NPV) over the 10-year holding period. "
                f"Progressive IRPF brackets (19–47%) are applied to the STR profit each year. "
                f"On the sell side, capital gains tax reduces proceeds from "
                f"€{ctx['npv_sell_pretax_eur']} to €{ctx['npv_sell_eur']}.\n\n⚠️ {DISCLAIMER}"
            )

        if "ibi" in q_lower or "property tax" in q_lower:
            explanation = ctx.get("ibi_explanation", "")
            method_label = {"cadastral": "exact cadastral value", "manual": "manual override", "estimated": "estimated from market price"}.get(
                ctx.get("ibi_method", "estimated"), "estimated from market price"
            )
            base = (
                f"IBI (Impuesto sobre Bienes Inmuebles) is the annual Spanish property tax. "
                f"For this property, we apply **€{ctx['ibi_eur_used']}/year** ({method_label}), "
                f"which is included in the cost breakdown."
            )
            if explanation:
                base += f"\n\n_{explanation}_"
            base += f"\n\n⚠️ {DISCLAIMER}"
            return base

        if any(kw in q_lower for kw in ("npv", "net present value", "value")):
            return (
                f"NPV comparison (10-year horizon):\n"
                f"- **Airbnb (after-tax):** €{ctx['npv_airbnb_p50_eur']}\n"
                f"- **Airbnb (pre-tax):** €{ctx['npv_airbnb_pretax_p50_eur']}\n"
                f"- **Sell (after CGT):** €{ctx['npv_sell_eur']}\n"
                f"- **Sell (pre-tax):** €{ctx['npv_sell_pretax_eur']}\n\n"
                f"The pre-tax figures show the gross economic value before Spanish income tax "
                f"and capital gains tax (CGT) are applied.\n\n⚠️ {DISCLAIMER}"
            )

        # Generic financial summary
        return (
            f"Financial summary: Annual gross €{ctx['gross_revenue_year_eur']}, "
            f"net (after-tax) €{ctx['net_revenue_year_eur']}. "
            f"IRR: {ctx['irr_str']}. "
            f"NPV Airbnb (10yr P50): €{ctx['npv_airbnb_p50_eur']} vs. "
            f"sell proceeds €{ctx['npv_sell_eur']}. "
            f"IBI: €{ctx['ibi_eur_used']}/yr. "
            f"Setup cost: €{ctx['setup_cost_eur_used']}.\n\n⚠️ {DISCLAIMER}"
        )
