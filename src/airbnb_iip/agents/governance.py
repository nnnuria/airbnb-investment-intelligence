"""
Governance Agent — output-bounds guardrails, disclaimers, model cards.
src/airbnb_iip/agents/governance.py

Per docs/UC2_Ordered_Task_Backlog.md Phase 10 #53: wraps Market Analyst
and Regulatory agent outputs before they reach the Coordinator. Pure,
dependency-free checks on plain dicts — this is a cross-cutting layer,
not another HTTP client or data service, so it has no __init__ state.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping

ROOT = Path(__file__).resolve().parents[3]

DISCLAIMER_FINANCIAL = "Indicative only — not financial advice."
DISCLAIMER_REGULATORY = (
    "Indicative only — not legal advice. "
    "Verify against current official municipal/regional sources."
)

# Plausibility bounds — deliberately generous (catch genuine errors, not
# flag every aggressive-but-real market). Per the diagram's explicit
# guardrails: "no negative revenue, no implausible yields".
MIN_PLAUSIBLE_NIGHTLY_EUR = 10.0
MAX_PLAUSIBLE_NIGHTLY_EUR = 2000.0
MAX_PLAUSIBLE_GROSS_YIELD = 0.25  # annual gross Airbnb revenue / sale value


@dataclass
class GuardrailViolation:
    field: str
    message: str
    severity: Literal["warning", "blocked"]

    def to_dict(self) -> dict[str, str]:
        return {"field": self.field, "message": self.message, "severity": self.severity}


@dataclass
class GovernanceReport:
    violations: list[GuardrailViolation] = field(default_factory=list)
    human_review_required: bool = False

    @property
    def passed(self) -> bool:
        """False if any *blocked*-severity violation is present. Warnings
        don't block — they're surfaced for transparency, not suppression."""
        return not any(v.severity == "blocked" for v in self.violations)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "human_review_required": self.human_review_required,
            "violations": [v.to_dict() for v in self.violations],
        }


class GovernanceGuard:
    """Output-bounds guardrails + disclaimers for the agent layer."""

    # ── Financial (Market Analyst) ──────────────────────────────────────────

    def check_financial_bounds(self, analysis: Mapping[str, Any]) -> list[GuardrailViolation]:
        """Check a Market Analyst ``gather_analysis``/``analyze`` result
        against plausibility bounds. Missing fields are skipped, not
        treated as failures — callers may pass a partial dict."""
        violations: list[GuardrailViolation] = []

        nightly = analysis.get("nightly_price_eur")
        if nightly is not None:
            if nightly < MIN_PLAUSIBLE_NIGHTLY_EUR:
                violations.append(GuardrailViolation(
                    "nightly_price_eur",
                    f"€{nightly:.0f}/night is below the plausible floor "
                    f"(€{MIN_PLAUSIBLE_NIGHTLY_EUR:.0f}) — likely a bad input spec.",
                    "blocked",
                ))
            elif nightly > MAX_PLAUSIBLE_NIGHTLY_EUR:
                violations.append(GuardrailViolation(
                    "nightly_price_eur",
                    f"€{nightly:.0f}/night exceeds the plausible ceiling "
                    f"(€{MAX_PLAUSIBLE_NIGHTLY_EUR:.0f}) — verify the input spec.",
                    "warning",
                ))

        net = analysis.get("annual_net_eur")
        if net is not None and net < 0:
            violations.append(GuardrailViolation(
                "annual_net_eur",
                f"Projected net revenue is negative (€{net:.0f}) — this property would "
                "lose money as an Airbnb under these assumptions.",
                "warning",
            ))

        gross = analysis.get("annual_gross_eur")
        sale_value = analysis.get("npv_sell_eur")
        if gross is not None and sale_value:
            gross_yield = gross / sale_value
            if gross_yield > MAX_PLAUSIBLE_GROSS_YIELD:
                violations.append(GuardrailViolation(
                    "annual_gross_eur",
                    f"Implied gross yield {gross_yield:.0%} exceeds the realistic ceiling "
                    f"({MAX_PLAUSIBLE_GROSS_YIELD:.0%}) for this market — treat with caution.",
                    "warning",
                ))

        p10, p90 = analysis.get("p10_eur"), analysis.get("p90_eur")
        if p10 is not None and p90 is not None and p10 > p90:
            violations.append(GuardrailViolation(
                "p10_eur",
                "P10 exceeds P90 — the uncertainty band is inverted, likely a bug upstream.",
                "blocked",
            ))

        return violations

    def apply_financial(self, analysis: Mapping[str, Any]) -> dict[str, Any]:
        """Wrap a Market Analyst result with a governance report and ensure
        the disclaimer is present. Does not mutate ``analysis``."""
        violations = self.check_financial_bounds(analysis)
        report = GovernanceReport(
            violations=violations,
            human_review_required=(
                analysis.get("recommendation") == "marginal"
                or any(v.severity == "blocked" for v in violations)
            ),
        )
        result = dict(analysis)
        result.setdefault("disclaimer", DISCLAIMER_FINANCIAL)
        result["governance"] = report.to_dict()
        return result

    # ── Regulatory ───────────────────────────────────────────────────────────

    def apply_regulatory(self, reg_result: Mapping[str, Any]) -> dict[str, Any]:
        """Wrap a Regulatory agent ``query``/``get_risk_flag`` result. Flags
        for human review when risk is HIGH or could not be determined."""
        result = dict(reg_result)
        result.setdefault("disclaimer", DISCLAIMER_REGULATORY)
        report = GovernanceReport(
            human_review_required=reg_result.get("risk_flag") in ("HIGH", "UNKNOWN"),
        )
        result["governance"] = report.to_dict()
        return result

    # ── Model cards ──────────────────────────────────────────────────────────

    def model_card_summary(self) -> list[dict[str, Any]]:
        """Lightweight registry of production models for a UI governance
        panel — pulled live from the saved model metadata files so the
        numbers can't drift out of sync with what's actually deployed."""
        cards: list[dict[str, Any]] = []

        price_meta_path = ROOT / "models" / "price_feature_cols.json"
        if price_meta_path.exists():
            meta = json.loads(price_meta_path.read_text())
            cards.append({
                "name": "Airbnb nightly price",
                "model": meta.get("best_model"),
                "target": "log1p(price)",
                "n_features": len(meta.get("selected_features", [])),
                "card_path": "docs/model_cards/price_model.md",
            })

        sale_meta_path = ROOT / "models" / "sale_feature_cols.json"
        if sale_meta_path.exists():
            meta = json.loads(sale_meta_path.read_text())
            cards.append({
                "name": "Sale price",
                "model": meta.get("best_model"),
                "target": meta.get("target"),
                "test_metrics": meta.get("test_metrics"),
                "card_path": "docs/model_cards/sale_model.md",
            })

        return cards


# Module-level convenience wrappers — most callers don't need their own
# GovernanceGuard instance (it carries no state).
_default_guard = GovernanceGuard()


def apply_financial_guardrails(analysis: Mapping[str, Any]) -> dict[str, Any]:
    return _default_guard.apply_financial(analysis)


def apply_regulatory_guardrails(reg_result: Mapping[str, Any]) -> dict[str, Any]:
    return _default_guard.apply_regulatory(reg_result)


def model_card_summary() -> list[dict[str, Any]]:
    return _default_guard.model_card_summary()
