"""Tests for the Governance layer."""

from __future__ import annotations

from airbnb_iip.agents.governance import (
    DISCLAIMER_FINANCIAL,
    DISCLAIMER_REGULATORY,
    GovernanceGuard,
    apply_financial_guardrails,
    apply_regulatory_guardrails,
    model_card_summary,
)

GOOD_ANALYSIS = {
    "recommendation": "Airbnb",
    "nightly_price_eur": 150.0,
    "annual_gross_eur": 30000.0,
    "annual_net_eur": 20000.0,
    "npv_sell_eur": 300000.0,
    "p10_eur": 15000.0,
    "p90_eur": 25000.0,
}


def test_clean_analysis_passes_with_no_violations():
    result = apply_financial_guardrails(GOOD_ANALYSIS)
    assert result["governance"]["passed"] is True
    assert result["governance"]["violations"] == []
    assert result["governance"]["human_review_required"] is False


def test_disclaimer_is_added_when_missing():
    result = apply_financial_guardrails(GOOD_ANALYSIS)
    assert result["disclaimer"] == DISCLAIMER_FINANCIAL


def test_existing_disclaimer_is_not_overwritten():
    analysis = {**GOOD_ANALYSIS, "disclaimer": "custom disclaimer"}
    result = apply_financial_guardrails(analysis)
    assert result["disclaimer"] == "custom disclaimer"


def test_apply_does_not_mutate_input():
    original = dict(GOOD_ANALYSIS)
    apply_financial_guardrails(GOOD_ANALYSIS)
    assert GOOD_ANALYSIS == original


def test_negative_net_revenue_is_flagged_as_warning():
    analysis = {**GOOD_ANALYSIS, "annual_net_eur": -500.0}
    result = apply_financial_guardrails(analysis)
    violations = result["governance"]["violations"]
    assert any(v["field"] == "annual_net_eur" and v["severity"] == "warning" for v in violations)
    assert result["governance"]["passed"] is True  # warning, not blocked


def test_implausible_gross_yield_is_flagged():
    analysis = {**GOOD_ANALYSIS, "annual_gross_eur": 150000.0, "npv_sell_eur": 300000.0}
    result = apply_financial_guardrails(analysis)
    violations = result["governance"]["violations"]
    assert any(v["field"] == "annual_gross_eur" for v in violations)


def test_nightly_price_below_floor_is_blocked():
    analysis = {**GOOD_ANALYSIS, "nightly_price_eur": 2.0}
    result = apply_financial_guardrails(analysis)
    assert result["governance"]["passed"] is False
    assert result["governance"]["human_review_required"] is True


def test_inverted_uncertainty_band_is_blocked():
    analysis = {**GOOD_ANALYSIS, "p10_eur": 30000.0, "p90_eur": 10000.0}
    result = apply_financial_guardrails(analysis)
    assert result["governance"]["passed"] is False


def test_marginal_recommendation_requires_human_review():
    analysis = {**GOOD_ANALYSIS, "recommendation": "marginal"}
    result = apply_financial_guardrails(analysis)
    assert result["governance"]["human_review_required"] is True


def test_partial_analysis_does_not_crash():
    """Missing fields are skipped, not treated as failures."""
    result = apply_financial_guardrails({"recommendation": "Airbnb"})
    assert result["governance"]["passed"] is True


def test_regulatory_disclaimer_and_high_risk_review():
    reg_result = {"risk_flag": "HIGH", "answer": "...", "sources": ["doc.pdf"]}
    result = apply_regulatory_guardrails(reg_result)
    assert result["disclaimer"] == DISCLAIMER_REGULATORY
    assert result["governance"]["human_review_required"] is True


def test_regulatory_low_risk_does_not_require_review():
    reg_result = {"risk_flag": "LOW", "answer": "...", "sources": ["doc.pdf"]}
    result = apply_regulatory_guardrails(reg_result)
    assert result["governance"]["human_review_required"] is False


def test_model_card_summary_reflects_saved_artifacts():
    cards = model_card_summary()
    names = {c["name"] for c in cards}
    assert "Airbnb nightly price" in names
    assert "Sale price" in names
    for card in cards:
        assert card["card_path"].startswith("docs/model_cards/")


def test_guard_is_stateless_across_instances():
    a, b = GovernanceGuard(), GovernanceGuard()
    assert a.check_financial_bounds(GOOD_ANALYSIS) == []
    assert b.check_financial_bounds(GOOD_ANALYSIS) == []
