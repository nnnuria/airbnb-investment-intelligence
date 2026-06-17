"""Tests for airbnb_iip.finance.costs."""
import pytest

from airbnb_iip.finance.costs import (
    annual_gross_revenue,
    cleaning_cost_annual,
    compute_noi,
    tourist_tax_annual,
)


# ── annual_gross_revenue ──────────────────────────────────────────────────────

def test_annual_gross_revenue_basic():
    assert annual_gross_revenue(100.0, 200.0) == pytest.approx(20_000.0)


def test_annual_gross_revenue_zero_nights():
    assert annual_gross_revenue(150.0, 0.0) == 0.0


@pytest.mark.parametrize("price,nights", [(-1.0, 100.0), (100.0, -1.0)])
def test_annual_gross_revenue_rejects_negative(price, nights):
    with pytest.raises(ValueError):
        annual_gross_revenue(price, nights)


# ── tourist_tax_annual ────────────────────────────────────────────────────────

def test_tourist_tax_barcelona_nonzero():
    # 4.40 EUR/guest/night * 2 guests * 100 nights = 880
    tax = tourist_tax_annual(100.0, city="barcelona", guests_per_night=2.0)
    assert tax == pytest.approx(880.0)


def test_tourist_tax_madrid_is_zero():
    assert tourist_tax_annual(100.0, city="madrid") == 0.0


def test_tourist_tax_malaga_is_zero():
    assert tourist_tax_annual(100.0, city="malaga") == 0.0


def test_tourist_tax_case_insensitive_city():
    a = tourist_tax_annual(50.0, city="Barcelona")
    b = tourist_tax_annual(50.0, city="barcelona")
    assert a == b


def test_tourist_tax_custom_rate_overrides_city():
    tax = tourist_tax_annual(10.0, city="madrid", rate_per_guest_night=1.0, guests_per_night=1.0)
    assert tax == pytest.approx(10.0)


def test_tourist_tax_rejects_negative_nights():
    with pytest.raises(ValueError):
        tourist_tax_annual(-5.0, city="barcelona")


def test_tourist_tax_rejects_nonpositive_guests():
    with pytest.raises(ValueError):
        tourist_tax_annual(10.0, city="barcelona", guests_per_night=0)


# ── cleaning_cost_annual ────────────────────────────────────────────────────

def test_cleaning_cost_basic():
    # 219 nights / 3 nights per stay = 73 turnovers; * 50 = 3650
    cost = cleaning_cost_annual(219.0, avg_length_of_stay=3.0, cost_per_clean_eur=50.0)
    assert cost == pytest.approx(3650.0)


def test_cleaning_cost_self_cleaning_is_zero():
    cost = cleaning_cost_annual(219.0, self_cleaning=True)
    assert cost == 0.0


def test_cleaning_cost_uses_config_default_when_not_given():
    from airbnb_iip.config import FINANCE

    cost = cleaning_cost_annual(90.0, avg_length_of_stay=3.0)
    expected = (90.0 / 3.0) * FINANCE["cleaning_fee_per_stay_eur"]
    assert cost == pytest.approx(expected)


def test_cleaning_cost_rejects_negative_nights():
    with pytest.raises(ValueError):
        cleaning_cost_annual(-1.0)


def test_cleaning_cost_rejects_nonpositive_avg_stay():
    with pytest.raises(ValueError):
        cleaning_cost_annual(100.0, avg_length_of_stay=0)


# ── compute_noi ──────────────────────────────────────────────────────────────

def test_compute_noi_basic_breakdown_sums_correctly():
    out = compute_noi(
        gross_revenue=20_000.0,
        property_value=300_000.0,
        platform_fee_rate=0.03,
        cleaning_cost_eur=2000.0,
        tourist_tax_eur=0.0,
        utilities_eur=3000.0,
        ibi_eur=500.0,
        basuras_eur=150.0,
        community_fee_eur=1200.0,
        insurance_eur=700.0,
        maintenance_rate=0.0125,
        management_fee_rate=0.0,
        accounting_fee_eur=1000.0,
        effective_tax_rate=0.20,
    )
    maintenance = 0.0125 * 300_000.0
    platform_fee = 0.03 * 20_000.0
    pre_tax = 20_000.0 - (
        platform_fee + 2000.0 + 0.0 + 3000.0 + 500.0 + 150.0 + 1200.0 + 700.0
        + maintenance + 0.0 + 1000.0
    )
    expected_tax = 0.20 * max(pre_tax, 0.0)
    assert out["pre_tax_noi"] == pytest.approx(pre_tax)
    assert out["tax"] == pytest.approx(expected_tax)
    assert out["noi"] == pytest.approx(pre_tax - expected_tax)


def test_compute_noi_loss_year_pays_no_tax():
    out = compute_noi(
        gross_revenue=1000.0,
        property_value=300_000.0,
        maintenance_rate=0.0125,  # 3750 alone already exceeds gross revenue
        management_fee_rate=0.0,
        effective_tax_rate=0.30,
    )
    assert out["pre_tax_noi"] < 0
    assert out["tax"] == 0.0
    assert out["noi"] == pytest.approx(out["pre_tax_noi"])


def test_compute_noi_management_fee_scales_with_revenue():
    base = compute_noi(gross_revenue=10_000.0, property_value=200_000.0, management_fee_rate=0.0)
    managed = compute_noi(gross_revenue=10_000.0, property_value=200_000.0, management_fee_rate=0.20)
    assert managed["management_fee"] == pytest.approx(2000.0)
    assert managed["noi"] < base["noi"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"platform_fee_rate": 1.5},
        {"maintenance_rate": -0.1},
        {"management_fee_rate": 2.0},
        {"effective_tax_rate": -0.01},
    ],
)
def test_compute_noi_rejects_invalid_rates(kwargs):
    with pytest.raises(ValueError):
        compute_noi(gross_revenue=10_000.0, property_value=100_000.0, **kwargs)


def test_compute_noi_rejects_negative_amounts():
    with pytest.raises(ValueError):
        compute_noi(gross_revenue=-10.0, property_value=100_000.0)