"""Tests for airbnb_iip.finance.scenarios — NPV comparison + uncertainty bands."""
import pytest

from airbnb_iip.finance.scenarios import (
    break_even_horizon,
    capital_gains_tax,
    monte_carlo,
    npv_airbnb,
    npv_sell,
    recommend,
)


# ── capital_gains_tax ──────────────────────────────────────────────────────────

def test_capital_gains_tax_known_value():
    # gain = 100,000 -> 6000*0.19 + 44000*0.21 + 50000*0.23 = 1140+9240+11500=21880
    assert capital_gains_tax(100_000.0) == pytest.approx(21_880.0)


def test_capital_gains_tax_zero_or_negative_gain():
    assert capital_gains_tax(0.0) == 0.0
    assert capital_gains_tax(-5_000.0) == 0.0


def test_capital_gains_tax_exempt():
    assert capital_gains_tax(100_000.0, exempt=True) == 0.0


def test_capital_gains_tax_within_first_bracket():
    assert capital_gains_tax(5_000.0) == pytest.approx(5_000.0 * 0.19)


def test_capital_gains_tax_rejects_non_increasing_brackets():
    with pytest.raises(ValueError):
        capital_gains_tax(10_000.0, brackets=[(50_000, 0.2), (6_000, 0.1)])


# ── npv_sell ───────────────────────────────────────────────────────────────────

def test_npv_sell_basic():
    out = npv_sell(
        400_000.0,
        purchase_price=300_000.0,
        agent_rate=0.05,
        notary_rate=0.01,
    )
    assert out["gain"] == pytest.approx(100_000.0)
    assert out["transaction_costs"] == pytest.approx(400_000.0 * 0.06)
    assert out["cgt"] == pytest.approx(capital_gains_tax(100_000.0))
    assert out["net_proceeds"] == pytest.approx(
        400_000.0 - out["transaction_costs"] - out["cgt"]
    )


def test_npv_sell_exempt_skips_cgt():
    out = npv_sell(400_000.0, purchase_price=300_000.0, cgt_exempt=True)
    assert out["cgt"] == 0.0


def test_npv_sell_rejects_negative_price():
    with pytest.raises(ValueError):
        npv_sell(-1.0, purchase_price=100_000.0)


# ── npv_airbnb ─────────────────────────────────────────────────────────────────

def test_npv_airbnb_scalar_noi():
    # constant noi=10000/yr for 2 years at r=0.10, no terminal/setup
    npv = npv_airbnb(10_000.0, years=2, discount_rate=0.10)
    expected = 10_000 / 1.10 + 10_000 / 1.10**2
    assert npv == pytest.approx(expected)


def test_npv_airbnb_requires_years_for_scalar():
    with pytest.raises(ValueError):
        npv_airbnb(10_000.0)


def test_npv_airbnb_series_infers_years():
    npv = npv_airbnb([5_000.0, 6_000.0, 7_000.0], discount_rate=0.07)
    expected = 5_000 / 1.07 + 6_000 / 1.07**2 + 7_000 / 1.07**3
    assert npv == pytest.approx(expected)


def test_npv_airbnb_includes_terminal_value():
    npv_no_term = npv_airbnb(10_000.0, years=1, discount_rate=0.10)
    npv_with_term = npv_airbnb(10_000.0, years=1, discount_rate=0.10, terminal_value_net=1_000.0)
    assert npv_with_term - npv_no_term == pytest.approx(1_000.0 / 1.10)


def test_npv_airbnb_subtracts_setup_cost():
    npv = npv_airbnb(10_000.0, years=1, discount_rate=0.10, setup_cost=2_000.0)
    assert npv == pytest.approx(10_000 / 1.10 - 2_000.0)


def test_npv_airbnb_rejects_invalid_discount_rate():
    with pytest.raises(ValueError):
        npv_airbnb(10_000.0, years=1, discount_rate=0.0)


def test_npv_airbnb_rejects_empty_series():
    with pytest.raises(ValueError):
        npv_airbnb([])


# ── break_even_horizon ────────────────────────────────────────────────────────

def test_break_even_horizon_finds_crossing_year():
    # noi=10000/yr, r=0.10, sell value low enough to cross within a few years
    t = break_even_horizon(10_000.0, npv_sell_value=15_000.0, discount_rate=0.10)
    assert t is not None
    npv_before = npv_airbnb(10_000.0, years=t - 1, discount_rate=0.10) if t > 1 else 0.0
    npv_at = npv_airbnb(10_000.0, years=t, discount_rate=0.10)
    assert npv_at > 15_000.0
    assert npv_before <= 15_000.0


def test_break_even_horizon_none_when_unreachable():
    t = break_even_horizon(100.0, npv_sell_value=1_000_000.0, discount_rate=0.10, max_years=5)
    assert t is None


def test_break_even_horizon_respects_terminal_value_fn():
    t_no_term = break_even_horizon(1_000.0, npv_sell_value=50_000.0, discount_rate=0.10, max_years=30)
    t_with_term = break_even_horizon(
        1_000.0,
        npv_sell_value=50_000.0,
        discount_rate=0.10,
        terminal_value_fn=lambda year: 60_000.0,
        max_years=30,
    )
    assert t_with_term <= (t_no_term or 31)


def test_break_even_horizon_rejects_non_positive_max_years():
    with pytest.raises(ValueError):
        break_even_horizon(1_000.0, npv_sell_value=10_000.0, max_years=0)


# ── recommend ──────────────────────────────────────────────────────────────────

def test_recommend_airbnb_when_greater():
    assert recommend(500_000.0, 400_000.0) == "Airbnb"


def test_recommend_sell_when_greater_or_equal():
    assert recommend(400_000.0, 500_000.0) == "Sell"
    assert recommend(400_000.0, 400_000.0) == "Sell"


# ── monte_carlo ────────────────────────────────────────────────────────────────

def test_monte_carlo_returns_ordered_percentiles():
    out = monte_carlo(
        price_hat=150.0,
        occ_hat=200.0,
        years=10,
        npv_sell_value=300_000.0,
        cost_kwargs={"property_value": 400_000.0},
        n_simulations=500,
        random_state=42,
    )
    assert out["p10"] <= out["p50"] <= out["p90"]
    assert 0.0 <= out["p_airbnb_gt_sell"] <= 1.0


def test_monte_carlo_is_reproducible_with_seed():
    kwargs = dict(
        price_hat=150.0,
        occ_hat=200.0,
        years=10,
        npv_sell_value=300_000.0,
        cost_kwargs={"property_value": 400_000.0},
        n_simulations=200,
        random_state=7,
    )
    out1 = monte_carlo(**kwargs)
    out2 = monte_carlo(**kwargs)
    assert out1 == out2


def test_monte_carlo_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        monte_carlo(
            price_hat=-1.0,
            occ_hat=200.0,
            years=10,
            npv_sell_value=300_000.0,
            cost_kwargs={"property_value": 400_000.0},
        )
    with pytest.raises(ValueError):
        monte_carlo(
            price_hat=150.0,
            occ_hat=200.0,
            years=10,
            npv_sell_value=300_000.0,
            cost_kwargs={"property_value": 400_000.0},
            n_simulations=0,
        )