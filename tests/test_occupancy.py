"""Tests for the San Francisco occupancy estimator."""
import math

import pandas as pd
import pytest

from airbnb_iip.config import OCCUPANCY
from airbnb_iip.data.occupancy import (
    DAYS_PER_MONTH,
    DAYS_PER_YEAR,
    MONTHS_PER_YEAR,
    estimate_occupancy,
    estimate_occupancy_l365d,
)


# ── Scalar form ───────────────────────────────────────────────────────────────

def test_estimate_occupancy_known_value():
    # reviews_per_month=1.0 with defaults (review_rate=0.5, los=3):
    # nights_per_month = (1 / 0.5) * 3 = 6
    # occupancy_rate   = 6 / 30 = 0.2
    out = estimate_occupancy(1.0)
    assert out["nights_booked_per_month"] == pytest.approx(6.0)
    assert out["occupancy_rate"] == pytest.approx(0.2)


def test_estimate_occupancy_defaults_come_from_config():
    # Recompute with the config dict directly — must match.
    out = estimate_occupancy(
        2.0,
        review_rate=OCCUPANCY["review_rate"],
        avg_length_of_stay=OCCUPANCY["avg_length_of_stay"],
        max_occupancy=OCCUPANCY["max_occupancy"],
    )
    default_out = estimate_occupancy(2.0)
    assert out == default_out


def test_estimate_occupancy_caps_at_max_occupancy():
    # reviews_per_month=10 with defaults would give (10/0.5)*3 = 60 nights
    # in a 30-day month, which is impossible. Cap = 0.7 * 30 = 21.
    out = estimate_occupancy(10.0)
    assert out["nights_booked_per_month"] == pytest.approx(21.0)
    assert out["occupancy_rate"] == pytest.approx(0.7)


def test_estimate_occupancy_zero_reviews_returns_zero():
    out = estimate_occupancy(0.0)
    assert out["nights_booked_per_month"] == 0.0
    assert out["occupancy_rate"] == 0.0


@pytest.mark.parametrize("value", [None, float("nan")])
def test_estimate_occupancy_none_or_nan_treated_as_zero(value):
    out = estimate_occupancy(value)
    assert out["nights_booked_per_month"] == 0.0
    assert out["occupancy_rate"] == 0.0


def test_estimate_occupancy_negative_reviews_clamped_to_zero():
    out = estimate_occupancy(-5.0)
    assert out["nights_booked_per_month"] == 0.0
    assert out["occupancy_rate"] == 0.0


def test_estimate_occupancy_respects_custom_review_rate():
    # review_rate=1.0 → every guest reviews → fewer inferred stays
    # nights/month = (4 / 1.0) * 3 = 12
    out = estimate_occupancy(4.0, review_rate=1.0)
    assert out["nights_booked_per_month"] == pytest.approx(12.0)


def test_estimate_occupancy_respects_custom_length_of_stay():
    # los=5 → (2/0.5)*5 = 20 nights/month
    out = estimate_occupancy(2.0, avg_length_of_stay=5)
    assert out["nights_booked_per_month"] == pytest.approx(20.0)


def test_estimate_occupancy_respects_custom_cap():
    # Tight 50% cap: max 15 nights/month even at high review activity
    out = estimate_occupancy(100.0, max_occupancy=0.5)
    assert out["nights_booked_per_month"] == pytest.approx(15.0)
    assert out["occupancy_rate"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"review_rate": 0.0},
        {"review_rate": -0.1},
        {"review_rate": 1.1},
        {"avg_length_of_stay": 0.0},
        {"avg_length_of_stay": -1},
        {"max_occupancy": 0.0},
        {"max_occupancy": 1.5},
    ],
)
def test_estimate_occupancy_rejects_invalid_assumptions(kwargs):
    with pytest.raises(ValueError):
        estimate_occupancy(1.0, **kwargs)


def test_estimate_occupancy_rejects_non_numeric_reviews():
    with pytest.raises(ValueError, match="reviews_per_month"):
        estimate_occupancy("not-a-number")


# ── Vectorised form ───────────────────────────────────────────────────────────

def test_estimate_occupancy_l365d_returns_named_int_series():
    df = pd.DataFrame({"reviews_per_month": [0.0, 1.0, 2.0]})
    s = estimate_occupancy_l365d(df)
    assert isinstance(s, pd.Series)
    assert s.name == "estimated_occupancy_l365d"
    assert s.dtype == "int64"
    assert len(s) == 3


def test_estimate_occupancy_l365d_matches_inside_airbnb_convention():
    # Inside Airbnb annualises with × 12, not × (365/30):
    # nights/year = (reviews_per_month / review_rate) * los * 12
    # reviews=1 → (1/0.5)*3*12 = 72 nights/year
    df = pd.DataFrame({"reviews_per_month": [1.0]})
    s = estimate_occupancy_l365d(df)
    assert s.iloc[0] == 72


def test_estimate_occupancy_l365d_caps_at_255_with_defaults():
    # floor(0.70 * 365) = 255 — matches the Inside Airbnb upper bound
    df = pd.DataFrame({"reviews_per_month": [50.0]})
    s = estimate_occupancy_l365d(df)
    assert s.iloc[0] == 255


def test_estimate_occupancy_l365d_handles_nan_and_negative():
    df = pd.DataFrame(
        {"reviews_per_month": [float("nan"), -1.0, 0.0, 1.0]}
    )
    s = estimate_occupancy_l365d(df)
    assert s.tolist() == [0, 0, 0, 72]


def test_estimate_occupancy_l365d_preserves_index():
    df = pd.DataFrame(
        {"reviews_per_month": [1.0, 2.0]},
        index=pd.Index(["a", "b"], name="listing_id"),
    )
    s = estimate_occupancy_l365d(df)
    assert s.index.tolist() == ["a", "b"]
    assert s.index.name == "listing_id"


def test_estimate_occupancy_l365d_accepts_custom_column_name():
    df = pd.DataFrame({"my_reviews_col": [1.0, 2.0]})
    s = estimate_occupancy_l365d(df, reviews_per_month_col="my_reviews_col")
    assert s.tolist() == [72, 144]


def test_estimate_occupancy_l365d_missing_column_raises():
    df = pd.DataFrame({"price": [100.0]})
    with pytest.raises(KeyError, match="reviews_per_month"):
        estimate_occupancy_l365d(df)


def test_estimate_occupancy_l365d_respects_custom_assumptions():
    df = pd.DataFrame({"reviews_per_month": [1.0]})
    # review_rate=1.0, los=4 → 1 * 4 * 12 = 48
    s = estimate_occupancy_l365d(df, review_rate=1.0, avg_length_of_stay=4)
    assert s.iloc[0] == 48


def test_estimate_occupancy_l365d_coerces_string_numbers():
    # Defensive: if a column comes in as object dtype with numeric strings,
    # we should coerce rather than crash.
    df = pd.DataFrame({"reviews_per_month": ["1.0", "2.0", "garbage"]})
    s = estimate_occupancy_l365d(df)
    assert s.tolist() == [72, 144, 0]


# ── Cross-check: scalar (×12) lines up with l365d for clean inputs ────────────

def test_scalar_x12_consistent_with_l365d_below_cap():
    # When the result is below the cap, scalar × 12 should equal l365d
    df = pd.DataFrame({"reviews_per_month": [0.5, 1.0, 2.0, 3.0]})
    vector = estimate_occupancy_l365d(df).tolist()
    scalar = [
        int(round(estimate_occupancy(r)["nights_booked_per_month"] * MONTHS_PER_YEAR))
        for r in df["reviews_per_month"]
    ]
    # The cap kicks in at high values; check the ones that stay below.
    annual_cap = math.floor(OCCUPANCY["max_occupancy"] * DAYS_PER_YEAR)
    monthly_cap = OCCUPANCY["max_occupancy"] * DAYS_PER_MONTH
    for r, v, s in zip(df["reviews_per_month"], vector, scalar):
        nights_month = (r / OCCUPANCY["review_rate"]) * OCCUPANCY["avg_length_of_stay"]
        if nights_month <= monthly_cap and v < annual_cap:
            assert v == s, f"mismatch at reviews_per_month={r}: vector={v} scalar={s}"


# ── Constants sanity ──────────────────────────────────────────────────────────

def test_constants_match_calendar_assumptions():
    assert DAYS_PER_YEAR == 365
    assert MONTHS_PER_YEAR == 12
    assert DAYS_PER_MONTH == 30
