"""Tests for the calendar-based occupancy helpers.

The legacy San Francisco review-based estimator has been retired; occupancy is
now (a) a target computed from Inside Airbnb calendar availability and (b) a
learned LightGBM model (see tests/test_occupancy_model.py). These tests cover
the lightweight calendar helpers in airbnb_iip.data.occupancy.
"""
import pandas as pd
import pytest

from airbnb_iip.data.occupancy import (
    BOOKED_FLAG,
    DAYS_PER_YEAR,
    occupancy_l365d_from_availability,
    occupancy_rate_from_calendar,
)


# ── occupancy_rate_from_calendar ──────────────────────────────────────────────

def _calendar():
    # listing 1: 3 booked of 4 nights → 0.75; listing 2: 0 of 2 → 0.0
    return pd.DataFrame({
        "listing_id": [1, 1, 1, 1, 2, 2],
        "date": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04",
                 "2026-01-01", "2026-01-02"],
        "available": ["f", "f", "f", "t", "t", "t"],
    })


def test_occupancy_rate_basic():
    out = occupancy_rate_from_calendar(_calendar())
    out = out.set_index("listing_id")
    assert out.loc[1, "occupancy_rate"] == pytest.approx(0.75)
    assert out.loc[2, "occupancy_rate"] == pytest.approx(0.0)
    assert out.loc[1, "calendar_days"] == 4
    assert out.loc[1, "occupancy_nights_l365d"] == round(0.75 * DAYS_PER_YEAR)


def test_occupancy_rate_f_is_booked():
    # Sanity: 'f' (not available) is the booked flag, 't' is free.
    assert BOOKED_FLAG == "f"
    df = pd.DataFrame({"listing_id": [9, 9], "available": ["f", "f"]})
    out = occupancy_rate_from_calendar(df)
    assert out.loc[0, "occupancy_rate"] == pytest.approx(1.0)


def test_occupancy_rate_handles_whitespace_and_custom_cols():
    df = pd.DataFrame({"lid": [1, 1], "avail": [" f ", "t"]})
    out = occupancy_rate_from_calendar(df, id_col="lid", available_col="avail")
    assert out.loc[0, "occupancy_rate"] == pytest.approx(0.5)


def test_occupancy_rate_missing_column_raises():
    with pytest.raises(KeyError):
        occupancy_rate_from_calendar(pd.DataFrame({"listing_id": [1]}))


# ── occupancy_l365d_from_availability ────────────────────────────────────────

def test_l365d_from_availability_complement():
    df = pd.DataFrame({"availability_365": [0, 110, 365]})
    s = occupancy_l365d_from_availability(df)
    assert s.name == "estimated_occupancy_l365d"
    assert s.dtype == "int64"
    assert s.tolist() == [365, 255, 0]


def test_l365d_from_availability_nan_and_clip():
    df = pd.DataFrame({"availability_365": [float("nan"), -5, 400]})
    s = occupancy_l365d_from_availability(df)
    # NaN → fully available → 0 booked; negatives/over-365 clamped.
    assert s.tolist() == [0, 365, 0]


def test_l365d_from_availability_preserves_index():
    df = pd.DataFrame(
        {"availability_365": [100, 200]},
        index=pd.Index(["a", "b"], name="listing_id"),
    )
    s = occupancy_l365d_from_availability(df)
    assert s.index.tolist() == ["a", "b"]


def test_l365d_from_availability_missing_column_raises():
    with pytest.raises(KeyError, match="availability_365"):
        occupancy_l365d_from_availability(pd.DataFrame({"price": [100.0]}))
