"""Tests for the seasonality module."""
from datetime import date, timedelta

import pandas as pd
import pytest

from airbnb_iip.features.seasonality import (
    DEFAULT_EXCLUDE_DAYS,
    MONTHS_PER_YEAR,
    compute_seasonal_multipliers,
    multipliers_to_dataframe,
)


# ── compute_seasonal_multipliers ─────────────────────────────────────────────

def _calendar(rows):
    """Helper: build a calendar DataFrame from (city, date, available) tuples."""
    return pd.DataFrame(rows, columns=["city", "date", "available"])


def test_seasonality_flat_demand_yields_all_ones():
    # All 12 months at the same 50% unavailability → multiplier = 1.0 everywhere.
    snap = date(2026, 1, 1)
    rows = []
    for month in range(1, 13):
        rows.append(("madrid", f"2027-{month:02d}-15", "t"))
        rows.append(("madrid", f"2027-{month:02d}-16", "f"))
    cal = _calendar(rows)
    out = compute_seasonal_multipliers(cal, snapshot_date=snap)
    assert set(out["madrid"]) == set(range(1, 13))
    for m in range(1, 13):
        assert out["madrid"][m] == pytest.approx(1.0)


def test_seasonality_peak_month_gets_higher_multiplier():
    # August fully unavailable, other 11 months fully available.
    # August unavail=1.0, others 0.0; mean=1/12; mult_aug = 1.0/(1/12) = 12.
    snap = date(2026, 1, 1)
    rows = []
    for month in range(1, 13):
        avail_flag = "f" if month == 8 else "t"
        rows.append(("madrid", f"2027-{month:02d}-15", avail_flag))
    cal = _calendar(rows)
    out = compute_seasonal_multipliers(cal, snapshot_date=snap)
    assert out["madrid"][8] > out["madrid"][1]
    assert out["madrid"][8] == pytest.approx(12.0)
    # Other months are 0.0 (zero unavail, normalised) — that's expected with
    # the extreme synthetic case; flat-demand fallback only fires when the
    # baseline itself is zero across the year.
    assert out["madrid"][1] == pytest.approx(0.0)


def test_seasonality_normalisation_average_to_one():
    # For a more realistic spread, the mean of multipliers per city = 1.0.
    snap = date(2026, 1, 1)
    rows = []
    # 12 months × 4 rows each: vary unavailability across months.
    unavail_rates = {1: 0.25, 2: 0.30, 3: 0.40, 4: 0.50, 5: 0.55, 6: 0.60,
                     7: 0.75, 8: 0.85, 9: 0.55, 10: 0.45, 11: 0.30, 12: 0.40}
    for month, rate in unavail_rates.items():
        for slot in range(20):
            avail = "f" if slot < int(rate * 20) else "t"
            rows.append(("madrid", f"2027-{month:02d}-{slot + 1:02d}", avail))
    cal = _calendar(rows)
    out = compute_seasonal_multipliers(cal, snapshot_date=snap)
    avg = sum(out["madrid"].values()) / 12
    assert avg == pytest.approx(1.0, abs=0.01)


def test_seasonality_excludes_first_60_days_by_default():
    # Inside the 60-day cutoff: 100% unavail. Outside: 50%.
    # If filter works, the included month's multiplier should reflect the
    # outside-window value only.
    snap = date(2026, 1, 1)
    inside_date = (snap + timedelta(days=10)).isoformat()      # excluded
    outside_date = (snap + timedelta(days=DEFAULT_EXCLUDE_DAYS + 10)).isoformat()
    rows = [
        ("madrid", inside_date, "f"),     # would dominate if not filtered
        ("madrid", inside_date, "f"),
        ("madrid", outside_date, "t"),
        ("madrid", outside_date, "f"),    # 1/2 unavail
    ]
    cal = _calendar(rows)
    out = compute_seasonal_multipliers(cal, snapshot_date=snap)
    # Only the outside-window month is present.
    out_month = pd.to_datetime(outside_date).month
    assert out_month in out["madrid"]
    # That month has 50% unavail; since it's the only month with signal,
    # its multiplier = unavail / mean(unavail) = 0.5 / 0.5 = 1.0.
    assert out["madrid"][out_month] == pytest.approx(1.0)


def test_seasonality_custom_exclude_days_zero_keeps_everything():
    snap = date(2026, 1, 1)
    rows = [
        ("madrid", "2026-01-15", "t"),
        ("madrid", "2026-01-16", "f"),
    ]
    cal = _calendar(rows)
    out = compute_seasonal_multipliers(cal, snapshot_date=snap, exclude_days=0)
    assert 1 in out["madrid"]


def test_seasonality_negative_exclude_raises():
    with pytest.raises(ValueError, match="exclude_days"):
        compute_seasonal_multipliers(_calendar([]), exclude_days=-1)


def test_seasonality_missing_columns_raise():
    cal = pd.DataFrame({"date": ["2027-01-01"], "available": ["t"]})
    with pytest.raises(KeyError, match="city"):
        compute_seasonal_multipliers(cal)


def test_seasonality_per_city_independent():
    snap = date(2026, 1, 1)
    rows = [
        # Madrid: full unavail in August
        ("madrid", "2027-08-15", "f"),
        ("madrid", "2027-08-16", "f"),
        ("madrid", "2027-01-15", "t"),
        # Barcelona: full unavail in January
        ("barcelona", "2027-01-15", "f"),
        ("barcelona", "2027-08-15", "t"),
    ]
    cal = _calendar(rows)
    out = compute_seasonal_multipliers(cal, snapshot_date=snap)
    assert "madrid" in out and "barcelona" in out
    assert out["madrid"][8] > out["madrid"][1]
    assert out["barcelona"][1] > out["barcelona"][8]


def test_seasonality_handles_boolean_available_column():
    snap = date(2026, 1, 1)
    cal = pd.DataFrame({
        "city": ["madrid"] * 4,
        "date": ["2027-06-15", "2027-06-16", "2027-12-15", "2027-12-16"],
        "available": [True, False, True, True],
    })
    out = compute_seasonal_multipliers(cal, snapshot_date=snap)
    # June unavail=0.5, December unavail=0 → June multiplier > December
    assert out["madrid"][6] > out["madrid"][12]


def test_seasonality_invalid_dates_dropped():
    snap = date(2026, 1, 1)
    cal = _calendar([
        ("madrid", "not-a-date", "t"),
        ("madrid", "2027-06-15", "f"),
        ("madrid", "2027-06-16", "t"),
    ])
    out = compute_seasonal_multipliers(cal, snapshot_date=snap)
    assert 6 in out["madrid"]


def test_seasonality_empty_calendar_returns_empty():
    out = compute_seasonal_multipliers(_calendar([]))
    assert out == {}


def test_seasonality_flat_fallback_when_baseline_zero():
    # All-available calendar → unavail=0 everywhere → baseline=0 → flat fallback.
    snap = date(2026, 1, 1)
    rows = [("madrid", f"2027-{m:02d}-15", "t") for m in range(1, 13)]
    cal = _calendar(rows)
    out = compute_seasonal_multipliers(cal, snapshot_date=snap)
    for m in range(1, 13):
        assert out["madrid"][m] == 1.0


# ── multipliers_to_dataframe ─────────────────────────────────────────────────

def test_multipliers_to_dataframe_round_trip():
    inp = {
        "madrid": {1: 0.9, 2: 0.95, 3: 1.0},
        "barcelona": {1: 1.1},
    }
    df = multipliers_to_dataframe(inp)
    assert set(df.columns) == {"city", "month", "multiplier"}
    assert len(df) == 4
    assert df.loc[df["city"] == "madrid", "month"].tolist() == [1, 2, 3]
    assert df["month"].dtype.kind == "i"


def test_constants_match_expected_values():
    assert DEFAULT_EXCLUDE_DAYS == 60
    assert MONTHS_PER_YEAR == 12
