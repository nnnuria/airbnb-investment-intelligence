"""Tests for the build_abt orchestrator."""
import json
from datetime import date

import pandas as pd
import pytest

from airbnb_iip.data.abt import (
    DEFAULT_OUTPUT_DIR,
    KNOWN_CITIES,
    build_abt,
    build_abts,
    output_path,
)
from airbnb_iip.features.amenities import AMENITY_COLUMNS


def _listings(n=3, city="madrid"):
    """Synthesise a tiny cleaned listings frame."""
    return pd.DataFrame({
        "id": list(range(1, n + 1)),
        "city": [city] * n,
        "amenities": [json.dumps(["Pool", "Air conditioning"])] * n,
        "latitude": [40.4168 + 0.001 * i for i in range(n)],
        "longitude": [-3.7038] * n,
        "reviews_per_month": [1.0] * n,
    })


# ── build_abt ────────────────────────────────────────────────────────────────

def test_build_abt_returns_engineered_frame_with_known_columns():
    df = _listings()
    out = build_abt(
        "madrid",
        listings=df,
        output_dir=None,         # don't write to disk
    )
    # All 22 amenity columns present
    for col in AMENITY_COLUMNS:
        assert col in out.columns
    # Density present
    assert "competitive_density_500m" in out.columns
    # Occupancy estimate present (computed since input didn't have it)
    assert "estimated_occupancy_l365d" in out.columns
    # Same row count as input
    assert len(out) == len(df)


def test_build_abt_writes_versioned_parquet(tmp_path):
    df = _listings()
    out = build_abt(
        "madrid",
        listings=df,
        output_dir=tmp_path,
        snapshot_date="2026-06-17",
    )
    written = tmp_path / "madrid_abt_2026-06-17.parquet"
    assert written.exists()
    roundtrip = pd.read_parquet(written)
    pd.testing.assert_frame_equal(
        out.reset_index(drop=True),
        roundtrip.reset_index(drop=True),
        check_dtype=False,
    )


def test_build_abt_filters_to_requested_city():
    df = pd.concat([_listings(2, "madrid"), _listings(3, "barcelona")], ignore_index=True)
    out = build_abt("barcelona", listings=df, output_dir=None)
    assert len(out) == 3
    assert (out["city"] == "barcelona").all()


def test_build_abt_preserves_existing_occupancy_column():
    df = _listings()
    df["estimated_occupancy_l365d"] = 999     # Inside Airbnb's pre-computed value
    out = build_abt("madrid", listings=df, output_dir=None)
    assert (out["estimated_occupancy_l365d"] == 999).all()


def test_build_abt_unknown_city_raises():
    with pytest.raises(ValueError, match="Unknown city"):
        build_abt("valencia", listings=_listings(), output_dir=None)


def test_build_abt_no_rows_for_city_raises():
    df = _listings(city="madrid")
    with pytest.raises(ValueError, match="No rows for city"):
        build_abt("barcelona", listings=df, output_dir=None)


def test_build_abt_missing_input_file_raises(tmp_path):
    missing = tmp_path / "does-not-exist.parquet"
    with pytest.raises(FileNotFoundError):
        build_abt("madrid", input_parquet=missing, output_dir=None)


def test_build_abt_idempotent_re_runs(tmp_path):
    df = _listings()
    first = build_abt(
        "madrid", listings=df, output_dir=tmp_path, snapshot_date="2026-06-17",
    )
    second = build_abt(
        "madrid", listings=df, output_dir=tmp_path, snapshot_date="2026-06-17",
    )
    pd.testing.assert_frame_equal(first, second, check_dtype=False)


def test_build_abt_case_insensitive_city():
    df = _listings()
    out = build_abt("MADRID", listings=df, output_dir=None)
    assert len(out) == len(df)


# ── build_abts ───────────────────────────────────────────────────────────────

def test_build_abts_runs_each_city(tmp_path):
    df = pd.concat([
        _listings(2, "madrid"),
        _listings(3, "barcelona"),
        _listings(4, "malaga"),
    ], ignore_index=True)
    out = build_abts(
        cities=KNOWN_CITIES,
        listings=df,
        output_dir=tmp_path,
        snapshot_date="2026-06-17",
    )
    assert set(out) == set(KNOWN_CITIES)
    assert len(out["madrid"]) == 2
    assert len(out["barcelona"]) == 3
    assert len(out["malaga"]) == 4


def test_build_abts_continues_after_a_failure(tmp_path, caplog):
    # One city has no rows → that city is skipped; others succeed.
    df = pd.concat([_listings(2, "madrid"), _listings(3, "barcelona")], ignore_index=True)
    with caplog.at_level("ERROR"):
        out = build_abts(
            cities=("madrid", "barcelona", "malaga"),
            listings=df,
            output_dir=tmp_path,
            snapshot_date="2026-06-17",
        )
    assert "madrid" in out and "barcelona" in out
    assert "malaga" not in out


# ── output_path ──────────────────────────────────────────────────────────────

def test_output_path_includes_snapshot_date():
    assert output_path("madrid", snapshot_date="2026-06-17").name == "madrid_abt_2026-06-17.parquet"


def test_output_path_defaults_to_today():
    p = output_path("madrid")
    assert date.today().isoformat() in p.name


def test_output_path_default_dir():
    assert output_path("madrid").parent == DEFAULT_OUTPUT_DIR


def test_known_cities_match_project_scope():
    assert KNOWN_CITIES == ("madrid", "barcelona", "malaga")
