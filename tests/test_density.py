"""Tests for the competitive density module."""
import numpy as np
import pandas as pd
import pytest

from airbnb_iip.features.density import (
    DEFAULT_OUTPUT_COL,
    DEFAULT_RADIUS_M,
    EARTH_RADIUS_M,
    add_competitive_density,
    compute_density_array,
)


# ── compute_density_array ────────────────────────────────────────────────────

def test_density_empty_input():
    arr = compute_density_array(
        cities=np.array([]),
        latitudes=np.array([]),
        longitudes=np.array([]),
    )
    assert arr.dtype == np.int32
    assert arr.shape == (0,)


def test_density_single_listing_has_zero_neighbours():
    arr = compute_density_array(
        cities=np.array(["madrid"]),
        latitudes=np.array([40.4168]),
        longitudes=np.array([-3.7038]),
    )
    assert arr.tolist() == [0]


def test_density_excludes_self():
    # Two listings at identical coordinates → each has exactly 1 neighbour
    arr = compute_density_array(
        cities=np.array(["madrid", "madrid"]),
        latitudes=np.array([40.4168, 40.4168]),
        longitudes=np.array([-3.7038, -3.7038]),
    )
    assert arr.tolist() == [1, 1]


def test_density_excludes_other_cities():
    # Listings in Madrid and Barcelona are far apart — never neighbours.
    arr = compute_density_array(
        cities=np.array(["madrid", "madrid", "barcelona"]),
        latitudes=np.array([40.4168, 40.4168, 41.3851]),
        longitudes=np.array([-3.7038, -3.7038, 2.1734]),
    )
    assert arr.tolist() == [1, 1, 0]


def test_density_outside_radius_excluded():
    # Two listings ~1 km apart with default 500 m radius → not neighbours.
    arr = compute_density_array(
        cities=np.array(["madrid", "madrid"]),
        latitudes=np.array([40.4168, 40.4258]),    # ~1.0 km north
        longitudes=np.array([-3.7038, -3.7038]),
        radius_m=500,
    )
    assert arr.tolist() == [0, 0]


def test_density_inside_radius_included():
    # Two listings ~200 m apart → neighbours at 500 m radius.
    arr = compute_density_array(
        cities=np.array(["madrid", "madrid"]),
        latitudes=np.array([40.4168, 40.4186]),    # ~0.2 km north
        longitudes=np.array([-3.7038, -3.7038]),
        radius_m=500,
    )
    assert arr.tolist() == [1, 1]


def test_density_radius_parameter_is_respected():
    # Same two points, but a tight 100 m radius excludes them.
    arr = compute_density_array(
        cities=np.array(["madrid", "madrid"]),
        latitudes=np.array([40.4168, 40.4186]),
        longitudes=np.array([-3.7038, -3.7038]),
        radius_m=100,
    )
    assert arr.tolist() == [0, 0]


def test_density_nan_coords_get_zero_and_do_not_count():
    arr = compute_density_array(
        cities=np.array(["madrid", "madrid", "madrid"]),
        latitudes=np.array([40.4168, 40.4168, np.nan]),
        longitudes=np.array([-3.7038, -3.7038, np.nan]),
    )
    # Rows 0 and 1 are at the same point → 1 neighbour each.
    # Row 2 has NaN coords → density 0, and doesn't appear in others' counts.
    assert arr.tolist() == [1, 1, 0]


def test_density_mismatched_lengths_raises():
    with pytest.raises(ValueError, match="same length"):
        compute_density_array(
            cities=np.array(["madrid"]),
            latitudes=np.array([40.0, 41.0]),
            longitudes=np.array([-3.0]),
        )


def test_density_constants_match_expected_values():
    assert EARTH_RADIUS_M == 6_371_000
    assert DEFAULT_RADIUS_M == 500
    assert DEFAULT_OUTPUT_COL == "competitive_density_500m"


# ── add_competitive_density ──────────────────────────────────────────────────

@pytest.fixture
def listings():
    return pd.DataFrame({
        "id": [1, 2, 3],
        "city": ["madrid", "madrid", "barcelona"],
        "latitude": [40.4168, 40.4168, 41.3851],
        "longitude": [-3.7038, -3.7038, 2.1734],
    })


def test_add_density_appends_default_column(listings):
    out = add_competitive_density(listings)
    assert DEFAULT_OUTPUT_COL in out.columns
    assert out[DEFAULT_OUTPUT_COL].tolist() == [1, 1, 0]


def test_add_density_does_not_mutate_input(listings):
    add_competitive_density(listings)
    assert DEFAULT_OUTPUT_COL not in listings.columns


def test_add_density_missing_required_column_raises(listings):
    with pytest.raises(KeyError):
        add_competitive_density(listings.drop(columns=["city"]))
    with pytest.raises(KeyError):
        add_competitive_density(listings.drop(columns=["latitude"]))


def test_add_density_custom_output_column(listings):
    out = add_competitive_density(
        listings, radius_m=500, output_col="density_500m",
    )
    assert "density_500m" in out.columns
    assert DEFAULT_OUTPUT_COL not in out.columns


def test_add_density_preserves_row_order_and_index(listings):
    listings_indexed = listings.set_index("id")
    out = add_competitive_density(listings_indexed)
    assert out.index.tolist() == [1, 2, 3]
