"""Tests for the amenity feature engineering module."""
import json

import numpy as np
import pandas as pd
import pytest

from airbnb_iip.features.amenities import (
    AMENITY_COLUMNS,
    AMENITY_PATTERNS,
    add_amenity_flags,
    parse_amenity_flags,
)


# ── Canonical contract ───────────────────────────────────────────────────────

def test_canonical_22_amenity_columns_locked():
    """Price model artefact depends on exactly these 22 columns. Don't drift."""
    assert len(AMENITY_COLUMNS) == 22
    expected = {
        "has_pool", "has_gym", "has_parking", "has_hot_tub", "has_beach",
        "has_view", "has_ac", "has_elevator", "has_washer", "has_dishwasher",
        "has_workspace", "has_self_checkin", "has_pets", "has_crib",
        "has_private_entrance", "has_balcony", "has_bathtub", "has_dryer",
        "has_ev_charger", "has_outdoor_space", "has_long_term_ok",
        "has_cleaning_service",
    }
    assert set(AMENITY_COLUMNS) == expected


# ── parse_amenity_flags ──────────────────────────────────────────────────────

def _amenities_json(items):
    return json.dumps(items)


def test_parse_flags_basic_matches():
    series = pd.Series([
        _amenities_json(["Wifi", "Kitchen", "Pool"]),
        _amenities_json(["Air conditioning", "Elevator", "Washer"]),
    ])
    flags = parse_amenity_flags(series)

    assert flags.loc[0, "has_pool"] == 1
    assert flags.loc[0, "has_ac"] == 0
    assert flags.loc[1, "has_ac"] == 1
    assert flags.loc[1, "has_elevator"] == 1
    assert flags.loc[1, "has_washer"] == 1
    assert flags.loc[1, "has_pool"] == 0


def test_parse_flags_case_insensitive():
    series = pd.Series([_amenities_json(["AIR CONDITIONING"])])
    flags = parse_amenity_flags(series)
    assert flags.loc[0, "has_ac"] == 1


def test_parse_flags_exclusion_regex_for_pool():
    # "pool table" must NOT trigger has_pool
    series = pd.Series([
        _amenities_json(["Pool table"]),
        _amenities_json(["Outdoor pool"]),
    ])
    flags = parse_amenity_flags(series)
    assert flags.loc[0, "has_pool"] == 0
    assert flags.loc[1, "has_pool"] == 1


def test_parse_flags_balcony_includes_synonyms():
    # has_balcony bundles balcony | patio | terrace
    for amenity in ("Balcony", "Patio", "Private terrace"):
        flags = parse_amenity_flags(pd.Series([_amenities_json([amenity])]))
        assert flags.loc[0, "has_balcony"] == 1, amenity


def test_parse_flags_word_boundaries_for_gym():
    # \bgym\b should NOT match "gymnastics" or "gymkhana"
    series = pd.Series([
        _amenities_json(["Gymnastics studio nearby"]),
        _amenities_json(["Shared gym"]),
    ])
    flags = parse_amenity_flags(series)
    assert flags.loc[0, "has_gym"] == 0
    assert flags.loc[1, "has_gym"] == 1


def test_parse_flags_self_checkin_alternatives():
    series = pd.Series([
        _amenities_json(["Self check-in"]),
        _amenities_json(["Smart lock"]),
        _amenities_json(["Lockbox"]),
        _amenities_json(["Keypad"]),
        _amenities_json(["Doorman"]),
    ])
    flags = parse_amenity_flags(series)
    assert flags["has_self_checkin"].tolist() == [1, 1, 1, 1, 0]


def test_parse_flags_dtypes_int8():
    series = pd.Series([_amenities_json(["Wifi"])])
    flags = parse_amenity_flags(series)
    for col in flags.columns:
        assert flags[col].dtype == np.int8, col


def test_parse_flags_returns_all_22_columns_even_for_empty_input():
    series = pd.Series([_amenities_json([])])
    flags = parse_amenity_flags(series)
    assert list(flags.columns) == list(AMENITY_COLUMNS)
    assert (flags.iloc[0] == 0).all()


def test_parse_flags_handles_nan_and_none():
    series = pd.Series([None, float("nan"), _amenities_json(["Pool"])])
    flags = parse_amenity_flags(series)
    assert (flags.iloc[0] == 0).all()
    assert (flags.iloc[1] == 0).all()
    assert flags.iloc[2]["has_pool"] == 1


def test_parse_flags_handles_malformed_json():
    series = pd.Series(["not-valid-json", "{}", "[",  _amenities_json(["Wifi"])])
    flags = parse_amenity_flags(series)
    # Malformed rows are treated as empty arrays, never raise
    assert (flags.iloc[0] == 0).all()
    assert (flags.iloc[1] == 0).all()  # dict ≠ list
    assert (flags.iloc[2] == 0).all()  # truncated json
    assert flags.iloc[3].sum() >= 0     # last row parses cleanly


def test_parse_flags_accepts_python_list_input():
    """If a caller already json-parsed the column, accept the list."""
    series = pd.Series([["Pool"], ["Elevator"]])
    flags = parse_amenity_flags(series)
    assert flags.loc[0, "has_pool"] == 1
    assert flags.loc[1, "has_elevator"] == 1


def test_parse_flags_preserves_index():
    series = pd.Series(
        [_amenities_json(["Pool"]), _amenities_json(["Elevator"])],
        index=pd.Index(["a", "b"], name="listing_id"),
    )
    flags = parse_amenity_flags(series)
    assert flags.index.tolist() == ["a", "b"]
    assert flags.index.name == "listing_id"


def test_parse_flags_custom_patterns():
    custom = {"has_wifi": (r"wifi", None)}
    series = pd.Series([_amenities_json(["Wifi", "Pool"])])
    flags = parse_amenity_flags(series, patterns=custom)
    assert list(flags.columns) == ["has_wifi"]
    assert flags.loc[0, "has_wifi"] == 1


# ── add_amenity_flags ────────────────────────────────────────────────────────

def test_add_amenity_flags_appends_to_df():
    df = pd.DataFrame({
        "listing_id": [1, 2],
        "amenities": [_amenities_json(["Pool"]), _amenities_json(["Elevator"])],
    })
    out = add_amenity_flags(df)
    assert "has_pool" in out.columns
    assert "has_elevator" in out.columns
    assert out.loc[0, "has_pool"] == 1
    assert out.loc[1, "has_elevator"] == 1


def test_add_amenity_flags_does_not_mutate_input():
    df = pd.DataFrame({"amenities": [_amenities_json(["Pool"])]})
    add_amenity_flags(df)
    assert "has_pool" not in df.columns


def test_add_amenity_flags_idempotent():
    df = pd.DataFrame({"amenities": [_amenities_json(["Pool"])]})
    out1 = add_amenity_flags(df)
    out2 = add_amenity_flags(out1)
    pd.testing.assert_frame_equal(out1, out2, check_dtype=False)


def test_add_amenity_flags_missing_column_raises():
    df = pd.DataFrame({"some_other_col": [1, 2]})
    with pytest.raises(KeyError, match="amenities"):
        add_amenity_flags(df)


def test_add_amenity_flags_custom_column_name():
    df = pd.DataFrame({"amen_str": [_amenities_json(["Pool"])]})
    out = add_amenity_flags(df, amenities_col="amen_str")
    assert out.loc[0, "has_pool"] == 1
