"""Tests for the Idealista sale loader/cleaner.

These use synthetic in-memory records so they run in CI without the raw
``data/raw/`` dumps (which are gitignored and absent there).
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from airbnb_iip.data.idealista_sale import (
    clean_sale_listings,
    parse_floor,
    tidy_records,
)


def _record(**overrides):
    base = {
        "propertyCode": "1",
        "price": 280_000,
        "size": 80,
        "priceByArea": 3500,
        "rooms": 3,
        "bathrooms": 2,
        "propertyType": "flat",
        "status": "good",
        "floor": "2",
        "hasLift": True,
        "exterior": True,
        "newDevelopment": False,
        "district": "Centro",
        "neighborhood": "Centro",
        "municipality": "Málaga",
        "province": "Málaga",
        "latitude": 36.72,
        "longitude": -4.42,
        "_city": "malaga",
    }
    base.update(overrides)
    return base


@pytest.mark.parametrize(
    "raw,expected",
    [("bj", 0.0), ("en", 0.5), ("ss", -1.0), ("st", -1.0), ("3", 3.0), ("-1", -1.0)],
)
def test_parse_floor_codes(raw, expected):
    assert parse_floor(raw) == expected


def test_parse_floor_unknown_is_nan():
    assert math.isnan(parse_floor(None))
    assert math.isnan(parse_floor("penthouse"))
    assert math.isnan(parse_floor(float("nan")))


def test_tidy_records_renames_types_and_derives_floor():
    df = tidy_records([_record(floor="bj")])
    assert "price" in df.columns and "size_m2" in df.columns and "city" in df.columns
    assert df.loc[0, "floor_num"] == 0.0
    assert df["price"].dtype.kind in "if"  # numeric (int or float)
    assert str(df["has_lift"].dtype) == "boolean"


def test_tidy_records_dedups_property_code():
    df = tidy_records([_record(propertyCode="1"), _record(propertyCode="1"),
                       _record(propertyCode="2")])
    assert len(df) == 2


def test_tidy_records_default_city_fallback():
    rec = _record()
    del rec["_city"]
    df = tidy_records([rec], default_city="madrid")
    assert df.loc[0, "city"] == "madrid"


def test_clean_drops_outliers_and_junk():
    df = tidy_records([
        _record(propertyCode="ok"),                          # keep
        _record(propertyCode="huge", size=32000),            # drop: size
        _record(propertyCode="cheap_ppm2", priceByArea=5),   # drop: €/m²
        _record(propertyCode="palace", price=8_500_000),     # drop: price
        _record(propertyCode="manyrooms", rooms=40),         # drop: rooms
    ])
    out = clean_sale_listings(df)
    assert list(out["property_code"]) == ["ok"]


def test_clean_is_pure():
    df = tidy_records([_record(), _record(propertyCode="huge", size=32000)])
    before = len(df)
    clean_sale_listings(df)
    assert len(df) == before  # original untouched
