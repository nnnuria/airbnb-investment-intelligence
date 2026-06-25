"""Unit tests for sale-listing amenity extraction (file-free, CI-safe)."""

from __future__ import annotations

import pandas as pd

from airbnb_iip.features.sale_amenities import (
    SALE_AMENITY_COLUMNS,
    add_sale_amenity_flags,
    parse_sale_amenity_flags,
)


def _flags(text: str) -> dict:
    df = parse_sale_amenity_flags(pd.Series([text]))
    return {c: int(df[c].iloc[0]) for c in df.columns}


def test_basic_matches():
    f = _flags("Piso con piscina, garaje y terraza. Aire acondicionado incluido.")
    assert f["has_pool"] == 1
    assert f["has_parking"] == 1
    assert f["has_terrace"] == 1
    assert f["has_ac"] == 1
    assert f["has_garden"] == 0


def test_negation_suppresses_flag():
    f = _flags("Vivienda sin piscina pero con garaje y trastero.")
    assert f["has_pool"] == 0          # "sin piscina"
    assert f["has_parking"] == 1
    assert f["has_storage"] == 1


def test_negation_only_local():
    # "sin garaje" must not suppress an unrelated pool mention.
    f = _flags("Amplia piscina comunitaria. Se vende sin garaje.")
    assert f["has_pool"] == 1
    assert f["has_parking"] == 0


def test_mojibake_accented_terms():
    # Scraped text stores accents as the U+FFFD replacement char.
    f = _flags("Bonito �tico con balc�n, jard�n y calefacci�n.")
    assert f["has_balcony"] == 1
    assert f["has_garden"] == 1
    assert f["has_heating"] == 1


def test_plurals_and_unaccented_variants():
    assert _flags("dos balcones y amplios jardines")["has_balcony"] == 1
    assert _flags("dos balcones y amplios jardines")["has_garden"] == 1


def test_empty_and_nan_safe():
    df = parse_sale_amenity_flags(pd.Series([None, "", "casa sin nada"]))
    assert (df.sum(axis=1) == 0).all()


def test_add_flags_combines_description_and_title():
    df = pd.DataFrame({
        "description": ["Piso reformado, muy luminoso."],
        "_ad_title": ["Piso con piscina en el centro"],
    })
    out = add_sale_amenity_flags(df)
    assert out["has_pool"].iloc[0] == 1            # signal came from the title
    assert all(c in out.columns for c in SALE_AMENITY_COLUMNS)


def test_add_flags_tolerates_missing_description():
    out = add_sale_amenity_flags(pd.DataFrame({"price": [100]}))
    assert all(out[c].iloc[0] == 0 for c in SALE_AMENITY_COLUMNS)
