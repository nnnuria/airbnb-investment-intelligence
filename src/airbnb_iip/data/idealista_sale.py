"""Idealista sale-listings loader & cleaner.

Reads the raw Apify JSONL dumps
(``data/raw/idealista/<city>/sale_<date>.jsonl``) into a tidy, analysis-ready
DataFrame for the **sell-price model**. Scope (v1) is the structured fields with
near-complete coverage; the rich ``features`` dict, ``detailedType`` and the
free-text ``description`` are deliberately left for a later iteration.

Design notes
------------
* The record→frame mapping lives in :func:`tidy_records` (pure, file-free) so
  it can be unit-tested without the raw dumps, which are gitignored and absent
  in CI.
* ``priceByArea`` (= price ÷ size) is loaded for EDA but is a **leakage trap** —
  it must never be used as a model feature.
* Cleaning is a separate pure step (:func:`clean_sale_listings`) so EDA can see
  the raw distribution (including outliers) before anything is dropped.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RAW_DIR = ROOT / "data" / "raw" / "idealista"
CITIES: tuple[str, ...] = ("madrid", "barcelona", "malaga")

# Raw Idealista field → tidy snake_case column.
COLUMN_MAP: dict[str, str] = {
    "propertyCode": "property_code",
    "price": "price",
    "size": "size_m2",
    "priceByArea": "price_per_m2",   # EDA only — leakage trap for modelling
    "rooms": "rooms",
    "bathrooms": "bathrooms",
    "propertyType": "property_type",
    "status": "status",
    "floor": "floor_raw",
    "hasLift": "has_lift",
    "exterior": "exterior",
    "newDevelopment": "new_development",
    "district": "district",
    "neighborhood": "neighborhood",
    "municipality": "municipality",
    "province": "province",
    "latitude": "latitude",
    "longitude": "longitude",
    "_city": "city",
}

_NUMERIC_COLS = ("price", "size_m2", "price_per_m2", "rooms", "bathrooms",
                 "latitude", "longitude")
_BOOL_COLS = ("has_lift", "exterior", "new_development")

# Idealista non-numeric floor codes → numeric level.
#   bj = bajo (ground), en = entresuelo (mezzanine), ss = semisótano, st = sótano
_FLOOR_CODES: dict[str, float] = {"bj": 0.0, "en": 0.5, "ss": -1.0, "st": -1.0}


def parse_floor(value) -> float:
    """Map an Idealista ``floor`` value to a numeric level (NaN if unknown)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    s = str(value).strip().lower()
    if s in _FLOOR_CODES:
        return _FLOOR_CODES[s]
    try:
        return float(int(s))
    except ValueError:
        return np.nan


def tidy_records(records: list[dict], default_city: str | None = None) -> pd.DataFrame:
    """Turn raw JSONL records into a tidy DataFrame. Pure — no file I/O.

    Selects + renames known columns, coerces dtypes, derives ``floor_num``,
    and drops duplicate ``property_code`` rows (keeping the first).
    """
    df = pd.DataFrame.from_records(records)
    if default_city is not None and "_city" not in df.columns:
        df["_city"] = default_city

    keep = [c for c in COLUMN_MAP if c in df.columns]
    df = df[keep].rename(columns={c: COLUMN_MAP[c] for c in keep})

    for col in _NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in _BOOL_COLS:
        if col in df.columns:
            df[col] = df[col].astype("boolean")

    df["floor_num"] = (
        df["floor_raw"].map(parse_floor) if "floor_raw" in df.columns else np.nan
    )

    if "property_code" in df.columns:
        df = df.drop_duplicates(subset=["property_code"], keep="first")
    return df.reset_index(drop=True)


def find_sale_files(
    raw_dir: Path | str = DEFAULT_RAW_DIR, cities: tuple[str, ...] = CITIES
) -> dict[str, Path]:
    """Return the latest ``sale_*.jsonl`` per city (by filename, which is dated)."""
    raw_dir = Path(raw_dir)
    found: dict[str, Path] = {}
    for city in cities:
        files = sorted((raw_dir / city).glob("sale_*.jsonl"))
        if files:
            found[city] = files[-1]
    return found


def load_sale_listings(
    raw_dir: Path | str = DEFAULT_RAW_DIR,
    cities: tuple[str, ...] = CITIES,
    paths: Mapping[str, Path | str] | None = None,
) -> pd.DataFrame:
    """Load raw sale JSONL into a tidy DataFrame (typing + dedup only).

    Parameters
    ----------
    raw_dir, cities
        Where to look and which cities to include (ignored if ``paths`` given).
    paths
        Optional explicit ``{city: file}`` mapping, bypassing discovery.

    Raises
    ------
    FileNotFoundError
        If no sale files are found.
    """
    files = dict(paths) if paths else find_sale_files(raw_dir, cities)
    if not files:
        raise FileNotFoundError(
            f"No sale_*.jsonl found under {raw_dir}. Run scripts/scrape_idealista.py "
            "with --operations sale first."
        )

    records: list[dict] = []
    for city, path in files.items():
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rec = json.loads(line)
                    rec.setdefault("_city", city)
                    records.append(rec)
    logger.info("Loaded %d raw sale records from %d cities", len(records), len(files))
    return tidy_records(records)


def clean_sale_listings(
    df: pd.DataFrame,
    *,
    price_range: tuple[float, float] = (20_000, 5_000_000),
    sqm_range: tuple[float, float] = (15, 1_000),
    ppm2_range: tuple[float, float] = (300, 20_000),
    rooms_max: int = 15,
    bathrooms_max: int = 10,
) -> pd.DataFrame:
    """Drop implausible rows / scrape artefacts. Pure — returns a filtered copy.

    Default ranges remove the obvious junk seen in EDA (a 32,000 m² listing, a
    €5/m² listing, etc.) while keeping genuine luxury/large stock. Tune via the
    keyword bounds; the model notebook may additionally percentile-clip price.
    """
    df = df.copy()
    n0 = len(df)
    mask = (
        df["price"].between(*price_range)
        & df["size_m2"].between(*sqm_range)
        & df["rooms"].between(0, rooms_max)
        & df["bathrooms"].between(0, bathrooms_max)
    )
    if "price_per_m2" in df.columns:
        mask &= df["price_per_m2"].between(*ppm2_range)
    df = df[mask].dropna(subset=["price", "size_m2", "district", "city"])
    logger.info("clean_sale_listings: %d → %d rows (dropped %d)", n0, len(df), n0 - len(df))
    return df.reset_index(drop=True)


def load_clean_sale(
    raw_dir: Path | str = DEFAULT_RAW_DIR,
    cities: tuple[str, ...] = CITIES,
    paths: Mapping[str, Path | str] | None = None,
) -> pd.DataFrame:
    """Convenience: :func:`load_sale_listings` → :func:`clean_sale_listings`."""
    return clean_sale_listings(load_sale_listings(raw_dir, cities, paths))
