"""ABT (Analytical Base Table) orchestrator.

The single importable entry point for the listings feature pipeline:

    >>> from airbnb_iip.data.abt import build_abt
    >>> path = build_abt("madrid")   # writes <date>-stamped parquet

What it does
============

1. Load the cleaned multi-city listings parquet (output of
   :mod:`airbnb_iip.data.cleaning`) and filter to one city.
2. Append amenity binary flags via :func:`airbnb_iip.features.amenities.add_amenity_flags`.
3. Append competitive density via
   :func:`airbnb_iip.features.density.add_competitive_density`.
4. Append the SF occupancy estimate via
   :func:`airbnb_iip.data.occupancy.estimate_occupancy_l365d` —
   only when ``estimated_occupancy_l365d`` is not already present (Inside
   Airbnb's pre-computed column wins when available).
5. Optionally write a versioned parquet to
   ``Data/processed/<city>_abt_<YYYY-MM-DD>.parquet``.

What it deliberately does NOT do
================================

* **No re-cleaning.** Cleaning is a separate pipeline
  (:mod:`airbnb_iip.data.cleaning`). build_abt assumes its input is the
  cleaned parquet.
* **No price prediction.** ``price_hat`` lives in
  ``Data/processed/listings_with_price_hat.parquet`` and is produced by a
  downstream step that loads the LightGBM artefact. Loading a heavy ML
  model would balloon this module's import cost and couple feature
  engineering to model-serving — keep them separate.
* **No seasonal multipliers on the row.** Seasonality is a city-level
  lookup ``{city: {month: multiplier}}`` produced by
  :mod:`airbnb_iip.features.seasonality`; the finance module joins it in
  per scenario, not per row.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd

from airbnb_iip.data.occupancy import estimate_occupancy_l365d
from airbnb_iip.features.amenities import add_amenity_flags
from airbnb_iip.features.density import add_competitive_density

logger = logging.getLogger(__name__)

DEFAULT_INPUT_PARQUET = Path("Data/processed/listings_all_cities.parquet")
DEFAULT_OUTPUT_DIR = Path("Data/processed")
KNOWN_CITIES: tuple[str, ...] = ("madrid", "barcelona", "malaga")


def build_abt(
    city: str,
    *,
    listings: pd.DataFrame | None = None,
    input_parquet: Path | str = DEFAULT_INPUT_PARQUET,
    output_dir: Path | str | None = DEFAULT_OUTPUT_DIR,
    snapshot_date: str | None = None,
    city_col: str = "city",
) -> pd.DataFrame:
    """Build the feature-engineered ABT for one city.

    Parameters
    ----------
    city
        City slug (``"madrid"``, ``"barcelona"``, ``"malaga"``). Lowercased
        before lookup.
    listings
        Optional pre-loaded DataFrame to skip parquet I/O — useful in tests
        and when chaining build_abt into a larger pipeline. When provided,
        ``input_parquet`` is ignored.
    input_parquet
        Cleaned multi-city listings parquet. Default
        ``Data/processed/listings_all_cities.parquet``.
    output_dir
        Where to write ``<city>_abt_<YYYY-MM-DD>.parquet``. Pass ``None``
        to skip writing (returns the DataFrame only).
    snapshot_date
        Override the filename date stamp. Defaults to today.
    city_col
        Column to filter on. Default ``"city"``.

    Returns
    -------
    DataFrame
        The feature-engineered city slice. Even when ``output_dir`` is
        provided, the in-memory frame is returned for chaining.

    Raises
    ------
    ValueError
        If ``city`` is unknown.
    KeyError
        If a required source column is missing.
    FileNotFoundError
        If ``listings`` is None and ``input_parquet`` doesn't exist.
    """
    city = city.lower()
    _validate_city(city)

    df = _load_listings(listings, input_parquet)
    df = _filter_city(df, city, city_col=city_col)
    logger.info("build_abt(%s): %d rows after city filter", city, len(df))

    df = add_amenity_flags(df)
    df = add_competitive_density(df)

    if "estimated_occupancy_l365d" not in df.columns:
        df["estimated_occupancy_l365d"] = estimate_occupancy_l365d(df)
        logger.info("build_abt(%s): added estimated_occupancy_l365d", city)
    else:
        logger.info(
            "build_abt(%s): estimated_occupancy_l365d already present "
            "(Inside Airbnb value preserved)", city,
        )

    if output_dir is not None:
        out_path = output_path(city, output_dir=output_dir, snapshot_date=snapshot_date)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)
        logger.info("Wrote ABT: %s (%d rows, %d cols)", out_path, len(df), df.shape[1])

    return df


def build_abts(
    cities: Iterable[str] = KNOWN_CITIES,
    **kwargs,
) -> dict[str, pd.DataFrame]:
    """Build ABTs for many cities; failures are logged and skipped."""
    out: dict[str, pd.DataFrame] = {}
    for city in cities:
        try:
            out[city] = build_abt(city, **kwargs)
        except Exception:
            logger.exception("build_abt failed for city=%s — skipping", city)
    return out


def output_path(
    city: str,
    *,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    snapshot_date: str | None = None,
) -> Path:
    """Return the versioned ABT parquet path for one city."""
    stamp = snapshot_date or date.today().isoformat()
    return Path(output_dir) / f"{city.lower()}_abt_{stamp}.parquet"


# ── internals ────────────────────────────────────────────────────────────────

def _validate_city(city: str) -> None:
    if city not in KNOWN_CITIES:
        raise ValueError(
            f"Unknown city {city!r}. Known: {list(KNOWN_CITIES)}. "
            "Add to KNOWN_CITIES if extending coverage."
        )


def _load_listings(
    listings: pd.DataFrame | None, input_parquet: Path | str,
) -> pd.DataFrame:
    if listings is not None:
        return listings.copy()
    path = Path(input_parquet)
    if not path.exists():
        raise FileNotFoundError(
            f"Cleaned listings parquet not found at {path}. "
            "Run the cleaning pipeline first or pass listings= explicitly."
        )
    return pd.read_parquet(path)


def _filter_city(df: pd.DataFrame, city: str, *, city_col: str) -> pd.DataFrame:
    if city_col not in df.columns:
        raise KeyError(
            f"DataFrame is missing {city_col!r} column. "
            "Cleaned listings must carry a city tag."
        )
    mask = df[city_col].astype(str).str.lower() == city
    if not mask.any():
        raise ValueError(
            f"No rows for city={city!r} in input. "
            f"Available: {sorted(df[city_col].astype(str).str.lower().unique())}"
        )
    return df.loc[mask].reset_index(drop=True)
