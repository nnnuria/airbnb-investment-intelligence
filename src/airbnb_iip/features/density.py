"""District competitive density via haversine BallTree.

For each listing, count the number of **other** listings within a fixed
ground-distance radius (default 500 m). The output column
``competitive_density_500m`` is part of the production price model's input
contract — preserve it exactly.

Implementation lifted verbatim from `notebooks/price_ml_model.ipynb` cell
5.3. Per-city trees, not one global tree, so neighbours never bleed across
cities.

Why per-city trees rather than per-district:

* Cities are well-separated geographically; haversine queries within a
  city tree never wander across cities.
* District boundaries are fuzzy. Building one tree per district would
  introduce edge artefacts near district borders. The 500 m radius is
  small enough that "competitive density" naturally tracks block-level
  market saturation regardless of which district a listing nominally
  belongs to.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EARTH_RADIUS_M = 6_371_000           # WGS84 mean radius
DEFAULT_RADIUS_M = 500
DEFAULT_OUTPUT_COL = "competitive_density_500m"


def add_competitive_density(
    df: pd.DataFrame,
    *,
    radius_m: float = DEFAULT_RADIUS_M,
    output_col: str = DEFAULT_OUTPUT_COL,
    city_col: str = "city",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
) -> pd.DataFrame:
    """Return a copy of ``df`` with a competitive density column appended.

    Parameters
    ----------
    df
        Listings DataFrame; must contain ``city_col``, ``lat_col``, ``lon_col``.
    radius_m
        Search radius in metres. Default 500 (the price model's contract).
        If overridden, the default output column name no longer applies —
        pass ``output_col`` explicitly.
    output_col
        Column name to write. Default ``"competitive_density_500m"``.
    city_col, lat_col, lon_col
        Column names to read.

    Returns
    -------
    DataFrame
        ``df`` plus one new int32 column. The count **excludes the listing
        itself** (a listing is not its own neighbour). Listings with NaN
        latitude or longitude get a density of 0 and are not included as
        neighbours for any other listing.

    Raises
    ------
    KeyError
        If any required column is missing.
    """
    for col in (city_col, lat_col, lon_col):
        if col not in df.columns:
            raise KeyError(f"DataFrame is missing column {col!r}")

    density = compute_density_array(
        df[city_col].to_numpy(),
        df[lat_col].to_numpy(),
        df[lon_col].to_numpy(),
        radius_m=radius_m,
    )
    out = df.copy()
    out[output_col] = density
    return out


def compute_density_array(
    cities: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    *,
    radius_m: float = DEFAULT_RADIUS_M,
) -> np.ndarray:
    """Return per-listing density counts as a 1-D ``int32`` array.

    Same length as the inputs. Listings whose lat/lon is NaN return 0 and
    do not count as neighbours for any other listing.

    Pandas/sklearn are imported lazily so importing this module is cheap.
    """
    from sklearn.neighbors import BallTree

    cities = np.asarray(cities)
    lats = np.asarray(latitudes, dtype=float)
    lons = np.asarray(longitudes, dtype=float)

    if not (len(cities) == len(lats) == len(lons)):
        raise ValueError("cities, latitudes, longitudes must be the same length")

    density = np.zeros(len(cities), dtype=np.int32)
    if len(cities) == 0:
        return density

    valid = np.isfinite(lats) & np.isfinite(lons)
    radius_rad = radius_m / EARTH_RADIUS_M

    for city in pd.unique(cities):
        in_city = cities == city
        usable = in_city & valid
        if not usable.any():
            continue
        coords_rad = np.radians(
            np.column_stack([lats[usable], lons[usable]])
        )
        tree = BallTree(coords_rad, metric="haversine")
        counts = tree.query_radius(coords_rad, r=radius_rad, count_only=True)
        density[usable] = counts.astype(np.int32) - 1
    return density
