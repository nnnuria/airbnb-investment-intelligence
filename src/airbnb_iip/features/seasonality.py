"""Seasonal multipliers from Inside Airbnb calendar data.

Inside Airbnb publishes a ``calendar.csv.gz`` per city: one row per
(listing × date) for the ~365 days following the scrape, with columns
``date``, ``available`` (``t``/``f``), ``price`` (often null), and stay
limits.

What this module computes
-------------------------

A **monthly seasonal multiplier per city** — a dict
``{city: {1: 0.92, 2: 0.95, ..., 12: 1.10}}`` where ``1.10`` for August
means listings in that city run ~10% above the annual baseline in August.

Why availability-based, not price-based
---------------------------------------

The natural choice would be ``median_price_in_month / annual_median_price``.
We don't do that because the ``price`` column is **100% null in Inside
Airbnb's Spanish exports** (verified across Madrid/Barcelona/Málaga). Until
that changes, we infer demand from availability instead:

    unavail_rate(month) = mean(1 - is_available)  filtered to date > today + N

Higher unavailability = stronger demand = higher multiplier. Normalised so
the annual mean equals 1.0, so the multiplier behaves like a price index.

Why filter the first 60 days
----------------------------

The first ~60 days after the scrape are **already booked** for many
listings — the availability signal is dominated by *near-term sold-out*
rather than *seasonal demand*. Inside Airbnb's own methodology
recommends excluding the scrape-date artefact window. 60 days is the
SPRINT_STATUS-prescribed default and is wide enough to clear most
short-horizon bookings without throwing away too much signal.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

DEFAULT_EXCLUDE_DAYS = 60
MONTHS_PER_YEAR = 12


# ── Public API ───────────────────────────────────────────────────────────────

def compute_seasonal_multipliers(
    calendar: pd.DataFrame,
    *,
    snapshot_date: str | date | None = None,
    exclude_days: int = DEFAULT_EXCLUDE_DAYS,
    city_col: str = "city",
    date_col: str = "date",
    available_col: str = "available",
) -> dict[str, dict[int, float]]:
    """Compute monthly seasonal multipliers per city.

    Parameters
    ----------
    calendar
        Long-form calendar dataframe with one row per (listing, date) and
        at least ``city_col``, ``date_col``, ``available_col``. The
        ``available_col`` is interpreted as boolean — values in
        ``{"t", "T", True, 1}`` are available; everything else is not.
    snapshot_date
        Anchor for the ``exclude_days`` filter. Defaults to today.
    exclude_days
        Drop calendar rows where ``date <= snapshot_date + exclude_days``.
        Default 60 (per SPRINT_STATUS guidance).
    city_col, date_col, available_col
        Override the column names if they differ from the defaults.

    Returns
    -------
    dict
        ``{city: {month_int (1-12): multiplier}}`` where the multiplier is
        the city's monthly unavailability rate normalised so the city's
        annual mean is 1.0. A city missing entirely from the input is
        absent from the output; a city missing a particular month gets
        ``1.0`` for that month.

    Raises
    ------
    KeyError
        If any required column is missing.

    Notes
    -----
    Output is a plain Python dict (not a DataFrame) because downstream
    consumers (finance module, FastAPI endpoint) treat it as a lookup
    table by (city, month). Convert to long-form with
    :func:`multipliers_to_dataframe` when needed.
    """
    for col in (city_col, date_col, available_col):
        if col not in calendar.columns:
            raise KeyError(f"calendar is missing column {col!r}")
    if exclude_days < 0:
        raise ValueError(f"exclude_days must be >= 0, got {exclude_days}")

    anchor = _coerce_snapshot_date(snapshot_date)
    cutoff = anchor + timedelta(days=exclude_days)

    dates = pd.to_datetime(calendar[date_col], errors="coerce", format="mixed")
    keep = dates.notna() & (dates > pd.Timestamp(cutoff))
    if not keep.any():
        return {}

    df = pd.DataFrame({
        "city": calendar.loc[keep, city_col].to_numpy(),
        "month": dates.loc[keep].dt.month.to_numpy(),
        "unavail": (~_as_bool(calendar.loc[keep, available_col])).astype("float64"),
    })

    monthly = df.groupby(["city", "month"], observed=True)["unavail"].mean()

    result: dict[str, dict[int, float]] = {}
    for city, by_month in monthly.groupby(level="city", observed=True):
        rates = by_month.droplevel("city")
        baseline = rates.mean()
        if baseline <= 0 or not np.isfinite(baseline):
            # No signal — flat seasonality is the safest fallback.
            result[str(city)] = {m: 1.0 for m in range(1, MONTHS_PER_YEAR + 1)}
            continue
        normalised = (rates / baseline).round(4)
        city_map = {m: 1.0 for m in range(1, MONTHS_PER_YEAR + 1)}
        for month, val in normalised.items():
            city_map[int(month)] = float(val)
        result[str(city)] = city_map
    return result


def multipliers_to_dataframe(
    multipliers: Mapping[str, Mapping[int, float]],
) -> pd.DataFrame:
    """Pivot the dict-of-dicts to a long DataFrame.

    Columns: ``city``, ``month``, ``multiplier``. Useful for joining onto
    listings by ``city`` + a target month, or for writing as Parquet/CSV.
    """
    rows = [
        {"city": city, "month": int(month), "multiplier": float(value)}
        for city, by_month in multipliers.items()
        for month, value in by_month.items()
    ]
    return pd.DataFrame(rows, columns=["city", "month", "multiplier"])


def load_calendar(
    paths: Mapping[str, Path | str],
    *,
    date_col: str = "date",
    available_col: str = "available",
    price_col: str = "price",
) -> pd.DataFrame:
    """Load one ``calendar.csv[.gz]`` per city and concatenate.

    Parameters
    ----------
    paths
        ``{city_name: path_to_calendar_file}``. Files may be ``.csv`` or
        ``.csv.gz`` — pandas detects from the extension.
    date_col, available_col, price_col
        Source column names in the raw files. Forwarded as-is to the
        output (no renaming) so the result is compatible with
        :func:`compute_seasonal_multipliers`.

    Returns
    -------
    DataFrame
        Concatenated calendar with one row per (city, listing, date) and
        a ``city`` column tagging each row.
    """
    frames: list[pd.DataFrame] = []
    for city, path in paths.items():
        df = pd.read_csv(
            path,
            usecols=lambda c: c in {"listing_id", date_col, available_col, price_col},
        )
        df["city"] = city
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["city", date_col, available_col])
    return pd.concat(frames, ignore_index=True)


# ── internals ────────────────────────────────────────────────────────────────

def _coerce_snapshot_date(value: str | date | None) -> date:
    if value is None:
        return date.today()
    if isinstance(value, date):
        return value
    return pd.to_datetime(value).date()


def _as_bool(series: pd.Series) -> pd.Series:
    """Interpret Inside Airbnb's t/f flag (or already-boolean) as bool."""
    if series.dtype == bool:
        return series
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"t", "true", "1"})
