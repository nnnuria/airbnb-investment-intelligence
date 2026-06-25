"""Calendar-based occupancy utilities.

The legacy **San Francisco** review-based estimator (``estimate_occupancy`` /
``estimate_occupancy_l365d`` — bookings inferred from review velocity) has been
**retired**. It barely tracked real availability (corr ≈ 0.13 with calendar
occupancy on this dataset) and has been replaced by a learned LightGBM model
whose target comes straight from Inside Airbnb calendar data:

    a night is "booked"  ⇔  available == 'f'
    occupancy_rate       =  booked nights / calendar nights

The model lives in :mod:`airbnb_iip.models.occupancy`
(:class:`~airbnb_iip.models.occupancy.OccupancyPredictor`,
:func:`~airbnb_iip.models.occupancy.get_occupancy_predictor`); it is trained by
``scripts/train_occupancy_model.py`` and documented in
``docs/model_cards/occupancy_model.md``.

This module keeps only the lightweight, dependency-free helpers used to
**build the calendar target** and to populate the ABT's
``estimated_occupancy_l365d`` column from forward availability — no ML import,
no review heuristics.
"""

from __future__ import annotations

from typing import Any

DAYS_PER_YEAR = 365

# Inside Airbnb encodes calendar availability as a single char: 't' = available
# (bookable), 'f' = not available, which we treat as a booked night.
BOOKED_FLAG = "f"


def occupancy_rate_from_calendar(
    calendar: Any,
    *,
    id_col: str = "listing_id",
    available_col: str = "available",
) -> Any:
    """Per-listing occupancy rate from an Inside Airbnb calendar frame.

    Occupancy is the share of a listing's forward-calendar nights that are
    **not available** (``available == 'f'`` ⇒ booked).

    Parameters
    ----------
    calendar
        A pandas DataFrame with one row per (listing, date) and at least
        ``id_col`` and ``available_col`` columns.
    id_col, available_col
        Column names. Default to Inside Airbnb's ``listing_id`` / ``available``.

    Returns
    -------
    pandas.DataFrame
        One row per listing with columns ``listing_id`` (named after
        ``id_col``), ``occupancy_rate`` (float in [0, 1]), ``calendar_days``
        (window length), and ``occupancy_nights_l365d`` (rate × 365, int).

    Raises
    ------
    KeyError
        If a required column is missing.
    """
    import pandas as pd

    for col in (id_col, available_col):
        if col not in calendar.columns:
            raise KeyError(f"calendar is missing required column {col!r}")

    booked = (
        calendar[available_col].astype(str).str.strip() == BOOKED_FLAG
    ).astype("int8")
    grouped = booked.groupby(calendar[id_col])
    out = grouped.mean().rename("occupancy_rate").reset_index()
    out["calendar_days"] = grouped.size().to_numpy()
    out["occupancy_nights_l365d"] = (
        (out["occupancy_rate"] * DAYS_PER_YEAR).round().astype("int64")
    )
    return out


def occupancy_l365d_from_availability(
    df: Any,
    *,
    availability_col: str = "availability_365",
) -> Any:
    """Booked nights over the next 365 days from forward availability.

    ``estimated_occupancy_l365d = 365 − availability_365``. Inside Airbnb's
    ``availability_365`` is the count of bookable nights in the next year, so
    its complement is the count of taken (booked/blocked) nights — a purely
    **calendar-derived** quantity, not a review heuristic.

    Drop-in replacement for the ABT fill that the retired SF estimator used to
    provide. Returns an ``int64`` Series named ``estimated_occupancy_l365d``,
    indexed like ``df``, values in ``[0, 365]``.

    Raises
    ------
    KeyError
        If ``availability_col`` is missing from ``df``.
    """
    import pandas as pd

    if availability_col not in df.columns:
        raise KeyError(
            f"DataFrame is missing column {availability_col!r}. "
            "Pass availability_col= to override the source column."
        )
    avail = (
        pd.to_numeric(df[availability_col], errors="coerce")
        .fillna(DAYS_PER_YEAR)
        .clip(lower=0, upper=DAYS_PER_YEAR)
    )
    return (
        (DAYS_PER_YEAR - avail)
        .round()
        .astype("int64")
        .rename("estimated_occupancy_l365d")
    )


__all__ = [
    "DAYS_PER_YEAR",
    "BOOKED_FLAG",
    "occupancy_rate_from_calendar",
    "occupancy_l365d_from_availability",
]
