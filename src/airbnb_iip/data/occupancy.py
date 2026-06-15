"""San Francisco occupancy estimator (SPRINT_STATUS task 2).

Inside Airbnb's exports include a per-listing ``estimated_occupancy_l365d``
column derived from the **San Francisco model** — the de-facto standard for
inferring bookings from review activity, because only guests who actually
stayed can leave reviews. This module reimplements that estimator as two
pure functions wired to our own auditable assumptions
(``airbnb_iip.config.OCCUPANCY``):

* :func:`estimate_occupancy` — scalar form for one listing or a hypothetical
  property (the decision-flow case where Inside Airbnb has no row).
* :func:`estimate_occupancy_l365d` — vectorised form that produces an
  ``estimated_occupancy_l365d`` Series matching the Inside Airbnb column,
  ready to drop into a listings dataframe or use as a target for the
  downstream occupancy ML model (task 4).

The model
=========

    nights_per_month  ≈  (reviews_per_month / review_rate) × avg_length_of_stay
    nights_per_year   ≈  nights_per_month × 12
    occupancy_rate    =  nights_per_year / 365     (capped at ``max_occupancy``)

Assumptions (defaults from :data:`airbnb_iip.config.OCCUPANCY`):

* ``review_rate`` = 0.50 — share of stays that leave a review. Inside
  Airbnb's published default; tune per city if a stronger anchor is
  available (e.g. Madrid/Barcelona STR registry).
* ``avg_length_of_stay`` = 3 — mean booking length in nights for urban
  short-term rentals.
* ``max_occupancy`` = 0.70 — realistic annual occupancy ceiling. Caps the
  estimate at ``floor(0.70 × 365) = 255`` nights/year, matching the upper
  bound observed in Inside Airbnb's pre-computed column.

This is an **indicative** estimate, not ground truth. Use it as a feature
or target for downstream models — never present it as a guaranteed
projection to the end user.
"""

from __future__ import annotations

import math
from typing import Any

from airbnb_iip.config import OCCUPANCY

DAYS_PER_YEAR = 365
MONTHS_PER_YEAR = 12
DAYS_PER_MONTH = 30


def estimate_occupancy(
    reviews_per_month: float | None,
    *,
    review_rate: float = OCCUPANCY["review_rate"],
    avg_length_of_stay: float = OCCUPANCY["avg_length_of_stay"],
    max_occupancy: float = OCCUPANCY["max_occupancy"],
) -> dict[str, float]:
    """Monthly occupancy estimate for one listing using the SF model.

    Parameters
    ----------
    reviews_per_month
        Average reviews per month for the listing. ``None`` and ``NaN`` are
        treated as zero (no review history → no inferred bookings).
        Negative inputs are clamped to zero.
    review_rate
        Share of stays that leave a review. Must be in ``(0, 1]``. Default
        from :data:`airbnb_iip.config.OCCUPANCY`.
    avg_length_of_stay
        Mean nights per booking. Must be ``> 0``. Default from config.
    max_occupancy
        Annual occupancy ceiling in ``(0, 1]``. Applied as a monthly cap of
        ``max_occupancy * 30`` nights. Default from config.

    Returns
    -------
    dict with:
        ``nights_booked_per_month`` : float
            Estimated booked nights/month, capped at ``max_occupancy * 30``.
        ``occupancy_rate`` : float
            ``nights_booked_per_month / 30``, in ``[0, max_occupancy]``.

    Raises
    ------
    ValueError
        If any of the assumption parameters are out of range, or if
        ``reviews_per_month`` is non-numeric (other than ``None`` / ``NaN``).
    """
    _validate_assumptions(review_rate, avg_length_of_stay, max_occupancy)

    r = _coerce_reviews_per_month(reviews_per_month)
    nights_per_month = (r / review_rate) * avg_length_of_stay
    ceiling = max_occupancy * DAYS_PER_MONTH
    nights = min(nights_per_month, ceiling)
    return {
        "nights_booked_per_month": nights,
        "occupancy_rate": nights / DAYS_PER_MONTH,
    }


def estimate_occupancy_l365d(
    df: Any,
    *,
    review_rate: float = OCCUPANCY["review_rate"],
    avg_length_of_stay: float = OCCUPANCY["avg_length_of_stay"],
    max_occupancy: float = OCCUPANCY["max_occupancy"],
    reviews_per_month_col: str = "reviews_per_month",
):
    """Vectorised SF estimator: estimated booked nights over the last 365 days.

    Drop-in replacement for Inside Airbnb's pre-computed
    ``estimated_occupancy_l365d`` column. Same name, same dtype (int64),
    same upper bound (``floor(max_occupancy * 365)``, i.e. 255 nights with
    the default 0.70 ceiling).

    Parameters
    ----------
    df
        A pandas DataFrame containing ``reviews_per_month_col``. NaN and
        negative values are treated as zero.
    review_rate, avg_length_of_stay, max_occupancy
        See :func:`estimate_occupancy`. Defaults from
        :data:`airbnb_iip.config.OCCUPANCY`.
    reviews_per_month_col
        Override the source column name. Defaults to ``"reviews_per_month"``
        — the Inside Airbnb canonical name.

    Returns
    -------
    pandas.Series
        Named ``estimated_occupancy_l365d``, indexed like ``df``, ``int64``
        dtype, values in ``[0, floor(max_occupancy * 365)]``.

    Raises
    ------
    ValueError
        If assumptions are out of range.
    KeyError
        If ``reviews_per_month_col`` is missing from ``df``.

    Notes
    -----
    Annualisation uses ``× 12`` (months in a year), matching Inside Airbnb's
    convention. This deliberately differs from a pure linear scaling of the
    monthly value to ``365/30 ≈ 12.17`` so the column lines up with the
    pre-computed one.
    """
    import pandas as pd

    _validate_assumptions(review_rate, avg_length_of_stay, max_occupancy)
    if reviews_per_month_col not in df.columns:
        raise KeyError(
            f"DataFrame is missing column {reviews_per_month_col!r}. "
            "Pass reviews_per_month_col= to override the source column."
        )

    reviews = (
        pd.to_numeric(df[reviews_per_month_col], errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0)
    )
    nights_year = (reviews / review_rate) * avg_length_of_stay * MONTHS_PER_YEAR
    ceiling = math.floor(max_occupancy * DAYS_PER_YEAR)
    capped = nights_year.clip(upper=ceiling)
    return (
        capped.round()
        .astype("int64")
        .rename("estimated_occupancy_l365d")
    )


def _validate_assumptions(
    review_rate: float, avg_length_of_stay: float, max_occupancy: float,
) -> None:
    if not 0 < review_rate <= 1:
        raise ValueError(f"review_rate must be in (0, 1], got {review_rate}")
    if avg_length_of_stay <= 0:
        raise ValueError(
            f"avg_length_of_stay must be > 0, got {avg_length_of_stay}"
        )
    if not 0 < max_occupancy <= 1:
        raise ValueError(f"max_occupancy must be in (0, 1], got {max_occupancy}")


def _coerce_reviews_per_month(value: Any) -> float:
    """Treat ``None``/``NaN``/negative as zero; raise on non-numeric input."""
    if value is None:
        return 0.0
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"reviews_per_month must be numeric or None, got {value!r}"
        ) from exc
    if math.isnan(v):
        return 0.0
    return max(v, 0.0)
