"""
Shared data-cleaning utilities for the Airbnb IIP project.

Each function is a pure transform: takes a DataFrame/Series, returns a
DataFrame/Series.  No global state, no city-specific paths or thresholds.
City-specific decisions (outlier cuts, extra column drops) belong in the
per-city EDA notebooks, applied *after* calling clean_listings().
"""

import re

import numpy as np
import pandas as pd


# ── Type conversions ──────────────────────────────────────────────────────────

def parse_price(series: pd.Series) -> pd.Series:
    """Strip currency symbols / commas and convert to float."""
    return series.str.replace(r"[$,]", "", regex=True).astype("float")


def parse_rate(series: pd.Series) -> pd.Series:
    """Strip '%' and return a float proportion in [0, 1]."""
    return series.str.replace("%", "", regex=False).astype("float") / 100


def parse_bool(series: pd.Series) -> pd.Series:
    """Convert Airbnb 't'/'f' strings to bool (NaN → False)."""
    return series == "t"


def parse_dates(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Convert named date columns to datetime64[ns] (coerces bad values to NaT)."""
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format="%Y-%m-%d", errors="coerce")
    return df


# ── Bathroom reconciliation ───────────────────────────────────────────────────

def parse_bathrooms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reconcile legacy bathrooms (float) with newer bathrooms_text (string).

    Produces:
      bathrooms_number      – float count extracted from text, falling back to
                              the legacy float column
      bathrooms_description – text descriptor (e.g. 'shared bath', 'private bath')

    Drops the originals (bathrooms, bathrooms_text) and fills any remaining
    NaN with the column mode.
    """
    df = df.copy()
    if "bathrooms_text" in df.columns:
        df["bathrooms_number"] = (
            df["bathrooms_text"]
            .str.extract(r"(^\d*\.?\d+)")[0]
            .astype(float)
        )
        df["bathrooms_description"] = (
            df["bathrooms_text"]
            .str.replace(r"^\d*\.?\d+\s*", "", regex=True)
            .replace("", np.nan)
        )
        if "bathrooms" in df.columns:
            df["bathrooms_number"] = df["bathrooms_number"].fillna(df["bathrooms"])
        df = df.drop(columns=[c for c in ["bathrooms", "bathrooms_text"] if c in df.columns])

    for col in ["bathrooms_number", "bathrooms_description"]:
        if col in df.columns:
            mode_val = df[col].mode()
            if not mode_val.empty:
                df[col] = df[col].fillna(mode_val[0])

    return df


# ── Temporal features ─────────────────────────────────────────────────────────

def add_temporal_features(
    df: pd.DataFrame, reference_col: str = "last_scraped"
) -> pd.DataFrame:
    """
    Derive host tenure, days-since-review, and review-span features.
    Date columns must already be parsed as datetime before calling this.
    """
    df = df.copy()
    ref = df[reference_col] if reference_col in df.columns else pd.to_datetime("today")
    if "host_since" in df.columns:
        df["host_tenure_years"] = (ref - df["host_since"]).dt.days / 365.25
    if "first_review" in df.columns:
        df["days_since_first_review"] = (ref - df["first_review"]).dt.days
    if "last_review" in df.columns:
        df["days_since_last_review"] = (ref - df["last_review"]).dt.days
    if "first_review" in df.columns and "last_review" in df.columns:
        df["review_span_years"] = (
            (df["last_review"] - df["first_review"]).dt.days / 365.25
        )
    return df


# ── Deduplication ─────────────────────────────────────────────────────────────

def drop_true_duplicates(
    df: pd.DataFrame, id_cols: list = None
) -> pd.DataFrame:
    """
    Drop rows identical across every column except identifier columns.
    Keeps the first occurrence.
    """
    if id_cols is None:
        id_cols = ["id", "listing_url"]
    subset = [c for c in df.columns if c not in id_cols]
    return df.drop_duplicates(subset=subset, keep="first").copy()


# ── Columns to drop ───────────────────────────────────────────────────────────

_REDUNDANT_COLS = [
    "listing_url", "picture_url", "host_url",
    "host_thumbnail_url", "host_picture_url",
    "host_neighbourhood", "host_listings_count",
    "host_total_listings_count",
]


def drop_url_cols(df: pd.DataFrame, extra_cols: list = None) -> pd.DataFrame:
    """Drop URL / thumbnail / redundant columns with no analytical value."""
    to_drop = [c for c in _REDUNDANT_COLS if c in df.columns]
    if extra_cols:
        to_drop += [c for c in extra_cols if c in df.columns]
    return df.drop(columns=to_drop)


# ── Missing value imputation ──────────────────────────────────────────────────

def impute_host_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hierarchically impute host-level fields:
      1. Mode within same host_id
      2. host_location gets a second fallback: mode within host_neighbourhood
      3. Remaining gaps → 'Unknown' (text) or 0 (rates / response_time)
    """
    df = df.copy()
    host_modal_cols = [
        "host_neighbourhood", "host_about", "host_acceptance_rate",
        "host_response_rate", "host_response_time", "host_location",
    ]
    for col in host_modal_cols:
        if col not in df.columns:
            continue
        df[col] = df[col].fillna(
            df.groupby("host_id")[col].transform(
                lambda x: x.mode().iat[0] if not x.mode().empty else np.nan
            )
        )

    if "host_location" in df.columns and "host_neighbourhood" in df.columns:
        df["host_location"] = df["host_location"].fillna(
            df.groupby("host_neighbourhood")["host_location"].transform(
                lambda x: x.mode().iat[0] if not x.mode().empty else np.nan
            )
        )

    for col in ["host_neighbourhood", "host_about", "host_location",
                "neighborhood_overview", "license"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    for col in ["host_acceptance_rate", "host_response_rate", "host_response_time"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return df


def impute_price(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hierarchically impute missing price via group means:
      1. (neighbourhood, property_type, room_type, accommodates)
      2. (neighbourhood, property_type, room_type)
      3. (neighbourhood,)
    """
    df = df.copy()
    for groupby_cols in [
        ["neighbourhood_cleansed", "property_type", "room_type", "accommodates"],
        ["neighbourhood_cleansed", "property_type", "room_type"],
        ["neighbourhood_cleansed"],
    ]:
        available = [c for c in groupby_cols if c in df.columns]
        if len(available) == len(groupby_cols):
            df["price"] = df["price"].fillna(
                df.groupby(groupby_cols)["price"].transform("mean")
            )
        if df["price"].isna().sum() == 0:
            break
    return df


def impute_beds_bedrooms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values via the chain:
      beds ← bedrooms ← accommodates
    Then cast both to int.
    """
    df = df.copy()
    if "bedrooms" in df.columns and "accommodates" in df.columns:
        df["bedrooms"] = df["bedrooms"].fillna(df["accommodates"]).astype(int)
    if "beds" in df.columns and "bedrooms" in df.columns:
        df["beds"] = df["beds"].fillna(df["bedrooms"]).astype(int)
    return df


_REVIEW_COLS = [
    "review_scores_rating", "review_scores_accuracy", "review_scores_cleanliness",
    "review_scores_checkin", "review_scores_communication",
    "review_scores_location", "review_scores_value",
    "review_span_years", "days_since_last_review", "days_since_first_review",
    "reviews_per_month",
]


def fill_review_zeros(df: pd.DataFrame) -> pd.DataFrame:
    """Fill review-score / activity NaN with 0 (listing has no reviews yet)."""
    df = df.copy()
    cols = [c for c in _REVIEW_COLS if c in df.columns]
    df[cols] = df[cols].fillna(0)
    return df


def fill_sentinel_dates(
    df: pd.DataFrame, cols: list = None
) -> pd.DataFrame:
    """Fill missing review date columns with the sentinel 1800-01-01."""
    df = df.copy()
    if cols is None:
        cols = ["first_review", "last_review"]
    sentinel = pd.Timestamp("1800-01-01")
    for col in cols:
        if col in df.columns:
            df[col] = df[col].fillna(sentinel)
    return df


# ── Feature engineering ───────────────────────────────────────────────────────

_HOTEL_KEYWORDS = [
    "hotel", "hostel", "aparthotel", "serviced apartment",
    "boutique hotel", "bed and breakfast", "pension", "ryokan",
]
_UNIQUE_KEYWORDS = [
    "tiny home", "hut", "yurt", "camper", "rv", "cave",
    "barn", "dome", "earthen home", "tent", "religious building",
]


def standardize_property_type(pt: str) -> str:
    """Map a raw property_type string to one of 6 standard categories."""
    if not isinstance(pt, str):
        return "Other"
    pt_lower = pt.lower()
    if "entire" in pt_lower:
        return "Entire place"
    if "private room" in pt_lower:
        return "Private room"
    if "shared room" in pt_lower:
        return "Shared room"
    if any(k in pt_lower for k in _HOTEL_KEYWORDS):
        return "Hotel / Hostel"
    if any(k in pt_lower for k in _UNIQUE_KEYWORDS):
        return "Unique stay"
    return "Other"


def add_property_type_std(df: pd.DataFrame) -> pd.DataFrame:
    """Add property_type_std column."""
    df = df.copy()
    if "property_type" in df.columns:
        df["property_type_std"] = df["property_type"].apply(standardize_property_type)
    return df


def rate_bucket(x) -> str:
    """Bin a [0, 1] rate to 'unknown' / 'low' / 'medium' / 'high'."""
    if pd.isna(x) or x == 0:
        return "unknown"
    if x < 0.5:
        return "low"
    if x < 0.8:
        return "medium"
    return "high"


def add_rate_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Add ordered categorical buckets for host response / acceptance rates."""
    df = df.copy()
    cat_dtype = pd.CategoricalDtype(["unknown", "low", "medium", "high"], ordered=True)
    for col in ["host_response_rate", "host_acceptance_rate"]:
        if col in df.columns:
            df[f"{col}_cat"] = pd.Categorical(
                df[col].apply(rate_bucket), dtype=cat_dtype
            )
    return df


def add_price_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add price_cat (low / medium / high / very_high) using the data's own
    quartiles (Q1, median, Q3) so thresholds adapt to each city automatically.
    """
    df = df.copy()
    q1, q2, q3 = df["price"].quantile([0.25, 0.50, 0.75])

    def _cat(x):
        if pd.isna(x):
            return "low"
        if x <= q1:
            return "low"
        if x <= q2:
            return "medium"
        if x <= q3:
            return "high"
        return "very_high"

    cat_dtype = pd.CategoricalDtype(
        ["low", "medium", "high", "very_high"], ordered=True
    )
    df["price_cat"] = pd.Categorical(df["price"].apply(_cat), dtype=cat_dtype)
    return df


def add_description_length(df: pd.DataFrame) -> pd.DataFrame:
    """Add description_length (character count)."""
    df = df.copy()
    if "description" in df.columns:
        df["description_length"] = df["description"].fillna("").str.len()
    return df


# ── Full cleaning pipeline ────────────────────────────────────────────────────

_DATE_COLS = [
    "last_scraped", "host_since", "calendar_last_scraped",
    "first_review", "last_review",
]
_BOOL_COLS = [
    "host_is_superhost", "host_has_profile_pic", "host_identity_verified",
    "has_availability", "instant_bookable",
]


def clean_listings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full standard cleaning pipeline on a raw listings DataFrame.

    Steps (in order):
      1.  Drop always-useless columns (calendar_updated, neighbourhood)
      2.  Drop exact duplicates (ignoring id / listing_url)
      3.  Parse dates
      4.  Parse prices, rates, booleans
      5.  Normalise host_name multi-host separator to '&'
      6.  Reconcile bathrooms fields
      7.  Impute host fields, price, beds/bedrooms
      8.  Fill review scores with 0 / dates with sentinel 1800-01-01
      9.  Add temporal features
      10. Add engineered features (property_type_std, rate_cat, price_cat,
          description_length)
      11. Drop URL / redundant columns

    City-specific decisions (outlier cuts, extra column drops) should be
    applied in the per-city EDA notebook *after* calling this function.
    """
    df = df.copy()

    # 1. Drop always-useless columns
    for col in ["calendar_updated", "neighbourhood"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # 2. Deduplication
    df = drop_true_duplicates(df)

    # 3. Dates
    df = parse_dates(df, _DATE_COLS)

    # 4. Type conversions
    if "price" in df.columns:
        df["price"] = parse_price(df["price"])
    for col in ["host_response_rate", "host_acceptance_rate"]:
        if col in df.columns:
            df[col] = parse_rate(df[col])
    for col in _BOOL_COLS:
        if col in df.columns:
            df[col] = parse_bool(df[col])

    # 5. Normalise multi-host name separators to '&'
    if "host_name" in df.columns:
        df["host_name"] = df["host_name"].str.replace(r" . ", " & ", regex=True)

    # 6. Bathrooms
    df = parse_bathrooms(df)

    # 7. Imputation
    df = impute_host_fields(df)
    df = impute_price(df)
    df = impute_beds_bedrooms(df)

    # 8. Review sentinels / zeros
    df = fill_sentinel_dates(df)
    df = fill_review_zeros(df)

    # 9. Temporal features (requires parsed dates)
    df = add_temporal_features(df)

    # 10. Engineered features
    df = add_property_type_std(df)
    df = add_rate_categories(df)
    df = add_price_category(df)
    df = add_description_length(df)

    # 11. Drop URL / redundant columns
    df = drop_url_cols(df)

    return df


# ── Reporting helpers ─────────────────────────────────────────────────────────

def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame of missing counts and percentages, sorted descending."""
    report = df.isna().sum().to_frame("missing_count")
    report["missing_percent"] = report["missing_count"] / len(df) * 100
    return report.sort_values("missing_percent", ascending=False)
