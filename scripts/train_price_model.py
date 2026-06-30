"""Train (or retrain) the nightly-price model — CLI entry point.

Mirrors ``train_sale_model.py`` / ``train_occupancy_model.py``: the **narrated
reference** is ``notebooks/price_ml_model_comparison.ipynb`` (the approach study
that selected the by-city LightGBM as the production model), and this script is
the reproducible, headless retrain of those **production** artefacts:

    models/price_city_madrid_model.pkl
    models/price_city_barcelona_model.pkl
    models/price_city_malaga_model.pkl
    models/price_city_encoders.pkl       (neighbourhood target map + cat encoders)
    models/price_city_artefacts.json     (feature list + city→file map)

consumed at serving by ``airbnb_iip.models.price.CityPricePredictor``.

Pipeline (faithful to the comparison notebook, §2–§7):
    load listings_segmented → target log1p(price) clipped at p99.5 → amenity
    flags (22, three `_ml`-suffixed so they don't clobber the parquet's existing
    has_crib/has_balcony/has_private_entrance) → booking flags + competitive
    density → drop leakage/metadata → encode (label / ordinal / neighbourhood
    target-encoding / median impute) → RF-importance RFE (≥ median) → LightGBM
    GridSearchCV for hyper-params → refit one LightGBM per city on the
    city-feature subset → save artefacts.

LightGBM is the production family selected by the notebook's 7-model bake-off
(``price_city_artefacts.json`` records ``best_model: LightGBM``); this retrain
trains that family directly rather than re-running the full comparison.

Safety: existing city artefacts are copied to ``*_legacy.*`` (once — never
overwritten) before the new ones are written, so a revert is just restoring them.

Run:
    python scripts/train_price_model.py
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.simplefilter("ignore")
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("train_price")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from airbnb_iip.features.amenities import AMENITY_PATTERNS, add_amenity_flags  # noqa: E402
from airbnb_iip.features.density import add_competitive_density                # noqa: E402

MODELS = ROOT / "models"
DATA_PATH = ROOT / "Data" / "processed" / "listings_segmented.parquet"
SEED = 42
TARGET = "price_log"

ARTEFACTS_JSON = MODELS / "price_city_artefacts.json"
ENCODERS_PKL = MODELS / "price_city_encoders.pkl"
CITIES = ["Madrid", "Barcelona", "Málaga"]

# The notebook suffixes three flags with `_ml` so they coexist with the
# segmented parquet's pre-existing has_crib / has_balcony / has_private_entrance
# columns (both the raw and `_ml` versions end up in the feature set). The regex
# bodies are identical to the shared patterns, so we just rename those keys.
_ML_RENAME = {
    "has_crib": "has_crib_ml",
    "has_private_entrance": "has_private_entrance_ml",
    "has_balcony": "has_balcony_ml",
}
NOTEBOOK_AMENITY_PATTERNS = {
    _ML_RENAME.get(k, k): v for k, v in AMENITY_PATTERNS.items()
}

# Leakage / metadata columns dropped before modelling (notebook §2.4).
DROP_COLS = [
    "id", "scrape_id", "host_id", "source",
    "name", "description", "neighborhood_overview", "picture_url",
    "host_name", "host_about", "host_location", "host_verifications",
    "amenities", "bathrooms_description", "license",
    "last_scraped", "first_review", "last_review", "calendar_last_scraped",
    "host_since", "property_type", "price_cat",
    "estimated_revenue_l365d", "estimated_occupancy_l365d",
    "neighbourhood_group_cleansed", "has_availability", "availability_eoy",
    "minimum_minimum_nights", "maximum_minimum_nights",
    "minimum_maximum_nights", "maximum_maximum_nights", "maximum_nights_avg_ntm",
    "availability_30", "availability_60", "availability_90", "availability_365",
    "days_since_first_review", "days_since_last_review", "review_span_years",
    "review_scores_rating", "review_scores_accuracy", "review_scores_cleanliness",
    "review_scores_checkin", "review_scores_communication",
    "review_scores_location", "review_scores_value",
    "Cluster_Final",
    "log1p_accommodates", "log1p_bedrooms", "log1p_beds",
    "log1p_bathrooms_number", "log1p_minimum_nights",
    "log1p_calculated_host_listings_count", "log1p_amenity_count",
    "n_outlier_cols",
    "host_acceptance_rate_ord", "host_response_rate_ord",
    "host_response_rate_cat", "host_acceptance_rate_cat",
]

CAT_COLS = ["city", "room_type", "property_type_std"]
RESPONSE_TIME_ORD = {
    "within an hour": 0, "within a few hours": 1,
    "within a day": 2, "a few days or more": 3, "Unknown": 2,
}
CITY_DROP_COLS = ["city", "is_madrid", "is_barcelona"]

# LightGBM hyper-parameter grid (notebook §5, LightGBM cell).
LGB_PARAM_GRID = {
    "n_estimators": [200, 400],
    "learning_rate": [0.05, 0.1],
    "num_leaves": [31, 63],
    "min_child_samples": [10, 30],
}


# ── 1. Build the modelling frame ─────────────────────────────────────────────

def build_frame() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found — generate the segmented listings parquet first."
        )
    df = pd.read_parquet(DATA_PATH)
    log.info("Loaded %s: %d rows", DATA_PATH.name, len(df))

    # Target: log1p(price), drop missing, clip the top 0.5%.
    df = df[df["price"].notna()].copy()
    df[TARGET] = np.log1p(df["price"])
    p995 = df[TARGET].quantile(0.995)
    n_clipped = int((df[TARGET] > p995).sum())
    df = df[df[TARGET] <= p995].copy().reset_index(drop=True)
    log.info("After price clip: %d rows (%d above p99.5 removed)", len(df), n_clipped)

    # Amenity flags (22, with the three `_ml` keys), booking flags, density.
    df = add_amenity_flags(df, patterns=NOTEBOOK_AMENITY_PATTERNS)
    df["is_long_stay"] = (df["minimum_nights"] >= 28).astype(np.int8)
    df["is_weekly"] = ((df["minimum_nights"] >= 7) & (df["minimum_nights"] < 28)).astype(np.int8)
    df["has_licence"] = df["license"].notna().astype(np.int8)
    df = add_competitive_density(df)
    return df


# ── 2. Encode features (notebook §2.4–2.5) ───────────────────────────────────

def encode(df: pd.DataFrame):
    feature_cols = [
        c for c in df.columns
        if c not in DROP_COLS
        and c not in [TARGET, "price", "Segment_Name", "neighbourhood_cleansed"]
    ]
    df_model = df[feature_cols + [TARGET, "Segment_Name"]].copy()

    for col in df_model.select_dtypes(include="bool").columns:
        df_model[col] = df_model[col].astype(np.int8)
    for col in df_model.select_dtypes(include="category").columns:
        df_model[col] = df_model[col].cat.codes.astype(np.int8)

    cat_encoders: dict[str, dict] = {}
    for col in [c for c in CAT_COLS if c in feature_cols]:
        if df_model[col].dtype == object:
            df_model[col] = df_model[col].astype(str).replace({"nan": None, "None": None})
            cats = sorted(df_model[col].dropna().unique())
            cat_encoders[col] = {c: i for i, c in enumerate(cats)}
            df_model[col] = df_model[col].map(cat_encoders[col])

    if "host_response_time" in df_model.columns:
        df_model["host_response_time"] = (
            df_model["host_response_time"].astype(str)
            .map(RESPONSE_TIME_ORD).fillna(2).astype(np.int8)
        )

    # Neighbourhood target-encoding: mean log-price per neighbourhood.
    neigh_target_map = df.groupby("neighbourhood_cleansed")[TARGET].mean()
    df_model["neighbourhood_target_enc"] = df["neighbourhood_cleansed"].map(neigh_target_map)

    for col in df_model.select_dtypes(include="object").columns:
        if col not in [TARGET, "Segment_Name"]:
            cats = sorted(df_model[col].dropna().unique())
            df_model[col] = df_model[col].map({c: i for i, c in enumerate(cats)})

    num_cols = [c for c in df_model.select_dtypes(include=[np.number]).columns if c != TARGET]
    df_model[num_cols] = df_model[num_cols].fillna(df_model[num_cols].median())

    return df_model, cat_encoders, neigh_target_map


# ── 3. RFE + LightGBM hyper-params + by-city training (notebook §3, §5, §7) ───

def train(df: pd.DataFrame, df_model: pd.DataFrame) -> dict:
    from lightgbm import LGBMRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import GridSearchCV, train_test_split

    all_features = [c for c in df_model.columns if c not in [TARGET, "Segment_Name"]]
    X_all = df_model[all_features]
    y_all = df_model[TARGET].to_numpy()
    city_all = df.loc[df_model.index, "city"].astype(str).to_numpy()

    X_train, X_test, y_train, y_test, city_train, city_test = train_test_split(
        X_all, y_all, city_all, test_size=0.20, random_state=SEED, stratify=city_all,
    )
    log.info("Train: %d  |  Test: %d", len(X_train), len(X_test))

    # RFE: keep features with RandomForest importance >= median.
    rf = RandomForestRegressor(
        n_estimators=100, max_depth=12, min_samples_leaf=10, random_state=SEED, n_jobs=-1,
    ).fit(X_train, y_train)
    importances = pd.Series(rf.feature_importances_, index=all_features)
    selected = importances[importances >= importances.median()].sort_values(ascending=False).index.tolist()
    log.info("RFE: %d → %d features", len(all_features), len(selected))

    # LightGBM hyper-parameter search on the selected features.
    lgb_cv = GridSearchCV(
        LGBMRegressor(random_state=SEED, n_jobs=-1, verbose=-1),
        LGB_PARAM_GRID, cv=5, scoring="r2", n_jobs=-1, refit=True,
    ).fit(X_train[selected], y_train)
    log.info("LightGBM best params: %s  (CV R²=%.4f)", lgb_cv.best_params_, lgb_cv.best_score_)

    # By-city: one LightGBM per city on the city-feature subset.
    city_features = [c for c in selected if c not in CITY_DROP_COLS]
    city_models: dict[str, object] = {}
    y_te_all, y_pred_all = [], []
    log.info("=== By-city LightGBM ===")
    for city in CITIES:
        tr, te = city_train == city, city_test == city
        model = LGBMRegressor(**lgb_cv.best_params_, random_state=SEED, n_jobs=-1, verbose=-1)
        model.fit(X_train.loc[tr, city_features], y_train[tr])
        city_models[city] = model
        y_pred = model.predict(X_test.loc[te, city_features])
        r2 = r2_score(y_test[te], y_pred)
        mae_eur = mean_absolute_error(np.expm1(y_test[te]), np.expm1(y_pred))
        log.info("  %-10s n_test=%4d  R²=%.4f  MAE=€%.1f", city, int(te.sum()), r2, mae_eur)
        y_te_all.append(y_test[te])
        y_pred_all.append(y_pred)

    combined_r2 = r2_score(np.concatenate(y_te_all), np.concatenate(y_pred_all))
    combined_rmse = np.sqrt(mean_squared_error(
        np.expm1(np.concatenate(y_te_all)), np.expm1(np.concatenate(y_pred_all))))
    log.info("Combined test: R²=%.4f  RMSE=€%.1f", combined_r2, combined_rmse)

    return {"city_models": city_models, "city_features": city_features}


# ── 4. Save serving artefacts (notebook §7) ──────────────────────────────────

def _backup_legacy() -> None:
    files = [ARTEFACTS_JSON, ENCODERS_PKL] + [
        MODELS / f"price_city_{_slug(c)}_model.pkl" for c in CITIES
    ]
    for f in files:
        legacy = f.with_name(f.stem + "_legacy" + f.suffix)
        if f.exists() and not legacy.exists():
            shutil.copy2(f, legacy)
            log.info("Backed up %s → %s", f.name, legacy.name)


def _slug(city: str) -> str:
    return city.lower().replace("á", "a")


def save_artefacts(sel: dict, cat_encoders: dict, neigh_target_map: pd.Series) -> None:
    _backup_legacy()
    MODELS.mkdir(exist_ok=True)

    city_model_files = {}
    for city, model in sel["city_models"].items():
        fname = f"price_city_{_slug(city)}_model.pkl"
        joblib.dump(model, MODELS / fname)
        city_model_files[city] = fname
    log.info("Saved %d city models", len(city_model_files))

    ARTEFACTS_JSON.write_text(json.dumps({
        "city_features": sel["city_features"],
        "best_model": "LightGBM",
        "city_model_files": city_model_files,
    }, ensure_ascii=False, indent=2))

    city_encoders = {
        "neighbourhood_cleansed": neigh_target_map.to_dict(),
        **{k: v for k, v in cat_encoders.items() if k in ("property_type_std", "host_response_time")},
    }
    joblib.dump(city_encoders, ENCODERS_PKL)
    log.info("Saved %s (%d neighbourhoods) and %s",
             ENCODERS_PKL.name, len(neigh_target_map), ARTEFACTS_JSON.name)


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    df = build_frame()
    df_model, cat_encoders, neigh_target_map = encode(df)
    sel = train(df, df_model)
    save_artefacts(sel, cat_encoders, neigh_target_map)
    log.info("Done — production price artefacts written to %s", MODELS)


if __name__ == "__main__":
    main()
