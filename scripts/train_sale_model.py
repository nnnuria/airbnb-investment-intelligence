"""Train (or retrain) the sell-price model — CLI entry point.

Mirrors the modelling logic documented in ``notebooks/sell_price_model.ipynb``
(the notebook stays the narrated reference; this script is the reproducible,
headless retrain). Compared to the original pipeline it adds an **amenity binary
feature group** mined from listing text (see
:mod:`airbnb_iip.features.sale_amenities`) so the model learns an amenity →
resale-value relationship.

Flow:
    load_clean_sale() (now carries amenity flags) → trim tails → preprocess
    (numeric / one-hot / target-enc / binary-passthrough) → compare
    Ridge / RandomForest / LightGBM / XGBoost → save best Pipeline + metadata.

Safety: the existing ``sale_best_model.pkl`` / ``sale_feature_cols.json`` are
copied to ``*_legacy.*`` (once — never overwritten) before the new artifacts are
written, so a revert is just restoring those two files.

Run:
    python scripts/train_sale_model.py
"""

from __future__ import annotations

import json
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from airbnb_iip.data.idealista_sale import load_clean_sale          # noqa: E402
from airbnb_iip.features.sale_amenities import SALE_AMENITY_COLUMNS  # noqa: E402

MODELS = ROOT / "models"
MODEL_PKL = MODELS / "sale_best_model.pkl"
META_JSON = MODELS / "sale_feature_cols.json"
SEED = 42

# Feature groups
NUM = ["size_m2", "rooms", "bathrooms", "floor_num", "latitude", "longitude"]
LOWCARD = ["property_type", "status", "city", "has_lift", "exterior", "new_development"]
HIGHCARD = ["district", "neighborhood"]          # target-encoded (cross-fitted)
BINARY = list(SALE_AMENITY_COLUMNS)              # text-mined amenity flags {0,1}
FEATURES = NUM + LOWCARD + HIGHCARD + BINARY
TARGET = "y"


def build_frame() -> pd.DataFrame:
    df = load_clean_sale()

    # Gentle target trim: drop the extreme 0.5% tails so a few mansions don't dominate.
    lo, hi = df["price"].quantile([0.005, 0.995])
    df = df[df["price"].between(lo, hi)].copy()

    # Nullable booleans -> labelled strings (own category for missing).
    for c in ["has_lift", "exterior", "new_development"]:
        df[c] = df[c].astype("object").where(df[c].notna(), "unknown").map(
            {True: "yes", False: "no", "unknown": "unknown"}
        )
    for c in ["district", "neighborhood", "status"]:
        df[c] = df[c].fillna("unknown")

    # Amenity flags: guarantee clean int {0,1} (loader yields int8 already).
    for c in BINARY:
        df[c] = df[c].fillna(0).astype(int)

    df["y"] = np.log1p(df["price"])
    print(f"model frame: {len(df):,} rows  (price EUR{lo:,.0f}-{hi:,.0f})")
    return df


def make_preprocessor(scale_numeric: bool):
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num_steps), NUM),
        ("low", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), LOWCARD),
        ("high", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("te", TargetEncoder(target_type="continuous", random_state=SEED)),
        ]), HIGHCARD),
        # Binary amenity flags: a missing/un-toggled flag is "not advertised" → 0,
        # NOT a median (which would leak ~10% premium into blank inputs).
        ("bin", SimpleImputer(strategy="constant", fill_value=0), BINARY),
    ])


def eur_metrics(yl_true, yl_pred) -> dict:
    from sklearn.metrics import (mean_absolute_error, root_mean_squared_error,
                                 r2_score, mean_absolute_percentage_error)
    yt, yp = np.expm1(yl_true), np.expm1(yl_pred)
    return {
        "MAE_eur":  round(mean_absolute_error(yt, yp)),
        "RMSE_eur": round(root_mean_squared_error(yt, yp)),
        "R2":       round(r2_score(yt, yp), 3),
        "MAPE_%":   round(100 * mean_absolute_percentage_error(yt, yp), 1),
    }


def train(df: pd.DataFrame):
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor

    X, y = df[FEATURES], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=df["city"]
    )
    print(f"train {len(X_tr):,} | test {len(X_te):,}")

    def build(model, scale):
        return Pipeline([("prep", make_preprocessor(scale)), ("model", model)])

    models = {
        "Ridge":        build(Ridge(alpha=1.0), scale=True),
        "RandomForest": build(RandomForestRegressor(n_estimators=300, max_depth=18,
                              min_samples_leaf=5, n_jobs=-1, random_state=SEED), scale=False),
        "LightGBM":     build(LGBMRegressor(n_estimators=800, learning_rate=0.05, num_leaves=63,
                              subsample=0.8, colsample_bytree=0.8, random_state=SEED, verbose=-1), scale=False),
        "XGBoost":      build(XGBRegressor(n_estimators=800, learning_rate=0.05, max_depth=6,
                              subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=-1), scale=False),
    }

    rows, fitted = [], {}
    for name, pipe in models.items():
        cv = -cross_val_score(pipe, X_tr, y_tr, cv=3, scoring="neg_mean_absolute_error")
        pipe.fit(X_tr, y_tr)
        fitted[name] = pipe
        m = eur_metrics(y_te, pipe.predict(X_te))
        m.update(model=name, cv_MAE_log=round(cv.mean(), 3))
        rows.append(m)
        print(f"{name:13} test MAE EUR{m['MAE_eur']:>8,} | MAPE {m['MAPE_%']:>4}% | R2 {m['R2']}")

    results = pd.DataFrame(rows).set_index("model").sort_values("MAE_eur")
    best_name = results.index[0]
    return best_name, fitted[best_name], (X_tr, X_te, y_tr, y_te)


def counterfactual_check(model) -> None:
    """Toggle each amenity on a fixed reference flat; report the EUR uplift."""
    base = {c: None for c in FEATURES}
    base.update(city="madrid", district="Chamberí", neighborhood="unknown",
                size_m2=90, rooms=3, bathrooms=2, floor_num=3,
                property_type="flat", status="good",
                has_lift="yes", exterior="yes", new_development="no")
    for c in BINARY:
        base[c] = 0

    def pred(spec):
        return float(np.expm1(model.predict(pd.DataFrame([spec]))[0]))

    p0 = pred(base)
    print(f"\ncounterfactual (Madrid / Chamberí / 90m² flat) base = EUR{p0:,.0f}")
    for c in BINARY:
        spec = dict(base); spec[c] = 1
        d = pred(spec) - p0
        print(f"  +{c:14s} {d:>+11,.0f} EUR  ({d / p0:+.1%})")


def backup_legacy() -> None:
    for src, dst in [(MODEL_PKL, MODELS / "sale_best_model_legacy.pkl"),
                     (META_JSON, MODELS / "sale_feature_cols_legacy.json")]:
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            print(f"backed up {src.name} -> {dst.name}")


def main() -> None:
    import joblib

    df = build_frame()
    best_name, best, (X_tr, X_te, y_tr, y_te) = train(df)
    print(f"\nbest model: {best_name}")
    counterfactual_check(best)

    backup_legacy()
    joblib.dump(best, MODEL_PKL)
    meta = {
        "best_model": best_name,
        "target": "log1p(price)",
        "input_features": FEATURES,
        "num": NUM, "lowcard_onehot": LOWCARD, "highcard_target_enc": HIGHCARD,
        "binary_amenities": BINARY,
        "excluded_leakage": ["price_per_m2"],
        "amenity_source": "text-mined from description + suggestedTexts.title",
        "test_metrics": eur_metrics(y_te, best.predict(X_te)),
        "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
    }
    META_JSON.write_text(json.dumps(meta, indent=2))
    print("\nsaved:")
    print(f"  {MODEL_PKL.relative_to(ROOT)}")
    print(f"  {META_JSON.relative_to(ROOT)}")
    print(json.dumps(meta["test_metrics"], indent=2))


if __name__ == "__main__":
    main()
