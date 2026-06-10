"""Rental-revenue model: interpretable baseline (Ridge) + gradient boosting (XGBoost).

Target: gross monthly rent (EUR), modelled in log space (log1p) because rent is
right-skewed. Annual revenue = monthly x 12 is reported alongside.

Location is the key signal: high-cardinality `neighborhood` / `municipality` are
target-encoded with sklearn's cross-fitted TargetEncoder (leakage-safe); low-card
`city` / `property_type` are one-hot encoded.

Usage:
    python src/model.py                      # train, evaluate, save artifacts
    from model import predict_rent
    predict_rent(city="Madrid", sq_m=80, n_rooms=2,
                 property_type="flat", neighborhood="Trafalgar")
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder
from xgboost import XGBRegressor

from features import build_model_frame

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

NUMERIC = ["sq_m", "n_rooms"]
LOWCARD = ["city", "property_type"]
HIGHCARD = ["neighborhood", "municipality"]
FEATURES = NUMERIC + LOWCARD + HIGHCARD
TARGET = "rent_eur_month"
SEED = 42


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC),
            ("low", OneHotEncoder(handle_unknown="ignore"), LOWCARD),
            ("high", TargetEncoder(smooth="auto"), HIGHCARD),
        ]
    )


def _wrap_log(estimator) -> TransformedTargetRegressor:
    """Model log1p(rent); predictions auto back-transformed to EUR."""
    return TransformedTargetRegressor(
        regressor=estimator, func=np.log1p, inverse_func=np.expm1
    )


def get_models() -> dict[str, Pipeline]:
    ridge = Pipeline(
        [("prep", make_preprocessor()), ("reg", _wrap_log(Ridge(alpha=1.0)))]
    )
    xgb = Pipeline(
        [
            ("prep", make_preprocessor()),
            (
                "reg",
                _wrap_log(
                    XGBRegressor(
                        n_estimators=600,
                        learning_rate=0.03,
                        max_depth=4,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        min_child_weight=3,
                        reg_lambda=1.0,
                        random_state=SEED,
                        n_jobs=-1,
                    )
                ),
            ),
        ]
    )
    return {"ridge": ridge, "xgboost": xgb}


def _metrics(y_true, y_pred) -> dict:
    return {
        "MAE_eur": round(mean_absolute_error(y_true, y_pred), 1),
        "RMSE_eur": round(root_mean_squared_error(y_true, y_pred), 1),
        "MAPE_pct": round(100 * mean_absolute_percentage_error(y_true, y_pred), 1),
        "R2": round(r2_score(y_true, y_pred), 3),
    }


def evaluate(model, X_test, y_test) -> dict:
    pred = model.predict(X_test)
    out = {"overall": _metrics(y_test, pred)}
    for city in X_test["city"].unique():
        m = X_test["city"] == city
        out[city] = _metrics(y_test[m], pred[m])
    return out


def train_and_evaluate(scope: str = "city", verbose: bool = True) -> dict:
    df = build_model_frame(scope=scope, verbose=verbose)
    X, y = df[FEATURES], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=X["city"]
    )

    results = {}
    fitted = {}
    for name, model in get_models().items():
        cv = cross_val_score(
            model, X_tr, y_tr, cv=5,
            scoring="neg_mean_absolute_error",
        )
        model.fit(X_tr, y_tr)
        results[name] = {
            "cv_MAE_eur": round(-cv.mean(), 1),
            "cv_MAE_std": round(cv.std(), 1),
            "test": evaluate(model, X_te, y_te),
        }
        fitted[name] = model
        if verbose:
            print(f"\n=== {name} ===")
            print(f"  CV MAE: {results[name]['cv_MAE_eur']} ± {results[name]['cv_MAE_std']}")
            print(f"  test  : {results[name]['test']['overall']}")

    # pick best by test overall MAE
    best = min(results, key=lambda n: results[n]["test"]["overall"]["MAE_eur"])
    best_model = fitted[best]

    # permutation importance (model-agnostic explainability) on the best model
    perm = permutation_importance(
        best_model, X_te, y_te, n_repeats=10, random_state=SEED,
        scoring="neg_mean_absolute_error",
    )
    importance = (
        pd.Series(perm.importances_mean, index=FEATURES)
        .sort_values(ascending=False)
        .round(2)
        .to_dict()
    )
    results["best_model"] = best
    results["permutation_importance"] = importance

    # persist artifacts
    MODELS.mkdir(exist_ok=True)
    joblib.dump(best_model, MODELS / "rent_model.pkl")
    (REPORTS).mkdir(exist_ok=True)
    with open(REPORTS / "model_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    if verbose:
        print(f"\nBest model: {best}")
        print("Top drivers (permutation importance, EUR MAE impact):")
        for k, v in importance.items():
            print(f"  {k:14s} {v}")
        print(f"\nSaved -> {MODELS/'rent_model.pkl'}")
    return results


def predict_rent(
    city: str,
    sq_m: float,
    n_rooms: float,
    property_type: str = "flat",
    neighborhood: str = "unknown",
    municipality: str | None = None,
) -> dict:
    """Predict monthly (and annual) gross rent for a single property."""
    model = joblib.load(MODELS / "rent_model.pkl")
    row = pd.DataFrame(
        [{
            "sq_m": sq_m,
            "n_rooms": n_rooms,
            "city": city,
            "property_type": property_type,
            "neighborhood": neighborhood,
            "municipality": municipality or city,
        }]
    )[FEATURES]
    monthly = float(model.predict(row)[0])
    return {
        "monthly_eur": round(monthly),
        "annual_eur": round(monthly * 12),
    }


def evaluate_against_live(live_df: pd.DataFrame) -> dict:
    """Score the trained (2022) model against fresh listings to quantify drift.

    `live_df` must contain the model FEATURES plus the true `rent_eur_month`
    (e.g. from a future idealista API pull or scrape). Returns headline metrics
    and the mean signed error (positive => current rents exceed 2022 predictions),
    which is a direct read on how stale the 2022 training data has become.
    """
    model = joblib.load(MODELS / "rent_model.pkl")
    pred = model.predict(live_df[FEATURES])
    true = live_df[TARGET].to_numpy()
    return {
        **_metrics(true, pred),
        "mean_signed_error_eur": round(float((true - pred).mean()), 1),
        "n": len(live_df),
    }


if __name__ == "__main__":
    train_and_evaluate(scope="city")
