"""Nightly-price model inference service.

Loads the trained LightGBM nightly-price model plus its preprocessing
artefacts (``models/price_*.pkl`` / ``price_feature_cols.json``) and exposes a
single :meth:`PricePredictor.predict` entry point that turns a *partial*
property spec into a predicted Airbnb nightly price in EUR.

This deliberately lives in the package (not in ``api/``) so the FastAPI
endpoints **and** the agent layer can both call the same service — the
architecture's "agents call services, not notebooks" rule.

Preprocessing contract
=======================
Replicated verbatim from ``notebooks/price_ml_model.ipynb`` so the served
predictions match the offline model:

1. Categorical integer-encoding for ``property_type_std`` and
   ``host_response_time`` via the saved ``price_cat_encoders.pkl`` maps.
2. Neighbourhood **target-encoding**: ``neighbourhood_target_enc`` is produced
   by mapping ``neighbourhood_cleansed`` (a name like "Salamanca") through the
   saved encoder; unknown neighbourhoods fall back to the global mean.
3. Missing numeric features are imputed with the column **median**, derived
   once at load time from the processed listings parquet.
4. Features are ordered to ``selected_features`` and passed **unscaled** to
   ``model.predict``. The production LightGBM was trained on unscaled
   features; the saved ``StandardScaler`` applied only to the linear baseline
   models, so applying it here would flatten predictions to a near-constant.
5. The model targets ``log1p(price)``; predictions are back-transformed with
   ``np.expm1`` to EUR/night.
"""

from __future__ import annotations

import json
import logging
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import numpy as np

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_DIR = ROOT / "models"
DEFAULT_LISTINGS = ROOT / "Data" / "processed" / "listings_with_price_hat.parquet"

# Features (within selected_features) that carry an integer category map in
# price_cat_encoders.pkl. city / room_type are encoded too but were dropped by
# RFE, so they never reach the scaler.
INT_ENCODED: tuple[str, ...] = ("property_type_std", "host_response_time")

# Target-encoded neighbourhood: the model feature vs. its raw source column.
NEIGHBOURHOOD_FEATURE = "neighbourhood_target_enc"
NEIGHBOURHOOD_SOURCE = "neighbourhood_cleansed"


class PricePredictor:
    """Loads the price model + artefacts and predicts EUR/night."""

    def __init__(
        self,
        model_dir: Path | str = DEFAULT_MODEL_DIR,
        listings_path: Path | str = DEFAULT_LISTINGS,
    ) -> None:
        import joblib

        model_dir = Path(model_dir)
        # The artefacts were pickled with an older scikit-learn; the version
        # banner is noisy but harmless. The StandardScaler is intentionally NOT
        # loaded — the LightGBM model was trained unscaled (see module docstring).
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = joblib.load(model_dir / "price_best_model.pkl")
            self.encoders: dict[str, dict] = joblib.load(
                model_dir / "price_cat_encoders.pkl"
            )

        cols = json.loads((model_dir / "price_feature_cols.json").read_text())
        self.features: list[str] = list(cols["selected_features"])
        self.best_model: str = cols.get("best_model", type(self.model).__name__)

        self._nbh_map: dict[str, float] = self.encoders.get(NEIGHBOURHOOD_SOURCE, {})
        self._nbh_default = (
            float(np.mean(list(self._nbh_map.values()))) if self._nbh_map else 0.0
        )
        self.medians: dict[str, float] = self._compute_medians(Path(listings_path))
        self._explainer: Any = None  # lazy shap.TreeExplainer, built on first explain()
        logger.info(
            "PricePredictor ready: %s, %d features, medians from %s",
            self.best_model, len(self.features),
            "parquet" if self.medians else "fallback (0.0)",
        )

    # ── public API ────────────────────────────────────────────────────────────

    def predict(self, prop: Mapping[str, Any]) -> float:
        """Predict the nightly price (EUR) for one property spec.

        Unknown / missing fields are imputed with training-set medians, so a
        minimal spec (e.g. just ``city`` + ``accommodates``) still returns a
        number — accuracy improves as more fields are supplied.
        """
        ordered = self._prepare_ordered_frame(prop)
        y_log = float(self.model.predict(ordered)[0])
        return float(np.expm1(y_log))

    def explain(self, prop: Mapping[str, Any], top_n: int = 6) -> list[dict[str, Any]]:
        """Per-prediction SHAP feature attribution — the agent layer's "why".

        Returns the ``top_n`` features by |SHAP value| (descending), each as
        ``{"feature", "shap_value", "direction"}``. SHAP values are in the
        model's own output space (log1p(price)), so they describe each
        feature's *relative* push on price, not a EUR amount — narrate them
        qualitatively (e.g. "neighbourhood is the strongest positive driver").
        """
        import shap

        ordered = self._prepare_ordered_frame(prop)
        if self._explainer is None:
            self._explainer = shap.TreeExplainer(self.model)
        shap_values = np.asarray(self._explainer.shap_values(ordered))[0]

        ranked = sorted(
            zip(self.features, shap_values.tolist()),
            key=lambda kv: -abs(kv[1]),
        )[:top_n]
        return [
            {
                "feature": feature,
                "shap_value": round(value, 4),
                "direction": "increases" if value > 0 else "decreases",
            }
            for feature, value in ranked
        ]

    # ── internals ───────────────────────────────────────────────────────────

    def _prepare_ordered_frame(self, prop: Mapping[str, Any]):
        """Raw spec → ordered, median-filled, UNSCALED feature frame.

        Shared by :meth:`predict` and :meth:`explain` so both build the
        identical feature vector. Named columns (not a bare ndarray) so
        LightGBM doesn't warn about missing feature names.
        """
        import pandas as pd

        feats = self._build_features_frame(pd.DataFrame([dict(prop)]))
        for col in self.features:
            fallback = self._nbh_default if col == NEIGHBOURHOOD_FEATURE else 0.0
            feats[col] = feats[col].fillna(self.medians.get(col, fallback))
        return feats[self.features]

    def _build_features_frame(self, df):
        """Map a raw-spec frame into the 29 encoded model columns (NaN-filled later)."""
        import pandas as pd

        out = pd.DataFrame(index=df.index)
        for feat in self.features:
            if feat == NEIGHBOURHOOD_FEATURE:
                out[feat] = (
                    df[NEIGHBOURHOOD_SOURCE].map(self._nbh_map)
                    if NEIGHBOURHOOD_SOURCE in df.columns
                    else np.nan
                )
            elif feat in INT_ENCODED:
                enc = self.encoders.get(feat, {})
                out[feat] = df[feat].map(enc) if feat in df.columns else np.nan
            elif feat in df.columns:
                series = df[feat]
                if series.dtype == bool:
                    series = series.astype(int)
                out[feat] = pd.to_numeric(series, errors="coerce")
            else:
                out[feat] = np.nan
        return out

    def _compute_medians(self, path: Path) -> dict[str, float]:
        import pandas as pd

        if not path.exists():
            logger.warning(
                "Listings parquet not found at %s — numeric medians default to "
                "0.0. Predictions remain valid but less accurate for sparse specs.",
                path,
            )
            return {}
        source_cols = {
            f for f in self.features if f != NEIGHBOURHOOD_FEATURE
        } | {NEIGHBOURHOOD_SOURCE}
        df = pd.read_parquet(path)
        present = [c for c in source_cols if c in df.columns]
        feats = self._build_features_frame(df[present])
        return feats.median(numeric_only=True).to_dict()


@lru_cache(maxsize=1)
def get_price_predictor() -> PricePredictor:
    """Process-wide singleton — loads artefacts once."""
    return PricePredictor()
