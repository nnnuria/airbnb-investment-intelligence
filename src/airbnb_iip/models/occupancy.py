"""Calendar-based occupancy model — inference service.

The **production occupancy model**. Replaces the legacy San Francisco
review-based estimator (which inferred bookings from review velocity and
barely tracked real availability — corr ≈ 0.13 with calendar occupancy).
This model is a LightGBM regressor whose target is computed directly from
Inside Airbnb **calendar** data: a night counts as booked when
``available == 'f'``, and a listing's occupancy rate is the share of booked
nights across its forward calendar window.

Trained and saved by ``scripts/train_occupancy_model.py``; see
``docs/model_cards/occupancy_model.md``.

Artefacts required in ``model_dir``
===================================
* ``occupancy_feature_cols.json`` — feature list, routing, metrics, clip range.
* ``occupancy_encoders.pkl``      — neighbourhood target-encoding map (mean
  occupancy), label-encoder maps (room_type / property_type_std / city),
  per-feature training medians, and the neighbourhood fallback value.
* ``occupancy_best_model.pkl``    — the global model (``approach == "global"``), or
* ``occupancy_city_{city}_model.pkl`` — one per city (``approach == "by_city"``).

Preprocessing contract (mirrors :class:`airbnb_iip.models.price.CityPricePredictor`)
-----------------------------------------------------------------------------------
1. ``neighbourhood_cleansed`` → ``neighbourhood_target_enc`` via the saved map;
   unknown neighbourhoods fall back to the global mean occupancy.
2. ``room_type`` / ``property_type_std`` / ``city`` label-encoded via saved maps;
   unknown category → ``-1``.
3. ``amenity_count`` derived from the 22 amenity flags when not supplied.
4. Missing numeric features imputed with training-set medians.
5. Features ordered to the artefact list and passed **unscaled** to LightGBM.
6. The raw prediction is clipped to the artefact's ``occupancy_clip`` range
   (default ``[0.02, 0.95]``) — a real operation is never 0% or 100% booked
   year-round.
"""

from __future__ import annotations

import json
import logging
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from airbnb_iip.features.amenities import AMENITY_COLUMNS

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_DIR = ROOT / "models"

_ARTEFACTS_NAME = "occupancy_feature_cols.json"
_ENCODERS_NAME = "occupancy_encoders.pkl"

# Sane fallback if artefacts are missing — the active-listing mean from training.
_FALLBACK_OCCUPANCY = 0.40


class OccupancyPredictor:
    """Loads the calendar-occupancy model + artefacts and predicts a rate in [0, 1]."""

    def __init__(self, model_dir: Path | str = DEFAULT_MODEL_DIR) -> None:
        import joblib

        model_dir = Path(model_dir)
        artefacts_path = model_dir / _ARTEFACTS_NAME
        encoders_path = model_dir / _ENCODERS_NAME

        self.available: bool = False
        if not artefacts_path.exists() or not encoders_path.exists():
            logger.warning(
                "Occupancy artefacts not found in %s — predictions fall back to "
                "%.0f%%. Run scripts/train_occupancy_model.py to generate them.",
                model_dir, _FALLBACK_OCCUPANCY * 100,
            )
            return

        art = json.loads(artefacts_path.read_text())
        self.approach: str = art.get("approach", "global")
        self.features: list[str] = art["features"]
        self.best_model: str = art.get("model", "LightGBM")
        self.neighbourhood_feature: str = art.get(
            "neighbourhood_feature", "neighbourhood_target_enc"
        )
        self.neighbourhood_source: str = art.get(
            "neighbourhood_source", "neighbourhood_cleansed"
        )
        self.label_encoded: list[str] = art.get("label_encoded", [])
        clip = art.get("occupancy_clip", [0.02, 0.95])
        self.clip_lo, self.clip_hi = float(clip[0]), float(clip[1])
        self.metrics: dict = art.get("metrics", {})

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            enc: dict = joblib.load(encoders_path)
            self._neigh_map: dict[str, float] = enc.get(self.neighbourhood_source, {})
            self._neigh_default: float = float(
                enc.get("_neighbourhood_default", _FALLBACK_OCCUPANCY)
            )
            self._medians: dict[str, float] = enc.get("_medians", {})
            self._label_maps: dict[str, dict] = {
                c: enc[c] for c in self.label_encoded if c in enc
            }

            self._models: dict[str, Any] = {}
            self._global_model: Any = None
            if self.approach == "global":
                self._global_model = joblib.load(model_dir / art["model_file"])
            else:
                for city, fname in art.get("city_model_files", {}).items():
                    path = model_dir / fname
                    if path.exists():
                        self._models[city] = joblib.load(path)
                    else:
                        logger.warning("Occupancy city model file not found: %s", path)

        self.available = True
        self._explainer: Any = None
        logger.info(
            "OccupancyPredictor ready: %s (%s), %d features, R²=%.3f",
            self.best_model, self.approach, len(self.features),
            self.metrics.get("chosen", {}).get("r2", float("nan")),
        )

    # ── public API ────────────────────────────────────────────────────────────

    def predict(self, prop: Mapping[str, Any]) -> float:
        """Predict the annual occupancy rate (fraction in [0, 1]) for one property.

        Unknown / missing fields are imputed with training medians, so a minimal
        spec still returns a number. The result is clipped to the artefact's
        plausibility range.
        """
        if not self.available:
            return _FALLBACK_OCCUPANCY
        model = self._model_for(prop)
        ordered = self._prepare_ordered_frame(prop)
        rate = float(model.predict(ordered)[0])
        return float(np.clip(rate, self.clip_lo, self.clip_hi))

    def predict_nights(self, prop: Mapping[str, Any], *, days: int = 365) -> int:
        """Convenience: occupancy rate × ``days``, rounded to whole nights."""
        return int(round(self.predict(prop) * days))

    def explain(self, prop: Mapping[str, Any], top_n: int = 6) -> list[dict[str, Any]]:
        """Per-prediction SHAP attribution (the agent layer's "why")."""
        if not self.available:
            return []
        import shap

        model = self._model_for(prop)
        ordered = self._prepare_ordered_frame(prop)
        explainer = shap.TreeExplainer(model)
        shap_values = np.asarray(explainer.shap_values(ordered))[0]
        ranked = sorted(
            zip(self.features, shap_values.tolist()), key=lambda kv: -abs(kv[1])
        )[:top_n]
        return [
            {
                "feature": f,
                "shap_value": round(v, 4),
                "direction": "increases" if v > 0 else "decreases",
            }
            for f, v in ranked
        ]

    # ── internals ───────────────────────────────────────────────────────────

    def _model_for(self, prop: Mapping[str, Any]):
        if self.approach == "global":
            return self._global_model
        city = str(prop.get("city", ""))
        model = self._models.get(city) or self._models.get(
            next((k for k in self._models if k.lower() == city.lower()), ""), None
        )
        if model is None:
            # Accent-insensitive match (e.g. "malaga" → "Málaga").
            import unicodedata

            def _slug(s: str) -> str:
                return unicodedata.normalize("NFKD", str(s)).encode(
                    "ascii", "ignore"
                ).decode().lower()

            city_slug = _slug(city)
            model = next(
                (m for k, m in self._models.items() if _slug(k) == city_slug), None
            )
        if model is None:
            logger.warning(
                "No occupancy model for city '%s' — using first available.", city
            )
            model = next(iter(self._models.values()))
        return model

    def _prepare_ordered_frame(self, prop: Mapping[str, Any]):
        import pandas as pd

        feats = self._build_features_frame(pd.DataFrame([dict(prop)]))
        for col in self.features:
            if col == self.neighbourhood_feature:
                fallback = self._neigh_default
            elif col in self._label_maps:
                fallback = -1.0
            else:
                fallback = self._medians.get(col, 0.0)
            feats[col] = feats[col].fillna(fallback)
        return feats[self.features]

    def _build_features_frame(self, df):
        import pandas as pd

        # Derive amenity_count from the flags when the caller didn't supply it.
        if "amenity_count" in self.features and "amenity_count" not in df.columns:
            present = [c for c in AMENITY_COLUMNS if c in df.columns]
            if present:
                df = df.copy()
                df["amenity_count"] = (
                    df[present].apply(pd.to_numeric, errors="coerce")
                    .fillna(0).astype(int).sum(axis=1)
                )

        out = pd.DataFrame(index=df.index)
        for feat in self.features:
            if feat == self.neighbourhood_feature:
                out[feat] = (
                    df[self.neighbourhood_source].astype(str).map(self._neigh_map)
                    if self.neighbourhood_source in df.columns
                    else np.nan
                )
            elif feat in self._label_maps:
                enc = self._label_maps[feat]
                out[feat] = (
                    df[feat].astype(str).map(enc) if feat in df.columns else np.nan
                )
            elif feat in df.columns:
                series = df[feat]
                if series.dtype == bool:
                    series = series.astype(int)
                out[feat] = pd.to_numeric(series, errors="coerce")
            else:
                out[feat] = np.nan
        return out


@lru_cache(maxsize=1)
def get_occupancy_predictor() -> OccupancyPredictor:
    """Process-wide singleton — loads artefacts once."""
    return OccupancyPredictor()
