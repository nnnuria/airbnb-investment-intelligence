"""Sale-price model inference service.

Wraps the trained sell-price **Pipeline** (`models/sale_best_model.pkl`) and
turns a partial property spec into a predicted total sale price in EUR. This is
the service the finance engine's *sell* scenario and the `/airbnb_vs_sell`
endpoint consume.

Because the whole sklearn Pipeline (preprocessing + model) is persisted, there
is no preprocessing to re-implement here — we just feed a DataFrame and call
``.predict`` (contrast the price model, which needed manual replication).

District-level imputation
=========================
The model leans on ``neighborhood`` and ``lat/lon`` for location. A minimal
``(city, district, size)`` query lacks those, which washes out the location
signal. :meth:`SalePredictor.predict` therefore fills missing fields from
``models/sale_district_defaults.json`` (per-district medians + modal
neighbourhood), falling back city → global. Caller-supplied values always win.
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
DEFAULT_MODEL = ROOT / "models" / "sale_best_model.pkl"
DEFAULT_META = ROOT / "models" / "sale_feature_cols.json"
DEFAULT_DEFAULTS = ROOT / "models" / "sale_district_defaults.json"

# District defaults are imputed for location/context, but NOT for these
# size-correlated counts: a district's modal unit is often small (e.g. Chamberí
# is mostly ~37 m² 1-room flats), so filling rooms/bathrooms/floor from the mode
# fights a larger user-supplied size. Left blank, the pipeline imputes a global
# median — coherent with whatever size the caller gives.
_SIZE_CORRELATED = frozenset({"rooms", "bathrooms", "floor_num"})


class SalePredictor:
    """Loads the sale Pipeline + district defaults and predicts EUR sale price."""

    def __init__(
        self,
        model_path: Path | str = DEFAULT_MODEL,
        meta_path: Path | str = DEFAULT_META,
        defaults_path: Path | str = DEFAULT_DEFAULTS,
    ) -> None:
        import joblib

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # cross-version unpickle banners
            self.model = joblib.load(model_path)

        meta = json.loads(Path(meta_path).read_text())
        self.features: list[str] = list(meta["input_features"])
        self.best_model: str = meta.get("best_model", type(self.model).__name__)

        d = json.loads(Path(defaults_path).read_text(encoding="utf-8"))
        self._by_city_district: dict = d.get("by_city_district", {})
        self._by_city: dict = d.get("by_city", {})
        self._global: dict = d.get("global", {})
        logger.info("SalePredictor ready: %s, %d features, %d district defaults",
                    self.best_model, len(self.features), len(self._by_city_district))

    # ── public API ────────────────────────────────────────────────────────────

    def predict(self, spec: Mapping[str, Any]) -> float:
        """Predict total sale price (EUR) for a property spec.

        Missing fields are imputed from district → city → global defaults, so a
        sparse ``{"city": "madrid", "district": "Chamberí", "size_m2": 90}``
        still reflects location. Caller-supplied (non-None) values override.
        """
        import pandas as pd

        resolved = self._resolve(spec)
        row = pd.DataFrame([{c: resolved.get(c) for c in self.features}])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # ColumnTransformer strips feature names
            return float(np.expm1(self.model.predict(row)[0]))

    # ── internals ───────────────────────────────────────────────────────────

    def _resolve(self, spec: Mapping[str, Any]) -> dict:
        city = (spec.get("city") or "").strip().lower()
        district = spec.get("district")

        def _fill(target: dict, src: dict) -> None:
            target.update({k: v for k, v in src.items() if k not in _SIZE_CORRELATED})

        merged: dict = {}
        _fill(merged, self._global)
        if city and city in self._by_city:
            _fill(merged, self._by_city[city])
        if city and district:
            key = f"{city}::{district}"
            if key in self._by_city_district:
                _fill(merged, self._by_city_district[key])
        # caller-supplied values win (skip None); size-correlated counts only
        # ever come from the caller — otherwise the pipeline's median imputes them.
        merged.update({k: v for k, v in spec.items() if v is not None})
        # city/district are the lookup keys; pass through as given
        if spec.get("city") is not None:
            merged["city"] = spec["city"]
        if district is not None:
            merged["district"] = district
        return merged


@lru_cache(maxsize=1)
def get_sale_predictor() -> SalePredictor:
    """Process-wide singleton — loads artefacts once."""
    return SalePredictor()


def estimate_sale_value(
    city: str,
    district: str | None = None,
    size_m2: float | None = None,
    **overrides: Any,
) -> float:
    """Indicative total sale price (EUR) — the finance engine's sell anchor.

    Convenience wrapper over :meth:`SalePredictor.predict`. Extra structured
    fields (``rooms``, ``bathrooms``, ``property_type``, ``status``, ...) can be
    passed as keyword overrides for a sharper estimate.
    """
    spec = {"city": city, "district": district, "size_m2": size_m2, **overrides}
    return get_sale_predictor().predict(spec)
