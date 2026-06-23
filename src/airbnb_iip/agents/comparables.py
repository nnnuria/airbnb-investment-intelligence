"""
Comparables Agent — structured filters + KNN over the cleaned ABT.
src/airbnb_iip/agents/comparables.py

Self-contained, like the Regulatory agent: it owns its own data (the
segmented listings table) rather than calling another service over HTTP,
because there is no existing "comparables" endpoint to call — this agent
*is* that service.

Flow:
    property spec -> structured filters (city, segment, room_type,
    neighbourhood) -> KNN on size/capacity features among the filtered
    candidates -> ranked comparable listings + benchmark stats
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_PATH = ROOT / "Data" / "processed" / "listings_segmented.parquet"

# Numeric features describing a listing's size/capacity — what "similar
# property" means once the structured filters have narrowed the pool.
NUMERIC_FEATURES: tuple[str, ...] = (
    "accommodates",
    "bedrooms",
    "bathrooms_number",
    "amenity_count",
)

# Columns surfaced on each comparable result, beyond the id/distance.
DISPLAY_COLUMNS: tuple[str, ...] = (
    "id",
    "name",
    "city",
    "neighbourhood_cleansed",
    "room_type",
    "property_type_std",
    "Segment_Name",
    "accommodates",
    "bedrooms",
    "bathrooms_number",
    "price",
    "estimated_revenue_l365d",
    "review_scores_rating",
)

MIN_CANDIDATES = 5  # below this, progressively relax structured filters


class ComparablesAgent:
    """Finds genuinely similar listings as benchmarks from the cleaned ABT."""

    def __init__(self, data_path: Path | str = DEFAULT_DATA_PATH):
        import pandas as pd

        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Comparables data not found at {path}. "
                "Run the segmentation pipeline first or pass data_path= explicitly."
            )
        df = pd.read_parquet(path)
        missing = [c for c in (*NUMERIC_FEATURES, "city") if c not in df.columns]
        if missing:
            raise KeyError(f"Comparables data is missing required columns: {missing}")

        self.df = df.reset_index(drop=True)
        # The data's city column keeps the display spelling ("Málaga"), but
        # callers pass the ASCII slug used everywhere else in this codebase
        # (config.py's TOURIST_TAX_EUR_PER_GUEST_NIGHT, the city_pattern in
        # api/schemas.py, etc.) — normalize once so "malaga" matches "Málaga".
        self._city_norm = self.df["city"].map(_normalize_city)
        self._fit_scaler()

    def _fit_scaler(self) -> None:
        from sklearn.preprocessing import StandardScaler

        numeric = self.df[list(NUMERIC_FEATURES)].apply(
            lambda s: s.fillna(s.median())
        )
        self._scaler = StandardScaler().fit(numeric)
        self._scaled = self._scaler.transform(numeric)

    # ── public API ────────────────────────────────────────────────────────────

    def find_comparables(
        self,
        property_spec: Mapping[str, Any],
        k: int = 5,
    ) -> dict[str, Any]:
        """Return up to ``k`` comparable listings for ``property_spec``.

        ``property_spec`` should include ``city`` (required for a sane
        result) and may include ``segment`` (``Segment_Name``), ``room_type``,
        and ``neighbourhood_cleansed`` as structured filters, plus any of
        :data:`NUMERIC_FEATURES` describing the property's size/capacity.

        Filters are relaxed in order (neighbourhood -> segment -> room_type)
        whenever fewer than :data:`MIN_CANDIDATES` rows survive, so a rare
        combination still returns a usable comp set instead of an empty one.
        """
        city = _normalize_city(property_spec.get("city", ""))
        if not city:
            raise ValueError("property_spec must include 'city'")

        filters_applied, mask = self._apply_filters(property_spec, city)
        candidates = self.df.loc[mask]
        scaled_candidates = self._scaled[mask.to_numpy()]

        if len(candidates) == 0:
            return {
                "comparables": [],
                "filters_applied": filters_applied,
                "benchmark": {},
            }

        distances, order = self._rank_by_similarity(
            property_spec, candidates, scaled_candidates, k,
        )
        top = candidates.iloc[order]

        comps = [
            {
                **{col: _to_native(row[col]) for col in DISPLAY_COLUMNS if col in row},
                "distance": round(float(dist), 4),
            }
            for (_, row), dist in zip(top.iterrows(), distances)
        ]

        return {
            "comparables": comps,
            "filters_applied": filters_applied,
            "benchmark": self._benchmark_stats(top),
        }

    # ── internals ────────────────────────────────────────────────────────────

    def _apply_filters(
        self, spec: Mapping[str, Any], city: str,
    ) -> tuple[dict[str, Any], Any]:
        mask = self._city_norm == city
        applied: dict[str, Any] = {"city": city}

        candidate_filters: list[tuple[str, str]] = []
        # "district" (e.g. "Salamanca") is the broader Inside-Airbnb
        # neighbourhood_group_cleansed; "neighbourhood_cleansed" is the
        # finer barrio within it (e.g. "Goya"). Callers may supply either.
        if spec.get("district") and "neighbourhood_group_cleansed" in self.df.columns:
            candidate_filters.append(("neighbourhood_group_cleansed", spec["district"]))
        if spec.get("neighbourhood_cleansed") and "neighbourhood_cleansed" in self.df.columns:
            candidate_filters.append(("neighbourhood_cleansed", spec["neighbourhood_cleansed"]))
        if spec.get("segment") and "Segment_Name" in self.df.columns:
            candidate_filters.append(("Segment_Name", spec["segment"]))
        if spec.get("room_type") and "room_type" in self.df.columns:
            candidate_filters.append(("room_type", spec["room_type"]))

        # Apply all filters, then relax the earliest ones first (neighbourhood
        # is the most specific / least essential to match) until enough
        # candidates remain.
        working_mask = mask.copy()
        working_filters = list(candidate_filters)
        for col, value in candidate_filters:
            trial_mask = working_mask & (self.df[col].astype(str) == str(value))
            if trial_mask.sum() == 0:
                continue
            working_mask = trial_mask
            applied[col] = value

        while working_mask.sum() < MIN_CANDIDATES and working_filters:
            col, _ = working_filters.pop(0)
            applied.pop(col, None)
            working_mask = mask.copy()
            for c, v in working_filters:
                if c in applied:
                    working_mask &= self.df[c].astype(str) == str(v)

        if working_mask.sum() == 0:
            working_mask = mask  # city-only is the final fallback
            applied = {"city": city}

        return applied, working_mask

    def _rank_by_similarity(
        self,
        spec: Mapping[str, Any],
        candidates,
        scaled_candidates,
        k: int,
    ) -> tuple[Sequence[float], Sequence[int]]:
        import pandas as pd
        from sklearn.neighbors import NearestNeighbors

        query = {feat: spec.get(feat) for feat in NUMERIC_FEATURES}
        medians = candidates[list(NUMERIC_FEATURES)].median()
        query_row = {
            feat: query[feat] if query[feat] is not None else medians[feat]
            for feat in NUMERIC_FEATURES
        }
        # Named columns (matching how the scaler was fit) avoid sklearn's
        # "X does not have valid feature names" warning.
        query_scaled = self._scaler.transform(pd.DataFrame([query_row], columns=NUMERIC_FEATURES))

        n_neighbors = min(k, len(candidates))
        nn = NearestNeighbors(n_neighbors=n_neighbors).fit(scaled_candidates)
        distances, indices = nn.kneighbors(query_scaled)
        return distances[0], indices[0]

    def _benchmark_stats(self, comps) -> dict[str, Any]:
        stats: dict[str, Any] = {}
        for col in ("price", "estimated_revenue_l365d", "review_scores_rating"):
            if col in comps.columns and comps[col].notna().any():
                stats[col] = {
                    "median": _to_native(comps[col].median()),
                    "p25": _to_native(comps[col].quantile(0.25)),
                    "p75": _to_native(comps[col].quantile(0.75)),
                }
        return stats


def _normalize_city(value: Any) -> str:
    """Accent-strip + lowercase a city name ("Málaga" -> "malaga").

    Matches the ASCII-slug convention used elsewhere in this codebase
    (e.g. config.py's TOURIST_TAX_EUR_PER_GUEST_NIGHT keys), even though the
    source data keeps the display spelling.
    """
    import unicodedata

    ascii_only = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore")
    return ascii_only.decode().strip().lower()


def _to_native(value: Any) -> Any:
    """numpy/pandas scalar -> plain Python type (JSON-serialisable)."""
    if value is None:
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


@lru_cache(maxsize=1)
def get_comparables_agent() -> ComparablesAgent:
    """Process-wide singleton — loads + fits the data once."""
    return ComparablesAgent()


# ── Quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    agent = ComparablesAgent()
    result = agent.find_comparables(
        {
            "city": "madrid",
            "neighbourhood_cleansed": "Salamanca",
            "room_type": "Entire home/apt",
            "accommodates": 4,
            "bedrooms": 2,
            "bathrooms_number": 1,
        },
        k=5,
    )
    print("Filters applied:", result["filters_applied"])
    print("Benchmark:", result["benchmark"])
    for comp in result["comparables"]:
        print(comp)
