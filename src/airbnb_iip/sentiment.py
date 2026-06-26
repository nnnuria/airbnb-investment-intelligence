"""Serve the precomputed multilingual review-sentiment artifacts (UC3).

Reads the outputs of ``scripts/run_sentiment.py`` (already on disk) and shapes
them for the ``GET /sentiment`` endpoint — no scoring/retraining at request
time. Artifacts in ``Data/processed/``:

* ``sentiment_summary.csv``   — per-city ``label_split`` (donut) + ``lang_split``
  (language bar) + model metrics. NOTE: those two columns are **Python dict
  reprs** (single-quoted), so they are parsed with ``ast.literal_eval``.
* ``sentiment_examples.json``  — per-city sample positive/negative review quotes.
* ``<city>_sentiment.parquet`` — per-listing ``mean_sentiment`` → histogram.
"""

from __future__ import annotations

import ast
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "Data" / "processed"  # capital-D, matching agents/comparables.py

CITIES = ("madrid", "barcelona", "malaga")
_N_BINS = 10  # mean_sentiment histogram buckets over [-1, 1]

DISCLAIMER = (
    "Sentiment is weak-labelled from review text by a multilingual lexicon + "
    "Naïve Bayes model (no human labels), so treat it as an indicative signal."
)


def _parse_dict(value: Any) -> dict[str, float]:
    """Parse a ``label_split``/``lang_split`` cell (a Python dict repr) safely."""
    if isinstance(value, dict):
        return {str(k): float(v) for k, v in value.items()}
    try:
        parsed = ast.literal_eval(str(value))
        return {str(k): float(v) for k, v in dict(parsed).items()}
    except (ValueError, SyntaxError, TypeError):
        return {}


def _normalize(city: str) -> str:
    return str(city).strip().lower()


@lru_cache(maxsize=len(CITIES))
def get_city_sentiment(city: str) -> dict[str, Any]:
    """Per-city sentiment payload. Raises ``FileNotFoundError`` if the artifacts
    are missing (the router maps that to a 503). ``ValueError`` for an unknown
    city or one absent from the summary."""
    import numpy as np
    import pandas as pd

    city = _normalize(city)
    if city not in CITIES:
        raise ValueError(f"Unknown city '{city}'. Expected one of {CITIES}.")

    summary_path = DATA / "sentiment_summary.csv"
    examples_path = DATA / "sentiment_examples.json"
    parquet_path = DATA / f"{city}_sentiment.parquet"
    for p in (summary_path, examples_path, parquet_path):
        if not p.exists():
            raise FileNotFoundError(f"Sentiment artifact missing: {p}")

    # ── city-level aggregates (donut + language + metrics) ────────────────────
    summary = pd.read_csv(summary_path)
    rows = summary[summary["city"].astype(str).str.lower() == city]
    if rows.empty:
        raise ValueError(f"No sentiment summary row for city '{city}'.")
    row = rows.iloc[0]

    label_split = _parse_dict(row.get("label_split"))
    lang_split = _parse_dict(row.get("lang_split"))
    metrics = {
        k: float(row[k])
        for k in ("nb_accuracy", "nb_f1_macro", "corr_sentiment_vs_rating")
        if k in row and pd.notna(row[k])
    }

    # ── sample review quotes ──────────────────────────────────────────────────
    examples_all = json.loads(examples_path.read_text(encoding="utf-8"))
    city_examples = examples_all.get(city, {}) or {}
    examples = {
        polarity: [
            {"text": str(ex.get("text", "")), "score": float(ex.get("score", 0.0))}
            for ex in (city_examples.get(polarity) or [])[:3]
        ]
        for polarity in ("positive", "negative")
    }

    # ── per-listing distribution → fixed-bin histogram ────────────────────────
    listings = pd.read_parquet(parquet_path, columns=["mean_sentiment"])
    scores = listings["mean_sentiment"].dropna().to_numpy()
    edges = np.linspace(-1.0, 1.0, _N_BINS + 1)
    counts, _ = np.histogram(scores, bins=edges)
    listing_histogram = [
        {"label": f"{edges[i]:.1f}", "count": int(counts[i])} for i in range(_N_BINS)
    ]

    return {
        "city": city,
        "reviews_scored": int(row["reviews_scored"]) if pd.notna(row.get("reviews_scored")) else 0,
        "label_split": label_split,
        "lang_split": lang_split,
        "metrics": metrics,
        "examples": examples,
        "listing_histogram": listing_histogram,
        "n_listings": int(len(scores)),
        "disclaimer": DISCLAIMER,
    }
