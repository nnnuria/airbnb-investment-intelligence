"""Generate models/sale_district_defaults.json.

The sell-price model leans on `neighborhood` and `lat/lon` for location. When a
caller (finance engine / API) supplies only `(city, district, size)`, those are
missing and the prediction loses location signal. This artefact provides
district-level fallbacks — median lat/lon/size/rooms/etc. and the modal
neighbourhood/condition per (city, district) — that :class:`SalePredictor`
fills in so a sparse query still reflects location.

It's a committed artefact so serving never needs the gitignored raw data. Run:

    python scripts/make_sale_district_defaults.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from airbnb_iip.data.idealista_sale import load_clean_sale

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "models" / "sale_district_defaults.json"

NUM_FIELDS = ["latitude", "longitude", "size_m2", "rooms", "bathrooms", "floor_num"]
CAT_FIELDS = ["neighborhood", "status", "property_type", "has_lift", "exterior",
              "new_development"]


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    """Match the model notebook's frame (booleans → labels, fill cat NaNs)."""
    df = df.copy()
    for c in ["has_lift", "exterior", "new_development"]:
        df[c] = df[c].astype("object").where(df[c].notna(), "unknown").map(
            {True: "yes", False: "no", "unknown": "unknown"}
        )
    for c in ["district", "neighborhood", "status"]:
        df[c] = df[c].fillna("unknown")
    return df


def _aggregate(g: pd.DataFrame) -> dict:
    out: dict = {}
    for f in NUM_FIELDS:
        val = g[f].median()
        if pd.notna(val):
            out[f] = round(float(val), 6)
    for f in CAT_FIELDS:
        vc = g[f].value_counts()
        if len(vc):
            out[f] = str(vc.index[0])  # mode
    return out


def main() -> None:
    df = _prep(load_clean_sale())

    by_city_district = {
        f"{city.lower()}::{district}": _aggregate(g)
        for (city, district), g in df.groupby(["city", "district"])
    }
    by_city = {str(city).lower(): _aggregate(g) for city, g in df.groupby("city")}
    glob = _aggregate(df)

    artefact = {
        "by_city_district": by_city_district,
        "by_city": by_city,
        "global": glob,
        "_meta": {"n_rows": int(len(df)), "n_districts": len(by_city_district)},
    }
    OUT.write_text(json.dumps(artefact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {OUT}  ({len(by_city_district)} city/district keys, "
          f"{len(by_city)} cities)")


if __name__ == "__main__":
    main()
