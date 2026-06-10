"""Shared loaders/cleaners for the three Idealista Kaggle sources.

Listing-type summary (verified during EDA):
  - madrid/Datos.csv ................. SALE (median ~620k EUR)
  - barcelona/*.csv .................. SALE ("... for sale ...", median ~305k EUR)
  - spain/rent_spain_scraping_dataset.csv ... RENT (monthly, median ~900 EUR)

Only the Spain file carries the target we need for rental-revenue prediction.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"

# property type keywords appearing at the start of Spanish idealista titles
_TYPE_PATTERNS = [
    ("studio", r"\bestudio\b"),
    ("penthouse", r"\b�tico|atico\b"),
    ("duplex", r"\bd�plex|duplex\b"),
    ("house", r"\bcasa|chalet\b"),
    ("flat", r"\bpiso\b"),
]


def _to_int(x) -> int | None:
    if pd.isna(x):
        return None
    m = re.search(r"\d+", str(x).replace(".", "").replace(",", ""))
    return int(m.group()) if m else None


def property_type_from_title(title: str) -> str:
    t = str(title).lower()
    for label, pat in _TYPE_PATTERNS:
        if re.search(pat, t):
            return label
    return "other"


def city_from_title(title: str) -> str:
    """Last comma-separated token is usually the municipality."""
    return str(title).split(",")[-1].strip()


def load_rent(city: str | None = None, dedup: bool = True) -> pd.DataFrame:
    """Load the Spain RENTAL dataset, cleaned.

    Parameters
    ----------
    city : optional, one of {"madrid", "barcelona"} to filter by province.
    dedup : drop exact duplicate rows (the raw file is ~87% duplicated in the
            Madrid/Barcelona subset).
    """
    df = pd.read_csv(RAW / "spain" / "rent_spain_scraping_dataset.csv")
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
    df = df.rename(
        columns={
            "precio": "rent_eur_month",
            "habitaciones": "n_rooms",
            "metros": "sq_m",
            "provincia": "province",
            "comunidad autonoma": "region",
            "titulo": "title",
            "total inmuebles/comunidad": "scrape_pool_size",
        }
    )
    if dedup:
        df = df.drop_duplicates()
    df["sq_m"] = df["sq_m"].map(_to_int)
    df["property_type"] = df["title"].map(property_type_from_title)
    df["municipality"] = df["title"].map(city_from_title)
    df["eur_per_sqm"] = df["rent_eur_month"] / df["sq_m"]
    if city:
        df = df[df["province"].str.lower().str.strip() == city.lower()]
    return df.reset_index(drop=True)


def load_madrid_sale() -> pd.DataFrame:
    """Madrid SALE listings (feature-rich; wrong target for rental)."""
    df = pd.read_csv(RAW / "madrid" / "Datos.csv")
    return df.rename(
        columns={
            "provincia": "province",
            "zona": "district",
            "titulo": "title",
            "PrecioActual": "price_eur",
            "PrecioAnterior": "price_prev_eur",
            "metros": "sq_m",
            "habitaciones": "n_rooms",
            "ascensor": "lift",
            "localizacion": "orientation",
            "planta": "floor",
            "ba�os": "n_baths",
        }
    )


def load_barcelona_sale() -> pd.DataFrame:
    """Barcelona SALE listings, richest structured file (feature-rich; sale target)."""
    return pd.read_csv(RAW / "barcelona" / "structured_dataw_description.csv")
