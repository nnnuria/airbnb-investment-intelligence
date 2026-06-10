"""Feature engineering for the rental-revenue model.

Builds a clean modelling frame from the Spain RENTAL dataset, with location
features parsed out of the structured `title` string:

    "{property_type} en {street + number}, {neighborhood}, {municipality}"

Key engineered field: `neighborhood` (parsed for ~97.5% of Madrid/Barcelona rows),
the strongest location signal available in this data.
"""
from __future__ import annotations

import re

import pandas as pd

from loaders import load_rent, property_type_from_title

# Municipalities that appear under province="Barcelona" but are out-of-region
# (Balearic Islands etc.) due to scraper mislabeling. Dropped in metro scope.
_OUT_OF_REGION = {
    "palma de mallorca", "eivissa", "llucmajor", "marratxi", "alaró", "alaro",
    "santanyi", "santanyí", "sant josep de sa talaia", "sant lluis", "sant lluís",
    "manacor", "inca", "calvià", "calvia", "pollença", "pollensa", "felanitx",
    "ibiza", "mahón", "mahon", "ciutadella de menorca", "andratx", "soller", "sóller",
}

_STREET_RE = re.compile(
    r"(?i)^(calle|avenida|avinguda|paseo|passeig|camino|cami|passatge|carrer|"
    r"plaza|plaça|placa|ronda|via|vía|gran v[ií]a|travesía|travessera)\b"
)


def parse_title(title: str) -> dict:
    """Extract street, neighborhood, municipality, property_type from a title."""
    raw = str(title)
    ptype = property_type_from_title(raw)
    body = re.sub(r"^.*? en ", "", raw)  # strip "<type> en "
    toks = [t.strip() for t in body.split(",") if t.strip()]
    municipality = toks[-1] if toks else None
    neighborhood = None
    if len(toks) >= 2:
        cand = toks[-2]
        neighborhood = None if cand.isdigit() else cand
    street = toks[0] if toks and _STREET_RE.match(toks[0]) else None
    return {
        "property_type": ptype,
        "street": street,
        "neighborhood": neighborhood,
        "municipality": municipality,
    }


def build_model_frame(
    scope: str = "city",
    rent_range: tuple[int, int] = (200, 8000),
    sqm_range: tuple[int, int] = (15, 400),
    rooms_max: int = 8,
    verbose: bool = True,
) -> pd.DataFrame:
    """Return a clean modelling frame for Madrid & Barcelona rentals.

    scope:
        "city"  -> Madrid city + Barcelona city proper (cleanest, ~2.8k rows)
        "metro" -> whole Madrid/Barcelona provinces minus out-of-region noise
    """
    # drop loaders' engineered cols; parse_title regenerates them cleanly
    df = load_rent().drop(columns=["municipality", "property_type"], errors="ignore")
    audit = {"loaded (dedup)": len(df)}

    # province gate
    prov = df["province"].str.lower().str.strip()
    df = df[prov.isin(["madrid", "barcelona"])].copy()
    audit["madrid+barcelona province"] = len(df)

    # parse title -> location features
    parsed = df["title"].apply(parse_title).apply(pd.Series)
    df = pd.concat([df.reset_index(drop=True), parsed.reset_index(drop=True)], axis=1)

    muni_l = df["municipality"].str.lower().str.strip()
    if scope == "city":
        df = df[muni_l.isin(["madrid", "barcelona"])].copy()
    elif scope == "metro":
        df = df[~muni_l.isin(_OUT_OF_REGION)].copy()
    else:
        raise ValueError("scope must be 'city' or 'metro'")
    audit[f"scope={scope}"] = len(df)

    # canonical city label
    df["city"] = df["province"].str.lower().str.strip().map(
        {"madrid": "Madrid", "barcelona": "Barcelona"}
    )

    # numeric cleaning + outlier gate
    df["rent_eur_month"] = pd.to_numeric(df["rent_eur_month"], errors="coerce")
    df["sq_m"] = pd.to_numeric(df["sq_m"], errors="coerce")
    df["n_rooms"] = pd.to_numeric(df["n_rooms"], errors="coerce")

    before = len(df)
    df = df[df["rent_eur_month"].between(*rent_range)]
    df = df[df["sq_m"].between(*sqm_range)]
    df = df[df["n_rooms"].fillna(0).le(rooms_max)]
    df = df.dropna(subset=["rent_eur_month", "sq_m"])
    audit["after outlier/NA filter"] = len(df)
    audit["dropped by filters"] = before - len(df)

    # fill rooms (studios often NaN -> 0) and tidy neighborhood
    df["n_rooms"] = df["n_rooms"].fillna(0)
    df["neighborhood"] = df["neighborhood"].fillna("unknown")

    keep = [
        "rent_eur_month", "city", "municipality", "neighborhood", "street",
        "property_type", "sq_m", "n_rooms",
    ]
    out = df[keep].reset_index(drop=True)

    if verbose:
        print("Row audit:")
        for k, v in audit.items():
            print(f"  {k:32s} {v}")
        print(f"\nFinal frame: {out.shape}")
        print("By city:\n", out["city"].value_counts().to_string())
        print(f"\nUnique neighborhoods: {out['neighborhood'].nunique()}")
    return out


if __name__ == "__main__":
    build_model_frame()
