"""Check the single most important question: which datasets are RENTAL vs SALE,
and what Madrid/Barcelona rental coverage we actually have."""
import pandas as pd
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
pd.set_option("display.width", 120)

print("\n##### MADRID Datos.csv — price scale & listing type #####")
m = pd.read_csv(RAW / "madrid" / "Datos.csv")
print("PrecioActual describe:")
print(m["PrecioActual"].describe().round(0))
print("sample titulos:", m["titulo"].head(3).tolist())

print("\n##### BARCELONA structured — price scale & listing type #####")
b = pd.read_csv(RAW / "barcelona" / "structured_dataw_description.csv")
print("price describe:")
print(b["price"].describe().round(0))
print("raw titles sample:")
br = pd.read_csv(RAW / "barcelona" / "raw_data.csv")
print(br["title"].head(3).tolist())

print("\n##### SPAIN rent dataset — price scale, provinces #####")
s = pd.read_csv(RAW / "spain" / "rent_spain_scraping_dataset.csv")
print("precio describe:")
print(s["precio"].describe().round(0))
print("\nrows total:", len(s))
print("\nprovincia value counts (top 15):")
print(s["provincia"].value_counts().head(15))

# filter to Madrid + Barcelona
prov = s["provincia"].str.lower().str.strip()
mask = prov.str.contains("madrid") | prov.str.contains("barcelona")
sm = s[mask]
print(f"\nMadrid+Barcelona rental rows: {len(sm)}")
print(sm.groupby(prov[mask])["precio"].describe().round(0))
print("\nmetros dtype/sample:", s["metros"].dtype, s["metros"].head(3).tolist())
print("\nDuplicate rows in spain (full):", s.duplicated().sum())
print("Duplicate ignoring index col:", s.drop(columns=["Unnamed: 0"]).duplicated().sum())
