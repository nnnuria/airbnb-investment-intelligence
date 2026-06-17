"""Generate notebooks/sell_price_eda.ipynb.

Authoring the EDA as a script keeps the notebook diff-friendly and
reproducible (same pattern as scripts/make_price_notebook.py). Run:

    python scripts/make_sell_eda_notebook.py

It builds the notebook from the cells below; execute it in Jupyter (or via
`jupyter nbconvert --execute`) to render the plots.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "sell_price_eda.ipynb"

nb = nbf.v4.new_notebook()
cells: list = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(src: str) -> None:
    cells.append(nbf.v4.new_code_cell(src.strip("\n")))


# ── Title ──────────────────────────────────────────────────────────────────────
md(r"""
# Sell-Price EDA — Idealista sale listings
**Airbnb Investment Intelligence Platform · sell-side model**

Sanity-checks the scraped Idealista **sale** data (Madrid, Barcelona, Málaga)
before modelling: coverage, distributions, €/m² by district, price–size
relationship, and outliers.

> **Leakage note:** `price_per_m2` (= price ÷ size) is shown here for
> intuition only. It must **never** be a feature in a model whose target is
> `price`.
""")

# ── Setup ──────────────────────────────────────────────────────────────────────
md("## 1. Setup & load")
code(r"""
%matplotlib inline
import warnings; warnings.simplefilter("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110

from airbnb_iip.data.idealista_sale import load_sale_listings, clean_sale_listings

raw   = load_sale_listings()
clean = clean_sale_listings(raw)
print(f"raw   : {len(raw):,} rows × {raw.shape[1]} cols")
print(f"clean : {len(clean):,} rows  (dropped {len(raw)-len(clean):,})")
clean.head(3)
""")

# ── Coverage ───────────────────────────────────────────────────────────────────
md("## 2. Data coverage & quality")
code(r"""
cov = pd.DataFrame({
    "non_null_%": (raw.notna().mean() * 100).round(1),
    "n_unique": raw.nunique(),
})
cov.sort_values("non_null_%", ascending=False)
""")
code(r"""
print("rows per city (clean):")
print(clean["city"].value_counts())
print("\nduplicate property_code in raw load:", raw["property_code"].duplicated().sum())
""")

# ── Target ─────────────────────────────────────────────────────────────────────
md("""
## 3. Target — sale price (€)

Price is heavily right-skewed, so the model will train on `log1p(price)`.
""")
code(r"""
display(clean.groupby("city")["price"].describe(percentiles=[.1,.5,.9])
        [["count","min","10%","50%","90%","max"]].round(0))

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].hist(clean["price"], bins=60, color="#4C72B0", edgecolor="white")
ax[0].set_title("price (€)"); ax[0].set_xlabel("€")
ax[1].hist(np.log1p(clean["price"]), bins=60, color="#55A868", edgecolor="white")
ax[1].set_title("log1p(price)  ← model target"); ax[1].set_xlabel("log1p €")
plt.tight_layout()
""")

# ── Size ───────────────────────────────────────────────────────────────────────
md("## 4. Size (m²)")
code(r"""
fig, ax = plt.subplots(figsize=(9, 4))
for city, g in clean.groupby("city"):
    sns.kdeplot(g["size_m2"], label=city, ax=ax, clip=(0, 400))
ax.set_title("Size distribution by city"); ax.set_xlabel("m²"); ax.legend()
plt.tight_layout()
""")

# ── €/m² ───────────────────────────────────────────────────────────────────────
md("""
## 5. Price per m² (the sell-side anchor)

This is the headline number the finance engine's *sell* scenario rests on
(`sale value ≈ €/m² × size`). Shown by city, then by district.
""")
code(r"""
display(clean.groupby("city")["price_per_m2"].describe(percentiles=[.1,.5,.9])
        [["min","10%","50%","90%","max"]].round(0))

fig, ax = plt.subplots(figsize=(9, 4))
for city, g in clean.groupby("city"):
    sns.kdeplot(g["price_per_m2"], label=city, ax=ax, clip=(0, 12000))
ax.set_title("€/m² by city"); ax.set_xlabel("€/m²"); ax.legend()
plt.tight_layout()
""")

md("### 6. €/m² ranked by district")
code(r"""
for city in ["madrid", "barcelona", "malaga"]:
    g = clean[clean["city"] == city]
    med = (g.groupby("district")["price_per_m2"].median()
             .sort_values(ascending=False))
    fig, ax = plt.subplots(figsize=(8, max(3, 0.28*len(med))))
    med.plot.barh(ax=ax, color="#C44E52")
    ax.invert_yaxis()
    ax.set_title(f"{city.title()} — median €/m² by district")
    ax.set_xlabel("€/m²")
    plt.tight_layout(); plt.show()
""")

# ── Price vs size ──────────────────────────────────────────────────────────────
md("""
## 7. Price vs size

If price scaled perfectly linearly with size, €/m² would be constant. It
isn't — larger flats have lower €/m² — which is exactly why we model total
price (with size as a feature) rather than €/m² × size.
""")
code(r"""
fig, ax = plt.subplots(figsize=(7, 6))
sns.scatterplot(data=clean.sample(min(4000, len(clean)), random_state=0),
                x="size_m2", y="price", hue="city", alpha=0.3, s=12, ax=ax)
ax.set(xscale="log", yscale="log", title="price vs size (log–log)")
plt.tight_layout()

corr = np.corrcoef(np.log1p(clean["size_m2"]), np.log1p(clean["price"]))[0, 1]
print(f"corr(log size, log price) = {corr:.3f}")
""")

# ── Type & condition ───────────────────────────────────────────────────────────
md("## 8. Property type & condition")
code(r"""
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
clean["property_type"].value_counts().plot.bar(ax=ax[0], color="#4C72B0")
ax[0].set_title("Listings by property_type"); ax[0].tick_params(axis="x", rotation=30)

(clean.groupby("property_type")["price"].median().sort_values()
      .plot.barh(ax=ax[1], color="#55A868"))
ax[1].set_title("Median price by property_type"); ax[1].set_xlabel("€")
plt.tight_layout()
""")
code(r"""
display(clean.groupby("status")["price"].agg(["count", "median"]).round(0))
sns.boxplot(data=clean.dropna(subset=["status"]), x="status", y="price", showfliers=False)
plt.yscale("log"); plt.title("Price by condition (status)"); plt.tight_layout()
""")

# ── Structural ─────────────────────────────────────────────────────────────────
md("## 9. Structural features (floor, lift, exterior)")
code(r"""
# Nullable booleans → labelled strings so NA is its own category (seaborn-safe).
d = clean.copy()
for c in ["has_lift", "exterior"]:
    d[c] = d[c].astype("object").where(d[c].notna(), "unknown").map(
        {True: "yes", False: "no", "unknown": "unknown"}
    )

fig, ax = plt.subplots(1, 3, figsize=(14, 4))
sns.boxplot(data=d, x="has_lift", y="price", order=["no","yes","unknown"],
            showfliers=False, ax=ax[0])
ax[0].set(yscale="log", title="has_lift")
sns.boxplot(data=d, x="exterior", y="price", order=["no","yes","unknown"],
            showfliers=False, ax=ax[1])
ax[1].set(yscale="log", title="exterior")
sub = d[d["floor_num"].between(-1, 12)]
sns.boxplot(data=sub, x="floor_num", y="price", showfliers=False, ax=ax[2])
ax[2].set(yscale="log", title="floor_num")
plt.tight_layout()
""")

# ── Correlations ───────────────────────────────────────────────────────────────
md("## 10. Numeric correlations")
code(r"""
num = ["price", "size_m2", "rooms", "bathrooms", "floor_num", "latitude", "longitude"]
corr = clean[num].assign(log_price=np.log1p(clean["price"])).corr()
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Correlation matrix"); plt.tight_layout()
""")

# ── Geography ──────────────────────────────────────────────────────────────────
md("## 11. Geography — price by location")
code(r"""
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, city in zip(axes, ["madrid", "barcelona", "malaga"]):
    g = clean[clean["city"] == city]
    sc = ax.scatter(g["longitude"], g["latitude"],
                    c=np.log1p(g["price"]), cmap="viridis", s=6, alpha=0.5)
    ax.set_title(city.title()); ax.set_xlabel("lon"); ax.set_ylabel("lat")
    plt.colorbar(sc, ax=ax, label="log1p(price)")
plt.tight_layout()
""")

# ── Summary ────────────────────────────────────────────────────────────────────
md("""
## 12. Sanity-check summary

Fill in after running:

- **Coverage** — core fields (price, size, rooms, baths, type, district,
  lat/lon) ~100%; `floor`/`exterior` partial (~85%).
- **Price/size** — right-skewed → `log1p(price)` target confirmed; `log size`
  is the strongest single correlate of `log price`.
- **€/m² by district** — ordering matches local knowledge (central/coastal
  districts highest). This is the sell-side anchor.
- **Outliers** — `clean_sale_listings` removed the obvious junk (32,000 m²
  listing, €5/m², €8.5M). Spot-check anything still surprising above.
- **Next** — feature engineering + model training in `sell_price_model.ipynb`.
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(f"Wrote {OUT} ({len(cells)} cells)")
