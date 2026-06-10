# Idealista Data-Source Assessment — Rental Revenue Prediction

**Scope:** Madrid & Barcelona. **Objective:** decide whether the three Kaggle Idealista datasets
hold what we need to build a **rental revenue** prediction model (the rental arm of the larger
*rent vs. Airbnb vs. sell* agentic system).

**TL;DR:** Only **one** of the three datasets is rental data. The other two are **sale** listings —
rich in features but the wrong target for predicting rent. We can build a *baseline* rental model
today, but the rental data is thin **and dated (~Sept 2022)**, so it will likely need supplementing
and/or market-index adjustment for production quality.

## Data vintage (important)

| Dataset | Type | Kaggle last-updated | Per-listing dates |
|---|---|---|---|
| **spain rental** (target) | RENT | **2022-09-25** (~Sept 2022 snapshot) | none |
| madrid sale | SALE | 2025-02-12 | none |
| barcelona sale | SALE | 2024-09-24 | none |

The rental file is a single point-in-time scrape from **~September 2022 (~3.7 yrs old as of 2026-06)**
with **no timestamp column** — listings cannot be date-filtered or trend-adjusted within the data.
Madrid/Barcelona rents have risen materially since 2022, so the dataset captures rent *structure*
(size/rooms/city → price) reliably but **absolute rent levels are stale** and need an external
rent-index uplift (e.g. INE / idealista index, 2022→present) before revenue figures are trustworthy.
Note the only rental source is also the *oldest*, while the sale datasets are recent (2024–2025).

---

## The three sources

| # | Kaggle dataset | File(s) used | Rows (post-dedup) | Listing type | Price median |
|---|---|---|---|---|---|
| 1 | `fjcob1/idealista-madrid` | `madrid/Datos.csv` | 11,826 | **SALE** | €620,000 |
| 2 | `victorianomh/idealista-barcelona-raw-scraped-data` | `barcelona/structured_dataw_description.csv` (+ `clean_data`, `raw_data`) | 14,359 | **SALE** | €305,000 |
| 3 | `laurabarreda/rental-listins-in-idealista-spain` | `spain/rent_spain_scraping_dataset.csv` | 79,749 → **3,562** for MAD+BCN | **RENT/month** | €900 |

> The Barcelona raw titles literally read *"Flat / apartment **for sale** in …"*, and the Madrid /
> Barcelona price medians (€620k / €305k) are sale prices. The Spain file's €900 median is monthly rent.

---

## Source 3 — the rental dataset (the only one with our target)

This is the **only** file with monthly rent, so it is the backbone for the rental model.

**Strengths**
- Has the **target**: `precio` = monthly rent (€).
- Covers both cities: **~1,773 Madrid** + **~1,789 Barcelona** unique listings (Madrid/Barcelona *provinces*, incl. metro towns like Pozuelo, Sitges, Badalona).
- `metros` (size) parses cleanly for **98.8%** of rows; `habitaciones` (rooms) present for **95%**.
- The free-text `title` encodes **property type** (Piso/Ático/Casa…) and **municipality**, both of
  which we extract in `src/loaders.py`.
- Rent ↔ size correlation is sensible and positive (sanity check passes).

**Weaknesses / risks**
- **Heavy duplication:** the Madrid+Barcelona subset is ~87% duplicate rows (27,091 → **3,562** unique).
  Must dedup before any analysis (handled in `load_rent`).
- **Feature-thin:** no **bathrooms**, no **amenities** (lift/AC/terrace/parking), no **condition/year**,
  no **neighbourhood/district** (only municipality + free-text street), no coordinates.
- Some **province mislabeling** (a minority of "Barcelona" rows are actually Balearic Islands) — clean
  by parsing the city from `title`.
- Small effective sample (~3.5k) limits model complexity.

**Verdict:** **Usable as the MVP target dataset.** Supports a baseline
`rent ~ city + property_type + rooms + m² (+ municipality)`. Not yet enough for amenity- or
neighbourhood-level accuracy.

---

## Sources 1 & 2 — the sale datasets (wrong target, but valuable)

These cannot train a *rental* model (their price is a sale price), but they are **feature-rich** and
worth keeping:

- **Madrid `Datos.csv`** — district (`zona`), size, rooms, **bathrooms**, floor, lift, orientation,
  tags, full description, listing URL.
- **Barcelona `structured_*`** — the richest schema of all: **neighborhood**, property type, size
  (built + floor area), rooms, **bathrooms**, floor, **year built**, **amenities** (lift, terrace,
  balcony, parking, pool, garden, AC, heating), **condition**, **energy certificate**, orientation,
  description.

**Role in the project**
1. **The *sell* arm** of the rent/Airbnb/sell decision — these *are* the right data for a sale-price model.
2. **Feature & neighbourhood reference** — schema and neighbourhood structure we can transfer to enrich
   the rental data.

---

## Feature coverage at a glance

| Feature | spain RENT | madrid SALE | barcelona SALE |
|---|:--:|:--:|:--:|
| **rent target** | ✅ | — | — |
| sale price | — | ✅ | ✅ |
| size m² | ✅ | ✅ | ✅ |
| rooms | ✅ | ✅ | ✅ |
| bathrooms | — | ✅ | ✅ |
| neighbourhood/district | city only | district | ✅ neighbourhood |
| property type | engineered | from title | ✅ |
| floor / year built | — | floor only | ✅ both |
| condition / renovation | — | tags | ✅ |
| amenities (lift/AC/terrace…) | — | partial | ✅ |
| energy certificate | — | — | ✅ |
| lat/long | — | — | — |
| free-text description | — | ✅ | ✅ |

---

## Recommendation / next steps

1. **Build the baseline rental model** on Source 3 (Madrid + Barcelona, deduplicated): target
   `rent_eur_month`, features `city, property_type, n_rooms, sq_m, municipality, eur_per_sqm` as EDA aid.
2. **Plan for more rental data.** To get bathrooms, amenities and neighbourhood granularity (the
   drivers of rent), supplement with a **targeted idealista rentals scrape** for Madrid/Barcelona or a
   richer rental dataset. The sale files show exactly which features are worth capturing.
3. **Keep the sale datasets** for the *sell* arm and as a neighbourhood/feature reference; don't mix
   their prices into the rental target.
4. **No geocoordinates anywhere** — if location precision matters, geocode from address/municipality.

See `notebooks/01_data_inventory_eda.ipynb` for the full analysis and figures
(`reports/figures/`).
