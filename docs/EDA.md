# Exploratory Data Analysis (EDA)
**Airbnb Investment Intelligence Platform · IE × KPMG Capstone 2026**

*Owner: Almu (Task #5 — Documentation). Source of truth: repo `nnnuria/airbnb-investment-intelligence` @ `d55fe1a`. Figures are drawn from the per-city notebooks, `notebooks/cross_city_comparison.ipynb`, the sell-side notebooks, and `docs/DATA_FINDINGS.md`, cross-checked against the external AirROI Madrid market report.*

---

## 1. Purpose & scope

The EDA underpins the platform's core decision — **"keep as an Airbnb or sell?"** — by characterising three Spanish short-term-rental (STR) markets and validating that the data can support the price, occupancy, and sell-side models. It covers four datasets:

| Dataset | Source | Role in the project |
|---|---|---|
| Airbnb listings | Inside Airbnb (`listings.csv.gz`) | Nightly-price model + occupancy estimator |
| Airbnb calendar | Inside Airbnb (`calendar.csv.gz`) | Seasonality multipliers (availability only — see §6) |
| Airbnb reviews | Inside Airbnb (`reviews.csv.gz`) | Occupancy proxy (SF model) + sentiment feature |
| Idealista sales | Apify scraper (`scripts/scrape_idealista.py`) | Sell-side price model (the sell scenario anchor) |
| Regulatory corpus | Official municipal/regional STR rules | Regulatory RAG agent (risk flag) |

Cities: **Madrid · Barcelona · Málaga.** Snapshot reference date: **14 Sep 2025** (`config/config.yaml`; Málaga scraped 30 Sep). The committed data uses this September-2025 snapshot; an earlier Barcelona snapshot recorded in `DATA_FINDINGS.md` is superseded.

---

## 2. Data volume & cleaning yield

The cleaning pipeline (`src/airbnb_iip/data/cleaning.py`, documented fully in `FEATURE_ENGINEERING.md`) reduces raw listings to a modelling-ready set, primarily by dropping rows with missing price and capping price outliers above the 99.5th percentile.

| City | Raw listings | Clean listings | Source notebook |
|---|---:|---:|---|
| Madrid | 25,000 | 18,862 | `notebooks/madrid/01_cleaning.ipynb` |
| Barcelona | 19,410 | 15,199 | `notebooks/barcelona/01_cleaning.ipynb` |
| Málaga | 9,714 | 8,762 | `notebooks/malaga/01_cleaning.ipynb` |
| **Combined ABT** | — | **42,823** | `notebooks/merge_listings_for_ml.ipynb` → `Data/processed/listings_all_cities.parquet` |

City-specific outliers removed during EDA: implausible records such as `beds = 40` and `bedrooms = 25`. Duplicate (name + host + coordinates) pairs — likely whole-building hosts — were identified (Madrid 196, Málaga 71) and reviewed.

> **Note on figure consistency:** `docs/DATA_FINDINGS.md` reports raw listing counts that differ slightly from the per-city cleaning notebooks (e.g. Barcelona 18,177 / 18,927 vs. 19,410). The difference traces to the Barcelona snapshot replacement and to whether counts are pre- or post-cleaning. The numbers above are the **cleaned** counts from the notebooks; treat DATA_FINDINGS figures as the raw-snapshot reference.

---

## 3. Listings — price & supply

| City | Median nightly price | Entire-home share | Notes |
|---|---:|---:|---|
| Madrid | €110 | 72% | Largest, deepest market; lowest availability (highest booked share) |
| Barcelona | €130 | 69% | Cleaned-ABT median (2025-09-14 snapshot) — see note below |
| Málaga | €102 | 89% | Most "whole-property" market — clearest Airbnb-vs-sell comparison |

- **Málaga is the cleanest test case for the decision flow:** 89% entire homes means most listings have a direct sale comparison. Madrid and Barcelona carry a larger private-room segment (~27–30%), which is less comparable to a sale.
- **Price is strongly right-skewed** in all three cities (raw skewness ≈ 3.2 Madrid / 3.8 Barcelona; Málaga much higher at ≈ 9.5, driven by luxury/coastal outliers). This is why all price modelling uses a **`log1p(price)`** target.
- **Commercial operators dominate:** ≈ 52–60% of listings per city are run by hosts with 6+ listings (`calculated_host_listings_count`) — relevant to the "professional vs peer-to-peer" framing.

> *Barcelona median (resolved):* Barcelona is part of the **2025-09-14** snapshot (per `config/config.yaml` and the listings' `last_scraped`), the same as Madrid. Its cleaned median nightly price (`Data/processed/listings_all_cities.parquet`, 15,199 listings) is **€130**. The **~€111** quoted earlier was the **all-cities pooled** median, not Barcelona's. **Use €130** as Barcelona's headline median everywhere (report, deck).

---

## 4. Occupancy & demand (calendar-derived)

Occupancy is measured directly from Inside Airbnb's **calendar** — a night with `available == 'f'` is treated as booked — and a LightGBM model learns it from listing features (see `FEATURE_ENGINEERING.md §5` and `model_cards/occupancy_model.md`). This replaced the earlier review-velocity (San Francisco) estimate, which correlated only ≈ 0.13 with actual calendar occupancy. EDA-level demand indicators:

| City | Calendar availability | Implied booked share | Median occupancy (days/yr) |
|---|---:|---:|---:|
| Madrid | 46% available | **highest** | 64 |
| Barcelona | 57% available | medium | 62 |
| Málaga | 57% available | medium | 42 |

**Madrid leads on demand and data depth** (25k listings, ~1.3M reviews, price populated) — which is why the team prioritised it for the first end-to-end slice and the demo.

### External validation — AirROI Madrid market report
To sanity-check the calendar-derived occupancy against an independent market source, we benchmarked Madrid against [AirROI](https://www.airroi.com/airbnb-data/spain/community-of-madrid/madrid) (2026 data) — the model's central Madrid estimates (~40–50%) line up well with it:

| Metric (AirROI, Madrid) | Value |
|---|---:|
| Market occupancy | **51.6%** |
| Median occupancy | 61% |
| Top-25% occupancy | 80%+ |
| Top-10% occupancy | 91%+ |
| Average daily rate (ADR) | ~$182 (≈ €168) |
| Average annual revenue | ~$28,580 (≈ €26.4k) |

This **validates the model's calibration**: the project's default occupancy assumption (0.55) sits right at the market average, and the 70% ceiling sits between the median (61%) and top quartile (80%) — only elite listings legitimately exceed it. ⚠️ AirROI figures are in **USD**; convert before any direct comparison with the EUR pipeline.

---

## 5. Sentiment (review text)

A multilingual pipeline (`notebooks/*/03_sentiment.ipynb`) scores per-listing review sentiment, intended as a standalone feature for the optimisation flow.

| City | Positive | Negative | Dominant languages | Corr. vs star rating |
|---|---:|---:|---|---:|
| Madrid | 60.6% | 3.4% | ES 42%, EN 39% | 0.23 |
| Barcelona | 71.7% | 3.1% | EN 59%, ES 12% | 0.29 |
| Málaga | 64.8% | 3.2% | EN 47%, ES 22% | 0.28 |

- **Multilingual processing was essential** — 42% of Madrid reviews are in Spanish; an English-only model would have mis-scored nearly half the data.
- **Negative-review rates are statistically indistinguishable** across cities (3.1–3.4%) — do *not* headline a "most negative city."
- **Honest caveat:** the lexicon-labelled classifier's high training accuracy (~0.94) reflects agreement with weak labels, not ground truth. Use the **correlation with real star ratings (0.23–0.29)** as the honest validation metric.

---

## 6. Calendar & seasonality

**Critical finding: the calendar `price` column is 100% null across all three cities** (~19M rows). This is expected Inside Airbnb behaviour, not a download error, and has two consequences:
1. **No price time series exists → no ARIMA/Prophet price model is possible or needed.** Nightly price comes from the cross-sectional listings model.
2. The calendar can only inform **demand seasonality** via the `available` flag.

**Scrape-date artefact:** because the calendar is forward-looking from one scrape date, the next ~2–4 weeks are already booked and read as "unavailable" regardless of season. Measured for Madrid: 0–2 days ahead shows 85.8% booked vs. 58.8% at 6–12 months. **Fix:** exclude dates within 60 days of the snapshot before computing multipliers.

### Monthly seasonality

Per-city monthly multipliers are produced by `src/airbnb_iip/features/seasonality.py` (availability-based, 60-day scrape-artefact exclusion, normalised to an annual mean of 1.0). **Specific monthly multiplier values are not reported here:** from a single forward snapshot they are dominated by the scrape-date booking horizon and are not stable enough across cities/snapshots to quote as fixed figures — see the discrepancy below.

> ⚠️ **Open discrepancy to resolve.** The repo's availability-derived seasonality, the demo's hardcoded seasonality (`mocks.py`), and the external AirROI report **disagree on the summer**. AirROI reports Madrid's **peak in April and trough in August**; the availability method reads August as *high demand* — likely because residents leave in August, so low *supply availability* is misread as high *tourist demand*. This matters for the occupancy model and should be called out explicitly in the deck rather than smoothed over.

---

## 7. Sell-side EDA (Idealista)

Source: `notebooks/sell_price_eda.ipynb`, from the scraped Idealista sale listings.

| Metric | Value |
|---|---:|
| Raw sale listings | 14,909 |
| Clean | 14,862 (dropped 47) |
| Madrid / Barcelona / Málaga | 5,030 / 4,993 / 4,839 |
| Field coverage (price, size, rooms, district, lat/lon) | ~100% |
| Floor / exterior / lift coverage | ~85% |
| corr(log size, log price) | 0.75–0.80 |
| **Model-ready split (committed)** | **11,774 train / 2,944 test** (`models/sale_feature_cols.json`) |

> The raw/clean/per-city counts above are the EDA stage; the **committed sell model** is trained on **11,774 train / 2,944 test** rows (the model-ready split per `models/sale_feature_cols.json`) — that split (and R² 0.868 / MAE €47,272 / MAPE 13.9%) is the authoritative figure.

**Key modelling insight:** price does **not** scale linearly with size — larger flats have lower €/m². This is why the sell model predicts **total price with size as a feature**, rather than the naïve `€/m² × size`. District-level €/m² ordering matches local knowledge (central/coastal districts highest), confirming the sell-side anchor is sound. `price_per_m2` (= price ÷ size) is used for EDA intuition only and is **excluded from the model as a leakage trap**.

---

## 8. Market segmentation & dataset separation

### 8.1 Should we train separate models? (Task #2A)
`notebooks/data_segmentation_eda.ipynb` checks whether any data slice is large enough (≥500 records) to justify a *separate* model:

| Split dimension | Viable slices | Verdict |
|---|---|---|
| By city | 3 / 3 (Madrid 18,862 · Barcelona 15,199 · Málaga 8,762) | **Selected** after the #2B empirical check — by-city R² 0.814 vs 0.810 pooled (pooled per-city R²: 0.797 / 0.819 / 0.752) |
| By city × property type | 6 / 18 (only Entire-place & Private-room clear 500) | **No** — 12 slices too small; do not split by property type |
| By market segment | 4 / 4 (all ≥500) | Use the segment as a **feature**, not a model router |

**Conclusion (updated after #2B):** the empirical check selected **city-specific** price models (by-city R² 0.814 vs 0.810 pooled); they serve the decision engine's scenario / NPV / recommendation path (the pooled model still backs the standalone `/predict_price` endpoints). Do not split by property type.

### 8.2 Market segments (K-means clustering) — executed end-to-end
`notebooks/segmentation.ipynb` (method: `docs/cluster_analysis_method.md`) clusters listings on *attributes only*, writing `Segment_Name` to `Data/processed/listings_segmented.parquet`. **The notebook was run end-to-end on the real data with no errors; the numbers below are its actual output**, not estimates.

- **Outliers:** multivariate-extreme listings (>1 IQR-outlier column, **9.9%**, 4,234) → manual **Ultra / Extreme** segment, never clustered.
- **Features:** lean **protected core** (capacity, quality, room-type, **centrality**) in **PCA(0.9)** space. City identity excluded by design (it's a model feature; including it drops silhouette to ≈0.18).
- **k:** silhouette peaks at the trivial **k=2** (74% blob), so k is constrained to ≥4; among k∈[4,6] with no <5% micro-clusters → **k=4, silhouette 0.365**, largest segment **24.8% of all listings** (vs. the prior run's 56.7% mega-cluster).
- **DBSCAN sanity check:** 63 clusters, 31% noise, silhouette 0.27 → K-means is the right choice.

**Segments are named by structure** (not price rank), grounded in the actual cluster profiles:

| Segment | n | % of total | mean price | What it is |
|---|---:|---:|---:|---|
| Budget private rooms | 9,873 | 23.1% | €75 | ~95% private rooms, small (~1.8 guests) |
| Central entire homes | 10,262 | 24.0% | €159 | entire homes, **100% central** |
| Non-central entire homes | 10,600 | 24.8% | €162 | entire homes, **0% central** |
| Premium entire homes | 7,842 | 18.3% | €239 | **large** entire homes (~5.3 guests) |
| Ultra / Extreme | 4,234 | 9.9% | €262 | multivariate outliers, lower rating |

> **The two ~€160 tiers are the same price but split by location** (central vs non-central entire homes) — exactly the signal `is_central` was protected to keep. Same price, different location is a real, defensible distinction, which is why the segments are named by structure rather than price (it also pre-empts the obvious "why are these two separate?" question). A coarser three-tier `k=3` view (Budget/Mid/Premium, silhouette 0.33, largest 42.7%) is available if a pure price story is preferred. See `NOTEBOOK_IMPROVEMENTS.md`.

---

## 9. Regulatory landscape (EDA of the corpus)

The regulatory reality is **decision-changing** and a headline EDA insight in its own right: in all three cities, a *typical central flat can no longer obtain a new STR licence*.

| City | Regime | Status |
|---|---|---|
| Madrid | VUT (Plan RESIDE, Aug 2025) | No new "dispersed" licences in the historic centre |
| Barcelona | HUT (moratorium since 2014) | All ~10,101 licences expire by 2028, no renewal (upheld Mar 2025) |
| Málaga | VFT (Decree 28/2016) | City-wide moratorium on new tourist dwellings from Aug 2025 |

Plus a **national registration number (NRA)** required since 2 Jan 2025 (RD 1312/2024). This is exactly the responsible-AI story the platform is built to tell: the finance engine may show Airbnb winning on NPV while the regulatory agent flags that a *new* legal STR isn't permitted.

---

## 10. Headline EDA takeaways (deck-ready)

1. **Three viable markets, one clear lead.** Madrid is the strongest demonstration market — deepest data, highest demand, prices populated.
2. **Price is log-normal and location-driven.** `log1p(price)` target; neighbourhood + capacity dominate (see §4 of `FEATURE_ENGINEERING.md` and the price-model SHAP drivers).
3. **Occupancy must be inferred, not read.** Calendar prices are 100% null; occupancy comes from review velocity, validated at ~52% against AirROI for Madrid.
4. **Seasonality is genuinely uncertain in summer** — internal signals and the external benchmark disagree on August. A documented limitation, not a bug.
5. **Sale price is non-linear in size** — model total price, not €/m² × size.
6. **Regulation can override the numbers** — the most defensible, KPMG-aligned insight in the analysis.
7. **Splitting by city was selected (#2B):** by-city LightGBM won the empirical comparison (R² 0.814 vs 0.810 pooled) and serves the decision engine; splitting by **property type** is not warranted (only 6/18 slices large enough).

---

## 11. Known limitations & data caveats

- **Snapshot misalignment:** the three cities were scraped on different dates, so cross-city month-vs-month seasonality comparisons are only indicative until re-pulled on a common date.
- **Barcelona median price (resolved):** cleaned-ABT median is **€130** (2025-09-14 snapshot); €111 was the pooled all-cities median. Use **€130** downstream.
- **Sentiment labels are weak** (lexicon-generated); trust the 0.23–0.29 rating correlation, not the 0.94 training accuracy.
- **Málaga has a missing column** (`neighbourhood_group_cleansed` is all-null), which breaks one crosstab in `malaga/02_analysis.ipynb`.
- **Summer seasonality** conflict (§6) is unresolved.
- **Mid-market segments are fuzzy** — the segmentation (§8.2) was executed end-to-end, but the two middle tiers land at near-identical price (€159/€162); only the Budget and Premium ends are crisp. Treat segments as indicative, and see the k=3-vs-k=4 choice noted in §8.2.
- **Amenity-pipeline divergence** — two amenity approaches now coexist (hand-curated 22-flag regex vs. a data-driven notebook that rewrites the shared parquet); see `FEATURE_ENGINEERING.md §2`.
