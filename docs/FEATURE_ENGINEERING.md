# Feature Engineering & Preprocessing
**Airbnb Investment Intelligence Platform · IE × KPMG Capstone 2026**

*Owner: Almu (Task #5 — Documentation). Source of truth: repo `nnnuria/airbnb-investment-intelligence` @ `d55fe1a`. Every step below is traced to the file that implements it.*

---

## 0. Pipeline overview

Raw Inside Airbnb + Idealista files flow through cleaning → feature engineering → the Analytical Base Table (ABT) → the models:

```
listings.csv.gz ─┐
                 ├─► cleaning.py ─► listings_all_cities.parquet ─┐
calendar.csv.gz ─┘   (clean_listings)                            │
                                                                 ├─► abt.py (build_abt) ─► <city>_abt.parquet
features/ amenities.py · density.py · occupancy.py ──────────────┘            │
features/ seasonality.py  (city-level lookup, joined per scenario) ───────────┤
                                                                              ▼
                                              price model (LightGBM) → price_hat
                                                                              ▼
                                              occupancy model (Stage 2 — planned)

idealista sale jsonl ─► idealista_sale.py (tidy + clean) ─► sell model (LightGBM Pipeline)
```

**Design principle (from `docs/structure.md`):** all reusable logic lives in `src/airbnb_iip/`; notebooks only *call* it. The ABT is built by a single importable function, `build_abt(city)`.

---

## 1. Cleaning pipeline — `src/airbnb_iip/data/cleaning.py`

`clean_listings(df)` runs 12 ordered, pure-function steps:

| # | Step | Detail |
|---|---|---|
| 1 | Drop useless columns | `calendar_updated`, `neighbourhood` |
| 2 | Deduplicate | Drop rows identical across all columns except `id` / `listing_url` |
| 3 | Parse dates | `last_scraped`, `host_since`, `first_review`, `last_review`, etc. → `datetime` (bad values → `NaT`) |
| 4 | Type conversions | `parse_price` (strip `$`/`,` → float), `parse_rate` (`%` → [0,1]), `parse_bool` (`t`/`f` → bool); **rows with missing price are dropped** |
| 5 | Normalise host names | Multi-host separator → `&` |
| 6 | Reconcile bathrooms | Extract `bathrooms_number` from `bathrooms_text` (handles "half-bath" → 0.5), keep `bathrooms_description`, fill remaining NaN with mode |
| 7 | Impute | Host fields hierarchically (mode within `host_id`, then `host_neighbourhood`, then `'Unknown'`/0); beds & bedrooms via `(room_type, accommodates)` median with capacity fallback. **Price imputation is deliberately disabled** — missing prices are dropped instead |
| 8 | Review sentinels | `reviews_per_month` NaN → 0; missing review dates → sentinel `1800-01-01` (so temporal features can flag "no reviews") |
| 9 | Temporal features | `host_tenure_years`, `days_since_first_review`, `days_since_last_review`, `review_span_years` (sentinel dates → NaN) |
| 10 | Engineered features | `property_type_std` (6 standard categories), `host_response_rate_cat` / `host_acceptance_rate_cat` (ordered buckets), `price_cat` (city-adaptive quartiles low/medium/high/very_high), `description_length` |
| 11 | Drop URL/redundant cols | `listing_url`, `host_url`, thumbnails, `host_listings_count`, etc. |
| 12 | Filter price outliers | Drop `price <= 0` and `price >` 99.5th percentile |

**Property-type standardisation** (`standardize_property_type`) collapses 50+ raw `property_type` strings into: *Entire place · Private room · Shared room · Hotel / Hostel · Unique stay · Other*.

---

## 2. Amenity flags — `src/airbnb_iip/features/amenities.py`

The raw `amenities` column is a JSON-array string with inconsistent vendor naming ("Wifi" / "Wi-Fi" / "Wireless internet"). We parse it into **22 binary flags** via case-insensitive **regex**, not exact match.

- **Synonym bundling:** `has_balcony` matches `balcony|patio|terrace`; `has_self_checkin` matches `self check.in|smart lock|lockbox|keypad`.
- **Negation guards:** `has_pool` excludes `pool table`; `has_crib` excludes `crib.*table`.
- **Robustness:** malformed JSON rows → empty (all flags 0), never raises.

The 22 columns are: `has_pool, has_gym, has_parking, has_hot_tub, has_beach, has_view, has_ac, has_elevator, has_washer, has_dishwasher, has_workspace, has_self_checkin, has_pets, has_crib, has_private_entrance, has_balcony, has_bathtub, has_dryer, has_ev_charger, has_outdoor_space, has_long_term_ok, has_cleaning_service`.

> **Contract note:** the task brief asked for "top-30 amenities by frequency," but the team deliberately kept a **hand-curated 22** to (a) avoid redundant synonym columns and (b) preserve the exact input contract of the already-trained price model (`models/price_best_model.pkl`). 12 of the 22 are in the price model's selected feature set; the other 10 are retained for the downstream occupancy and comparables work. **This 22-column set is locked by `tests/test_amenities.py`** — changing it is a separate migration, not a refactor.

> ⚠️ **Divergence (new):** a second, **data-driven** amenity pipeline now exists in `notebooks/amenity_feature_engineering.ipynb` — it derives `amenity_count` + ~11 correlation-selected flags + Jaccard "bundle" flags and **back-patches them into `Data/processed/listings_all_cities.parquet`** (dropping prior `has_*`/`bundle_*` columns first). This is *separate* from the 22-flag regex set the price model was trained on. **Risk:** re-running that notebook with different thresholds silently changes the columns the price model and `segmentation.ipynb` consume. The two should be reconciled — or clearly scoped (regex-22 for the price model, data-driven set for segmentation) — before the demo.

---

## 3. Competitive density — `src/airbnb_iip/features/density.py`

`competitive_density_500m` = the number of *other* listings within a 500 m ground radius of each listing.

- **Method:** haversine `BallTree` (scikit-learn), `query_radius` with `r = 500 / 6,371,000` rad.
- **Per-city trees**, not one global tree — neighbours never bleed across cities.
- **Edge handling:** a listing excludes itself (`count − 1`); listings with NaN lat/lon get density 0 and aren't counted as neighbours.
- **Why 500 m / per-city (not per-district):** district borders are fuzzy and would create edge artefacts; a 500 m radius captures block-level saturation regardless of nominal district.

This column is part of the price model's input contract (lifted verbatim from the price notebook).

---

## 4. Seasonality multipliers — `src/airbnb_iip/features/seasonality.py`

Produces a **city-level lookup** `{city: {month: multiplier}}`, where 1.10 means listings run ~10% above the annual baseline that month. It is **not** a per-row feature — the finance module joins it in per scenario.

- **Availability-based, not price-based** — the calendar `price` column is 100% null (see `EDA.md §6`), so demand is inferred from `unavail_rate(month) = mean(1 − available)`.
- **60-day exclusion:** drops calendar rows within 60 days of the snapshot to remove the scrape-date artefact (near-term committed bookings masquerading as seasonal demand).
- **Normalised** so each city's annual mean = 1.0 (behaves like a price index). Cities/months with no signal fall back to a flat 1.0.

> ⚠️ See `EDA.md §6` for the unresolved **summer-seasonality conflict** between this availability method, the demo's hardcoded multipliers, and the external AirROI benchmark.

---

## 5. Occupancy model — `src/airbnb_iip/models/occupancy.py`

> **Replaced the San Francisco review-based estimator (2026-06).** The old model inferred bookings from review velocity and correlated only **≈ 0.13** with actual calendar occupancy on this dataset. It has been retired; `data/occupancy.py` now keeps only the lightweight calendar helpers.

Occupancy is now a **learned LightGBM model** whose target comes straight from Inside Airbnb **calendar** availability — a night is booked when `available == 'f'`:

```
occupancy_rate = booked_nights / calendar_nights      (per listing, first 120 calendar days from scrape)
nights_booked_year = round(occupancy_rate × 365)
```

- **Approach:** by-city models (Madrid / Barcelona / Málaga); chosen over a single global model on held-out R².
- **Calendar window:** the forward calendar's far tail is unreliable and the near-term weeks are always heavily booked, so the target uses each city's **first 120 days from its scrape date** (`--horizon-days 120`) — an equal-length, equal-horizon window that is **cross-city comparable**. Málaga uses a season-aligned **June-2025 scrape** (`--calendar-override`) so all three windows fall in spring/summer demand (raw means: Madrid 0.59 / Barcelona 0.58 / Málaga 0.52); a residual seasonal offset remains — see the model card.
- **Features (36):** neighbourhood target-encoding (mean occupancy), lat/lon, competitive density, nightly **price** (demand signal — the predicted price at serve time), `minimum_nights`, `instant_bookable`, capacity (accommodates / bedrooms / bathrooms / beds), `room_type` / `property_type_std` / `city`, the 22 amenity flags + `amenity_count`. **No review or host-quality features** — that severs the review→occupancy link and keeps every feature available for a hypothetical property.
- **Performance (held-out):** R² **0.29** (per-city Madrid 0.27 / Barcelona 0.33 / Málaga 0.29), MAE **0.18** (~66 nights/yr) vs a city-mean baseline R² of ≈0.00. Per-city and full detail in [`docs/model_cards/occupancy_model.md`](model_cards/occupancy_model.md).
- **Training:** `scripts/train_occupancy_model.py --horizon-days 120 --calendar-override Malaga=<june_calendar.csv.gz>` (target from the three `calendar.csv.gz` dumps, features from `listings_with_price_hat.parquet`, active-listing filter, neighbourhood TE fit on train only).
- **Serving:** `OccupancyPredictor` / `get_occupancy_predictor()`; the decision engine calls `engine.predict_occupancy(prop)`, so the dashboard, `/scenario`, `/estimate_occupancy`, and the optimisation agent all share one occupancy source.

> **External validation:** the model's central Madrid estimates (~40–50%) sit much closer to AirROI's independent Madrid market occupancy of **51.6%** than the old demo's ~15%.

The ABT's `estimated_occupancy_l365d` column is now filled from forward availability (`365 − availability_365`, also calendar-derived) by `data.occupancy.occupancy_l365d_from_availability` — Inside Airbnb's pre-computed value still wins when present.

---

## 6. ABT orchestration — `src/airbnb_iip/data/abt.py`

`build_abt(city)` is the single entry point that assembles the feature-engineered table for one city:

1. Load the cleaned multi-city parquet, filter to the city.
2. Append amenity flags (`add_amenity_flags`).
3. Append competitive density (`add_competitive_density`).
4. Append SF occupancy (`estimate_occupancy_l365d`) — **only if** `estimated_occupancy_l365d` isn't already present (Inside Airbnb's value wins when available).
5. Optionally write a versioned `Data/processed/<city>_abt_<date>.parquet`.

**Deliberately out of scope** for `build_abt`: re-cleaning (separate pipeline), price prediction (loading the LightGBM artefact would balloon import cost), and per-row seasonality (it's a city-level lookup applied later by the finance engine).

---

## 7. Price-model features — `notebooks/price_ml_model.ipynb` (generated by `scripts/make_price_notebook.py`)

**Target:** `log1p(price)`, clipped at the 99.5th percentile. **Selection:** a fast Random Forest ranks features by importance; everything **above the median importance** is kept → **~29 selected features**, served by `src/airbnb_iip/models/price.py`.

**Engineered/encoded inputs:**
- 22 amenity flags + `competitive_density_500m` (§2–3)
- `has_reviews` flag; review scores imputed to **city median** for listings with no reviews
- `is_long_stay` (≥28 nights), `is_weekly` (7–27), `has_licence`
- Categoricals **integer-encoded** (`property_type_std`, `host_response_time`); **`neighbourhood_cleansed` target-encoded** → `neighbourhood_target_enc` (unknown neighbourhoods → global mean)
- Numeric nulls → training-set median (computed once at serve time)

**Top SHAP drivers** (mean |SHAP|, log-price scale): `property_type_std` (0.216) · `accommodates` (0.133) · `minimum_nights` (0.126) · `neighbourhood_target_enc` (0.076) · `latitude` (0.064) · `bathrooms_number` (0.053) · `reviews_per_month` (0.047).

**Leakage exclusions (explicitly dropped):** `availability_30/60/90/365`, `estimated_occupancy_l365d`, `estimated_revenue_l365d`, `price_cat`, `days_since_last_review`, collinear review sub-scores, `scrape_id`, `source`. These are reserved (occupancy target) or are functions of price.

> ⚠️ **Reproducibility flag for the team:** the shipped artifact `price_best_model.pkl` is a verified **LightGBM** model (and the docs report a LightGBM/XGBoost comparison table — LightGBM Test R² 0.803). However, the committed generator `scripts/make_price_notebook.py` trains **HistGradientBoosting** (labelled "XGBoost-equivalent"), not LightGBM/XGBoost. The generator is therefore **stale** — re-running it will not reproduce the shipped model or the documented numbers. Document the artifact as LightGBM; the generator needs updating. (The *sell*-model generator is consistent.)

**Per-city breakdown of the pooled model (LightGBM):** Madrid R² 0.797 · Barcelona 0.819 · Málaga 0.752.

**Production model: city-specific (by-city).** A pooled model was compared against three city-specific LightGBM models on the same hold-out (the #2B empirical check); the by-city set was **selected** — Test R² **0.814** vs 0.810 pooled (RMSE €68.5, MAE €30.8). The by-city models (`CityPricePredictor`, `models/price_city_{madrid,barcelona,malaga}_model.pkl`) serve the **decision engine** — scenario, NPV, and the Airbnb-vs-sell recommendation (`engine.py` via `get_city_price_predictor()`). The pooled `PricePredictor` (`price_best_model.pkl`) still backs the standalone `/predict_price`, `/explain_price`, and `/estimate_revenue` endpoints (`api/dependencies.py` → `get_price_predictor()`).

---

## 8. Sell-model features — `notebooks/sell_price_model.ipynb` + `src/airbnb_iip/data/idealista_sale.py`

**Loader/cleaner** (`idealista_sale.py`): maps raw Idealista JSONL → tidy snake_case, parses Spanish floor codes (`bj`→0, `en`→0.5, `ss/st`→−1), dedupes on `property_code`, and filters implausible rows (price €20k–5M, size 15–1,000 m², €/m² 300–20k).

**Target:** `log1p(price)`. Persisted as a **single fitted sklearn `Pipeline`** (`models/sale_best_model.pkl`) so serving never re-implements preprocessing:
- **Numeric** (`size_m2, rooms, bathrooms, floor_num, latitude, longitude`): median impute (+ scale for the linear baseline)
- **Low-cardinality** (`property_type, status, city, has_lift, exterior, new_development`): one-hot
- **High-cardinality** (`district, neighborhood`): cross-fitted **target encoding**
- **Excluded:** `price_per_m2` (= price ÷ size → leakage)

**District-level imputation** (`models/sale_district_defaults.json`, built by `scripts/make_sale_district_defaults.py`): a sparse `(city, district, size)` query is enriched with district-median lat/lon and the modal neighbourhood so location signal isn't washed out. Size-correlated counts (rooms/bathrooms/floor) are *not* imputed from district mode (would fight a user-supplied size).

**Performance (LightGBM):** R² **0.868**, MAE **€47,272**, MAPE **13.9%** (held-out test n=2,944; train n=11,774 — per `models/sale_feature_cols.json` and `docs/model_cards/sale_model.md`); per-district predicted €/m² correlates ≥0.97 with actuals.

---

## 9. Configuration & governance note

All tunable assumptions are *intended* to live in config, for auditability:

| Where | Status |
|---|---|
| `src/airbnb_iip/config.py` | **The one actually imported** (`OCCUPANCY`, `FINANCE`, taxes, CGT brackets) |
| `config/config.yaml` | Duplicates the values but **is not loaded by any code** — effectively dead |
| `api/routers/revenue.py` | Hardcodes a `0.55` occupancy fallback (a *third* place the assumption lives) |

> **Governance gap to flag:** `docs/structure.md` claims `config.py` "loads `config.yaml`" — it does not. Occupancy/finance assumptions live in 2–3 places that can silently drift. Consolidating to a single source is a cheap, high-credibility fix for the governance story.

---

## 10. Market segmentation feature (`Segment_Name`)

`notebooks/segmentation.ipynb` (spec: `docs/cluster_analysis_method.md`) produces a market-segment label used as an additional grouping/benchmarking feature for the models.

- **Input:** `Data/processed/listings_all_cities.parquet`, **attributes only** — targets and leakage (`price`, `estimated_occupancy_l365d`, reviews/availability) are excluded from clustering.
- **Outlier split:** rows with >1 IQR-outlier column (~10%, per-city IQR bounds) are carved off as a manual **Ultra / Extreme** segment and never clustered.
- **Final features (verified):** a **lean protected core** — `log1p_accommodates`, `log1p_bedrooms`, `log1p_bathrooms_number`, `review_scores_rating`, `is_entire_place`, `is_private_room`, `is_central` — reached via the CV elimination loop (which strips the low-signal amenity flags), then clustered in **PCA(0.9) space**. City identity (`is_madrid`/`is_barcelona`) is excluded by design (it's a model feature; including it drops silhouette ≈0.36 → 0.18).
- **k & quality (executed end-to-end, no errors):** silhouette peaks at the trivial k=2, so k is constrained to ≥4; validation picks **k=4, silhouette 0.365**, largest segment **27.5% of clean / 24.8% of total** (vs the prior 56.7% mega-cluster). DBSCAN check (63 clusters, 31% noise, sil 0.27) confirms K-means.
- **Output (actual):** `Segment_Name` named by **structure** — Budget private rooms (€75) · Central entire homes (€159) · Non-central entire homes (€162) · Premium entire homes (€239) · Ultra/Extreme (€262); `Cluster_Final` → `Data/processed/listings_segmented.parquet` (42,811 rows, regenerated).
- **Validation:** PCA-space elbow + silhouette sweep; DBSCAN sanity check with **auto-`eps`** from the k-NN knee.

> Caveats: (1) consumes `has_*`/`bundle_*` columns from the amenity notebook (§2) — re-run both together; (2) the two ~€160 tiers (**Central vs Non-central entire homes**) are the same price but split by location — a real, defensible distinction, hence the structural names; `k=3` is the coarser price-only alternative. See `NOTEBOOK_IMPROVEMENTS.md`.

---

## 11. Summary — feature inventory by stage

| Stage | Features produced | Implemented in |
|---|---|---|
| Cleaning | price/rate/bool parsing, bathrooms, beds/bedrooms, temporal, `property_type_std`, `price_cat`, rate buckets, `description_length` | `data/cleaning.py` |
| Amenities | 22 binary flags | `features/amenities.py` |
| Spatial | `competitive_density_500m` | `features/density.py` |
| Occupancy | `estimated_occupancy_l365d` | `data/occupancy.py` |
| Seasonality | city × month multipliers | `features/seasonality.py` |
| Price model | `has_reviews`, stay-length flags, target/integer encodings, ~29 selected | `make_price_notebook.py` → `models/price.py` |
| Sell model | floor parsing, OHE + target-encoding, district imputation | `idealista_sale.py` → `models/sale.py` |
| Segmentation | `Segment_Name`, `Cluster_Final` | `segmentation.ipynb` → `listings_segmented.parquet` |

**Three reproducibility/governance items to hand back to the team:** the stale price generator (§7), the multi-source config (§9), and the amenity-pipeline divergence (§2).
