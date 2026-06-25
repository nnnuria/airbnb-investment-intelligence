# Model Card — Airbnb Occupancy (calendar-based)

## Model details
- **Type:** LightGBM gradient-boosted trees (regression), **by-city** (one model per Madrid / Barcelona / Málaga)
- **Target:** `occupancy_rate` ∈ [0, 1] — the share of a listing's forward-calendar nights that are **booked**, where Inside Airbnb's `available == 'f'` is treated as a booked night
- **Selected via:** compared a single **global** model (city as a feature) against a **by-city** model set; by-city won on held-out R² (0.293 vs 0.281) and is served in production
- **Artifacts:** `models/occupancy_city_{madrid,barcelona,malaga}_model.pkl`, `models/occupancy_encoders.pkl`, `models/occupancy_feature_cols.json` (a single `occupancy_best_model.pkl` is written instead when the global approach wins)
- **Serving code:** `src/airbnb_iip/models/occupancy.py` (`OccupancyPredictor`, `get_occupancy_predictor`)
- **Training code:** `scripts/train_occupancy_model.py`

## Why this replaced the San Francisco model
The previous occupancy estimate was the **San Francisco review-based model** (`nights ≈ reviews / review_rate × length_of_stay`, capped at 70%). On this dataset that estimate correlated only **≈ 0.13** with the actual calendar-derived occupancy — it barely tracked real availability. The calendar signal it ignored validates well against the listings' own `availability_365` field (corr ≈ 0.72). The learned model lifts occupancy explanatory power from a city-mean baseline of **R² ≈ 0.00** to **R² = 0.29** using only features available for a hypothetical property.

## Intended use
Predicts a property's **annual occupancy rate** from listing attributes (location, capacity, nightly price, amenities, room type, min-nights policy). Feeds:
- the decision engine (`engine.predict_occupancy` → `compute_scenario`): `occupancy_rate_annual`, `nights_booked_year`, and every downstream revenue / NOI / NPV / IRR / payback figure and the Airbnb-vs-sell recommendation
- the `/estimate_occupancy`, `/scenario`, and `/airbnb_vs_sell` API endpoints, and the optimisation agent's projected-nights fallback

**Not intended for:** guaranteeing a specific listing's bookings, or any market outside Madrid, Barcelona, and Málaga. All outputs are **indicative**.

## Training data
- **Target source:** Inside Airbnb **calendar** dumps for the three cities (`calendar.csv.gz`); `occupancy_rate = mean(available == 'f')` over each city's **first 120 calendar days from its scrape date** (`--horizon-days 120`). An equal-length, equal-horizon window per city neutralises the *near-term-booking-horizon* artifact (the next few weeks are always heavily booked while far-future dates read as available), so occupancy is **cross-city comparable**. The far-future, unreliable calendar tail is excluded.
- **Per-city window:** Madrid 2026-03-25→07-22, Barcelona 2026-03-21→07-18, Málaga 2025-06-29→10-26 — all 120 days. To keep the *season* roughly aligned (not just the horizon length), Málaga uses a **June-2025 scrape** (`--calendar-override Malaga=…`) so its window falls in summer→early-autumn, close to Madrid/Barcelona's spring→summer windows. (A September Málaga scrape would have placed its window in autumn→winter, its off-season, exaggerating the gap — see the caveat below.)
- **Feature source:** `Data/processed/listings_with_price_hat.parquet`, joined to the calendar by `id == listing_id`.
- **Sample:** 31,196 active listings after filtering — `has_availability == True`, ≥ 60 days of calendar within the window, dropping the fully-blocked (`rate ≥ 0.999`) cohort. Mean active occupancy **0.48** (Madrid 0.59, Barcelona 0.58, Málaga 0.52 at the raw-target level — now well aligned).
- **Split:** 80/20, stratified by city (n_train 24,956 / n_test 6,240). Neighbourhood target-encoding is fit on the train split only for honest metrics.

## Features (36)
No review counts/scores or host-quality fields are used — that deliberately severs the review→occupancy link the old model relied on, and keeps every feature available for a *hypothetical* property.
- **Location:** `neighbourhood_target_enc` (mean occupancy per neighbourhood), `latitude`, `longitude`, `competitive_density_500m`
- **Demand/price:** `price` (the model-predicted nightly rate at serve time), `minimum_nights`, `instant_bookable`
- **Capacity:** `accommodates`, `bedrooms`, `bathrooms_number`, `beds`
- **Type:** `room_type`, `property_type_std`, `city` (label-encoded)
- **Amenities:** the 22 `has_*` flags + `amenity_count`

**Top drivers (by gain):** price, competitive density, longitude/latitude, neighbourhood encoding, minimum-nights, amenity count. Behaviour is economically coherent — a higher nightly price and a longer minimum-nights policy both push occupancy down.

## Performance (held-out test set)
| Scope | n_test | R² | MAE (rate) | MAE (nights/yr) |
|---|---:|---:|---:|---:|
| **Combined (by-city)** | 6,240 | **0.293** | **0.181** | **66.2** |
| Madrid | 2,767 | 0.266 | 0.187 | 68.2 |
| Barcelona | 1,923 | 0.329 | 0.184 | 67.2 |
| Málaga | 1,550 | 0.290 | 0.168 | 61.5 |
| Global model | 6,240 | 0.281 | 0.187 | 68.3 |
| City-mean baseline | 6,240 | ≈ 0.00 | 0.230 | — |

*(Uniform 120-day-per-city window, with Málaga on a season-aligned June-2025 scrape. Aligning the season lifted the Málaga model from R² 0.19 — when it was trained on the autumn/winter window — to **0.29**, and balanced the three cities. For reference, alternative windows scored: full forward year R² 0.273; spring-only absolute cutoff at 2026-05-31 R² 0.376 — but that cutoff inflated Madrid/Barcelona by extrapolating a 2-month spring rate to a full year and was not cross-city comparable, so it was rejected.)*

Per-prediction SHAP attribution is available via `OccupancyPredictor.explain()`.

## Limitations & caveats
- **Blocked ≠ booked.** `available == 'f'` covers both guest bookings and host-blocked dates, so occupancy is an **upper bound** on true bookings. We mitigate the worst case by dropping fully-blocked listings, and clip predictions to `[0.02, 0.95]`.
- **Residual seasonal weighting + annualisation.** The 120-day windows are now season-aligned (all three in spring/summer demand), so the raw target means are close (Madrid 0.59 / Barcelona 0.58 / Málaga 0.52) and the cross-city comparison is fair. There is still a ~2–3 month seasonal *offset* (Málaga's window is Jun–Oct vs Madrid/Barcelona's Apr–Jul) and the cities are scraped in different years (Málaga 2025, Madrid/Barcelona 2026), so year-over-year demand shifts are not controlled. The engine also annualises as `nights = rate × 365`, treating each warm-season 120-day window as representative of the whole year, which slightly **overstates annual occupancy for all three** (the window is demand-weighted high). A true seasonally-balanced annual figure would require a full, season-matched 12-month window per city.
- **Forward calendar.** The target is *future* availability, not realised history. Modest R² (0.29) reflects genuine irreducible variance in forward occupancy from structural features alone; it sits between the noisier full-year window (0.27) and the seasonally-inflated spring cutoff (0.38).
- **Feature staleness.** Listing features come from the 2025-09 processed snapshot while the Madrid/Barcelona calendars are a 2026-03 forward window (Málaga is closer to contemporaneous). Structural features are stable; nightly price may drift slightly.
- **Unseen neighbourhoods** fall back to the city/global mean occupancy; unknown categorical values map to `-1`.
- **Indicative only** — never present as a guaranteed projection; every agent response carrying it includes the standard disclaimer.

## Governance / reproducibility
- Retrain when a new Inside Airbnb snapshot is adopted:
  `python scripts/train_occupancy_model.py --calendar-dir <dir> --horizon-days 120 --calendar-override Malaga=<june_malaga_calendar.csv.gz>`
  (expects `Madrid/ Barcelona/ Malaga/` calendar dumps under `--calendar-dir`; `--horizon-days N` uses each city's first N days for a cross-city-comparable window; `--calendar-override Folder=PATH` swaps one city's calendar file — here a season-aligned June Málaga scrape; `--cutoff-date` uses an absolute date instead; omitting all three uses the full forward window). Existing artefacts are backed up to `*_legacy.*` before being overwritten; the window and any overrides are recorded in `occupancy_feature_cols.json` under `occupancy_horizon_days` / `calendar_cutoff` / `calendar_overrides`.
- Update `config/config.yaml` `snapshot_date` and this card's metrics in the same change.
