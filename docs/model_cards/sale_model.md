# Model Card — Sale Price

## Model details
- **Type:** LightGBM gradient-boosted trees (regression), persisted as a full sklearn `Pipeline` (preprocessing + model in one artifact)
- **Target:** `log1p(price)` — back-transformed with `expm1` at inference
- **Features (22):**
  - *Structural/location (14):* `size_m2`, `rooms`, `bathrooms`, `floor_num`, `latitude`, `longitude`, `property_type`, `status`, `city`, `has_lift`, `exterior`, `new_development`, `district`, `neighborhood`
  - *Amenity flags (8), text-mined:* `has_pool`, `has_parking`, `has_terrace`, `has_ac`, `has_garden`, `has_storage`, `has_heating`, `has_balcony`
  - see `models/sale_feature_cols.json`
- **Excluded for leakage:** `price_per_m2` (derived from the target)
- **Artifacts:** `models/sale_best_model.pkl`, `models/sale_feature_cols.json`, `models/sale_district_defaults.json`; previous version retained as `models/sale_best_model_legacy.pkl` + `sale_feature_cols_legacy.json`
- **Serving code:** `src/airbnb_iip/models/sale.py` (`SalePredictor`)
- **Retrain:** `python scripts/train_sale_model.py` (backs up the live model to `*_legacy.*` first); `notebooks/sell_price_model.ipynb` is the narrated reference

## Intended use
Predicts a property's indicative total sale price (EUR) from listing attributes. This is the **sell-side anchor** for the Airbnb-vs-sell decision engine (`finance/scenarios.py`'s `npv_sell`) and the `/airbnb_vs_sell` endpoint.

**Not intended for:** a formal valuation, mortgage underwriting, or any market outside Madrid, Barcelona, and Málaga.

## Training data
- Source: Idealista sale listings, collected via `scripts/scrape_idealista.py` (Apify), JSONL output in `Data/sample/idealista_sample.jsonl` (schema documented in `docs/idealista_schema.md`)
- Split: 11,774 training rows / 2,944 test rows
- Amenity flags are mined from the free-text `description` + `suggestedTexts.title` (`src/airbnb_iip/features/sale_amenities.py`): Spanish keyword regexes with negation handling (`sin piscina` → 0) and robustness to the scrape's baked-in accent corruption (`balcón` → `balc�n`).

## Performance (held-out test set)
| Metric | Value |
|---|---|
| R² | 0.868 |
| MAE | €47,272 |
| RMSE | €92,041 |
| MAPE | 13.9% |

Amenity uplift sanity check (counterfactual on a Madrid/Chamberí 90 m² flat): pool +3.9%, parking +4.7%, A/C +4.5%, balcony +5.3%, terrace +1.3%; garden/storage/heating land near zero (ubiquitous or peripheral signals carry no premium after controlling for location/size).

## Limitations & caveats
- A sparse query (e.g. just `city` + `district` + `size_m2`) is filled from `sale_district_defaults.json` (district → city → global medians for location/context fields). Size-correlated fields (`rooms`, `bathrooms`, `floor_num`) are deliberately **not** auto-filled from district mode — they're left for the pipeline's own median imputation so they don't fight a caller-supplied size.
- District-level defaults are only as good as the underlying scrape's coverage in that district; sparsely-listed districts have noisier defaults.
- MAE of ~€47k is large in absolute terms for lower-value properties — relative error (MAPE 13.9%) is the more representative figure across the price range.
- **Amenity flags are a noisy lower bound.** A flag of 0 means "not advertised in the listing text", not "verified absent" — so the learned premium reflects *advertised* amenities. At serving, only amenities with a UI toggle are passed through (`has_pool`, `has_parking`, `has_ac`, `has_balcony`, and `has_garden` ← outdoor-space); `has_terrace`/`has_storage`/`has_heating` have no toggle and serve at the 0 baseline.

## Governance notes
- Every output feeding the Airbnb-vs-sell recommendation must carry the "indicative only, not financial advice" disclaimer (`agents/governance.py`).
- Sale value also sets the denominator for the gross-yield guardrail (`annual_gross_eur / npv_sell_eur` in `agents/governance.py`) — an under- or over-estimated sale price directly shifts whether that guardrail fires.
- Idealista's terms of service restrict scraping; re-scraping for a refreshed snapshot should go through the documented Apify pipeline, not ad-hoc scraping, and the snapshot date should be recorded alongside the data.
