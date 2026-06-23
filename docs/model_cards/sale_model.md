# Model Card — Sale Price

## Model details
- **Type:** LightGBM gradient-boosted trees (regression), persisted as a full sklearn `Pipeline` (preprocessing + model in one artifact)
- **Target:** `log1p(price)` — back-transformed with `expm1` at inference
- **Features (14):** `size_m2`, `rooms`, `bathrooms`, `floor_num`, `latitude`, `longitude`, `property_type`, `status`, `city`, `has_lift`, `exterior`, `new_development`, `district`, `neighborhood` — see `models/sale_feature_cols.json`
- **Excluded for leakage:** `price_per_m2` (derived from the target)
- **Artifacts:** `models/sale_best_model.pkl`, `models/sale_feature_cols.json`, `models/sale_district_defaults.json`
- **Serving code:** `src/airbnb_iip/models/sale.py` (`SalePredictor`)

## Intended use
Predicts a property's indicative total sale price (EUR) from listing attributes. This is the **sell-side anchor** for the Airbnb-vs-sell decision engine (`finance/scenarios.py`'s `npv_sell`) and the `/airbnb_vs_sell` endpoint.

**Not intended for:** a formal valuation, mortgage underwriting, or any market outside Madrid, Barcelona, and Málaga.

## Training data
- Source: Idealista sale listings, collected via `scripts/scrape_idealista.py` (Apify), JSONL output in `Data/sample/idealista_sample.jsonl` (schema documented in `docs/idealista_schema.md`)
- Split: 11,774 training rows / 2,944 test rows

## Performance (held-out test set)
| Metric | Value |
|---|---|
| R² | 0.859 |
| MAE | €48,543 |
| RMSE | €95,318 |
| MAPE | 14.4% |

## Limitations & caveats
- A sparse query (e.g. just `city` + `district` + `size_m2`) is filled from `sale_district_defaults.json` (district → city → global medians for location/context fields). Size-correlated fields (`rooms`, `bathrooms`, `floor_num`) are deliberately **not** auto-filled from district mode — they're left for the pipeline's own median imputation so they don't fight a caller-supplied size.
- District-level defaults are only as good as the underlying scrape's coverage in that district; sparsely-listed districts have noisier defaults.
- MAE of ~€48.5k is large in absolute terms for lower-value properties — relative error (MAPE 14.4%) is the more representative figure across the price range.

## Governance notes
- Every output feeding the Airbnb-vs-sell recommendation must carry the "indicative only, not financial advice" disclaimer (`agents/governance.py`).
- Sale value also sets the denominator for the gross-yield guardrail (`annual_gross_eur / npv_sell_eur` in `agents/governance.py`) — an under- or over-estimated sale price directly shifts whether that guardrail fires.
- Idealista's terms of service restrict scraping; re-scraping for a refreshed snapshot should go through the documented Apify pipeline, not ad-hoc scraping, and the snapshot date should be recorded alongside the data.
