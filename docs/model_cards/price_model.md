# Model Card — Airbnb Nightly Price

## Model details
- **Type:** LightGBM gradient-boosted trees (regression)
- **Target:** `log1p(price)` — back-transformed with `expm1` at inference
- **Selected via:** compared against OLS/Ridge baseline (Test R² = 0.602) and XGBoost/Random Forest; LightGBM had the best held-out error
- **Features:** 29, selected via VIF (multicollinearity) + Random Forest RFE — see `models/price_feature_cols.json` for the exact list and `docs/FEATURE_ENGINEERING.md` for the selection method
- **Artifacts:** `models/price_best_model.pkl`, `models/price_cat_encoders.pkl`, `models/price_feature_cols.json`
- **Serving code:** `src/airbnb_iip/models/price.py` (`PricePredictor`)

## Intended use
Predicts a property's Airbnb nightly rate (EUR) from listing attributes (location, capacity, amenities, host/review signals). Feeds:
- the Market Analyst agent's revenue projection and Airbnb-vs-sell recommendation
- the `/predict_price` and `/explain_price` API endpoints

**Not intended for:** setting a live listing's actual price without human review, or for any city/market outside Madrid, Barcelona, and Málaga.

## Training data
- Source: Inside Airbnb detailed listings snapshots for Madrid, Barcelona, Málaga (snapshot date `2025-09-14`, per `config/config.yaml`)
- Cleaning: `src/airbnb_iip/data/cleaning.py` — price parsing, bathroom/date standardisation, hierarchical neighbourhood-level imputation, IQR/percentile outlier capping, deduplication

## Performance (held-out test set)
| Metric | Value |
|---|---|
| R² | 0.803 |
| MAE | €34.6 / night |
| MdAPE | 16% |

SHAP global + per-prediction explainability is available (`models/12_shap_summary.png`, `13_shap_importance.png`, and `PricePredictor.explain()` for live per-prediction attribution).

## Limitations & caveats
- Unknown/missing input fields are imputed with training-set medians — predictions for very sparse specs (e.g. just `city`) are directionally reasonable but less precise than a fully-specified property.
- Neighbourhood is target-encoded from training data; an unseen neighbourhood name falls back to the global mean (silently — see governance note below).
- Trained on a single snapshot date; does not capture seasonal demand directly (seasonality is layered on separately in the finance engine, not in this model).
- SHAP values are in the model's `log1p(price)` output space, not EUR — they describe *relative* direction/magnitude of each feature's push, not an exact EUR contribution.

## Governance notes
- All outputs are indicative, not a pricing guarantee — every agent response carrying this model's prediction must include the "indicative only, not financial advice" disclaimer (`agents/governance.py`).
- A caller passing a district name (e.g. "Salamanca") instead of the finer Inside Airbnb neighbourhood (e.g. "Goya") into `neighbourhood_cleansed` will silently miss the target-encoding lookup and fall back to the global mean — no error is raised. Downstream UI should source the finer neighbourhood name, or this model should be extended to accept and resolve district-level input.
- Re-train when a new Inside Airbnb snapshot is adopted; update `config/config.yaml`'s `snapshot_date` and this card's metrics in the same change.
