# Rental Revenue Model — Results (v1)

**Target:** gross monthly rent (EUR); annual = ×12. **Scope:** Madrid & Barcelona city proper,
**2,595** cleaned/deduplicated listings from the ~Sept-2022 idealista scrape. **Approach:**
interpretable Ridge baseline + XGBoost, held-out 20% test, 5-fold CV. See
`notebooks/02_rental_revenue_model.ipynb`.

## Headline metrics (held-out test)

| Model | CV MAE | Test MAE | RMSE | MAPE | R² |
|---|---|---|---|---|---|
| Ridge (baseline) | €455 | €498 | €867 | 23.2% | 0.52 |
| **XGBoost (best)** | **€417** | **€447** | **€748** | **21.0%** | **0.643** |

Per-city (XGBoost) is broadly similar across Madrid and Barcelona. Typical prediction lands within
~€450/month of actual — good enough to **rank/compare** properties and feed the rent/Airbnb/sell
layer, not yet a precise valuation tool.

## What drives rent (permutation importance, € MAE impact)

| Feature | Impact | Note |
|---|---|---|
| `sq_m` | **486** | Size dominates, as expected |
| **`neighborhood`** | **133** | **Engineered from the `title` string — the #2 driver** |
| municipality | 11 | ~constant within city scope |
| property_type | 6 | |
| n_rooms | 6 | Collinear with size |
| city | 2 | |

The **neighbourhood feature parsed out of `title` (97.5% coverage) is the single biggest win** beyond
raw size — it captures the Salamanca-vs-Carabanchel / Eixample-vs-Nou-Barris gap that drives rent.

## Sanity-check predictions (2022 levels)

| City | m² | Rooms | Neighbourhood | Pred. €/mo | €/yr |
|---|---|---|---|---|---|
| Madrid | 80 | 2 | Trafalgar | 1,723 | 20,680 |
| Madrid | 45 | 1 | Lavapiés-Embajadores | 956 | 11,470 |
| Barcelona | 90 | 3 | La Dreta de l'Eixample | 2,102 | 25,224 |
| Barcelona | 60 | 2 | El Raval | 1,221 | 14,652 |

Believable for 2022; today's market is likely ~20-30% higher (the vintage gap).

## Limitations & next steps

- **2022 vintage** — absolute levels are stale. When fresh data arrives, call
  `model.evaluate_against_live(df)`; its `mean_signed_error` quantifies the drift.
- **Thin features** — no bathrooms, amenities, floor, condition. Biggest accuracy unlock is a richer
  scrape / idealista API pull.
- **Under-predicts the luxury tail** (see residual plot) — sparse high-end training data.
- **City proper only** — metro towns excluded for v1 (one flag flip: `build_model_frame(scope="metro")`).
- Optional: geocode `street` (68%) for distance-to-centre; enrich with sale-data neighbourhood €/m².

## Artifacts
`src/features.py`, `src/model.py`, `models/rent_model.pkl`, `reports/model_metrics.json`,
`notebooks/02_rental_revenue_model.ipynb`, figures `reports/figures/04-07`.
