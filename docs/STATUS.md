# Project Status
**Airbnb Investment Intelligence Platform · IE × KPMG Capstone 2026**

Last updated: **2026-06-24**

---

## Current state snapshot

Legend: ✅ Done · 🔄 In progress · ⬜ Not started

### Infrastructure & setup
| Component | Status | Notes |
|---|---|---|
| Repo structure & `pyproject.toml` | ✅ | Installable package: `pip install -e .` |
| GitHub Actions CI (pytest) | ✅ | Runs on push to main |
| `.env.example` + secrets handling | ✅ | Gemini API key + Apify token templated |
| Docker / docker-compose | ⬜ | Not yet implemented |

### Data acquisition
| Component | Status | Notes |
|---|---|---|
| Inside Airbnb — Madrid | ✅ | `listings`, `calendar`, `reviews`, `neighbourhoods` |
| Inside Airbnb — Barcelona | ✅ | Replacement snapshot (12 Jun 2025); price ~79% populated |
| Inside Airbnb — Málaga | ✅ | Downloaded and validated |
| Idealista scraper (sale + rent) | ✅ | `scripts/scrape_idealista.py`; 6 jobs × 2,500 cap; JSONL output |
| Regulatory corpus | ✅ | Madrid, Barcelona, Málaga + national rules in `docs/regulatory/` |

### Data engineering
| Component | Status | Notes |
|---|---|---|
| Cleaning pipeline (`cleaning.py`) | ✅ | Price, baths, beds, imputation, outlier capping, deduplication |
| Feature engineering (amenity dummies) | ✅ | `src/airbnb_iip/features/amenities.py` + `amenity_feature_engineering.ipynb` |
| Occupancy estimator (SF model) | ✅ | `src/airbnb_iip/data/occupancy.py` |
| Density features (BallTree) | ✅ | `src/airbnb_iip/features/density.py` |
| Seasonality features | ✅ | `src/airbnb_iip/features/seasonality.py` |
| `build_abt(city)` | ✅ | `src/airbnb_iip/data/abt.py` |
| Idealista sale price data | ✅ | `src/airbnb_iip/data/idealista_sale.py`; district defaults in `models/sale_district_defaults.json` |

### Exploratory analysis
| Component | Status | Notes |
|---|---|---|
| Cross-city comparison | ✅ | `notebooks/cross_city_comparison.ipynb` — 7 sections, city scorecard, investment atlas |
| Segmentation | ✅ | `notebooks/segmentation.ipynb` — K-means; 5 structural segments named |
| Sell price EDA | ✅ | `notebooks/sell_price_eda.ipynb` |
| Amenity feature engineering | ✅ | `notebooks/amenity_feature_engineering.ipynb` |

### Modelling
| Component | Status | Notes |
|---|---|---|
| Feature selection (VIF + RFE) | ✅ | ~30 features selected |
| Price model — general (OLS baseline) | ✅ | Linear Regression + Ridge; benchmark |
| Price model — general (LightGBM) | ✅ | Test R² = 0.810, RMSE = €69.5, MAE = €31.6; `models/price_best_model.pkl` |
| Price model — city-specific (production) | ✅ | 3× LightGBM; R² = 0.814, RMSE = €68.5, MAE = €30.8; `models/price_city_*.pkl` |
| Price model approach comparison | ✅ | `notebooks/price_ml_model_comparison.ipynb` — city-specific selected |
| Segmentation (K-means) | ✅ | 5 structural segments; `models/price_cluster_*.pkl` |
| Sale price model | ✅ | Regression on Idealista data; `models/sale_best_model.pkl` |
| Sentiment model (NB + TF-IDF) | ✅ | Multilingual; scores in `Data/processed/<city>_sentiment.parquet` |
| MLflow experiment tracking | ⬜ | Models saved locally; MLflow logging not wired |

### Primary flow — Airbnb-vs-sell decision engine
| Component | Status | Notes |
|---|---|---|
| Finance scenarios (revenue + sell) | ✅ | `src/airbnb_iip/finance/scenarios.py` — P10/P50/P90 bands |
| Cost model (Spain-specific) | ✅ | `src/airbnb_iip/finance/costs.py` — IBI, IRPF, tourist tax, Airbnb fee, etc. |
| Unit tests for finance engine | ✅ | `tests/test_finance_scenarios.py` · `tests/test_finance_costs.py` |

### AI / agent layer
| Component | Status | Notes |
|---|---|---|
| Market Analyst agent | ✅ | `src/airbnb_iip/agents/market_analyst.py` — narrates comparison, surfaces SHAP |
| Regulatory agent (RAG / FAISS) | ✅ | `src/airbnb_iip/agents/regulatory.py` — retrieval over `docs/regulatory/` corpus |
| Comparables agent (KNN) | ✅ | `src/airbnb_iip/agents/comparables.py` |
| Governance layer | ✅ | `src/airbnb_iip/agents/governance.py` — guardrails, disclaimers, model cards |
| LangGraph coordinator | ⬜ | Individual agents done; LangGraph routing not yet wired |

### API
| Component | Status | Notes |
|---|---|---|
| FastAPI — `/predict_price` | ✅ | `api/routers/predict.py` |
| FastAPI — `/estimate_occupancy` | ✅ | `api/routers/predict.py` |
| FastAPI — `/estimate_revenue` | ✅ | `api/routers/revenue.py` |
| FastAPI — `/airbnb_vs_sell` | ✅ | `api/routers/revenue.py` |
| FastAPI — `/optimise` | 🔄 | `api/routers/optimise.py` — stub |

### UI
| Component | Status | Notes |
|---|---|---|
| Streamlit — Home (landing page) | ✅ | `app/Home.py` — feature overview + CTAs |
| Streamlit — New Analysis | ✅ | `app/pages/1_New_Analysis.py` — full input form, scenario output, agents |
| Streamlit — Chat | ✅ | `app/pages/2_Chat.py` |
| Streamlit — Saved Properties | ✅ | `app/pages/3_Saved_Properties.py` |
| Streamlit — Optimisation | ✅ | `app/pages/4_Optimisation.py` |
| Amenity bundles + district inputs | ✅ | Categorised amenity presets + per-city district dropdowns |
| Comparables section | ✅ | Wired into New Analysis page |
| Governance / regulatory guardrails | ✅ | Wired into New Analysis page |
| Pre-cached demo outputs (JSON) | ⬜ | Required before KPMG demo |

### MLOps
| Component | Status | Notes |
|---|---|---|
| Docker / docker-compose | ⬜ | Not yet implemented |
| Evidently drift report | ⬜ | |
| MLflow experiment tracking | ⬜ | |

---

## Changelog

Entries are newest-first.

**2026-06-24** — ML approach comparison → city-specific models promoted to production
- `notebooks/price_ml_model_comparison.ipynb` — systematic comparison of 5 modelling approaches; city-specific LightGBM wins (R² 0.814, RMSE €68.5, MAE €30.8)
- `src/airbnb_iip/models/price.py` — `CityPricePredictor` added; `app/components/engine.py` updated to use city-specific models
- `reports/price_approach_comparison.png` — comparison chart

**2026-06-23** — City districts and amenity bundles added to New Analysis page
- `app/pages/1_New_Analysis.py` — per-city district dropdowns, 5 categorised amenity bundles, persistent state tracking

**2026-06-22** — Governance and regulatory agents wired into UI
- `src/airbnb_iip/agents/governance.py` + `regulatory.py` — implemented and connected
- Financial guardrails integrated into New Analysis page

**2026-06-21** — Market comparables section added to New Analysis page
- `src/airbnb_iip/agents/comparables.py` — KNN comparables agent
- Comparables output section added to `app/pages/1_New_Analysis.py`

**2026-06-20** — Real ML engine wired into the app
- `app/components/engine.py` — replaced mock with real price + sale models
- `src/airbnb_iip/agents/market_analyst.py` — implemented

**2026-06-19** — Streamlit app built (4 pages)
- `app/Home.py`, `app/pages/`, `app/components/` — full branded Streamlit app

**2026-06-18** — FastAPI model services built
- `api/main.py`, `api/routers/` — predict, revenue, optimise (stub) endpoints

**2026-06-17** — Finance module implemented
- `src/airbnb_iip/finance/scenarios.py` + `costs.py` — full Spain-specific cost model; P10/P50/P90 revenue bands
- Unit tests added

**2026-06-16** — Sale price model + Idealista sale data pipeline
- `notebooks/sell_price_eda.ipynb` + `sell_price_model.ipynb`
- `src/airbnb_iip/data/idealista_sale.py` + `models/sale_best_model.pkl`

**2026-06-15** — Segmentation and cluster-specific price models
- `notebooks/segmentation.ipynb` — K-means; 5 structural segments named
- `models/price_cluster_*.pkl` — per-segment price models

**2026-06-14** — Price ML model complete
- `notebooks/price_ml_model.ipynb` — LightGBM selected; Test R² = 0.803, SHAP global + per-prediction
- Artefacts saved: `models/price_best_model.pkl`, `price_feature_cols.json`, `price_cat_encoders.pkl`

**2026-06-14** — Investment decision framework documented
- `docs/INVESTMENT_DECISION_FRAMEWORK.md` — NPV comparison structure, cost model, ML model plan

**2026-06-13** — Idealista scraper merged
- `scripts/scrape_idealista.py` + `src/airbnb_iip/data/scrapers/idealista.py`

**~2026-06-09** — Regulatory corpus added
- `docs/regulatory/` — official STR rules for Madrid, Barcelona, Málaga, national level

**~2026-06-08** — Cleaning pipeline + EDA
- `src/airbnb_iip/data/cleaning.py`, per-city EDA notebooks, cross-city comparison

**~2026-06-08** — Repo scaffolded
- Package skeleton, `pyproject.toml`, GitHub Actions CI
