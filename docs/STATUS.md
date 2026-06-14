# Project Status
**Airbnb Investment Intelligence Platform · IE × KPMG Capstone 2026**

This document tracks the current state of every component and logs what changed, when.
Update it whenever you merge something significant — one bullet per meaningful change, dated.

---

## Current state snapshot

Legend: ✅ Done · 🔄 In progress · ⬜ Not started

### Infrastructure & setup
| Component | Status | Notes |
|---|---|---|
| Repo structure & `pyproject.toml` | ✅ | Installable package: `pip install -e .` |
| GitHub Actions CI (pytest) | ✅ | Runs on push to main |
| `.env.example` + secrets handling | ✅ | LLM key + Apify token templated |
| Docker / docker-compose | ⬜ | Scaffolded in plan; not yet implemented |

### Data acquisition
| Component | Status | Notes |
|---|---|---|
| Inside Airbnb — Madrid | ✅ | `listings`, `calendar`, `reviews`, `neighbourhoods` |
| Inside Airbnb — Barcelona | ✅ | Replacement snapshot (12 Jun 2025); price ~79% populated |
| Inside Airbnb — Málaga | ✅ | Downloaded and validated |
| External sale price €/m² by district | ⬜ | Needed for sell-side of Airbnb-vs-sell comparison |
| Idealista scraper (sale + rent) | ✅ | `scripts/scrape_idealista.py`; 6 jobs × 2,500 cap; JSONL output |
| Regulatory corpus | ✅ | Madrid, Barcelona, Málaga + national rules in `docs/regulatory/` |

### Data engineering
| Component | Status | Notes |
|---|---|---|
| Cleaning pipeline (`cleaning.py`) | ✅ | Price, baths, beds, imputation, outlier capping, deduplication; `host_name` regex bug fixed |
| Feature engineering | 🔄 | Amenity dummies, host tenure, neighbourhood aggregates — partially in notebooks |
| Occupancy estimator (SF model) | ⬜ | Defined in plan; not yet implemented as a function |
| `build_abt(city)` — unified ABT | ⬜ | Cleaning done; full pipeline function not yet wired |
| Spatial join (listings ↔ polygons) | ⬜ | |
| External data join (€/m²) | ⬜ | Blocked on external data acquisition above |
| Schema validation + CI tests | ⬜ | |

### Exploratory analysis
| Component | Status | Notes |
|---|---|---|
| Madrid EDA | ✅ | `notebooks/madrid/` — cleaning, analysis, sentiment |
| Barcelona EDA | ✅ | `notebooks/barcelona/` — cleaning, analysis, sentiment |
| Málaga EDA | ✅ | `notebooks/malaga/` — cleaning, analysis, sentiment |
| Cross-city comparison | ✅ | `notebooks/cross_city_comparison.ipynb` — 7 sections, city scorecard, investment atlas, radar chart |
| Airbnb-vs-sell EDA (revenue vs. sale value by district) | ⬜ | The primary flow narrative |

### Modelling
| Component | Status | Notes |
|---|---|---|
| Feature selection (VIF + RFE) | ✅ | VIF computed; RF-based importance threshold used for final selection; ~30 features kept |
| Price model (OLS baseline) | ✅ | Linear Regression + Ridge; Test R² = 0.602; saved as benchmark |
| Price model (XGBoost/LightGBM + SHAP) | ✅ | **LightGBM selected** — Test R² = 0.803, MAE = €34.6/night, MdAPE = 16%; SHAP global + per-prediction; model saved to `models/price_best_model.pkl` |
| Segmentation (K-means / hierarchical / DBSCAN) | ⬜ | |
| Sentiment model (NB + TF-IDF) | ✅ | Multilingual; scores in `data/processed/<city>_sentiment.parquet` |
| Demand / seasonality (Prophet/ARIMA) | ⬜ | Calendar seasonality multipliers derived (see framework doc); occupancy model not yet trained |
| Apriori amenity bundles | ⬜ | Feeds optimisation flow recommendations |
| MLflow experiment tracking | ⬜ | Models saved locally; MLflow logging not yet wired |

### Primary flow — Airbnb-vs-sell decision engine
| Component | Status | Notes |
|---|---|---|
| Airbnb net revenue scenario (pure functions) | ⬜ | `finance/scenarios.py` |
| Sell scenario (€/m² × size, break-even) | ⬜ | Needs external price data |
| Recommendation logic + uncertainty bands | ⬜ | |
| Unit tests for finance engine | ⬜ | |

### Optimisation flow — Airbnb revenue maximisation
| Component | Status | Notes |
|---|---|---|
| Feature gap analysis vs. comparables | ⬜ | |
| Amenity recommendations (Apriori-driven) | ⬜ | |
| Renovation / remodelling opportunities | ⬜ | |
| Pricing strategy recommendations | ⬜ | |
| Prioritised action list (ROI-ranked) | ⬜ | |

### AI / agent layer
| Component | Status | Notes |
|---|---|---|
| Market Analyst agent | ⬜ | |
| Optimisation agent | ⬜ | |
| Regulatory agent (RAG / FAISS) | ⬜ | Corpus ready; retrieval not yet implemented |
| Comparables agent (KNN) | ⬜ | |
| Coordinator (LangGraph) | ⬜ | |
| Governance layer + eval harness | ⬜ | |

### MLOps
| Component | Status | Notes |
|---|---|---|
| FastAPI services | ⬜ | `/predict_price`, `/estimate_revenue`, `/airbnb_vs_sell`, `/optimise` |
| Docker images | ⬜ | |
| Evidently drift report | ⬜ | |

### UI
| Component | Status | Notes |
|---|---|---|
| Streamlit — decision tab (Airbnb vs. sell) | ⬜ | |
| Streamlit — optimisation tab | ⬜ | |
| Streamlit — chat tab | ⬜ | |
| Pre-cached demo outputs (JSON) | ⬜ | Required before KPMG demo |

---

## Changelog

Entries are newest-first. One bullet = one meaningful unit of work (PR, task, or session).
Format: `**YYYY-MM-DD** — What changed (file or component) · who/branch if relevant`

---

**2026-06-14** — Price ML model complete (`notebooks/price_ml_model.ipynb`, branch `Price_ML_Modelling`)
- 5 models trained end-to-end: Linear Regression, Ridge, Random Forest, Gradient Boosting, XGBoost, LightGBM
- **LightGBM selected** as production model: Test R² = 0.803, RMSE = €75.5/night, MAE = €34.6/night, MdAPE = 16%
- Per-city: Madrid R² 0.797 · Barcelona R² 0.819 · Málaga R² 0.752
- Top SHAP drivers: `property_type_std` (0.216), `accommodates` (0.133), `minimum_nights` (0.126), `neighbourhood_target_enc` (0.076)
- Artefacts saved to `models/`: `price_best_model.pkl`, `price_feature_cols.json`, `price_cat_encoders.pkl`, `price_scaler.pkl`
- `listings_with_price_hat.parquet` written to `Data/processed/` for downstream occupancy model

**2026-06-14** — Investment decision framework documented
- `docs/INVESTMENT_DECISION_FRAMEWORK.md` — Full analytical design: NPV comparison structure, cost model (Spain-specific: IBI, basuras, community fee, IRPF, CGT brackets), revenue model, seasonality layer, ML model plan (price + occupancy, LightGBM, SHAP), NPV engine design, Monte Carlo plan, implementation sequence
- Key data findings incorporated: calendar prices 100% null across all 3 cities (no time series model); scrape-date artifact in availability data (filter to 60d+ for unbiased seasonality); one combined model with `city` as feature (not per-city)

**2026-06-13** — Cross-city comparison notebook created and debugged
- `notebooks/cross_city_comparison.ipynb` — 7-section notebook: market size, nightly rates, revenue & occupancy, top districts (investment atlas), competitive landscape, regulatory environment, city scorecard + radar chart
- `src/airbnb_iip/data/cleaning.py` — fixed `host_name` regex bug (`regex=True` with unescaped `.` → `regex=False`; was silently replacing any ` X ` pattern)
- Fixed notebook relative paths (`../../Data/interim/` → `../Data/interim/`) and adaptive district minimum threshold (city-scale `max(5, n//100)` replaces hard-coded `n >= 20` that excluded all Málaga districts)

**2026-06-13** — Updated product framing across all docs to reflect two-flow structure
- `README.md` — Repositioned as "Airbnb or sell?" + optimisation flow; updated use-case table, how-it-works diagram, data section, and disclaimer
- `docs/UC2_Ordered_Task_Backlog.md` — Renamed and restructured; Phase 7 split into decision engine + Phase 7b (optimisation flow with 5 sub-tasks); Phase 10–11 updated with optimisation agent and UI tabs; removed long-term rental as a third scenario
- `docs/DATA_FINDINGS.md` — Sentiment section relabelled; Málaga framing updated; sentiment use in optimisation flow documented
- `docs/structure.md` — API endpoints, agent list, runtime flow, and `build_abt()` comments updated

**2026-06-13** — Idealista scraper merged (`feature/idealista-scraper`)
- `scripts/scrape_idealista.py` — Apify-backed scraper for sale + rent listings; 6 jobs (3 cities × 2 operations); JSONL output under `data/raw/idealista/`
- `src/airbnb_iip/data/scrapers/idealista.py` — `load_jsonl_as_dataframe()` loader with canonical snake_case columns
- `tests/test_idealista_scraper.py` — Test coverage for scraper
- `docs/idealista_schema.md` + `Data/sample/idealista_sample.jsonl` — Schema docs and synthetic sample

**~2026-06-09** — Regulatory corpus added (`regulatory-corpus` branch)
- `docs/regulatory/` — Official STR rules for Madrid, Barcelona, Málaga, and national level
- `docs/regulatory/sources.md` — Source index with URLs and retrieval dates

**~2026-06-09** — Data findings documented (`feature/data-findings` branch)
- `docs/DATA_FINDINGS.md` — Calendar availability, listings pricing, and sentiment findings for all 3 cities
- Barcelona pricing issue resolved: replaced empty-price snapshot with 12 Jun 2025 snapshot (~79% coverage)
- Sentiment model run: multilingual NB+TF-IDF pipeline scoring 40k reviews/city; outputs in `data/processed/<city>_sentiment.parquet`

**~2026-06-08** — Cleaning pipeline implemented
- `src/airbnb_iip/data/cleaning.py` — Full cleaning pipeline: price parsing, baths/beds standardisation, imputation, outlier capping, deduplication, price filtering
- `notebooks/madrid/`, `notebooks/barcelona/`, `notebooks/malaga/` — EDA, cleaning, and sentiment notebooks for all 3 cities

**~2026-06-08** — Repo scaffolded
- Repo structure, `pyproject.toml`, `requirements.txt`, `.env.example`, `config.py`, GitHub Actions CI
- `src/airbnb_iip/` package skeleton: `data/`, `features/`, `models/`, `finance/`, `agents/`
- `docs/structure.md`, `docs/UC2_Ordered_Task_Backlog.md` — Engineering conventions and task backlog
