# Sprint Status & Remaining Work
**Airbnb Investment Intelligence Platform · IE × KPMG Capstone 2026**
**As of 15 June 2026 · 10 days to demo (25 June)**

---

## What's done

| Area | Detail |
|---|---|
| Repo & CI | Structure, `pyproject.toml`, GitHub Actions pytest, `.env.example` |
| Raw data | Inside Airbnb for Madrid, Barcelona, Málaga (listings, calendar, reviews, neighbourhoods) |
| Idealista scraper | `scripts/scrape_idealista.py` — sale + rent listings via Apify, JSONL output |
| Regulatory corpus | Official STR rules for Madrid, Barcelona, Málaga + national level in `docs/regulatory/` |
| Cleaning pipeline | `src/airbnb_iip/data/cleaning.py` — price parsing, imputation, outlier capping, deduplication |
| EDA | Per-city notebooks (Madrid, Barcelona, Málaga) + cross-city comparison notebook |
| Feature selection | VIF + Random Forest RFE; ~30 features selected |
| Price ML model | LightGBM — Test R² = 0.803, MAE = €34.6/night, MdAPE = 16%; SHAP global + per-prediction; artifacts saved to `models/` |
| Sentiment model | Multilingual Naïve Bayes + TF-IDF; scores in `Data/processed/<city>_sentiment.parquet` |

---

## What's missing

### CRITICAL PATH — primary "Airbnb or sell?" flow

These are blockers. The demo cannot work without them.

**1. External sale price data (blocks the entire sell scenario)**
- Need €/m² by district for Madrid, Barcelona, Málaga
- Source: Madrid City Council open data / published index (prefer over scraping — ToS risk)
- Output: clean CSV with source + snapshot date, joined onto listings by district
- The sell scenario has no anchor without this

**2. Occupancy estimator function**
- San Francisco model: `bookings/mo ≈ (reviews_per_month / review_rate) × avg_length_of_stay`
- Implement as a documented function in `src/airbnb_iip/data/`
- Cap at realistic ceiling; assumptions in docstring

**3. Feature engineering (partially in notebooks — needs to become code)**
- Amenity dummies: parse JSON string → top-30 binary flags
- Seasonal multipliers from calendar (filter to 60d+ dates to avoid scrape-date artifact)
- District competitive density (BallTree, 500m radius)
- `build_abt(city)` — single importable function that runs the full pipeline and emits a versioned parquet ABT

**4. Occupancy ML model (stage 2 of two-stage plan)**
- Input: `Data/processed/listings_with_price_hat.parquet` (already written by price model)
- Target: `estimated_occupancy_l365d`
- `price_hat` must be a feature (demand elasticity; no leakage since it's a prediction)
- Save artifact to `models/`

**5. Finance module — `src/airbnb_iip/finance/` is empty**
- Airbnb net scenario: `predicted_price × occupancy × seasonality − costs` → P10/P50/P90 bands
  - Costs: Airbnb 3% host fee, cleaning, management %, vacancy, IRPF (19–47%), Barcelona tourist tax (€4.40/night/person), STR insurance
- Sell scenario: `€/m² × size` → one-off capital realisation; break-even timeline (years of Airbnb net income to match sale value)
- Recommendation logic: compare the two, output uncertainty bands, never presented as financial advice
- All of the above as **pure functions with unit tests** — this is the product's core

---

### HIGH PRIORITY — needed before the demo

**6. FastAPI endpoints (`api/` is empty)**
- `/predict_price`, `/estimate_occupancy`, `/estimate_revenue`, `/airbnb_vs_sell`, `/optimise`
- Agents call these; notebooks do not run in production

**7. AI / agent layer (`src/airbnb_iip/agents/` is empty)**
- Market Analyst agent: calls revenue + scenario services, narrates with SHAP values
- Coordinator (LangGraph): routes primary (Airbnb vs sell) vs secondary (optimisation) flow
- Governance layer: output-bounds guardrails (no negative revenue, no implausible yields), "indicative, not financial/legal advice" disclaimer, model cards
- Regulatory agent: RAG over `docs/regulatory/` corpus (FAISS) with source citations — corpus is ready, retrieval not built
- Comparables agent: structured filters (segment, size, neighbourhood) + KNN baseline

**8. Streamlit UI (`app/` is empty)**
- Decision tab: property input → Airbnb-vs-sell comparison + break-even chart + recommendation with confidence bands
- Optimisation tab: improvement recommendations (amenities, renovation, pricing) each with revenue uplift + cost estimate
- Chat tab: routed through coordinator, supports both flows conversationally
- **Pre-cached demo outputs (JSON) — hard requirement.** Never run live LLM/API calls during the KPMG demo. Cache all agent outputs beforehand.

---

### SECONDARY — optimisation flow (after primary works)

**9. Optimisation flow**
- Feature gap analysis vs top-performing comparables in same segment + neighbourhood
- Amenity recommendations via Apriori association rules (support/confidence/lift), ranked by revenue uplift
- Pricing strategy: optimal base price + seasonal adjustments vs comparable listings
- Renovation opportunities: cost vs projected revenue impact
- ROI-ranked prioritised action list

---

### NICE TO HAVE

**10. MLOps**
- MLflow experiment tracking (models currently saved locally only)
- Docker / docker-compose for one-command local run (scaffolded in plan, not built)
- Evidently drift report (strong demo moment for KPMG's "Measure & Monitor" pillar)

**11. Remaining analysis**
- UC2-specific EDA: gross yield, Airbnb revenue vs sale value ratios by district — the primary flow narrative

---

## Suggested sprint plan (10 days)

| Dates | Focus |
|---|---|
| 15–16 Jun | External €/m² data + district join; occupancy estimator; `build_abt()` |
| 16–18 Jun | Finance module (revenue engine + sell scenario + recommendation logic) + unit tests |
| 18–20 Jun | Occupancy ML model; FastAPI endpoints; thin end-to-end slice running |
| 20–22 Jun | Market Analyst agent + Coordinator; Streamlit decision tab |
| 22–23 Jun | Optimisation flow; regulatory + comparables agents; remaining UI tabs |
| 23–24 Jun | Pre-cache all demo outputs as JSON; freeze code |
| 24–25 Jun | Dry runs + presentation deck |

---

## Single biggest risk

**External sale price data.** Without €/m² by district the sell scenario has no anchor and the primary recommendation — the centrepiece of the demo — cannot produce a number. This should be the first thing resolved on 15 June.

---

## Key files & artefacts for reference

| File | What it is |
|---|---|
| `docs/INVESTMENT_DECISION_FRAMEWORK.md` | Full analytical design: NPV structure, cost model, ML plan, seasonality layer |
| `docs/UC2_Ordered_Task_Backlog.md` | Full task backlog with phase breakdown |
| `models/price_best_model.pkl` | Production price model (LightGBM) |
| `models/price_feature_cols.json` | Feature list expected by the price model |
| `Data/processed/listings_with_price_hat.parquet` | All listings with `price_hat` — input for occupancy model |
| `Data/processed/listings_all_cities.parquet` | Cleaned + merged ABT (pre-feature-engineering) |
| `config/config.yaml` | Cost assumptions, editable without touching code |
| `src/airbnb_iip/data/cleaning.py` | Cleaning pipeline |
