# Airbnb Investment Intelligence Platform

> A governed, multi-agent AI system that helps property owners decide whether to
> list on Airbnb or sell — and, if they choose Airbnb, how to maximise their revenue.
>
> **IE × KPMG Lighthouse — Capstone 2026**

The platform takes a property you already own and answers a concrete question:
**"Should I list this on Airbnb or sell it?"** It combines structured Airbnb data,
real-estate market data, and a regulatory knowledge base behind a conversational interface,
returning a **data-grounded, explainable recommendation** — not a black-box number.
Once a decision is made, owners who choose Airbnb can continue into an **optimisation
flow** that surfaces the highest-impact improvements to their listing.

Cities covered: **Madrid · Barcelona · Málaga** · Data snapshot: **14 Sep 2025** (Inside Airbnb).

---

## Product flows

| | Flow | Question answered |
|---|---|---|
| **Primary** ⭐ | Airbnb or sell? | *"I own a property — should I list it on Airbnb or sell it?"* → compares projected Airbnb net revenue against the property's indicative sale value, with break-even timelines, confidence bands, and a clear recommendation. |
| **Secondary** | Optimise for Airbnb | *"I've decided to Airbnb — how do I maximise revenue?"* → property-specific recommendations: amenities to add, renovations with the best ROI, pricing strategy, and gap analysis vs comparable top performers. |
| **Stretch** | Pre-purchase screening | *"If I buy this flat, what will it earn on Airbnb?"* → predicted nightly rate, seasonal occupancy, revenue P10/P50/P90, yield, and regulatory risk. |

---

## How it works

```
Owner inputs property details
   → Streamlit UI (4 pages: New Analysis · Chat · Saved Properties · Optimisation)
      → Analysis engine (app/components/engine.py) orchestrates
         ├─ Price model       → city-specific LightGBM (Madrid / Barcelona / Málaga) + SHAP
         ├─ Sale model        → Idealista-backed district price-per-m² regression
         ├─ Finance engine    → Airbnb net revenue vs. sell scenario (P10/P50/P90 bands)
         ├─ Market Analyst    → narrates the comparison, surfaces SHAP "why"
         ├─ Regulatory (RAG)  → municipal STR rules with source citations
         └─ Comparables       → genuinely similar performing listings (KNN)
      → explainable brief: numbers from the models, narration from the LLM,
        with uncertainty bands and cited sources

   [If owner chooses Airbnb]
      → Optimisation flow
         ├─ Feature gap analysis vs top-performing comparables
         ├─ Amenity bundle recommendations
         └─ Prioritised action list with estimated revenue impact
```

A **Governance layer** wraps every agent response: output-bounds guardrails,
mandatory "indicative, not financial advice" framing, and model cards per model.

Full diagram: [`docs/Architecture_Diagram.svg`](docs/Architecture_Diagram.svg).

---

## Tech stack

- **Data & ML:** pandas, scikit-learn (HistGradientBoosting / LightGBM), SHAP, statsmodels, scipy, plotly
- **AI layer:** LangChain + FAISS (RAG), sentence-transformers, langchain-google-genai (Gemini narration), LangGraph
- **API:** FastAPI (model serving — `/predict_price`, `/estimate_occupancy`, `/estimate_revenue`, `/airbnb_vs_sell`)
- **UI:** Streamlit (4-page app)
- **MLOps:** pytest, GitHub Actions (CI/CD)

---

## Repository structure

```
src/airbnb_iip/       installable package — all reusable logic
  config.py           project settings: paths, finance & occupancy assumptions
  data/               cleaning · abt · occupancy · idealista_sale · scrapers/
  features/           amenities · density · seasonality
  models/             price (city-specific + general) · sale (Idealista-backed)
  finance/            costs · scenarios — pure functions, the core engine
  agents/             market_analyst · regulatory (RAG) · comparables · governance
api/                  FastAPI model services
  routers/            predict · revenue · optimise
app/                  Streamlit application
  Home.py             landing page
  pages/              1_New_Analysis · 2_Chat · 3_Saved_Properties · 4_Optimisation
  components/         branding · engine · storage · styling
models/               trained artefacts: price_city_*.pkl · sale_best_model.pkl · …
scripts/              scrape_idealista · make_price_notebook · make_sell_model_notebook
notebooks/            price_ml_model · price_ml_model_comparison · sell_price_eda ·
                      sell_price_model · segmentation · amenity_feature_engineering · …
config/               config.yaml (editable cost assumptions)
data/                 raw · interim · processed · external · regulatory · sample  (mostly gitignored)
docs/                 architecture · DATA_FINDINGS · EDA · FEATURE_ENGINEERING ·
                      model_cards/ · regulatory/ · structure
tests/                pytest suite (14 modules)
```

See **[`docs/structure.md`](docs/structure.md)** for engineering conventions.

---

## Getting started

**Prerequisites:** Python 3.11, Git.

```bash
# 1. Clone and create an environment
git clone <repo-url> && cd airbnb-investment-intelligence
python -m venv .venv && source .venv/bin/activate

# 2. Install dependencies + the package (editable)
pip install -r requirements.txt -r requirements-dev.txt
pip install -e .

# 3. Configure secrets
cp .env.example .env        # add GOOGLE_API_KEY (Gemini) and APIFY_API_TOKEN
```

> If `geopandas` fights pip on your machine, install it via conda first.

---

## Running the project

```bash
# Run the Streamlit app (primary interface)
streamlit run app/Home.py              # → http://localhost:8501

# Run the FastAPI model-serving layer
uvicorn api.main:app --reload          # → http://localhost:8000/docs

# Scrape fresh Idealista sale/rent data
python scripts/scrape_idealista.py    # requires APIFY_API_TOKEN in .env
```

**For the KPMG demo:** agent/LLM outputs are pre-cached as JSON and the app runs in cached mode
by default (a "live mode" toggle exists). Never rely on live LLM calls during the presentation.

---

## Data

- **Inside Airbnb** (public): detailed `listings.csv.gz`, `calendar.csv.gz`, `reviews.csv.gz`,
  and `neighbourhoods.geojson` per city. Not committed — fetch manually from insideairbnb.com.
- **Idealista** (scraped): district-level sale and rent listings for all three cities, collected
  via the Apify actor. Anchors the sell scenario with real €/m² data.
- **Regulatory corpus** (RAG): official municipal short-term-rental rules per city stored in
  `docs/regulatory/`. These change — always verify against current official sources.

The occupancy/revenue estimate uses the **San Francisco model** (bookings inferred from review
frequency, not naïve calendar availability) and is reported as **P10/P50/P90 bands**.

---

## Idealista scraper (data-collection agent)

A scraper agent collects Idealista apartment listings for Madrid, Barcelona, and Málaga, in
both **sale (venta)** and **rent (alquiler)** flavours, via the
[`igolaizola/idealista-scraper`](https://apify.com/igolaizola/idealista-scraper) Apify actor.

**Setup.** Add your Apify token to `.env`:

```bash
APIFY_API_TOKEN=apify_api_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

**Run.**

```bash
# All six (city × operation) jobs with defaults (cap = 2,500 per job)
python scripts/scrape_idealista.py

# A single slice
python scripts/scrape_idealista.py --cities madrid --operations rent

# Push past Idealista's ~1,800-per-query cap
python scripts/scrape_idealista.py --max-items 5000
```

**Outputs.** One JSONL file per (city, operation) under
`data/raw/idealista/<city>/<operation>_<YYYY-MM-DD>.jsonl` (git-ignored).

**Loading into pandas.**

```python
from airbnb_iip.data.scrapers.idealista import load_jsonl_as_dataframe

df = load_jsonl_as_dataframe("data/raw/idealista/madrid/sale_2026-06-10.jsonl")
# canonical snake_case columns: listing_id, price_eur, size_m2, latitude, …
```

Full canonical schema in [`docs/idealista_schema.md`](docs/idealista_schema.md).

---

## ML models

### Nightly price model

Three city-specific LightGBM models (Madrid · Barcelona · Málaga), selected over a single
pooled model by the approach comparison study
(`notebooks/price_ml_model_comparison.ipynb`):

| Approach | Test R² | RMSE | MAE |
|---|---|---|---|
| City-specific (production) | **0.814** | €68.5 | €30.8 |
| General pooled | 0.810 | €69.5 | €31.6 |

Top SHAP drivers: `property_type_std`, `accommodates`, `minimum_nights`,
`neighbourhood_target_enc`. Artefacts in `models/price_city_*.pkl`.

### Sale price model

Regression model backed by Idealista listings, predicting property sale value from size,
district, property type, and condition. Artefact in `models/sale_best_model.pkl`.

Model cards: [`docs/model_cards/price_model.md`](docs/model_cards/price_model.md) ·
[`docs/model_cards/sale_model.md`](docs/model_cards/sale_model.md).

---

## Master-technique coverage

The project exercises the full master curriculum: feature engineering & ABT design,
**VIF** and **RFE** feature selection, **PCA/Factor Analysis**, segmentation (**K-means /
hierarchical / DBSCAN** with silhouette & elbow), **association analysis (Apriori)** for amenity
bundle recommendations, linear/OLS regression, **LightGBM** with SHAP, **Naïve Bayes +
TF-IDF** sentiment for guest review analysis, **KNN** for comparable listings, and a complete
RAG + agent-orchestration stack. The full mapping is in
[`docs/UC2_Ordered_Task_Backlog.md`](docs/UC2_Ordered_Task_Backlog.md).

---

## Project documents

- [`docs/INVESTMENT_DECISION_FRAMEWORK.md`](docs/INVESTMENT_DECISION_FRAMEWORK.md) — NPV comparison design, cost model, ML model plan
- [`docs/UC2_Ordered_Task_Backlog.md`](docs/UC2_Ordered_Task_Backlog.md) — ordered build tasks (primary + optimisation flows)
- [`docs/structure.md`](docs/structure.md) — setup & engineering conventions
- [`docs/Architecture_Diagram.svg`](docs/Architecture_Diagram.svg) — system architecture diagram
- [`docs/DATA_FINDINGS.md`](docs/DATA_FINDINGS.md) — calendar, listings, and sentiment findings
- [`docs/EDA.md`](docs/EDA.md) — exploratory data analysis notes
- [`docs/FEATURE_ENGINEERING.md`](docs/FEATURE_ENGINEERING.md) — feature engineering decisions
- [`docs/idealista_schema.md`](docs/idealista_schema.md) — Idealista scraper data contract
- [`docs/model_cards/`](docs/model_cards/) — model cards for price and sale models

---

## Responsible AI & disclaimer

This is an **academic capstone project** using public Inside Airbnb data and published market
indices. All agent answers are **grounded, source-cited, and reported with uncertainty bands**;
high-value recommendations pass through a human-review gate. The platform provides **decision
support only and is not financial, legal, or investment advice**. Revenue projections and
optimisation recommendations are indicative; actual results depend on individual property
characteristics and market conditions. Regulatory information must be verified against current
official municipal sources.

---

## Team

Five-member IE capstone team, in collaboration with KPMG Lighthouse. Working conventions
(branching, the data contract, the bootstrap sequence) are in [`docs/structure.md`](docs/structure.md).

*Document Classification: KPMG Confidential.*
