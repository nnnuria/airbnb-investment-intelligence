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

We build the **Airbnb-vs-sell decision** first (the primary deliverable); the **optimisation
flow** is the natural second step for owners who choose Airbnb. UC1 (pre-purchase screening)
is a stretch goal that comes almost for free once the revenue engine exists.

| | Flow | Question answered |
|---|---|---|
| **Primary** ⭐ | Airbnb or sell? | *"I own a property — should I list it on Airbnb or sell it?"* → compares projected Airbnb net revenue against the property's indicative sale value, with break-even timelines, confidence bands, and a clear recommendation. |
| **Secondary** | Optimise for Airbnb | *"I've decided to Airbnb — how do I maximise revenue?"* → property-specific recommendations: amenities to add, renovations with the best ROI, pricing strategy, and gap analysis vs comparable top performers. |
| **Stretch** | Pre-purchase screening | *"If I buy this flat, what will it earn on Airbnb?"* → predicted nightly rate, seasonal occupancy, revenue P10/P50/P90, yield, and regulatory risk. |

---

## How it works

```
Owner inputs property details
   → Streamlit UI
      → Coordinator agent (LangGraph) routes & aggregates
         ├─ Market Analyst  → price + occupancy + Airbnb-vs-sell scenario engine (+ SHAP "why")
         ├─ Regulatory (RAG) → municipal STR rules, with source citations
         └─ Comparables (RAG)→ genuinely similar performing listings
      → explainable brief: numbers from the models, narration from the LLM,
        with uncertainty bands and cited sources

   [If owner chooses Airbnb]
      → Optimisation flow
         ├─ Feature gap analysis vs top-performing comparables
         ├─ Amenity recommendations (occupancy uplift + nightly-rate impact)
         ├─ Renovation / remodelling opportunities with cost-vs-revenue trade-offs
         └─ Prioritised action list with estimated revenue impact
```

Two cross-cutting layers wrap the whole stack: **MLOps** (MLflow · FastAPI · Docker · GitHub
Actions · Evidently) and **Governance / Trusted AI** (guardrails · citations · uncertainty ·
human-in-the-loop · model cards · eval harness).

Full diagram: [`docs/architecture.svg`](docs/architecture.svg).

---

## Tech stack

- **Data & ML:** pandas, geopandas, scikit-learn, XGBoost / LightGBM, SHAP, Prophet / ARIMA, mlxtend (Apriori), NLTK / spaCy (NLP)
- **AI layer:** LangGraph (orchestration), FAISS (vector store), RAG, an LLM API for narration
- **MLOps:** MLflow (tracking + registry), FastAPI (serving), Docker, GitHub Actions (CI/CD), pytest, Evidently (drift)
- **UI:** Streamlit

---

## Repository structure

```
src/airbnb_iip/   installable package — all reusable logic
  config.py       project settings: paths, finance & occupancy assumptions
  data/           loader · clean · abt (build_abt)
  features/       engineering · occupancy (SF model) · selection (VIF/RFE)
  models/         price · demand · segmentation · nlp
  finance/        scenarios — pure functions, the UC2 core
  agents/         coordinator · market_analyst · regulatory · comparables
api/              FastAPI model services
app/              Streamlit application
scripts/          download_data · build_abts · train_all
notebooks/        exploration only (logic belongs in src/)
data/             raw · interim · processed · external · regulatory · sample  (mostly gitignored)
docs/             schema.md (data contract) · model_cards · architecture.svg
tests/            pytest suite
```

See **[`docs/structure.md`](docs/structure.md)** for the full conventions (the *how we build*
reference) and **`schema.md`** for the ABT data contract.

---

## Getting started

**Prerequisites:** Python 3.11, Git, Docker (for the demo runtime).

```bash
# 1. Clone and create an environment
git clone <repo-url> && cd airbnb-investment-intelligence
python -m venv .venv && source .venv/bin/activate

# 2. Install dependencies + the package (editable)
pip install -r requirements.txt -r requirements-dev.txt
pip install -e .

# 3. Configure secrets
cp .env.example .env        # then add your LLM API key

# 4. Sanity check
python -c "import airbnb_iip; print('ok')"
```

> If `geopandas` or `prophet` fight pip on your machine, install those two via conda — it's the
> reliable escape hatch.

---

## Running the project

```bash
# Download the raw datasets (Inside Airbnb + external indices)
python scripts/download_data.py

# Build the Analytical Base Table (cleaned, feature-engineered, modelling-ready) per city
python scripts/build_abts.py            # writes data/processed/<city>.parquet

# Train models and log everything to MLflow
python scripts/train_all.py

# Inspect experiments and the model registry
mlflow ui                                # → http://localhost:5000

# Run the full app (API + UI together)
docker-compose up

# …or run the UI alone in dev
streamlit run app/streamlit_app.py
```

**For the KPMG demo:** agent/LLM outputs are pre-cached as JSON and the app runs in cached mode
by default (a "live mode" toggle exists). Never rely on live LLM calls during the presentation.

---

## Data

- **Inside Airbnb** (public): detailed `listings.csv.gz`, `calendar.csv.gz`, `reviews.csv.gz`,
  and `neighbourhoods.geojson` per city. Not committed — fetched by `scripts/download_data.py`.
- **External market data**: district-level **sale price per m²** curated from published/open-data
  sources (not scraped). Used to model the sell-side of the Airbnb-vs-sell comparison. Source +
  date documented in `data/external/`.
- **Regulatory corpus** (RAG): official municipal short-term-rental rules per city. These change
  — always verify against current official sources and cite them in output.

The occupancy/revenue estimate uses the documented **San Francisco model** (bookings inferred
from review frequency, not naïve calendar availability) and is reported as **P10/P50/P90 bands**,
not a single point estimate.

---

## Idealista scraper (data-collection agent)

A scraper agent collects Idealista apartment listings for Madrid, Barcelona, and Málaga, in
both **sale (venta)** and **rent (alquiler)** flavours, via the
[`igolaizola/idealista-scraper`](https://apify.com/igolaizola/idealista-scraper) Apify actor.
It calls the Apify HTTP API programmatically (no Apify CLI required), streams the dataset
to disk as JSON Lines, and tags each record with `_city` / `_operation` / `_scraped_at` for
downstream joins.

**Setup.** Add your Apify token to `.env` (gitignored; template in `.env.example`):

```bash
APIFY_API_TOKEN=apify_api_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

The token is read from the environment — it is never hardcoded or accepted on the command line.

**Run.**

```bash
# All six (city × operation) jobs with defaults (cap = 2,500 per job)
python scripts/scrape_idealista.py

# A single slice
python scripts/scrape_idealista.py --cities madrid --operations rent

# Push past Idealista's ~1,800-per-query cap — triggers the actor's
# sub-location splitting (slower; ordering changes).
python scripts/scrape_idealista.py --max-items 5000
```

**Outputs.** One JSONL file per (city, operation), under
`data/raw/idealista/<city>/<operation>_<YYYY-MM-DD>.jsonl` (git-ignored).
Example layout after a full run on 2026-06-10:

```
data/raw/idealista/
├── madrid/
│   ├── sale_2026-06-10.jsonl
│   └── rent_2026-06-10.jsonl
├── barcelona/
│   ├── sale_2026-06-10.jsonl
│   └── rent_2026-06-10.jsonl
└── malaga/
    ├── sale_2026-06-10.jsonl
    └── rent_2026-06-10.jsonl
```

**What's in each record.** Per-listing fields include identifiers (`propertyCode`, `url`),
pricing (`price`, `pricePerM2`), size and layout (`size`, `rooms`, `bathrooms`),
type (`propertyType`, `homeType`, `floor`), location (`address`, `district`,
`neighborhood`, `municipality`, `province`, `latitude`, `longitude`), narrative
(`title`, `description`), media (`images`, `virtualTour`), amenity tags (`features`),
and advertiser info (`agency`, `phone`). Full canonical schema in
[`docs/idealista_schema.md`](docs/idealista_schema.md); a synthetic three-record
sample lives at [`Data/sample/idealista_sample.jsonl`](Data/sample/idealista_sample.jsonl).

**Loading into pandas.**

```python
from airbnb_iip.data.scrapers.idealista import load_jsonl_as_dataframe

df = load_jsonl_as_dataframe("data/raw/idealista/madrid/sale_2026-06-10.jsonl")
# canonical snake_case columns: listing_id, price_eur, size_m2, latitude, …
```

**Limits and cost.** Idealista shows ~1,800 results per search query; the actor caps a
single job at this naturally. Set `--max-items` above 2,500 to trigger the actor's
automatic sub-location splitting and go beyond the cap (slower; ordering changes).
The actor itself is a paid Apify actor — check pricing on its store page before large
runs and start with a small `--max-items` to verify behaviour.

---

## Master-technique coverage

The project deliberately exercises the full master curriculum: feature engineering & ABT design,
**VIF** and **RFE** feature selection, **PCA/Factor Analysis**, segmentation (**K-means /
hierarchical / DBSCAN** with silhouette & elbow), **association analysis (Apriori)** for amenity
bundle recommendations, linear/OLS regression, **XGBoost/LightGBM** with SHAP, **Naïve Bayes +
TF-IDF** sentiment for guest review analysis, **KNN** for comparable listings, time series
(**Prophet/ARIMA**), and a complete MLOps + RAG + agent-orchestration stack. The full mapping is
in [`docs/Capstone_Plan.md`](docs/Capstone_Plan.md).

---

## Project documents

- [`docs/Capstone_Plan.md`](docs/Capstone_Plan.md) — full plan + master-coverage matrix
- [`docs/UC2_Ordered_Task_Backlog.md`](docs/UC2_Ordered_Task_Backlog.md) — ordered build tasks (primary + optimisation flows)
- [`docs/structure.md`](docs/structure.md) — setup & engineering conventions
- [`docs/architecture.svg`](docs/architecture.svg) — system architecture diagram
- [`docs/schema.md`](docs/schema.md) — the ABT data contract

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
