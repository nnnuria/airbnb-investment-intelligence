# Airbnb Investment Intelligence Platform

> A governed, multi-agent AI system that helps individuals and small investors make
> data-driven decisions about short-term rental properties in Spanish cities.
>
> **IE × KPMG Lighthouse — Capstone 2026**

The platform takes a property — one you're thinking of buying, one you already own, or one
you already host — and returns a **data-grounded, explainable investment brief** instead of a
black-box number. It combines structured Airbnb data, external market data, and a regulatory
knowledge base behind a conversational interface, with a full MLOps pipeline and responsible-AI
guardrails.

Cities covered: **Madrid · Barcelona · Málaga** · Data snapshot: **14 Sep 2025** (Inside Airbnb).

---

## Use cases

We build **UC2 first** (it's the primary deliverable); UC1 and UC3 are stretch goals — and UC1
comes almost for free once UC2's revenue engine exists.

| | Use case | Question answered |
|---|---|---|
| **UC2** ⭐ *(primary)* | Own-it optimiser | *"I own a flat — should I Airbnb it, sell it, or keep it as a long-term rental?"* → compares Airbnb net revenue vs long-term rent vs indicative sale value, with break-even timelines. |
| **UC1** | Pre-purchase screening | *"If I buy this flat, what will it earn on Airbnb and what's the gross yield?"* → predicted nightly rate, seasonal occupancy, revenue P10/P50/P90, yield, regulatory risk. |
| **UC3** | Host improvement engine | *"I'm hosting at €120/night, 4.3★ — what should I fix?"* → review-sentiment gaps vs top comparables + prioritised actions with €-uplift. |

---

## How it works

```
User question
   → Streamlit UI
      → Coordinator agent (LangGraph) routes & aggregates
         ├─ Market Analyst  → price + occupancy + financial scenario engine (+ SHAP "why")
         ├─ Regulatory (RAG) → municipal STR rules, with source citations
         └─ Comparables (RAG)→ genuinely similar performing listings
      → explainable brief: numbers from the models, narration from the LLM,
        with uncertainty bands and cited sources
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
- **External market data** (UC2): district-level **sale price per m²** and **long-term rent
  index**, curated from published/open-data sources (not scraped). Source + date documented in
  `data/external/`.
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
hierarchical / DBSCAN** with silhouette & elbow), **association analysis (Apriori)**, linear/OLS
regression, **XGBoost/LightGBM** with SHAP, **Naïve Bayes + TF-IDF** sentiment, **KNN**, time
series (**Prophet/ARIMA**), and a complete MLOps + RAG + agent-orchestration stack. The full
mapping is in [`docs/Capstone_Plan.md`](docs/Capstone_Plan.md).

---

## Project documents

- [`docs/Capstone_Plan.md`](docs/Capstone_Plan.md) — full plan + master-coverage matrix
- [`docs/UC2_Ordered_Task_Backlog.md`](docs/UC2_Ordered_Task_Backlog.md) — ordered build tasks
- [`docs/structure.md`](docs/structure.md) — setup & engineering conventions
- [`docs/architecture.svg`](docs/architecture.svg) — system architecture diagram
- [`docs/schema.md`](docs/schema.md) — the ABT data contract

---

## Responsible AI & disclaimer

This is an **academic capstone project** using public Inside Airbnb data and published market
indices. All agent answers are **grounded, source-cited, and reported with uncertainty bands**;
high-value recommendations pass through a human-review gate. The platform provides **decision
support only and is not financial, legal, or investment advice**. Regulatory information is
indicative and must be verified against current official municipal sources.

---

## Team

Five-member IE capstone team, in collaboration with KPMG Lighthouse. Working conventions
(branching, the data contract, the bootstrap sequence) are in [`docs/structure.md`](docs/structure.md).

*Document Classification: KPMG Confidential.*
