# Project Structure & Setup — Team Reference
**Airbnb Investment Intelligence Platform · IE × KPMG Capstone 2026**

This is the shared reference for how our repo is organised and how we all work in it.
Read this before writing any code. If something here is unclear or wrong, fix it in a PR —
this doc is the source of truth for *how* we build (the *what* lives in the plan and the
task backlog).

---

## 1. Two principles everything follows

**Principle 1 — separate the "offline" world from the "online" world.**
There are two systems in one repo:
- *Offline* = exploring, cleaning data, training models. Output = artifacts logged to MLflow.
- *Online* = what runs at demo time: Streamlit UI → agents → FastAPI services → registered models.

Notebooks and training scripts **never run during the demo**. Keeping this boundary clean is
what lets five people work in parallel and what makes our MLOps story coherent.

**Principle 2 — agree the interface before writing the code behind it.**
The reason teams stall is everyone waiting on the data. We avoid that by fixing the *contract*
on day one: `docs/schema.md` (exact ABT columns + types) plus the signature
`build_abt(city) -> DataFrame`, plus a tiny **synthetic sample** in `data/sample/` that matches
it. Then everyone builds against the sample immediately while the data lead fills in the real
`build_abt()` behind the same signature. When real data lands, it just works.

> **Golden rule:** anything reusable lives in `src/airbnb_iip/`. Notebooks only *call* it.

---

## 2. Repository layout

```
airbnb-investment-intelligence/
├── README.md
├── pyproject.toml              # makes src/ installable: pip install -e .
├── requirements.txt            # pinned versions
├── requirements-dev.txt        # pytest, ruff, nbstripout…
├── .gitignore                  # data/, mlruns/, .env, __pycache__
├── .env.example                # template only; real .env is gitignored
├── config/
│   └── config.yaml             # paths, cost assumptions, occupancy params, thresholds
├── data/                       # NOT in git (except sample/ and small external/)
│   ├── raw/                    # downloaded Inside Airbnb + external
│   ├── interim/                # cleaned intermediates
│   ├── processed/              # ABT parquet per city
│   ├── external/               # small curated price-€/m² + rent CSVs (commit if tiny)
│   ├── regulatory/             # corpus docs for RAG
│   └── sample/                 # tiny SYNTHETIC dataset, COMMITTED, matches schema.md
├── docs/
│   ├── schema.md               # THE data contract
│   ├── model_cards/
│   └── architecture.svg
├── notebooks/                  # exploration only; numbered + owner-prefixed
├── src/airbnb_iip/             # the installable package
│   ├── config.py               # loads config.yaml
│   ├── data/        loader.py · clean.py · abt.py        # build_abt(city)
│   ├── features/    engineering.py · occupancy.py · selection.py   # SF-model, VIF/RFE
│   ├── models/      price.py · demand.py · segmentation.py · nlp.py
│   ├── finance/     scenarios.py     # PURE functions — the UC2 core
│   └── agents/      coordinator.py · market_analyst.py · regulatory.py · comparables.py
├── api/main.py                 # FastAPI: /predict_price, /estimate_revenue, /compare_scenarios
├── app/streamlit_app.py        # UI: UC2 tab + chat
├── scripts/        download_data.py · build_abts.py · train_all.py
├── tests/          test_loader.py · test_finance.py · test_schema.py
├── docker/         Dockerfile.api · Dockerfile.app
├── docker-compose.yml
└── .github/workflows/ci.yml
```

`src/` is an **installable package** so imports work identically in notebooks, the API, and
tests — no `sys.path` hacks. After cloning, everyone runs `pip install -e .` once.

---

## 3. The data contract: `schema.md` + `build_abt()`

`build_abt(city)` is the single pipeline from raw files to a clean, modelling-ready
**Analytical Base Table** (one row per listing, consistent columns across all three cities).
It is the function everyone imports — nobody re-implements cleaning. See §"What is the ABT"
at the bottom for the full explanation.

```python
from airbnb_iip.data.abt import build_abt
abt = build_abt("madrid")        # returns the modelling-ready DataFrame
```

`docs/schema.md` lists every column it outputs (name, type, meaning). That file is the
contract: model code, the API, and the UI all assume those columns exist. Change a column →
update `schema.md` in the same PR.

---

## 4. Environment & dependencies

- Pin one Python version (**3.11** recommended) and write it in the README.
- A single pinned `requirements.txt` for local dev; `requirements-dev.txt` for tooling.
- **Docker is the source of truth** for anything that ships (API + app images). This kills
  "works on my machine" for the demo.
- Install-pain to expect: `geopandas`, `prophet`, `faiss`, plus `xgboost`, `lightgbm`, `shap`,
  `mlflow`, `evidently`, `langgraph`, `streamlit`. If `geopandas`/`prophet` fight pip locally,
  use a conda env for those two — pragmatic escape hatch.
- Don't adopt Poetry/uv unless someone already knows it. On a 19-day clock, tooling you must
  learn is a tax.

Standard onboarding:
```bash
git clone <repo> && cd airbnb-investment-intelligence
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
pip install -e .
cp .env.example .env          # then add your LLM API key
python -c "import airbnb_iip; print('ok')"
```

---

## 5. Config & secrets

- **Every tunable number lives in `config/config.yaml`**, never hardcoded: data paths,
  occupancy params (`review_rate`, `avg_length_of_stay`, occupancy cap), and especially the
  **financial assumptions** (cleaning cost, management %, platform fee, tax rate). The finance
  engine's credibility depends on these being visible and adjustable, and governance wants
  assumptions auditable.
- The **LLM API key** goes in `.env` (gitignored), loaded with `python-dotenv`. Commit only
  `.env.example` with the variable *names*. **Never commit a key** — a leaked key in a student
  repo is a classic, costly mistake.

---

## 6. Data handling

- Raw Inside Airbnb files (the `.gz`, `calendar.csv` especially) are large → **never in git.**
  Gitignore `data/raw/`, `data/interim/`, `data/processed/`, `mlruns/`.
- Commit `scripts/download_data.py` + a README note with the **snapshot date** (14 Sep 2025).
- The only committed data: the synthetic `data/sample/` (tests + early dev run anywhere) and
  the small curated external CSVs if they're a few KB.
- DVC is overkill; a download script + gitignored data is enough and easier to explain.

---

## 7. MLflow conventions

- Local file-based store (`mlruns/`, gitignored). Run `mlflow ui` for the demo + screenshots.
- Experiment per model family: `price_madrid`, `demand_madrid`, …
- Register the production-bound model under a **stable name** (e.g. `price_model`) so the API
  loads it by name + stage, not by run ID.
- `scripts/train_all.py` is the only place that trains + logs. **The API never trains.**

---

## 8. Git workflow

- Short feature branches → PR into a **protected `main`**; CI must pass to merge.
- Keep PRs small; review lightly but always.
- **Notebook merge conflicts are the #1 hazard.** Two rules:
  1. Install `nbstripout` as a git filter so notebook *outputs* aren't committed.
  2. Two people never edit the same notebook (number + prefix them by owner).
  The deeper fix is the layout: real logic in `src/` keeps notebooks thin and disposable.

---

## 9. How it wires together at runtime

Demo-time flow:
`Streamlit → Coordinator (LangGraph) → agent → FastAPI endpoint → loads MLflow model →
finance engine assembles 3 scenarios → agent narrates with SHAP + citations.`

`docker-compose up` brings API + app up together.

**For the KPMG demo: pre-cache agent/LLM outputs as JSON.** Live LLM calls fail at the worst
moment. Ship a cached, deterministic demo with a clearly-labelled "live mode" toggle.

---

## 10. Testing & CI

- `pytest`, run on every PR via `.github/workflows/ci.yml`.
- Highest-value tests (cheap, deterministic): the finance engine (`test_finance.py`), schema
  validation of the ABT (`test_schema.py`), and a model-loads-and-predicts smoke test.
- Everyone writes tests for their own module; CI green is the merge gate.

---

## 11. Bootstrap sequence (first two days)

1. One person scaffolds the repo (tree above, `pyproject.toml`, `.gitignore`, `.env.example`,
   empty `ci.yml` running `pytest`, README skeleton). Everyone `pip install -e .` and confirms
   `import airbnb_iip` works.
2. Write `docs/schema.md` + the `build_abt()` signature; commit synthetic `data/sample/`.
   **This is the moment the team unblocks.**
3. Finance engineer starts `finance/scenarios.py` as pure functions + tests against the sample
   (needs no trained models → UC2 centerpiece progresses from hour one).
4. Data lead fills in the real loader/clean/ABT behind the agreed signature.
5. Everyone else builds their `models/` module against `sample/`, then re-runs on the real ABT.

This means 3–4 people are productive before the real data is even cleaned.

---

## What is the ABT, and what does `build_abt()` do?

**ABT = Analytical Base Table** (from CRISP-DM, ML1 notes). A flat, two-dimensional,
modelling-ready table: **one row per instance** (here, one Airbnb **listing**), columns =
features describing it (+ a target like `price` for supervised models). It's the table that
feeds directly into any model. A poorly-built ABT produces weak models regardless of algorithm.

`build_abt(city)` is the one function that turns raw Inside Airbnb files into that table for a
given city. In order, it:

1. **Loads** the raw files for the city — `listings.csv.gz` (one row per listing),
   `calendar.csv.gz` (per-day price/availability), `reviews.csv.gz` (review text + dates).
2. **Cleans** (reusing the Madrid EDA logic): `$`-strings → float price; `t`/`f` → bool;
   bathrooms-text standardisation; dates → datetime; hierarchical neighbourhood-level price
   imputation; IQR/percentile outlier capping; drops known bad rows.
3. **Engineers features:** amenity count + key amenity dummies; `host_tenure_days`;
   `reviews_per_month`; price-per-person; neighbourhood aggregates; calendar-derived
   seasonality; the **estimated occupancy** (San Francisco model).
4. **Enriches** by joining external **district-level** data — sale €/m² and long-term rent
   index — onto each listing (the UC2 inputs).
5. **Attaches the `segment` label** from the clustering step (used as a feature + for benchmarking).
6. **Returns** a single DataFrame: one row per listing, the **same columns for all three cities**
   (the contract in `schema.md`), ready for any model.

Why it matters:
- It's the **single source of truth** for clean data. No two team members clean differently →
  no silent divergence between everyone's results.
- It's the **day-3 unblocker.** Once its signature + the synthetic sample exist, every other
  track can build in parallel.

Practical detail — make it **cached and reproducible**: set random seeds, and have it write the
result to `data/processed/<city>.parquet`, loading that if it already exists instead of
recomputing every time:

```python
def build_abt(city: str, force: bool = False) -> pd.DataFrame:
    out = PROCESSED_DIR / f"{city}.parquet"
    if out.exists() and not force:
        return pd.read_parquet(out)        # fast path: reuse cached ABT
    listings = load_city(city)             # 1. load
    cal      = load_calendar(city)
    rev      = load_reviews(city)
    df = clean(listings)                   # 2. clean
    df = add_features(df, cal, rev)        # 3. engineer (+ occupancy, seasonality)
    df = join_external(df, city)           # 4. enrich (€/m², rent index)
    df = attach_segments(df)               # 5. segment label
    validate_schema(df)                    # guard: matches docs/schema.md
    df.to_parquet(out)                     # 6. cache + return
    return df
```
