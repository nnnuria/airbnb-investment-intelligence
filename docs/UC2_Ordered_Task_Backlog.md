# UC2-First — Ordered Task Backlog
**Build order for the Airbnb Investment Intelligence Platform · no member split · ~19 days to 25 June**

UC2 = *"I own a flat — should I Airbnb it, sell it, or keep it as a long-term rental?"*
Do UC2 fully first (mentor's favourite). UC1 and UC3 are stretch — and note **UC1 is almost
free once UC2 exists**, because UC2 already contains the Airbnb-revenue engine; UC1 just adds
a yield ratio on top.

**Two decisions already locked in (so the tasks below assume them):**
- **One global price model**, with `segment`, `neighbourhood`, `room_type` as features.
  Segmentation is a feature + benchmarking tool, **not** a prediction router (avoids the
  price×availability circularity and data fragmentation).
- **The MVP vertical slice for UC2** = loader + external-data join → revenue engine →
  financial scenario engine → one agent → Streamlit tab. Get this thin slice working
  end-to-end early (~day 10–12), then widen.

Legend: ★ = on the UC2 critical path · ☆ = stretch / later use case.

---

## Phase 0 — Project setup
1. Create the GitHub repo; agree folder structure (`data/ src/ models/ api/ agents/ app/ tests/ notebooks/`), branch strategy, and a `requirements.txt`/`pyproject` + virtual env.
2. ★ Lock the **UC2 demo script** in one paragraph — the exact story you'll tell on 25 June. Build backwards from it.
3. Add a minimal **GitHub Actions** workflow that just runs `pytest` on push (green from day 1; fill in tests as you go).
4. ★ Write `data/schema.md` — the agreed column names/types every track codes against. This is the contract; agree it before anyone writes models.
5. Set up a shared task board (GitHub Projects/Trello) reflecting these phases.

## Phase 1 — Data acquisition
6. ★ Download Inside Airbnb **detailed** files for **Madrid first** (`listings.csv.gz`, `calendar.csv.gz`, `reviews.csv.gz`, `neighbourhoods.geojson`, `neighbourhoods.csv`), then Barcelona and Málaga.
7. ★ **Acquire external market data (now critical path, because UC2):**
   - Sale **price per m²** by district (Madrid City Council open data / published index).
   - **Long-term rent index** by district (INE / regional reference index / Idealista index).
   - Curate both into clean CSVs with a documented **source + snapshot date**. Prefer published indices over scraping (Idealista scraping breaches ToS — risky for a graded demo).
8. ☆ Begin collecting the **regulatory corpus** (official municipal STR rules per city) for the later Regulatory agent. Lower priority for UC2 but start the folder.

## Phase 2 — Data engineering (the unblocker)
9. ★ Build the **cleaning pipeline** (generalise the existing Madrid EDA logic into functions): dollar-string→float price; `t`/`f`→bool; bathrooms-text standardisation; dates→datetime; hierarchical neighbourhood-level **price imputation**; IQR/percentile outlier capping; drop the known bad rows (`beds=40`, `bedrooms=25`).
10. ★ **Feature engineering:** amenity count + key amenity dummies (dishwasher, AC, washer, parking, pool, lift…), `host_tenure_days`, `reviews_per_month`, price-per-person, neighbourhood aggregates, calendar-derived seasonality features.
11. ★ Build the **occupancy estimator** (San Francisco model) as a documented function:
    `bookings/mo ≈ (reviews_per_month / review_rate) × avg_length_of_stay`; convert to occupancy, cap at a realistic ceiling. Write the assumptions in a docstring.
12. ★ Emit a versioned **ABT per city** (parquet) via one importable `build_abt(city)` function. **This unblocks every other track — finish it by ~day 3.**
13. ★ **Spatial join** listings ↔ neighbourhood polygons (for maps and for matching external district data).
14. ★ **Join external data** (sale €/m², rent index) onto each listing's district — the UC2 enrichment.
15. Add **data-validation tests** (schema match, value ranges, null thresholds) and wire them into CI.

## Phase 3 — Exploratory analysis
16. Extend the EDA notebook to **all three cities** via the loader; refactor plots into reusable helpers the UI can call later.
17. Add a **cross-city comparison** section (price levels, room-type mix, regulatory intensity) — a slide-worthy insight on its own.
18. ★ **UC2-specific EDA:** distributions of gross yield, Airbnb-revenue vs long-term-rent vs sale-value ratios by district; identify where Airbnb beats long-term and where it doesn't. This *is* the UC2 narrative.

## Phase 4 — Feature selection methodology (curriculum showcase)
19. Correlation matrix + **VIF**; identify and treat multicollinearity (drop / combine / PCA).
20. **Recursive Feature Elimination (RFE)** + a short comparison of filter vs wrapper vs embedded selection; document the final feature set and *why*.
21. ☆ **PCA / Factor Analysis** on the numeric block — both as multicollinearity treatment and as the input space for clustering.

## Phase 5 — Segmentation (analytical deliverable, not a router)
22. Cluster on PCA/FA factor scores: **K-means** (elbow + **silhouette**), **hierarchical** (dendrogram), **DBSCAN**; compare.
23. **Profile and name** the market segments ("budget private rooms", "premium family entire-homes", "tourist-core studios"…).
24. Add the `segment` label as a **feature** in the modelling table and use segments for **neighbourhood/peer benchmarking** (feeds the Comparables agent).
25. **Documented experiment:** train per-segment price models vs the global model; compare RMSE/MAE on held-out data; report the result in the deck (this directly answers "why not a model per segment?").

## Phase 6 — Airbnb revenue engine (shared core; the Airbnb leg of UC2)
26. **OLS baseline** on `log(price)` with VIF/RFE features + assumption checks (homoscedasticity, normality, residuals). Read off interpretable amenity coefficients.
27. **XGBoost / LightGBM** tuned with CV + early stopping; **Random Forest** as the bagging comparison. Metrics: RMSE / MAE / R² on a held-out test set (scaler fit on train only — no leakage).
28. **SHAP** global + per-prediction explainability (per-prediction values become the agent's "why").
29. Select the production model; **log everything to MLflow** (params, metrics, artifacts) and register it.
30. ★ Compose **`predicted_price × occupancy × seasonality → annual revenue` with P10/P50/P90 bands** — the Airbnb scenario output UC2 needs.

## Phase 7 — UC2 financial scenario engine (the centerpiece)
31. ★ **Airbnb net scenario:** gross revenue minus documented costs (cleaning, management %, vacancy, platform fee, tax assumptions). Keep assumptions explicit and editable.
32. ★ **Long-term rental scenario:** annual net rent from the rent index for the property's district/size.
33. ★ **Sale scenario:** indicative sale value = price-per-m² × size; framed as one-off capital vs recurring income.
34. ★ **Break-even timeline** per scenario; optional payback/NPV for extra rigour.
35. ★ **Recommendation logic:** which scenario wins given the property spec + neighbourhood, with **uncertainty bands**, never as financial advice.
36. ★ Implement all of the above as **pure functions** with **unit tests** (deterministic → easy, high-value CI coverage).

## Phase 8 — Secondary ML (richness + later use cases)
37. ☆ **Demand/seasonality time series:** Prophet/ARIMA on reviews-per-month → seasonal multipliers feeding the revenue bands.
38. ☆ **Review sentiment** (Naïve Bayes + TF-IDF; **reuse the existing translated reviews**, don't re-run the 3-hour translation) + **aspect mining** (cleanliness/location/noise/wifi/host) for UC3.
39. ☆ **Apriori** amenity bundles (support/confidence/lift) for UC3 upgrade suggestions.

## Phase 9 — MLOps
40. **MLflow** registry populated; promote best models to `Production`.
41. **FastAPI** services: `/predict_price`, `/estimate_occupancy`, `/estimate_revenue`, `/compare_scenarios`. Agents call these, not notebooks.
42. **Dockerise** the API (and later the UI); `docker-compose` for one-command local run.
43. **pytest** suite (loader, financial engine, schema, model-loads-and-predicts smoke test) wired into CI.
44. **Evidently** drift report — simulate drift by feeding a later snapshot; document the retraining trigger. (Strong demo moment for KPMG's "Measure & Monitor" pillar.)

## Phase 10 — AI / agent layer
45. ★ **Market Analyst agent:** calls the revenue + scenario services, narrates with SHAP. For UC2 it produces the 3-way comparison brief.
46. ☆ **Regulatory agent:** RAG over the corpus (FAISS), answers with **source citations**.
47. ☆ **Comparables agent:** structured filters (segment, size, neighbourhood) + semantic retrieval; KNN as a baseline sanity check.
48. ★ **Coordinator (LangGraph):** routes the question, aggregates agent outputs, returns one explainable brief.
49. ★ **Governance layer:** output-bounds guardrails (no negative revenue/implausible yields), source citations, uncertainty surfaced, **human-review gate + "indicative, not financial/legal advice" disclaimer**, model cards, and a ~20-question **eval harness** run in CI.

## Phase 11 — User interface
50. ★ **Streamlit UC2 tab:** property input → three-scenario comparison + break-even chart + map + recommendation with bands.
51. ★ **Chat tab** routed through the coordinator.
52. ☆ Add **UC1 tab** (reuses the revenue engine + adds yield) and ☆ **UC3 tab** if time allows.
53. ★ **Pre-cache agent/LLM outputs as JSON** for the demo — **never run live API calls during the KPMG demo.**

## Phase 12 — Polish & deliver
54. End-to-end test on several **real example properties** (one clear "Airbnb wins", one "sell wins", one "rent wins").
55. Build the deck: problem → KPMG agentic-vision framing → live demo → architecture (use `Architecture_Diagram.svg`) → master-coverage matrix → governance → limitations/future.
56. Two full **dry-runs**; freeze code; tag a release.

---

### The five things to start tomorrow
1. Repo + structure + empty CI (Phase 0).
2. Download Madrid detailed files **and** kick off external-data sourcing — that's the new UC2 bottleneck (Phase 1).
3. Generalise the Madrid cleaning into `build_abt()` + finalise `schema.md` (Phase 2).
4. Start the financial scenario engine as pure functions with tests — it's the UC2 centerpiece and doesn't need the ML models finished to begin (Phase 7, can run in parallel with modelling).
5. Run VIF/RFE + OLS baseline on the Madrid ABT the moment it lands (Phases 4 & 6).
