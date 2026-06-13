# Product Task Backlog — Airbnb or Sell + Optimisation Flow
**Build order for the Airbnb Investment Intelligence Platform · no member split · ~19 days to 25 June**

**Primary flow:** *"I own a property — should I list it on Airbnb or sell it?"*
The platform compares projected Airbnb net revenue against the property's indicative sale value
and returns a clear, data-grounded recommendation with confidence bands.

**Secondary (optimisation) flow:** *"I've decided to Airbnb — how do I maximise revenue?"*
Triggered after the owner receives the primary recommendation and chooses Airbnb.
Returns property-specific improvements: amenities to add, renovation opportunities, pricing
strategy, and a prioritised action list with cost-vs-revenue trade-off analysis.

Build the primary flow fully first (mentor's favourite). The optimisation flow is the natural
next step once the revenue engine exists. UC1 (pre-purchase screening) is a stretch goal —
almost free once the primary flow is done, since it reuses the same revenue engine.

**Two decisions already locked in (so the tasks below assume them):**
- **One global price model**, with `segment`, `neighbourhood`, `room_type` as features.
  Segmentation is a feature + benchmarking tool, **not** a prediction router (avoids the
  price×availability circularity and data fragmentation).
- **The MVP vertical slice** = loader + external-data join → revenue engine →
  Airbnb-vs-sell scenario engine → one agent → Streamlit tab. Get this thin slice working
  end-to-end early (~day 10–12), then widen into the optimisation flow.

Legend: ★ = on the primary critical path · ☆ = optimisation flow / stretch.

---

## Phase 0 — Project setup
~~1. Create the GitHub repo; agree folder structure (`data/ src/ models/ api/ agents/ app/ tests/ notebooks/`), branch strategy, and a `requirements.txt`/`pyproject` + virtual env.~~
2. ★ Lock the **demo script** in one paragraph — the exact story you'll tell on 25 June: owner inputs property → platform recommends Airbnb or sell → owner chooses Airbnb → platform returns optimisation recommendations. Build backwards from it.
3. Add a minimal **GitHub Actions** workflow that just runs `pytest` on push (green from day 1; fill in tests as you go).
4. ★ Write `data/schema.md` — the agreed column names/types every track codes against. This is the contract; agree it before anyone writes models.
5. Set up a shared task board (GitHub Projects/Trello) reflecting these phases.

## Phase 1 — Data acquisition
~~6. ★ Download Inside Airbnb **detailed** files for **Madrid first** (`listings.csv.gz`, `calendar.csv.gz`, `reviews.csv.gz`, `neighbourhoods.geojson`, `neighbourhoods.csv`), then Barcelona and Málaga.~~
7. ★ **Acquire external market data (critical path — needed for the sell-side comparison):**
   - Sale **price per m²** by district (Madrid City Council open data / published index) — this
     is the sell scenario anchor in the Airbnb-vs-sell recommendation.
   - Curate into a clean CSV with a documented **source + snapshot date**. Prefer published
     indices over scraping (Idealista scraping breaches ToS — risky for a graded demo).
8. ☆ Begin collecting the **regulatory corpus** (official municipal STR rules per city) for the Regulatory agent. Lower priority but start the folder early.

## Phase 2 — Data engineering (the unblocker)
~~9. ★ Build the **cleaning pipeline** (generalise the existing Madrid EDA logic into functions): dollar-string→float price; `t`/`f`→bool; bathrooms-text standardisation; dates→datetime; hierarchical neighbourhood-level **price imputation**; IQR/percentile outlier capping; drop the known bad rows (`beds=40`, `bedrooms=25`).~~
10. ★ **Feature engineering:** amenity count + key amenity dummies (dishwasher, AC, washer, parking, pool, lift…), `host_tenure_days`, `reviews_per_month`, price-per-person, neighbourhood aggregates, calendar-derived seasonality features.
11. ★ Build the **occupancy estimator** (San Francisco model) as a documented function:
    `bookings/mo ≈ (reviews_per_month / review_rate) × avg_length_of_stay`; convert to occupancy, cap at a realistic ceiling. Write the assumptions in a docstring.
12. ★ Emit a versioned **ABT per city** (parquet) via one importable `build_abt(city)` function. **This unblocks every other track — finish it by ~day 3.**
13. ★ **Spatial join** listings ↔ neighbourhood polygons (for maps and for matching external district data).
14. ★ **Join external data** (sale €/m²) onto each listing's district — the sell-side anchor for the Airbnb-vs-sell comparison.
15. Add **data-validation tests** (schema match, value ranges, null thresholds) and wire them into CI.

## Phase 3 — Exploratory analysis
~~16. Extend the EDA notebook to **all three cities** via the loader; refactor plots into reusable helpers the UI can call later.~~
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

## Phase 7 — Decision engine: Airbnb vs. sell (the primary flow centerpiece)
31. ★ **Airbnb net scenario:** gross revenue minus documented costs (cleaning, management %, vacancy, platform fee, tax assumptions). Keep assumptions explicit and editable in `config.yaml`.
32. ★ **Sell scenario:** indicative sale value = price-per-m² × size; framed as one-off capital realisation vs recurring Airbnb income, with break-even timeline (how many years of Airbnb revenue equals the sale value).
33. ★ **Recommendation logic:** compare Airbnb net annual income against the opportunity cost of not selling, factoring in the owner's stated goals (income vs. capital). Output includes **uncertainty bands** (P10/P50/P90), never presented as financial advice.
34. ★ Implement all of the above as **pure functions** with **unit tests** (deterministic → easy, high-value CI coverage).

## Phase 7b — Optimisation flow (secondary, unlocked after the primary recommendation)
35. ☆ **Feature gap analysis:** compare the owner's property attributes against top-performing comparables in the same segment and neighbourhood. Surface the gaps (missing amenities, lower review scores, suboptimal pricing).
36. ☆ **Amenity recommendations:** use Apriori association rules to identify amenity bundles that correlate with higher occupancy and nightly rates. Rank by expected revenue uplift and ease of implementation.
37. ☆ **Renovation / remodelling opportunities:** identify property improvements (e.g. adding a bedroom, upgrading bathroom, installing AC) with estimated cost vs. projected revenue impact. Flag trade-offs explicitly.
38. ☆ **Pricing strategy:** recommend an optimal base price and seasonal pricing adjustments based on comparable listings and demand patterns.
39. ☆ **Prioritised action list:** rank all recommendations by estimated net revenue impact minus estimated cost, so the owner can act on the highest-ROI improvements first.

## Phase 8 — Secondary ML (feeds both the decision engine and the optimisation flow)
40. ☆ **Demand/seasonality time series:** Prophet/ARIMA on reviews-per-month → seasonal multipliers feeding the revenue bands in the primary recommendation.
41. ☆ **Review sentiment** (Naïve Bayes + TF-IDF; **reuse the existing translated reviews**, don't re-run the 3-hour translation) + **aspect mining** (cleanliness/location/noise/wifi/host) — feeds the optimisation flow's gap analysis.
42. ☆ **Apriori** amenity bundles (support/confidence/lift) — the engine behind optimisation flow amenity recommendations.

## Phase 9 — MLOps
43. **MLflow** registry populated; promote best models to `Production`.
44. **FastAPI** services: `/predict_price`, `/estimate_occupancy`, `/estimate_revenue`, `/airbnb_vs_sell`, `/optimise`. Agents call these, not notebooks.
45. **Dockerise** the API (and later the UI); `docker-compose` for one-command local run.
46. **pytest** suite (loader, financial engine, schema, model-loads-and-predicts smoke test) wired into CI.
47. **Evidently** drift report — simulate drift by feeding a later snapshot; document the retraining trigger. (Strong demo moment for KPMG's "Measure & Monitor" pillar.)

## Phase 10 — AI / agent layer
48. ★ **Market Analyst agent:** calls the revenue + scenario services, narrates with SHAP. Produces the Airbnb-vs-sell recommendation brief for the primary flow.
49. ☆ **Optimisation agent:** calls the feature-gap and amenity-recommendation services; produces a ranked, cost-aware improvement plan for the secondary flow.
50. ☆ **Regulatory agent:** RAG over the corpus (FAISS), answers with **source citations**. Surfaces STR licensing requirements relevant to the property's city and district.
51. ☆ **Comparables agent:** structured filters (segment, size, neighbourhood) + semantic retrieval; KNN as a baseline sanity check. Feeds both the primary benchmarking and the optimisation gap analysis.
52. ★ **Coordinator (LangGraph):** routes the user's intent (decision vs. optimisation), aggregates agent outputs, returns one explainable brief per flow.
53. ★ **Governance layer:** output-bounds guardrails (no negative revenue/implausible yields), source citations, uncertainty surfaced, **human-review gate + "indicative, not financial/legal advice" disclaimer**, model cards, and a ~20-question **eval harness** run in CI.

## Phase 11 — User interface
54. ★ **Streamlit primary tab:** property input → Airbnb-vs-sell comparison + break-even chart + map + recommendation with confidence bands. Clear CTA: "Proceed to optimisation" if Airbnb is recommended or chosen.
55. ★ **Optimisation tab:** triggered after the primary recommendation. Shows improvement recommendations grouped by category (amenities, renovation, pricing), each with estimated revenue uplift and cost estimate.
56. ★ **Chat tab** routed through the coordinator — supports both flows conversationally.
57. ☆ Add **Pre-purchase screening tab** (reuses the revenue engine + adds yield) if time allows.
58. ★ **Pre-cache agent/LLM outputs as JSON** for the demo — **never run live API calls during the KPMG demo.**

## Phase 12 — Polish & deliver
59. End-to-end test on several **real example properties** (one clear "Airbnb wins", one "sell wins", one strong optimisation story).
60. Build the deck: problem → KPMG agentic-vision framing → live demo → architecture (use `Architecture_Diagram.svg`) → master-coverage matrix → governance → limitations/future.
61. Two full **dry-runs**; freeze code; tag a release.

---

### The five things to start tomorrow
1. Repo + structure + empty CI (Phase 0).
2. Download Madrid detailed files **and** kick off external sale-price data sourcing — that's the sell-side anchor for the primary recommendation (Phase 1).
3. Generalise the Madrid cleaning into `build_abt()` + finalise `schema.md` (Phase 2).
4. Start the Airbnb-vs-sell scenario engine as pure functions with tests — it's the primary flow centerpiece and doesn't need the ML models finished to begin (Phase 7, can run in parallel with modelling).
5. Run VIF/RFE + OLS baseline on the Madrid ABT the moment it lands (Phases 4 & 6).
