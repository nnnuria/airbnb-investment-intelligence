# Data Findings — Calendar · Listings · Sentiment

*Airbnb Investment Intelligence Platform — IE × KPMG Lighthouse*
*Cities: Madrid · Barcelona · Málaga. Source: Inside Airbnb (snapshots vary by city — see below).*

City→file mapping confirmed by 100% `listing_id` overlap across calendar, listings, and reviews.
Each metric below is computed on the downloaded data; full code in `notebooks/02_calendar_eda.ipynb`
and `src/airbnb_iip/models/nlp.py`.

---

## Section 1 — Calendar (availability & seasonality)

| City | Listings | Calendar span | Availability rate | Price column |
|------|---------:|---------------|------------------:|--------------|
| Madrid    | 25,000 | 14 Sep 2025 → 14 Sep 2026 | 46% | empty |
| Barcelona *(superseded)* | 18,177 | 14 Dec 2025 → 14 Dec 2026 | 57% | empty |
| Málaga    |  9,714 | 30 Sep 2025 → 29 Sep 2026 | 57% | empty |

- **Availability** is fully populated and usable — it's the occupancy/seasonality signal. Madrid has the lowest availability (46% available = highest booked share).
- **⚠️ `price` is 100% empty in every calendar** (~19M rows). Expected Inside Airbnb behaviour, not a
  download error. Nightly price must come from `listings.csv`. (Occupancy is derived from calendar
  *availability* — `available == 'f'` — not calendar price; the earlier review-frequency estimate was
  retired, having correlated only ≈ 0.13 with calendar occupancy.)
- **Committed snapshot = September 2025.** The Barcelona row above is a *superseded* earlier
  snapshot (see the Section 2 note); the committed/modelled data is the September snapshot —
  `last_scraped` Madrid/Barcelona 14 Sep, Málaga 30 Sep 2025 in the committed parquet — and the
  occupancy model additionally uses an equal 120-day per-city window. Treat single-snapshot
  seasonality as indicative.

## Section 2 — Listings (price & supply)

| City | Listings | Median nightly price | Price coverage | Entire-home share |
|------|---------:|---------------------:|----------------|-------------------|
| Madrid    | 25,000 | €110 | ~76% ✅ | 67% |
| Barcelona | 18,927 | €143 *(superseded)* | ~79% ✅ | 62% *(superseded)* |
| Málaga    |  9,714 | €102 | ~91% ✅ | 88% |

- Málaga is the most "whole-property" market (88% entire homes) — most relevant for the
  Airbnb-vs-sell decision framing, where entire-home listings have the clearest comparison to
  a sale. Madrid and Barcelona have a larger private-room segment (~33–38%).
- **Cleaned-ABT figures (use these downstream).** The table above reports the **raw** snapshots (pre-cleaning). After cleaning (`Data/processed/listings_all_cities.parquet`, 42,823 listings; 15,199 in Barcelona), the figures the report and deck should quote are: median nightly price **Madrid €110 / Barcelona €130 / Málaga €102**, and entire-home share **72% / 69% / 89%**. (The Barcelona €143 / 62% in the Section 2 table are from a **superseded** earlier snapshot — see the note below.)
- **Barcelona snapshot (resolved).** The committed/modelled Barcelona data is the **September 2025** snapshot — `last_scraped` 14–15 Sep 2025 in `Data/processed/listings_all_cities.parquet` (same as Madrid; `config/config.yaml` `snapshot_date 2025-09-14`), price populated, **15,199 cleaned listings, median €130**. The September snapshot's **raw count is 19,410** (per the committed `notebooks/barcelona/01_cleaning.ipynb`: `Shape (19410, 79)` → **15,199** cleaned). Earlier Barcelona snapshots recorded in this doc — **Dec 2025** (18,177, empty price) and **Jun 2025** (18,927, €143) — are superseded.

## Section 3 — Sentiment (review analysis, optimisation flow)

Multilingual pipeline (handles EN + ES/FR/DE/IT/PT) → per-listing sentiment feature. 40k reviews
scored per city for this summary; full run via `scripts/run_sentiment.py --full`.

| City | Positive | Neutral | Negative | Dominant languages | Validation (corr vs rating) |
|------|---------:|--------:|---------:|--------------------|----------------------------:|
| Madrid    | 60.6% | 36.0% | 3.4% | ES 42%, EN 39% | 0.23 |
| Barcelona | 71.7% | 25.2% | 3.1% | EN 59%, ES 12% | 0.29 |
| Málaga    | 64.8% | 32.0% | 3.2% | EN 47%, ES 22% | 0.28 |

**Key observations**
- **Happiest guests: Barcelona** (71.7% positive).
- **Negative-review rates are statistically similar** across all three (3.1–3.4%) — don't headline a
  "most negative city"; the differences are within noise.
- **Multilingual was essential:** Madrid reviews are 42% Spanish — an English-only model would have
  mis-scored nearly half the data.

**Output produced:** per-listing sentiment (`mean_sentiment`, `pct_positive`, `pct_negative`,
`n_reviews`) in `data/processed/<city>_sentiment.parquet` — ready to join into the ABT as a
**feature for the revenue model** (primary flow) and as a **gap signal** in the optimisation
flow's guest-experience recommendations, keyed on `listing_id`.

**Honest caveat on the model:** the README-specified TF-IDF + Naïve Bayes is trained on
lexicon-generated labels, so its high training accuracy (~0.94) reflects agreement with those weak
labels, not ground truth. Use the **correlation with real star ratings (0.23–0.29)** as the honest
validation metric. Natural v2 upgrade: a multilingual transformer benchmarked against this baseline.

---

## Recommendations (what to do with this)

1. **Lead the build with Madrid.** Madrid has the lowest availability (highest demand)
   *and* the deepest data (25k listings, 1.3M reviews, price populated). It's the strongest city to
   demonstrate the occupancy-driven investment case on — prioritise it for the first end-to-end UC2
   slice and the KPMG demo.
2. **Add sentiment to the ABT as a standalone feature, not a rating proxy.** Star ratings are
   sparse and clustered near 5.0★, so they carry little discriminating signal; the per-listing
   sentiment score varies independently and can separate listings the ratings can't. Merge
   `data/processed/<city>_sentiment.parquet` on `listing_id` and let the model weigh it directly.
   In the optimisation flow, sentiment aspect scores (cleanliness, location, wifi, host) also
   surface targeted improvement recommendations.
3. **Barcelona pricing is resolved — proceed with BCN in price-based outputs.** The committed
   Barcelona data is the 2025-09-14 snapshot with price populated (cleaned median €130); the
   earlier empty-price / €143 Barcelona snapshots are superseded. No imputation or de-scoping needed.
4. **Snapshot dates (committed).** The committed snapshots are Madrid/Barcelona 14 Sep and
   Málaga 30 Sep 2025 (within ~2 weeks).

## Status summary

- ✅ **Availability + seasonality** ready for occupancy modelling (all 3 cities).
- ✅ **Sentiment feature** ready to merge into the ABT (all 3 cities).
- ✅ **Barcelona price** — RESOLVED; Barcelona uses the 2025-09-14 snapshot (price populated, cleaned median €130). All cities priced.
- ✅ **Snapshot dates** — committed snapshots are all September 2025 (Madrid/Barcelona 14 Sep, Málaga 30 Sep).
