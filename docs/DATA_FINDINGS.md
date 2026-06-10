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
| Barcelona | 18,177 | 14 Dec 2025 → 14 Dec 2026 | 57% | empty |
| Málaga    |  9,714 | 30 Sep 2025 → 29 Sep 2026 | 57% | empty |

- **Availability** is fully populated and usable — it's the occupancy/seasonality signal. Madrid is
  the tightest market (46% available = highest booked share).
- **⚠️ `price` is 100% empty in every calendar** (~19M rows). Expected Inside Airbnb behaviour, not a
  download error. Nightly price must come from `listings.csv`. (Consistent with the SF occupancy
  model, which already infers bookings from review frequency rather than calendar price.)
- **Seasonality caveat:** the three snapshots start on different dates, so a clean month-by-month
  peak comparison across cities isn't valid yet from a single snapshot. Treat per-city seasonality
  as indicative until snapshots are aligned.

## Section 2 — Listings (price & supply)

| City | Listings | Median nightly price | Price coverage | Entire-home share |
|------|---------:|---------------------:|----------------|-------------------|
| Madrid    | 25,000 | €110 | ~76% ✅ | 67% |
| Barcelona | 18,177 | — | **0% ⚠️** | 65% |
| Málaga    |  9,714 | €102 | ~91% ✅ | 88% |

- Málaga is the most "whole-property" market (88% entire homes) — relevant for UC2's sell-vs-rent
  framing. Madrid and Barcelona have a larger private-room segment (~32–34%).
- **⚠️ Barcelona has no price anywhere** — empty in both calendar and listings. Open team decision
  before price-dependent UC2 work on BCN: alternate snapshot · impute from comparables / external
  €-per-m² · or de-scope BCN for price-dependent outputs (keep availability + sentiment).

## Section 3 — Sentiment (review analysis, UC3)

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
**feature for the investment model** (UC2/UC3), keyed on `listing_id`.

**Honest caveat on the model:** the README-specified TF-IDF + Naïve Bayes is trained on
lexicon-generated labels, so its high training accuracy (~0.94) reflects agreement with those weak
labels, not ground truth. Use the **correlation with real star ratings (0.23–0.29)** as the honest
validation metric. Natural v2 upgrade: a multilingual transformer benchmarked against this baseline.

---

## Recommendations (what to do with this)

1. **Lead the build with Madrid.** It has the lowest availability (highest demand), the deepest data coverage (25k listings, 1.3M reviews), and populated pricing data.
   *and* the deepest data (25k listings, 1.3M reviews, price populated). It's the strongest city to
   demonstrate the occupancy-driven investment case on — prioritise it for the first end-to-end UC2
   slice and the KPMG demo.
2. **Add sentiment to the ABT as a standalone feature, not a rating proxy.** Star ratings are
   sparse and clustered near 5.0★, so they carry little discriminating signal; the per-listing
   sentiment score varies independently and can separate listings the ratings can't. Merge
   `data/processed/<city>_sentiment.parquet` on `listing_id` and let the model weigh it directly.
3. **Resolve Barcelona pricing before any price-based recommendation includes BCN.** Until then,
   scope BCN to availability + sentiment outputs only, so no price-dependent number ships on data we
   don't have. Recommended path: source an alternate Inside Airbnb BCN snapshot with price populated.
4. **Re-pull all three cities at one snapshot date** if cross-city seasonality comparison matters
   for the deliverable — the current snapshots start months apart, which makes month-vs-month
   demand comparisons unreliable.

## Status summary

- ✅ **Availability + seasonality** ready for occupancy modelling (all 3 cities).
- ✅ **Sentiment feature** ready to merge into the ABT (all 3 cities).
- ⚠️ **Barcelona price** — needs a team decision before BCN price-dependent UC2 work.
- ⚠️ **Snapshot alignment** — confirm whether to re-pull all cities at one date for clean
  cross-city seasonality.
