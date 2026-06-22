# Notebook Improvements — #2A & Segmentation

*By Almu. Improvements to Nuria's `data_segmentation_eda.ipynb` and `segmentation.ipynb`.
Verified by executing `notebooks/segmentation.ipynb` end-to-end on the real data
(`nbconvert --execute`, no errors) and regenerating `listings_segmented.parquet`.*

> ✅ **`segmentation.ipynb` has been executed end-to-end** on the real data (no
> errors) and `listings_segmented.parquet` regenerated with the new method.
> **`data_segmentation_eda.ipynb`** (additive, no method change) still needs a
> routine run to refresh its displayed outputs.

---

## `data_segmentation_eda.ipynb` (#2A) — additive, no computation changed
1. **Justified the 500-record threshold** in the intro (why 500 → stable train/val/test split).
2. **§7 Segment-level feasibility** — reads `listings_segmented.parquet`; all 5 market segments clear 500, so segment-aware modelling is viable (unlike property-type).
3. **§8 Empirical check** — cites the combined price model's per-city R² (0.797 / 0.819 / 0.752) as evidence city-splitting is feasible but *unnecessary*.
4. **§9 Conclusion & recommendation** — the written verdict the notebook was missing: **one combined model with `city` (+ `Segment_Name`) as features; do not split by property type; #2B to confirm empirically.**

## `segmentation.ipynb` — method fix + reproducibility
1. **Method fix (Phase 8):** added `PROTECTED_FEATURES` (location, capacity/structure, quality, listing type) that the CV-based elimination **never drops**. The prior run dropped `is_central`/`is_madrid`/`is_barcelona` and kept low-signal amenity flags (`has_hangers`, `has_hair_dryer`, …) — so segments ignored location, the #1 price driver. Now segments stay aligned with what moves price/occupancy.
2. **Reproducibility (Phase 9):** `auto_select_iteration()` replaces the hard-coded `WINNING_ITERATION = 22` (max silhouette among no-micro-cluster, interpretable-k iterations; manual override still possible).
3. **Reproducibility (Phase 13):** DBSCAN `eps` is auto-selected from the k-NN distance **knee** instead of the hard-coded `EPS = 1.5`.
4. **Robustness (Phases 5 & 7):** removed deprecated `fillna(..., inplace=True)` chained-assignment patterns.
5. **New "Limitations & Interpretation" cell:** moderate silhouette, mega-cluster caveat, protected-feature rationale, and the amenity-flag coupling warning.

### Verified effect — actual executed notebook output

Final config: **lean protected core (capacity + quality + room-type + centrality) + PCA(0.9), k constrained ≥4**. Numbers are the real `nbconvert --execute` output on the 42,811-row dataset, **not** estimates.

| | Committed (manual pick) | **Executed improved** |
|---|---|---|
| Silhouette | 0.36 | **0.365** (k=4) |
| Largest segment | **56.7%** mega-cluster | **24.8%** of total (27.5% of clean) |
| <5% micro-clusters | — | **0** |
| Location in segments | dropped | **kept** (centrality) |
| Amenity trivia (hangers, hair dryer) | present | **gone** |
| Reproducible | manual `=22` / `eps=1.5` | **deterministic, validated** |
| DBSCAN check | — | 63 clusters / 31% noise → K-means chosen |

Actual `Segment_Name` output (named by **structure**): **Budget private rooms** 9,873 (€75, 95% private) · **Central entire homes** 10,262 (€159, 100% central) · **Non-central entire homes** 10,600 (€162, 0% central) · **Premium entire homes** 7,842 (€239, large) · **Ultra/Extreme** 4,234.

### Bug caught in audit (and fixed)
My first version of Phase 10 chose k by "max silhouette, no micro-clusters" over k=2..8 — which selects the **trivial k=2 split (74% blob)**, worse than the original. Fixed by **constraining k≥4** (a 2–3-way split is too coarse for a market segmentation). Verified: it now picks k=4.

### Mid-market — resolved (team feedback incorporated)
Earlier I called the two mid tiers (€159/€162) a "fuzzy" mid-market. Profiling the actual clusters shows the split is **clean and meaningful**: **Central entire homes (100% central)** vs **Non-central entire homes (0% central)** — same price, different location, which is exactly the signal `is_central` was protected to preserve. Segments are now **named by structure** so this is self-evident on a slide, and it answers the strongest likely challenge ("why are two same-price tiers separate?").

What is *not* on offer: a single silhouette much above ~0.36 — STR listings sit on continuous gradients, so segments are a useful coarse grouping, not crisp natural classes. A coarser **k=3** (Budget/Mid/Premium, silhouette 0.33, largest 42.7%) remains available if a pure price story is ever preferred.

## Known coupling to flag
`segmentation.ipynb` auto-includes whatever `has_*`/`bundle_*` columns
`amenity_feature_engineering.ipynb` writes into `listings_all_cities.parquet`.
If that notebook is re-run with different thresholds, **re-run segmentation too**.
This is the same amenity-pipeline divergence noted in `FEATURE_ENGINEERING.md §2`.
