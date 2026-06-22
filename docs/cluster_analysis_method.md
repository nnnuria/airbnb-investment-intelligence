# Cluster Analysis Methodology — Reusable Spec

This document describes the **exact segmentation approach** to replicate on a new
dataset. Follow the steps in order. Anything marked **[ADAPT]** is dataset-specific
and must be re-decided for the new data; the *method* itself stays identical.

The defining characteristics of this approach are:
1. Split the data into a **"clean" subset** and an **"extreme-outlier" subset**, and
   treat the extreme group as its own manually-defined segment (never fed to K-Means).
2. **Exclude the target/outcome variable** from the clustering features.
3. Select clustering features via an **iterative backward-elimination loop** driven by
   the **Coefficient of Variation (CV)** of cluster means, re-optimising `k` each step.
4. Name the final segments by ranking them on the held-out target variable.

---

## Phase 0 — Setup

- Standard stack: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scipy.stats`,
  `sklearn` (`KMeans`, `DBSCAN`, `silhouette_score`, `StandardScaler`,
  `KNNImputer`, `SimpleImputer`).
- Set `RANDOM_STATE = 42` and reuse it everywhere a seed is accepted.
- Load the raw data into `df_raw`, work on a copy `df = df_raw.copy()`.

## Phase 1 — Data audit (quick)

- `df.info()`, missing-value share per column (`df.isna().mean()` sorted desc),
  and `df.describe()` on numeric columns.
- Goal is just to know shape, dtypes, missingness, and obvious scale issues.

## Phase 2 — Cleaning & obvious-error fixes

- Inspect the extreme tail of each key variable (sort desc, look at top rows).
- Drop **physically impossible / data-entry-error** rows only (e.g. a floor value
  far above any plausible real value). This is *not* statistical outlier removal —
  it's removing corrupt records. **[ADAPT]** the impossible-value thresholds.

## Phase 3 — Distribution checks & transforms

- Classify columns into three groups programmatically:
  - **binary**: numeric, exactly 2 unique values, subset of `{0,1}`.
  - **continuous**: numeric, not binary, not an ID column.
  - **categorical**: object / category dtype.
- Plot a histogram (with KDE) grid for the continuous columns.
- For any **right-skewed** continuous variable, create a `log1p_<var>` version
  (`np.log1p(...)`) and plot it to confirm the skew is reduced. Use the log1p version
  downstream (log1p handles zero values safely). **[ADAPT]** which variables get logged.

## Phase 4 — Outlier handling via IQR flag-counting (KEY STEP)

This step *partitions* the data; it does not delete outliers.

1. For each continuous column compute IQR bounds:
   `lower = Q1 - 1.5*IQR`, `upper = Q3 + 1.5*IQR`.
2. Build a boolean flag matrix (one column per continuous var) marking out-of-bound
   values, then `n_outlier_cols = flags.sum(axis=1)` per row.
3. **Partition:**
   - `df_clean` = rows with `n_outlier_cols <= 1` (0 or 1 outlier column).
   - `df_multi_outliers` = rows with `n_outlier_cols > 1`.
4. `df_multi_outliers` becomes a **separate, manually-profiled segment** (the
   "extreme/ultra" group). It is NOT clustered. Only `df_clean` goes into K-Means.

> Rationale: multivariate-extreme records distort centroid-based clustering, so they
> are carved off and described on their own.

## Phase 5 — Missing-value imputation

Impute `df_clean` and `df_multi_outliers` **separately** (same logic, computed
independently on each subset — no leakage between them). Choose an imputation order
based on logical dependency between variables, simplest/most-independent first.

- **Categorical / area-type fields:** fill with the **group mode** within a sensible
  grouping key (e.g. `groupby(location).transform(fill with mode))`.
- **Continuous fields with a natural predictor:** fill with the **group mean** within
  a fine grouping key, then fall back to a coarser grouping key for anything still
  missing (e.g. `groupby([region, size]).mean()` → then `groupby([region]).mean()`).
- **Structural binary fields** (e.g. has-elevator, is-exterior): use **`KNNImputer`**
  (`n_neighbors=5`, `weights="distance"`) over a fixed set of related structural
  features (one-hot encode any categoricals first), then **threshold the imputed
  value at 0.5 and cast back to `int`** so the column stays binary.

**[ADAPT]** the grouping keys, the imputation order, and the KNN feature set.

## Phase 6 — Domain feature engineering

Create interpretable engineered flags from raw fields (binary 0/1 is ideal for
clustering). In the source project these were location/quality flags, e.g.
`is_<premium_category>`, `is_<high_demand_category>`, and a within-group
"top-quartile of target" flag built with
`groupby(key)[target].transform(lambda x: (x.rank(pct=True) >= 0.75))`.

**[ADAPT]** entirely to the new domain. **Important:** if a flag is derived from the
**target variable**, do NOT use it as a clustering feature (see Phase 7).

## Phase 7 — Build the segmentation feature table (X)

- Work from `df_clean` → `df_segmentation = df_clean.copy()`.
- Drop very granular / identifier columns (address, record number, raw ID).
- **Exclude the target/outcome variable** (e.g. price/rent) from clustering features —
  cluster on *attributes*, not on the thing you'll later predict. Keep the target in
  the dataframe for profiling and the supervised phase, just don't feed it to K-Means.
- Define candidate feature lists:
  - `CONTINUOUS` = the (possibly log-transformed) continuous attributes.
  - `BINARY` = the structural + engineered binary flags.
- `CLUSTER_FEATURES = [c for c in CONTINUOUS + BINARY if c in df_segmentation]`.
- Coerce to numeric and fill any residual NaNs with the column median (safety net).

## Phase 8 — Iterative K-Means feature selection (THE CORE METHOD)

Backward elimination by Coefficient of Variation. Scale features each iteration with
`StandardScaler` (K-Means is distance-based, so scaling is mandatory).

```
current_features = list(CLUSTER_FEATURES)
iteration_results = []
iteration = 1

while len(current_features) >= 2:
    X = StandardScaler().fit_transform(df_segmentation[current_features])

    # 1) pick best k in [2, 5] by silhouette
    best_k, best_sil = 2, -1
    for k in range(2, 6):
        labels = KMeans(n_clusters=k, random_state=42, n_init=20).fit_predict(X)
        sil = silhouette_score(X, labels)
        if sil > best_sil:
            best_sil, best_k = sil, k

    # 2) fit with best k, profile cluster means
    labels = KMeans(n_clusters=best_k, random_state=42, n_init=20).fit_predict(X)
    df_segmentation['KMeans_Iter_Cluster'] = labels
    cluster_means = df_segmentation.groupby('KMeans_Iter_Cluster')[current_features].mean()

    # 3) flag tiny (<5%) clusters as a quality signal
    sizes = df_segmentation['KMeans_Iter_Cluster'].value_counts()
    small_clusters = int((sizes / len(df_segmentation) < 0.05).sum())

    # 4) Coefficient of Variation per feature across cluster means
    #    CV = std(cluster means) / (|mean of cluster means| + 1e-9)
    cv = {c: cluster_means[c].std() / (cluster_means[c].abs().mean() + 1e-9)
          for c in current_features}
    cv_series = pd.Series(cv).sort_values()
    feature_to_drop = cv_series.index[0]          # lowest CV = least discriminative

    # 5) record metrics BEFORE dropping
    iteration_results.append({
        'Iteration': iteration, 'Num_Features': len(current_features),
        'Best_k': best_k, 'Silhouette': round(best_sil, 4),
        '< 5% Clusters': small_clusters,
        'Dropped_Feature': feature_to_drop, 'CV_of_Dropped': round(cv_series.iloc[0], 4),
        'Remaining_Features': ", ".join(current_features),
    })

    # 6) drop least discriminative feature, continue
    current_features.remove(feature_to_drop)
    iteration += 1

summary_df = pd.DataFrame(iteration_results)   # display this table
```

Optionally, in each iteration, draw a bar-chart grid of cluster means per feature
(title each with its CV) to visually inspect separation.

## Phase 9 — Choose the winning iteration

From `summary_df`, pick the iteration that best balances:
- **High silhouette** (well-separated clusters),
- **Interpretable `k`** (small number of meaningful segments),
- **A sensible feature count** (not too sparse),
- **Few/no <5% micro-clusters**, and
- the feature it would drop next still having a **non-trivial CV** (i.e. you're not
  yet throwing away anything discriminative).

Record the exact feature set for that iteration as `FINAL_CLUSTER_FEATURES`. Note the
subtlety from the source notebook: the feature dropped *after* an iteration's
silhouette is recorded is still part of that iteration's solution — be careful that
`FINAL_CLUSTER_FEATURES` matches the chosen iteration's feature list exactly.

## Phase 10 — Validate final k (elbow + silhouette)

On `FINAL_CLUSTER_FEATURES` (scaled), loop `k` in `range(2, 11)` collecting
`inertia_` and `silhouette_score`. Plot the elbow (inertia) and silhouette curves
side by side, mark the chosen `k`. Confirm the chosen `k` is defensible.

## Phase 11 — Fit the final model & name segments

- Refit `KMeans(n_clusters=K_FINAL, random_state=42, n_init=25, max_iter=500)` on the
  scaled `FINAL_CLUSTER_FEATURES`; store labels in `df_segmentation['Cluster_Final']`.
- Report final silhouette and cluster sizes / percentages.
- **Name segments by the target:** compute mean target per cluster, sort ascending,
  and map cluster IDs to business-friendly names from lowest to highest
  (e.g. "Standard" → … → "Premium"). Store as `Segment_Name`.
- Re-attach `df_multi_outliers` as the top manually-named segment
  (e.g. "Ultra / Extreme").

## Phase 12 — Profile & visualise segments

- Build a profile table: `groupby(Segment_Name)[profile_features].mean()`, then append
  the mean row of `df_multi_outliers` as the extreme segment. Order segments by target.
- Visuals used: a **bar-chart grid** (one subplot per profile feature, percent-scale
  the binary flags to 0–100) and a **radar/spider chart** (min-max scale each feature
  to 0–1 across segments). Use a fixed colour-per-segment palette.

## Phase 13 — Sanity-check against DBSCAN

Run `DBSCAN` on the same scaled final feature set as a robustness check. In the source
project it produced many tiny, high-noise, non-interpretable micro-clusters, so
**iterative K-Means was selected** on the basis of higher silhouette + clean,
interpretable, business-meaningful segments. Report this comparison briefly.

---

## Quick checklist for Claude Code

- [ ] Load, audit, drop only impossible records.
- [ ] Auto-classify binary / continuous / categorical; log-transform skewed vars.
- [ ] IQR flag-count → split `df_clean` (≤1 flag) vs `df_multi_outliers` (>1 flag).
- [ ] Impute the two subsets separately (group mode / group mean / KNN-for-binaries).
- [ ] Engineer domain flags; do not derive clustering features from the target.
- [ ] Build X excluding the target and identifiers; scale with StandardScaler.
- [ ] Run the iterative CV-based backward-elimination loop; produce the summary table.
- [ ] Choose the winning iteration; validate k with elbow + silhouette.
- [ ] Refit final K-Means; name segments by target rank; re-add the extreme segment.
- [ ] Profile with bar grid + radar; sanity-check vs DBSCAN.
