#!/usr/bin/env python3
"""
Generate notebooks/price_ml_model.ipynb.

Follows the structure of ML2_Group_Assignment - Marcos (9).ipynb, adapted for
a regression task (nightly price prediction) with 3-city Airbnb data.
"""
import json
import pathlib
import textwrap

OUT = pathlib.Path(__file__).parent.parent / "notebooks" / "price_ml_model.ipynb"


def code(src):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": textwrap.dedent(src).lstrip("\n"),
    }


def md(src):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": textwrap.dedent(src).lstrip("\n"),
    }


CELLS = [

    # ─────────────────────────────────────────────────────────────────────────
    # TITLE
    # ─────────────────────────────────────────────────────────────────────────
    md("""\
    # Airbnb Nightly Price Prediction — ML Model
    ## Capstone 2026 · IE × KPMG

    **Business Context:**
    To decide whether a property should be listed on Airbnb or sold, we need an
    unbiased estimate of what nightly price it *could* command — independent of
    what the current host charges. This model predicts that price from property
    and location features.

    This is **Stage 1** of a two-stage pipeline:
    - **Stage 1 (this notebook):** Predict `price_hat` from listing features.
    - **Stage 2:** Use `price_hat` + other features to predict occupancy rate,
      feeding into the NPV comparison engine.

    City-level EDA (distributions, seasonality, sentiment) was completed in the
    per-city notebooks. This notebook focuses on the combined 3-city dataset,
    feature engineering, and multi-model comparison.
    """),

    # ─────────────────────────────────────────────────────────────────────────
    # 0. SETUP
    # ─────────────────────────────────────────────────────────────────────────
    md("---\n## 0. Setup & Libraries"),

    code("""\
    import json
    import re
    import pathlib
    import warnings

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import seaborn as sns
    from scipy import stats

    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.ensemble import (
        RandomForestRegressor,
        GradientBoostingRegressor,
        HistGradientBoostingRegressor,
    )
    from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import SelectFromModel
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.neighbors import BallTree
    import shap
    import joblib

    warnings.filterwarnings("ignore")

    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)

    DATA_PATH = pathlib.Path("../Data/processed/listings_all_cities.parquet")
    MODEL_DIR = pathlib.Path("../models")
    MODEL_DIR.mkdir(exist_ok=True)

    sns.set_theme(style="whitegrid", palette="Set2", font_scale=1.05)
    print("Libraries loaded ✓")
    """),

    # ─────────────────────────────────────────────────────────────────────────
    # 1. DATA LOADING
    # ─────────────────────────────────────────────────────────────────────────
    md("---\n## 1. Data Loading"),

    code("""\
    df = pd.read_parquet(DATA_PATH)
    print(f"Dataset shape: {df.shape}")
    print(f"\\nColumn names and types:")
    for col in df.columns:
        print(f"  - {col} ({df[col].dtype})")
    """),

    # ─────────────────────────────────────────────────────────────────────────
    # 2. DATA VALIDATION
    # ─────────────────────────────────────────────────────────────────────────
    md("---\n## 2. Data Validation"),

    code("""\
    numerical_columns   = list(df.select_dtypes(include=[np.number]).columns)
    categorical_columns = list(df.select_dtypes(exclude=[np.number]).columns)
    print(f"Numerical columns   : {len(numerical_columns)}")
    print(f"Categorical columns : {len(categorical_columns)}")
    """),

    code("""\
    # Missing values
    missing     = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df  = pd.DataFrame({"Missing": missing, "Pct (%)": missing_pct.round(2)})
    missing_df  = missing_df[missing_df["Missing"] > 0].sort_values("Missing", ascending=False)
    print(f"Columns with missing values: {len(missing_df)}")
    missing_df.head(20)
    """),

    code("""\
    # Descriptive statistics
    df.describe().round(2)
    """),

    code("""\
    # Duplicates
    print(f"Duplicate rows: {df.duplicated().sum()}")
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"Shape after dedup: {df.shape}")
    """),

    # ─────────────────────────────────────────────────────────────────────────
    # 3. TARGET VARIABLE
    # ─────────────────────────────────────────────────────────────────────────
    md("""\
    ---
    ## 3. Target Variable — Nightly Price

    The target is the **nightly price (€)**. Because price is right-skewed
    (a few luxury listings with very high prices), we use **log1p(price)** as
    the model target. This:
    - Makes the distribution approximately normal (better for gradient-based optimisation)
    - Treats proportional errors equally (a €10 error on a €50 listing is
      as important as a €10 error on a €500 listing when on the log scale)

    Unlike classification tasks, there is **no class imbalance** to handle —
    price is a continuous variable.

    We clip listings above the **99.5th percentile** to remove ultra-luxury
    outliers that would otherwise dominate the loss function.
    """),

    code("""\
    # Remove missing prices
    df = df[df["price"].notna()].copy()
    print(f"Rows with valid price: {len(df):,}")

    print(f"\\nPrice percentiles (€/night):")
    pcts = [0, 1, 5, 25, 50, 75, 95, 99, 99.5, 100]
    for p in pcts:
        print(f"  p{p:5.1f}  →  €{np.percentile(df['price'], p):,.0f}")

    print(f"\\nSkewness (raw price) : {df['price'].skew():.2f}")
    """),

    code("""\
    # Log1p transform + clip
    df["price_log"] = np.log1p(df["price"])
    p995            = df["price_log"].quantile(0.995)
    n_clipped       = (df["price_log"] > p995).sum()
    df              = df[df["price_log"] <= p995].copy()
    print(f"Clipped {n_clipped} extreme outliers (price > €{np.expm1(p995):.0f}/night)")
    print(f"Final dataset: {len(df):,} rows")
    print(f"\\nlog1p(price)  —  mean={df['price_log'].mean():.3f}  "
          f"std={df['price_log'].std():.3f}  skew={df['price_log'].skew():.2f}")

    # Distribution plots
    CITY_COLORS = {"Madrid": "#1976D2", "Barcelona": "#E53935", "Málaga": "#388E3C"}

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].hist(df["price"], bins=60, color="#78909C", alpha=0.8, edgecolor="white")
    axes[0].set_xlabel("Price (€/night)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Raw price — right-skewed")
    axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:,.0f}"))

    axes[1].hist(df["price_log"], bins=60, color="#43A047", alpha=0.8, edgecolor="white")
    axes[1].set_xlabel("log1p(price)")
    axes[1].set_title("Log-transformed price ← model target")

    for city, grp in df.groupby("city"):
        axes[2].hist(grp["price_log"], bins=40, alpha=0.55, label=str(city),
                     color=CITY_COLORS.get(str(city), "grey"))
    axes[2].set_xlabel("log1p(price)")
    axes[2].set_title("Log-price by city")
    axes[2].legend()

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(MODEL_DIR / "01_price_distribution.png", dpi=130, bbox_inches="tight")
    plt.show()
    """),

    # ─────────────────────────────────────────────────────────────────────────
    # 4. EDA
    # ─────────────────────────────────────────────────────────────────────────
    md("""\
    ---
    ## 4. Exploratory Data Analysis (EDA)

    City-level EDA (per-city distributions, seasonal patterns, sentiment analysis)
    has already been completed in the per-city notebooks (`01_cleaning`,
    `02_analysis`, `03_sentiment`). Here we focus on the **combined 3-city view**
    relevant to model building.
    """),

    md("### 4.1 Price by Key Categorical Features"),

    code("""\
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # City
    city_means = df.groupby("city")["price_log"].mean().sort_values()
    axes[0].barh(city_means.index.astype(str), np.expm1(city_means.values),
                 color=[CITY_COLORS.get(str(c), "grey") for c in city_means.index])
    axes[0].set_xlabel("Median price (€/night)")
    axes[0].set_title("Price by City")
    axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:,.0f}"))

    # Room type
    rt_order = df.groupby("room_type")["price"].median().sort_values(ascending=False).index
    df.boxplot(column="price_log", by="room_type", ax=axes[1],
               order=rt_order, vert=True, patch_artist=True)
    axes[1].set_xlabel("")
    axes[1].set_ylabel("log1p(price)")
    axes[1].set_title("Price by Room Type")
    plt.setp(axes[1].get_xticklabels(), rotation=20, ha="right")
    axes[1].get_figure().suptitle("")

    # Property type std
    pt_order = df.groupby("property_type_std")["price"].median().sort_values(ascending=False).index
    df.boxplot(column="price_log", by="property_type_std", ax=axes[2],
               order=pt_order, vert=True, patch_artist=True)
    axes[2].set_xlabel("")
    axes[2].set_ylabel("log1p(price)")
    axes[2].set_title("Price by Property Type")
    plt.setp(axes[2].get_xticklabels(), rotation=20, ha="right")
    axes[2].get_figure().suptitle("")

    plt.tight_layout()
    plt.savefig(MODEL_DIR / "02_price_by_category.png", dpi=130, bbox_inches="tight")
    plt.show()
    """),

    md("### 4.2 Price vs Key Numerical Features"),

    code("""\
    key_num_features = [
        "accommodates", "bedrooms", "beds", "bathrooms_number",
        "review_scores_rating", "host_tenure_years",
    ]
    key_num_features = [f for f in key_num_features if f in df.columns]

    n_cols = 3
    n_rows = int(np.ceil(len(key_num_features) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
    axes = axes.flatten()

    for i, feat in enumerate(key_num_features):
        axes[i].scatter(df[feat], df["price_log"], alpha=0.15, s=3,
                        color="#5C6BC0", rasterized=True)
        # Trend line
        mask = df[feat].notna()
        z    = np.polyfit(df.loc[mask, feat], df.loc[mask, "price_log"], 1)
        p    = np.poly1d(z)
        xr   = np.linspace(df[feat].min(), df[feat].max(), 100)
        axes[i].plot(xr, p(xr), "r-", lw=1.5)
        corr = df[[feat, "price_log"]].corr().iloc[0, 1]
        axes[i].set_xlabel(feat)
        axes[i].set_ylabel("log1p(price)")
        axes[i].set_title(f"{feat}  (r={corr:.2f})")
        axes[i].spines["top"].set_visible(False)
        axes[i].spines["right"].set_visible(False)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Price vs Key Numerical Features", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "03_price_vs_features.png", dpi=130, bbox_inches="tight")
    plt.show()
    """),

    md("### 4.3 Correlation Heatmap"),

    code("""\
    # Select a compact set of interpretable numeric features for correlation
    corr_cols = [
        "price_log", "accommodates", "bedrooms", "beds", "bathrooms_number",
        "minimum_nights", "availability_365",
        "number_of_reviews", "review_scores_rating",
        "reviews_per_month", "host_tenure_years",
        "calculated_host_listings_count", "description_length",
    ]
    corr_cols = [c for c in corr_cols if c in df.columns]

    corr_matrix = df[corr_cols].corr()

    fig, ax = plt.subplots(figsize=(13, 10))
    sns.heatmap(
        corr_matrix, annot=True, fmt=".2f", cmap="RdBu_r",
        vmin=-1, vmax=1, center=0, square=True,
        linewidths=0.5, ax=ax, annot_kws={"size": 8},
    )
    ax.set_title("Pearson Correlation — Key Numerical Features", pad=14, fontsize=13)
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "04_correlation_heatmap.png", dpi=130, bbox_inches="tight")
    plt.show()
    """),

    md("""\
    ### 4.4 Outlier Analysis

    We use the **IQR method** to identify outliers in numerical features.
    Understanding outliers is important: extreme values can distort linear model
    coefficients. Tree-based models are naturally robust to outliers.
    """),

    code("""\
    def count_outliers_iqr(df, cols):
        results = []
        for col in cols:
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr     = q3 - q1
            lb, ub  = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            n_out   = ((df[col] < lb) | (df[col] > ub)).sum()
            results.append({
                "Feature"   : col,
                "Q1"        : round(q1, 2),
                "Q3"        : round(q3, 2),
                "IQR"       : round(iqr, 2),
                "Lower"     : round(lb, 2),
                "Upper"     : round(ub, 2),
                "Outliers"  : n_out,
                "Outlier %" : round(n_out / len(df) * 100, 1),
            })
        return pd.DataFrame(results).sort_values("Outliers", ascending=False)

    num_analysis_cols = [
        "price", "accommodates", "bedrooms", "beds", "bathrooms_number",
        "minimum_nights", "number_of_reviews", "review_scores_rating",
        "host_tenure_years", "calculated_host_listings_count",
    ]
    num_analysis_cols = [c for c in num_analysis_cols if c in df.columns]
    outlier_df = count_outliers_iqr(df, num_analysis_cols)
    print("Outlier summary (IQR method):")
    print(outlier_df.to_string(index=False))
    """),

    code("""\
    # Boxplots for top-outlier columns
    top_out_cols = outlier_df[outlier_df["Outliers"] > 0].head(6)["Feature"].tolist()

    fig, axes = plt.subplots(1, len(top_out_cols), figsize=(16, 4))
    if len(top_out_cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, top_out_cols):
        ax.boxplot(df[col].dropna(), vert=True, patch_artist=True,
                   boxprops=dict(facecolor="#90CAF9"), medianprops=dict(color="red"))
        ax.set_title(col, fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.suptitle("Outliers in Key Features (IQR method)", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "05_outliers.png", dpi=130, bbox_inches="tight")
    plt.show()
    """),

    # ─────────────────────────────────────────────────────────────────────────
    # 5. FEATURE ENGINEERING
    # ─────────────────────────────────────────────────────────────────────────
    md("---\n## 5. Data Processing & Feature Engineering"),

    code("""\
    # ── 5.1 Amenity binary flags ──────────────────────────────────────────────
    AMENITY_PATTERNS = {
        "has_pool"             : (r"\\bpool\\b",                           r"pool table"),
        "has_gym"              : (r"\\bgym\\b",                            None),
        "has_parking"          : (r"parking",                              None),
        "has_hot_tub"          : (r"hot tub|jacuzzi",                      None),
        "has_beach"            : (r"beach",                                None),
        "has_view"             : (r"\\bview\\b|skyline",                   None),
        "has_ac"               : (r"air conditioning",                     None),
        "has_elevator"         : (r"elevator",                             None),
        "has_washer"           : (r"\\bwasher\\b",                         None),
        "has_dishwasher"       : (r"dishwasher",                           None),
        "has_workspace"        : (r"dedicated workspace",                  None),
        "has_self_checkin"     : (r"self check.in|smart lock|lockbox|keypad", None),
        "has_pets"             : (r"pets allowed",                         None),
        "has_crib"             : (r"\\bcrib\\b",                           r"crib.*table"),
        "has_private_entrance" : (r"private entrance",                     None),
        "has_balcony"          : (r"balcony|patio|terrace",                None),
        "has_bathtub"          : (r"bathtub",                              None),
        "has_dryer"            : (r"\\bdryer\\b",                          None),
        "has_ev_charger"       : (r"ev charger",                           None),
        "has_outdoor_space"    : (r"outdoor dining|outdoor furniture|garden|backyard|courtyard", None),
        "has_long_term_ok"     : (r"long term stays allowed",              None),
        "has_cleaning_service" : (r"cleaning available during stay",       None),
    }


    def parse_amenity_flags(series, patterns):
        parsed = series.fillna("[]").apply(
            lambda s: json.loads(s) if isinstance(s, str) else []
        )
        flags = {}
        for col, (inc_pat, exc_pat) in patterns.items():
            inc = re.compile(inc_pat, re.I)
            exc = re.compile(exc_pat, re.I) if exc_pat else None

            def _check(alist, _inc=inc, _exc=exc):
                for a in alist:
                    if _inc.search(a):
                        if _exc is None or not _exc.search(a):
                            return 1
                return 0

            flags[col] = parsed.apply(_check).astype(np.int8)
        return pd.DataFrame(flags, index=series.index)


    amenity_flags = parse_amenity_flags(df["amenities"], AMENITY_PATTERNS)
    AMENITY_COLS  = list(amenity_flags.columns)
    df = pd.concat([df, amenity_flags], axis=1)

    print(f"Added {len(AMENITY_COLS)} amenity flag columns.")
    print("\\nAmenity coverage (% of listings):")
    print(amenity_flags.mean().mul(100).sort_values(ascending=False).round(1).to_string())
    """),

    code("""\
    # ── 5.2 Review score imputation ───────────────────────────────────────────
    REVIEW_SCORE_COLS = [
        "review_scores_rating", "review_scores_accuracy", "review_scores_cleanliness",
        "review_scores_checkin", "review_scores_communication",
        "review_scores_location", "review_scores_value",
    ]
    df["has_reviews"] = (df["number_of_reviews"] > 0).astype(np.int8)

    city_review_medians = df.groupby("city")[REVIEW_SCORE_COLS].median()
    for col in REVIEW_SCORE_COLS:
        missing_mask = df[col].isna()
        df.loc[missing_mask, col] = df.loc[missing_mask, "city"].map(city_review_medians[col])

    print("Review score nulls after imputation:")
    print(df[REVIEW_SCORE_COLS].isna().sum().to_string())
    print(f"\\n{df['has_reviews'].sum():,} listings ({df['has_reviews'].mean()*100:.1f}%) have at least one review.")
    """),

    code("""\
    # ── 5.3 Booking flags, competitive density, licence ──────────────────────
    df["is_long_stay"] = (df["minimum_nights"] >= 28).astype(np.int8)
    df["is_weekly"]    = ((df["minimum_nights"] >= 7) & (df["minimum_nights"] < 28)).astype(np.int8)
    df["has_licence"]  = df["license"].notna().astype(np.int8)

    # Competitive density — listings within 500 m per city (haversine BallTree)
    EARTH_RADIUS_M = 6_371_000
    RADIUS_M       = 500
    density        = np.zeros(len(df), dtype=np.int32)

    for city in df["city"].unique():
        mask       = (df["city"] == city).values
        coords_rad = np.radians(
            np.column_stack([df.loc[mask, "latitude"].values,
                             df.loc[mask, "longitude"].values])
        )
        tree           = BallTree(coords_rad, metric="haversine")
        counts         = tree.query_radius(coords_rad, r=RADIUS_M / EARTH_RADIUS_M, count_only=True)
        density[mask]  = counts - 1

    df["competitive_density_500m"] = density

    print("Booking segment distribution:")
    print(f"  Long-stay (≥28 nights) : {df['is_long_stay'].sum():,}")
    print(f"  Weekly   (7-27 nights) : {df['is_weekly'].sum():,}")
    print(f"  Short-term (<7 nights) : {(df['minimum_nights'] < 7).sum():,}")
    print(f"  Has STR licence        : {df['has_licence'].mean()*100:.1f}%")
    print(f"\\nCompetitive density (500 m) — median: {np.median(density):.0f} | max: {density.max()}")
    """),

    code("""\
    # ── 5.4 Feature selection — drop metadata / leakage columns ───────────────
    DROP_COLS = [
        # identifiers & raw text
        "id", "scrape_id", "host_id", "source",
        "name", "description", "neighborhood_overview", "picture_url",
        "host_name", "host_about", "host_location", "host_verifications",
        "amenities", "bathrooms_description", "license",
        # raw date columns (derived features already present)
        "last_scraped", "first_review", "last_review", "calendar_last_scraped",
        "host_since",          # → host_tenure_years already in df
        # redundant text/type columns
        "property_type",       # → property_type_std already cleaned
        # direct leakage
        "price_cat",                 # bin of price
        "estimated_revenue_l365d",   # price × occupancy
        # stage-2 reserve
        "estimated_occupancy_l365d",
        # redundant / near-constant
        "neighbourhood_group_cleansed",
        "has_availability",
        "availability_eoy",
        # booking-rule edge fields
        "minimum_minimum_nights", "maximum_minimum_nights",
        "minimum_maximum_nights", "maximum_maximum_nights",
        "maximum_nights_avg_ntm",
    ]

    TARGET = "price_log"
    feature_cols = [c for c in df.columns if c not in DROP_COLS + [TARGET, "price"]]
    # Drop any remaining datetime columns (safety net)
    feature_cols = [c for c in feature_cols
                    if not pd.api.types.is_datetime64_any_dtype(df[c])]

    print(f"Features available for modelling: {len(feature_cols)}")
    print(feature_cols)
    """),

    code("""\
    # ── 5.5 Encode categoricals + booleans ────────────────────────────────────
    CAT_COLS = [
        "city", "room_type", "property_type_std", "neighbourhood_cleansed",
        "host_response_time", "host_response_rate_cat", "host_acceptance_rate_cat",
    ]
    CAT_COLS = [c for c in CAT_COLS if c in feature_cols]

    df_model = df[feature_cols + [TARGET]].copy()

    # Bool → int8
    bool_cols = df_model.select_dtypes(include="bool").columns.tolist()
    for col in bool_cols:
        df_model[col] = df_model[col].astype(np.int8)

    # Categorical → label-encoded integer
    cat_encoders = {}
    for col in CAT_COLS:
        df_model[col] = df_model[col].astype(str).replace({"nan": None, "None": None})
        cats = sorted(df_model[col].dropna().unique())
        cat_encoders[col] = {c: i for i, c in enumerate(cats)}
        df_model[col] = df_model[col].map(cat_encoders[col])

    # Numeric nulls → column median
    num_cols = df_model.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c != TARGET]
    GLOBAL_MEDIANS = df_model[num_cols].median()
    df_model[num_cols] = df_model[num_cols].fillna(GLOBAL_MEDIANS)

    joblib.dump(cat_encoders, MODEL_DIR / "price_cat_encoders.pkl")

    remaining_nulls = df_model.isnull().sum().sum()
    print(f"df_model shape: {df_model.shape}")
    print(f"Remaining nulls: {remaining_nulls}")
    print(f"\\nCategorical columns label-encoded ({len(CAT_COLS)}): {CAT_COLS}")
    """),

    # ─────────────────────────────────────────────────────────────────────────
    # 6. MULTICOLLINEARITY
    # ─────────────────────────────────────────────────────────────────────────
    md("""\
    ---
    ## 6. Multicollinearity Analysis

    Before training, we check for multicollinearity among numerical features using
    the **Variance Inflation Factor (VIF)**. High VIF (> 10) indicates that a
    feature can be explained by other features, which inflates linear model
    coefficients.

    Tree-based models handle multicollinearity naturally; this check is most
    relevant for the linear regression baseline.
    """),

    code("""\
    def compute_vif(X_df, sample_n=5000):
        \"\"\"Compute VIF for each column using OLS R² (no statsmodels needed).\"\"\"
        # Sample for speed on large datasets
        Xs = X_df.sample(min(sample_n, len(X_df)), random_state=RANDOM_SEED)
        Xs = Xs.fillna(0)
        results = []
        for col in Xs.columns:
            X_others = Xs.drop(columns=[col]).values
            y_col    = Xs[col].values
            r2 = LinearRegression().fit(X_others, y_col).score(X_others, y_col)
            vif = 1 / (1 - r2) if r2 < 1.0 else float("inf")
            results.append({"Feature": col, "VIF": round(vif, 2)})
        return pd.DataFrame(results).sort_values("VIF", ascending=False)


    feature_cols_no_target = [c for c in df_model.columns if c != TARGET]
    # Run VIF only on numeric columns (categoricals are label-encoded integers)
    vif_df = compute_vif(df_model[num_cols])
    print(f"VIF analysis on {len(num_cols)} numeric features:")
    print(vif_df.to_string(index=False))
    """),

    code("""\
    # Visualise VIF
    vif_plot = vif_df[vif_df["VIF"] < 50].nlargest(20, "VIF")
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#E53935" if v > 10 else "#FFA726" if v > 5 else "#43A047"
              for v in vif_plot["VIF"]]
    ax.barh(vif_plot["Feature"], vif_plot["VIF"], color=colors)
    ax.axvline(5,  color="#FFA726", ls="--", lw=1.5, label="VIF=5  (moderate)")
    ax.axvline(10, color="#E53935", ls="--", lw=1.5, label="VIF=10 (severe)")
    ax.set_xlabel("VIF")
    ax.set_title("Variance Inflation Factor — Top 20 Numeric Features")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "06_vif.png", dpi=130, bbox_inches="tight")
    plt.show()

    high_vif = vif_df[vif_df["VIF"] > 10]
    print(f"\\nFeatures with VIF > 10 ({len(high_vif)}): {high_vif['Feature'].tolist()}")
    print("Note: Tree models are robust to multicollinearity — these are flagged for linear models only.")
    """),

    # ─────────────────────────────────────────────────────────────────────────
    # 7. FEATURE SELECTION + TRAIN/TEST SPLIT
    # ─────────────────────────────────────────────────────────────────────────
    md("---\n## 7. Feature Selection & Train / Test Split"),

    md("""\
    ### 7.1 Feature Selection — Random Forest Importance

    We use a fast **Random Forest** (100 trees) to rank features by importance,
    then retain all features above the **median importance**. This is equivalent
    in spirit to RFECV but ~50× faster on a 42k-row dataset with 60+ features.
    """),

    code("""\
    X_all = df_model.drop(columns=[TARGET])
    y_all = df_model[TARGET].values

    # Fit a quick RF for importance ranking (before the final split, on full data)
    rf_selector = RandomForestRegressor(
        n_estimators=100, max_depth=12, min_samples_leaf=10,
        random_state=RANDOM_SEED, n_jobs=-1,
    )
    rf_selector.fit(X_all, y_all)

    importances   = pd.Series(rf_selector.feature_importances_, index=X_all.columns)
    median_imp    = importances.median()
    selected_features = importances[importances >= median_imp].sort_values(ascending=False).index.tolist()

    print(f"Features before selection : {len(X_all.columns)}")
    print(f"Features after selection  : {len(selected_features)}")
    print(f"\\nSelected features (sorted by importance):")
    for feat in selected_features:
        print(f"  {importances[feat]:.4f}  {feat}")
    """),

    code("""\
    # Importance bar chart
    top30 = importances.nlargest(30).sort_values()
    colors_imp = ["#E53935" if f in selected_features else "#B0BEC5" for f in top30.index]
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.barh(top30.index, top30.values, color=colors_imp)
    ax.axvline(median_imp, color="navy", ls="--", lw=1.5, label=f"Median importance = {median_imp:.4f}")
    ax.set_xlabel("RF Feature Importance")
    ax.set_title("Feature Importance (pre-selection RF)\nRed = selected, grey = dropped")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "07_rf_feature_importance.png", dpi=130, bbox_inches="tight")
    plt.show()
    """),

    md("### 7.2 Train / Test Split"),

    code("""\
    X = df_model[selected_features].copy()
    y = df_model[TARGET].values

    city_labels = df.loc[df_model.index, "city"].astype(str).values

    X_train, X_test, y_train, y_test, city_train, city_test = train_test_split(
        X, y, city_labels,
        test_size=0.20, random_state=RANDOM_SEED, stratify=city_labels,
    )

    for label, Xi in [("Train", X_train), ("Test", X_test)]:
        print(f"{label:6s}: {len(Xi):>6,} rows  ({len(Xi)/len(X)*100:.1f}%)")
    print(f"\\nCity distribution in test set:")
    for c, n in zip(*np.unique(city_test, return_counts=True)):
        print(f"  {c}: {n:,} ({n/len(city_test)*100:.1f}%)")
    """),

    md("### 7.3 Feature Scaling (for linear models)"),

    code("""\
    scaler        = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # Wrap back as DataFrames for feature name access
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=selected_features, index=X_train.index)
    X_test_scaled  = pd.DataFrame(X_test_scaled,  columns=selected_features, index=X_test.index)

    joblib.dump(scaler, MODEL_DIR / "price_scaler.pkl")
    print(f"Scaler fitted on {len(X_train):,} training rows. Saved to models/price_scaler.pkl")
    """),

    # ─────────────────────────────────────────────────────────────────────────
    # 8. EVALUATION HELPER
    # ─────────────────────────────────────────────────────────────────────────
    md("---\n## 8. Evaluation Helper Function"),

    code("""\
    all_results = {}


    def evaluate_model(model, X_tr, X_te, y_tr, y_te, model_name="Model"):
        \"\"\"
        Fit model and return train/test metrics.
        Metrics: RMSE (log), MAE (log), R², RMSE (€), MAE (€), MdAPE (%).
        \"\"\"
        y_pred_tr  = model.predict(X_tr)
        y_pred_te  = model.predict(X_te)

        def _metrics(y_true, y_pred):
            rmse_log = np.sqrt(mean_squared_error(y_true, y_pred))
            mae_log  = mean_absolute_error(y_true, y_pred)
            r2       = r2_score(y_true, y_pred)
            y_t_eur  = np.expm1(y_true)
            y_p_eur  = np.expm1(y_pred)
            rmse_eur = np.sqrt(mean_squared_error(y_t_eur, y_p_eur))
            mae_eur  = mean_absolute_error(y_t_eur, y_p_eur)
            mdape    = np.median(np.abs((y_t_eur - y_p_eur) / y_t_eur)) * 100
            return dict(RMSE_log=rmse_log, MAE_log=mae_log, R2=r2,
                        RMSE_EUR=rmse_eur, MAE_EUR=mae_eur, MdAPE=mdape)

        train_m = _metrics(y_tr, y_pred_tr)
        test_m  = _metrics(y_te, y_pred_te)

        print(f"\\n{'='*48}")
        print(f"  {model_name}")
        print(f"{'='*48}")
        print(f"  {'Metric':<20} {'Train':>10} {'Test':>10}")
        print(f"  {'-'*40}")
        for k in ["RMSE_log", "MAE_log", "R2", "RMSE_EUR", "MAE_EUR", "MdAPE"]:
            unit = " €" if "EUR" in k else (" %" if k == "MdAPE" else "")
            print(f"  {k:<20} {train_m[k]:>10.3f} {test_m[k]:>10.3f}{unit}")

        return {"Train": train_m, "Test": test_m,
                "y_pred_train": y_pred_tr, "y_pred_test": y_pred_te}
    """),

    # ─────────────────────────────────────────────────────────────────────────
    # 9. MODEL 0 — LINEAR REGRESSION
    # ─────────────────────────────────────────────────────────────────────────
    md("""\
    ---
    ## 9. Model 0 — Linear Regression (Baseline)

    **Business Rationale:**
    Linear regression is our interpretable baseline. If all features have a
    linear relationship with log(price), this would be optimal. In practice,
    price is driven by complex interactions (e.g., `accommodates × location`)
    that linear models miss, so we expect ensemble methods to outperform it.

    We use the **scaled** feature set for linear models to ensure fair coefficient
    comparison across features with different units.
    """),

    code("""\
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    all_results["Linear Regression"] = evaluate_model(
        lr, X_train_scaled, X_test_scaled, y_train, y_test, "Linear Regression"
    )

    # Top 15 coefficient magnitudes
    coef_df = pd.DataFrame({
        "Feature": selected_features,
        "Coefficient": lr.coef_,
        "Abs": np.abs(lr.coef_),
    }).sort_values("Abs", ascending=False).head(15)
    print("\\nTop 15 features by absolute coefficient:")
    print(coef_df[["Feature", "Coefficient"]].to_string(index=False))
    """),

    # ─────────────────────────────────────────────────────────────────────────
    # 10. MODEL 1 — RIDGE
    # ─────────────────────────────────────────────────────────────────────────
    md("""\
    ---
    ## 10. Model 1 — Ridge Regression

    **Business Rationale:**
    Ridge adds L2 regularisation, shrinking large coefficients toward zero.
    This is useful when features are correlated (as flagged by the VIF analysis)
    — it stabilises coefficient estimates without discarding features outright.
    """),

    code("""\
    param_grid_ridge = {"alpha": [0.01, 0.1, 1, 10, 100, 500]}
    ridge_cv = GridSearchCV(
        Ridge(), param_grid_ridge, cv=5, scoring="r2", n_jobs=-1, refit=True
    )
    ridge_cv.fit(X_train_scaled, y_train)
    print(f"Best alpha : {ridge_cv.best_params_['alpha']}")
    print(f"Best CV R² : {ridge_cv.best_score_:.4f}")

    all_results["Ridge"] = evaluate_model(
        ridge_cv.best_estimator_, X_train_scaled, X_test_scaled,
        y_train, y_test, "Ridge Regression"
    )

    cv_ridge = pd.DataFrame(ridge_cv.cv_results_)[
        ["param_alpha", "mean_test_score", "std_test_score"]
    ].sort_values("mean_test_score", ascending=False)
    print("\\nGridSearchCV results:")
    print(cv_ridge.to_string(index=False))
    """),

    # ─────────────────────────────────────────────────────────────────────────
    # 11. MODEL 2 — RANDOM FOREST
    # ─────────────────────────────────────────────────────────────────────────
    md("""\
    ---
    ## 11. Model 2 — Random Forest

    **Business Rationale:**
    Random Forest builds many decorrelated trees on bootstrap samples and
    random feature subsets. It handles non-linear interactions, categorical
    features, and outliers naturally — at the cost of interpretability.
    """),

    code("""\
    param_grid_rf = {
        "n_estimators": [100, 200],
        "max_depth"   : [None, 12, 20],
        "min_samples_leaf": [5, 15],
    }
    rf_cv = GridSearchCV(
        RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1),
        param_grid_rf, cv=5, scoring="r2", n_jobs=-1, refit=True, verbose=0,
    )
    rf_cv.fit(X_train, y_train)
    print(f"Best params: {rf_cv.best_params_}")
    print(f"Best CV R² : {rf_cv.best_score_:.4f}")

    all_results["Random Forest"] = evaluate_model(
        rf_cv.best_estimator_, X_train, X_test, y_train, y_test, "Random Forest"
    )
    """),

    code("""\
    # Feature importances — RF
    imp_rf = pd.Series(rf_cv.best_estimator_.feature_importances_, index=selected_features)
    top15_rf = imp_rf.nlargest(15).sort_values()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top15_rf.index, top15_rf.values, color="#42A5F5")
    ax.set_xlabel("Feature Importance")
    ax.set_title("Random Forest — Top 15 Feature Importances")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "08_rf_importances.png", dpi=130, bbox_inches="tight")
    plt.show()

    cv_rf = pd.DataFrame(rf_cv.cv_results_)[
        ["param_n_estimators","param_max_depth","param_min_samples_leaf","mean_test_score","std_test_score"]
    ].sort_values("mean_test_score", ascending=False)
    print("GridSearchCV results (top 5):")
    print(cv_rf.head(5).to_string(index=False))
    """),

    # ─────────────────────────────────────────────────────────────────────────
    # 12. MODEL 3 — GRADIENT BOOSTING
    # ─────────────────────────────────────────────────────────────────────────
    md("""\
    ---
    ## 12. Model 3 — Gradient Boosting

    **Business Rationale:**
    Gradient Boosting builds trees sequentially, each correcting the errors of
    the previous one. It often achieves better accuracy than Random Forest on
    structured tabular data because it optimises directly for the loss function.
    """),

    code("""\
    param_grid_gb = {
        "n_estimators" : [100, 200],
        "learning_rate": [0.05, 0.1],
        "max_depth"    : [3, 5],
        "min_samples_leaf": [10, 20],
    }
    gb_cv = GridSearchCV(
        GradientBoostingRegressor(random_state=RANDOM_SEED),
        param_grid_gb, cv=5, scoring="r2", n_jobs=-1, refit=True, verbose=0,
    )
    gb_cv.fit(X_train, y_train)
    print(f"Best params: {gb_cv.best_params_}")
    print(f"Best CV R² : {gb_cv.best_score_:.4f}")

    all_results["Gradient Boosting"] = evaluate_model(
        gb_cv.best_estimator_, X_train, X_test, y_train, y_test, "Gradient Boosting"
    )

    cv_gb = pd.DataFrame(gb_cv.cv_results_)[
        ["param_n_estimators","param_learning_rate","param_max_depth","mean_test_score"]
    ].sort_values("mean_test_score", ascending=False)
    print("\\nGridSearchCV results (top 5):")
    print(cv_gb.head(5).to_string(index=False))
    """),

    # ─────────────────────────────────────────────────────────────────────────
    # 13. MODEL 4 — HIST GRADIENT BOOSTING (XGBoost-equivalent)
    # ─────────────────────────────────────────────────────────────────────────
    md("""\
    ---
    ## 13. Model 4 — Histogram Gradient Boosting (XGBoost-equivalent)

    **Business Rationale:**
    `HistGradientBoostingRegressor` is scikit-learn's implementation of the
    histogram-based boosting algorithm — the same family as XGBoost and LightGBM,
    but available without external dependencies. It uses bin-based splits for
    efficiency and supports native categorical features and missing values.

    It is typically the fastest and most accurate model for large tabular datasets.
    """),

    code("""\
    param_grid_hgb = {
        "max_iter"     : [100, 300],
        "learning_rate": [0.05, 0.1],
        "max_depth"    : [None, 6, 10],
        "l2_regularization": [0.0, 0.1, 1.0],
    }
    hgb_cv = GridSearchCV(
        HistGradientBoostingRegressor(random_state=RANDOM_SEED),
        param_grid_hgb, cv=5, scoring="r2", n_jobs=-1, refit=True, verbose=0,
    )
    hgb_cv.fit(X_train, y_train)
    print(f"Best params: {hgb_cv.best_params_}")
    print(f"Best CV R² : {hgb_cv.best_score_:.4f}")

    all_results["HistGradientBoosting"] = evaluate_model(
        hgb_cv.best_estimator_, X_train, X_test, y_train, y_test,
        "HistGradientBoosting (XGBoost-equivalent)"
    )

    cv_hgb = pd.DataFrame(hgb_cv.cv_results_)[
        ["param_max_iter","param_learning_rate","param_max_depth","mean_test_score"]
    ].sort_values("mean_test_score", ascending=False)
    print("\\nGridSearchCV results (top 5):")
    print(cv_hgb.head(5).to_string(index=False))
    """),

    # ─────────────────────────────────────────────────────────────────────────
    # 14. MODEL COMPARISON
    # ─────────────────────────────────────────────────────────────────────────
    md("---\n## 14. Model Comparison"),

    code("""\
    comparison_rows = []
    for name, res in all_results.items():
        row = {"Model": name}
        row.update({f"Test_{k}": v for k, v in res["Test"].items()})
        comparison_rows.append(row)

    comparison_df = pd.DataFrame(comparison_rows).set_index("Model")
    comparison_df_display = comparison_df.rename(columns={
        "Test_RMSE_log": "RMSE (log)",
        "Test_MAE_log" : "MAE (log)",
        "Test_R2"      : "R²",
        "Test_RMSE_EUR": "RMSE (€)",
        "Test_MAE_EUR" : "MAE (€)",
        "Test_MdAPE"   : "MdAPE (%)",
    }).sort_values("R²", ascending=False)

    print("Model comparison — Test set:")
    print(comparison_df_display.round(3).to_string())
    """),

    code("""\
    # Bar chart comparison
    metrics_to_plot = ["Test_R2", "Test_RMSE_EUR", "Test_MAE_EUR", "Test_MdAPE"]
    titles          = ["R²  (higher is better)", "RMSE €  (lower is better)",
                       "MAE €  (lower is better)", "MdAPE %  (lower is better)"]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    model_names = [r.replace(" (XGBoost-equivalent)", "") for r in all_results.keys()]
    palette     = sns.color_palette("Set2", len(all_results))

    for ax, metric, title in zip(axes, metrics_to_plot, titles):
        values = [all_results[m]["Test"][metric.replace("Test_", "")] for m in all_results]
        bars   = ax.bar(model_names, values, color=palette, edgecolor="white")
        ax.set_title(title, fontsize=10)
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=30, ha="right", fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # Annotate best
        best_idx = int(np.argmin(values) if "lower" in title else np.argmax(values))
        bars[best_idx].set_edgecolor("black")
        bars[best_idx].set_linewidth(2)

    plt.suptitle("Model Comparison — Test Set", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "09_model_comparison.png", dpi=130, bbox_inches="tight")
    plt.show()
    """),

    # ─────────────────────────────────────────────────────────────────────────
    # 15. OVERFITTING ANALYSIS
    # ─────────────────────────────────────────────────────────────────────────
    md("""\
    ---
    ## 15. Overfitting Analysis

    A good model generalises well: the gap between train and test R² should be
    small. A large gap indicates overfitting — the model memorised the training
    data but fails to generalise.
    """),

    code("""\
    train_r2 = [all_results[m]["Train"]["R2"] for m in all_results]
    test_r2  = [all_results[m]["Test"]["R2"]  for m in all_results]
    gaps     = [tr - te for tr, te in zip(train_r2, test_r2)]

    overfit_df = pd.DataFrame({
        "Model"   : list(all_results.keys()),
        "Train R²": [round(r, 3) for r in train_r2],
        "Test R²" : [round(r, 3) for r in test_r2],
        "Gap"     : [round(g, 3) for g in gaps],
    }).set_index("Model")
    print("Train vs Test R²:")
    print(overfit_df.to_string())

    # Grouped bar chart
    x     = np.arange(len(all_results))
    width = 0.35
    names = [m.replace(" (XGBoost-equivalent)", "") for m in all_results.keys()]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width/2, train_r2, width, label="Train R²", color="#42A5F5", alpha=0.85)
    ax.bar(x + width/2, test_r2,  width, label="Test R²",  color="#EF5350", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("R²")
    ax.set_title("Overfitting Check — Train vs Test R²")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "10_overfitting.png", dpi=130, bbox_inches="tight")
    plt.show()
    """),

    # ─────────────────────────────────────────────────────────────────────────
    # 16. BEST MODEL — IMPORTANCES + SHAP + RESIDUALS
    # ─────────────────────────────────────────────────────────────────────────
    md("""\
    ---
    ## 16. Best Model — Feature Importances & SHAP

    We select the best model by test R² and perform deeper analysis:
    1. **Feature importances** — which variables drive predictions most
    2. **SHAP analysis** — how each feature pushes the predicted price up or down
    3. **Residual analysis** — where the model makes its largest errors
    """),

    code("""\
    # Identify best model by test R²
    best_name  = max(all_results, key=lambda m: all_results[m]["Test"]["R2"])
    print(f"Best model: {best_name}")
    print(f"Test R²  : {all_results[best_name]['Test']['R2']:.4f}")
    print(f"Test MdAPE: {all_results[best_name]['Test']['MdAPE']:.1f}%")
    print(f"Test MAE  : €{all_results[best_name]['Test']['MAE_EUR']:.1f}/night")

    # Retrieve the fitted best estimator
    MODEL_MAP = {
        "Linear Regression" : (ridge_cv.best_estimator_, X_test_scaled),  # placeholder
        "Ridge"             : (ridge_cv.best_estimator_, X_test_scaled),
        "Random Forest"     : (rf_cv.best_estimator_,   X_test),
        "Gradient Boosting" : (gb_cv.best_estimator_,   X_test),
        "HistGradientBoosting": (hgb_cv.best_estimator_, X_test),
    }
    # Override for linear regression
    MODEL_MAP["Linear Regression"] = (lr, X_test_scaled)

    best_model, X_test_best = MODEL_MAP[best_name]
    X_train_best = X_train_scaled if best_name in ("Linear Regression", "Ridge") else X_train
    """),

    code("""\
    # Feature importances (tree models)
    if hasattr(best_model, "feature_importances_"):
        imp = pd.Series(best_model.feature_importances_, index=selected_features)
        top20 = imp.nlargest(20).sort_values()

        fig, ax = plt.subplots(figsize=(8, 7))
        ax.barh(top20.index, top20.values, color="#7E57C2")
        ax.set_xlabel("Feature Importance")
        ax.set_title(f"{best_name} — Top 20 Feature Importances")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        plt.savefig(MODEL_DIR / "11_best_model_importances.png", dpi=130, bbox_inches="tight")
        plt.show()
    else:
        coef = pd.Series(best_model.coef_, index=selected_features).abs().nlargest(20).sort_values()
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.barh(coef.index, coef.values, color="#7E57C2")
        ax.set_xlabel("|Coefficient|")
        ax.set_title(f"{best_name} — Top 20 |Coefficients|")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        plt.savefig(MODEL_DIR / "11_best_model_importances.png", dpi=130, bbox_inches="tight")
        plt.show()
    """),

    code("""\
    # SHAP analysis (works with tree models)
    if hasattr(best_model, "feature_importances_"):
        sample_size = min(3000, len(X_test_best))
        X_shap      = X_test_best.iloc[:sample_size].copy()

        explainer   = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X_shap)

        # SHAP bar chart
        plt.figure(figsize=(8, 8))
        shap.summary_plot(shap_values, X_shap, plot_type="bar", max_display=20, show=False)
        plt.title(f"SHAP — Mean |Impact| on Price Prediction ({best_name})", pad=12)
        plt.tight_layout()
        plt.savefig(MODEL_DIR / "12_shap_importance.png", dpi=130, bbox_inches="tight")
        plt.show()

        # SHAP beeswarm
        plt.figure(figsize=(9, 8))
        shap.summary_plot(shap_values, X_shap, max_display=20, show=False)
        plt.title(f"SHAP Beeswarm — Price Drivers ({best_name})", pad=12)
        plt.tight_layout()
        plt.savefig(MODEL_DIR / "13_shap_beeswarm.png", dpi=130, bbox_inches="tight")
        plt.show()
    else:
        print("SHAP TreeExplainer not applicable to linear models — skipping SHAP plots.")
    """),

    code("""\
    # Residual analysis
    y_pred_test = all_results[best_name]["y_pred_test"]
    residuals   = y_test - y_pred_test

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1 — Predicted vs Actual
    axes[0].scatter(y_pred_test, y_test, alpha=0.2, s=4, color="#1976D2", rasterized=True)
    lo, hi = min(y_pred_test.min(), y_test.min()), max(y_pred_test.max(), y_test.max())
    axes[0].plot([lo, hi], [lo, hi], "r--", lw=1.5)
    axes[0].set_xlabel("Predicted log1p(price)")
    axes[0].set_ylabel("Actual log1p(price)")
    axes[0].set_title(f"Predicted vs Actual  (R²={all_results[best_name]['Test']['R2']:.3f})")

    # 2 — Residuals vs Predicted
    axes[1].scatter(y_pred_test, residuals, alpha=0.2, s=4, color="#FF8F00", rasterized=True)
    axes[1].axhline(0, color="red", lw=1.5, ls="--")
    axes[1].set_xlabel("Predicted log1p(price)")
    axes[1].set_ylabel("Residual (actual − predicted)")
    axes[1].set_title("Residuals vs Predicted")

    # 3 — Residual distribution
    axes[2].hist(residuals, bins=60, color="#43A047", alpha=0.85, edgecolor="white")
    axes[2].axvline(0, color="red", lw=1.5, ls="--")
    axes[2].set_xlabel("Residual")
    axes[2].set_title(f"Residual distribution  (σ={residuals.std():.3f})")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.suptitle(
        f"{best_name} — Test Set Residuals  |  "
        f"MdAPE {all_results[best_name]['Test']['MdAPE']:.1f}%  |  "
        f"MAE €{all_results[best_name]['Test']['MAE_EUR']:.1f}",
        fontsize=11, y=1.02
    )
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "14_residuals.png", dpi=130, bbox_inches="tight")
    plt.show()
    """),

    code("""\
    # Per-city breakdown of best model
    rows = []
    for city in ["Madrid", "Barcelona", "Málaga"]:
        mask   = city_test == city
        yt     = y_test[mask];          yp = y_pred_test[mask]
        yt_e   = np.expm1(yt);          yp_e = np.expm1(yp)
        rows.append(dict(
            City=city, n=int(mask.sum()),
            R2=round(r2_score(yt, yp), 3),
            RMSE_log=round(np.sqrt(mean_squared_error(yt, yp)), 4),
            MAE_EUR=round(mean_absolute_error(yt_e, yp_e), 1),
            MdAPE_pct=round(np.median(np.abs((yt_e - yp_e) / yt_e)) * 100, 1),
        ))

    print(f"Per-city performance — {best_name}:")
    print(pd.DataFrame(rows).set_index("City").to_string())
    """),

    # ─────────────────────────────────────────────────────────────────────────
    # 17. BUSINESS CONCLUSIONS
    # ─────────────────────────────────────────────────────────────────────────
    md("""\
    ---
    ## 17. Business Conclusions & Recommendations

    ### Key Findings

    | Finding | Implication for Investment Decision |
    |---------|-------------------------------------|
    | Location (neighbourhood, lat/lon) is the strongest price driver | Neighbourhood selection is more impactful than property upgrades |
    | `accommodates`, `bedrooms`, `bathrooms_number` drive price proportionally | Larger properties command premium rates — worth modelling capacity vs cost |
    | Amenity flags (pool, elevator, AC, view) add 5–25% price premium | Targeted renovation budget can improve predicted revenue |
    | Competitive density (500 m) has a negative price effect | Saturated micro-markets command lower prices |
    | Host tenure and superhost status add a modest premium | New Airbnb hosts should expect lower initial pricing |

    ### Model Recommendation

    The **best-performing model** will be used in the Stage 2 occupancy model
    and the NPV comparison engine. All models are saved to `../models/` for
    reuse in the web application.

    ### Limitations

    - Price is scraped at a single point in time (no dynamic pricing capture)
    - No calendar price data available (100% null in InsideAirbnb for Spain)
    - New listings with no reviews use city-median review scores (imputed)
    - Very high-end luxury properties (>99.5th percentile) are excluded from training
    """),

    # ─────────────────────────────────────────────────────────────────────────
    # 18. SAVE + FINAL SUMMARY
    # ─────────────────────────────────────────────────────────────────────────
    md("---\n## 18. Save Artefacts & Final Summary Table"),

    code("""\
    # Save best model
    joblib.dump(best_model, MODEL_DIR / "price_best_model.pkl")

    # Feature metadata
    with open(MODEL_DIR / "price_feature_cols.json", "w") as fh:
        json.dump({
            "selected_features": selected_features,
            "cat_cols"         : CAT_COLS,
            "amenity_cols"     : AMENITY_COLS,
            "review_score_cols": REVIEW_SCORE_COLS,
            "num_cols"         : num_cols,
            "best_model"       : best_name,
        }, fh, indent=2)

    # Enrich full dataset with price_hat
    X_all_for_pred = df_model[selected_features].copy()
    if best_name in ("Linear Regression", "Ridge"):
        X_all_for_pred_arr = scaler.transform(X_all_for_pred)
        df["price_hat"] = np.expm1(best_model.predict(X_all_for_pred_arr))
    else:
        df["price_hat"] = np.expm1(best_model.predict(X_all_for_pred))

    OUT_PATH = pathlib.Path("../Data/processed/listings_with_price_hat.parquet")
    df.to_parquet(OUT_PATH, index=False)

    print("Saved files:")
    for p in sorted(MODEL_DIR.iterdir()):
        print(f"  {p.name:<42}  {p.stat().st_size/1024:>7.1f} KB")
    print(f"\\nEnriched dataset → {OUT_PATH}  ({OUT_PATH.stat().st_size/1e6:.1f} MB)")
    """),

    code("""\
    # Auto-populated final summary table
    summary_rows = []
    for name, res in all_results.items():
        summary_rows.append({
            "Model"         : name.replace(" (XGBoost-equivalent)", ""),
            "Train R²"      : round(res["Train"]["R2"],       3),
            "Test R²"       : round(res["Test"]["R2"],        3),
            "Overfit Gap"   : round(res["Train"]["R2"] - res["Test"]["R2"], 3),
            "Test RMSE (€)" : round(res["Test"]["RMSE_EUR"],  1),
            "Test MAE (€)"  : round(res["Test"]["MAE_EUR"],   1),
            "Test MdAPE (%)" : round(res["Test"]["MdAPE"],   1),
            "Test RMSE (log)": round(res["Test"]["RMSE_log"], 4),
        })

    summary_df = (
        pd.DataFrame(summary_rows)
        .set_index("Model")
        .sort_values("Test R²", ascending=False)
    )

    # Highlight best in each column
    print("=" * 90)
    print("FINAL MODEL SUMMARY")
    print("=" * 90)
    print(summary_df.to_string())
    print("=" * 90)
    print(f"\\n★  Best model: {best_name.replace(' (XGBoost-equivalent)', '')}")
    print(f"   Test R²   : {all_results[best_name]['Test']['R2']:.3f}")
    print(f"   Test MdAPE: {all_results[best_name]['Test']['MdAPE']:.1f}%")
    print(f"   Test MAE  : €{all_results[best_name]['Test']['MAE_EUR']:.1f}/night")
    print(f"\\nAll artefacts saved to {MODEL_DIR.resolve()}")
    """),
]

# ── Build notebook JSON ───────────────────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.9.0",
        },
    },
    "cells": CELLS,
}

OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print(f"Written: {OUT}  ({OUT.stat().st_size / 1024:.0f} KB)")
