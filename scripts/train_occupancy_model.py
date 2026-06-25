"""Train (or retrain) the calendar-based occupancy model — CLI entry point.

Replaces the legacy *San Francisco* review-based occupancy estimator
(``airbnb_iip.data.occupancy.estimate_occupancy``) with a learned LightGBM
model whose **target is computed directly from Inside Airbnb calendar data**:
a night is "booked" when ``available == 'f'``, and a listing's occupancy rate
is the share of booked nights across its forward calendar window.

This is the "Stage 2" model described in ``docs/FEATURE_ENGINEERING.md §5``.

Flow
====
1. ``occupancy_target`` — load the three city calendars
   (``<city>/calendar.csv.gz``), and for each ``listing_id`` compute
   ``occupancy_rate = mean(available == 'f')`` over its forward window.
2. Join the target onto ``Data/processed/listings_with_price_hat.parquet`` by
   ``id`` (the calendar ``listing_id`` == the listings ``id``). Keep **active**
   listings only (``has_availability == True``) and drop the fully-blocked
   cohort (rate == 1.0 across the whole window) which is indistinguishable from
   delisted/paused inventory.
3. Engineer a leakage-free feature set (no review counts/scores — the whole
   point is to sever the review→occupancy link): neighbourhood **target
   encoding** (mean occupancy), city / room_type / property_type label
   encoding, capacity, location, nightly price, min-nights, amenity flags +
   count, and competitive density.
4. Train LightGBM two ways — one **global** model (city as a feature) and one
   **by-city** model set — and keep whichever wins on held-out R². Metrics are
   computed leakage-free (neighbourhood map fit on the train split only).
5. Save the serving artefacts to ``models/`` (consumed by
   ``airbnb_iip.models.occupancy.OccupancyPredictor``):
   ``occupancy_best_model.pkl`` (+ per-city pkls), ``occupancy_encoders.pkl``,
   ``occupancy_feature_cols.json``.

Safety: any existing occupancy artefacts are copied to ``*_legacy.*`` (once,
never overwritten) before the new ones are written, so a revert is just
restoring those files.

Run:
    python scripts/train_occupancy_model.py
    python scripts/train_occupancy_model.py --calendar-dir ~/Desktop   # default
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.simplefilter("ignore")
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("train_occupancy")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from airbnb_iip.features.amenities import AMENITY_COLUMNS  # noqa: E402

MODELS = ROOT / "models"
LISTINGS_PARQUET = ROOT / "Data" / "processed" / "listings_with_price_hat.parquet"

ARTEFACTS_JSON = MODELS / "occupancy_feature_cols.json"
ENCODERS_PKL = MODELS / "occupancy_encoders.pkl"
GLOBAL_MODEL_PKL = MODELS / "occupancy_best_model.pkl"
SEED = 42

# Calendar dumps live one dir per city. Folder name → canonical city label used
# in the listings parquet ("Málaga" carries the accent there).
CITY_DIRS = {"Madrid": "Madrid", "Barcelona": "Barcelona", "Malaga": "Málaga"}

# Calendar-derived occupancy guardrails. A listing blocked for *every* night of
# its window is almost certainly inactive/paused, not booked solid — exclude it
# so the model learns genuine demand, not delisted inventory.
FULLY_BLOCKED = 0.999
MIN_CALENDAR_DAYS = 60          # need a meaningful window to trust the rate

# ── Feature groups (NO review counts/scores — deliberately) ──────────────────
NEIGHBOURHOOD_FEATURE = "neighbourhood_target_enc"
NEIGHBOURHOOD_SOURCE = "neighbourhood_cleansed"
LABEL_ENCODED = ["room_type", "property_type_std", "city"]   # city dropped for by-city models

NUMERIC = [
    "accommodates", "bedrooms", "bathrooms_number", "beds",
    "latitude", "longitude", "price", "minimum_nights",
    "competitive_density_500m", "amenity_count",
]
BINARY = list(AMENITY_COLUMNS) + ["instant_bookable"]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Target: occupancy rate from the calendar
# ─────────────────────────────────────────────────────────────────────────────

def occupancy_target(
    calendar_dir: Path,
    cutoff_date: str | None = None,
    horizon_days: int | None = None,
    overrides: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Per-listing calendar occupancy across all three cities.

    The Inside Airbnb calendar is forward-looking, so the far-future tail is
    unreliable. Two ways to bound the window (``horizon_days`` wins if both set):

    * ``horizon_days`` (e.g. ``120``) — keep each city's **first N days** from
      its own scrape date. This gives every city an equal-length, equal-horizon
      window, so occupancy is cross-city comparable (it neutralises the
      near-term-booking-horizon artifact). The calendar *months* still differ
      because scrape dates differ, so a genuine seasonal difference remains.
    * ``cutoff_date`` (e.g. ``"2026-05-31"``) — keep nights on or before an
      absolute date; the per-city window length then depends on the scrape date
      (not comparable across cities).

    Returns a frame with ``id``, ``occupancy_rate``, ``occupancy_nights_l365d``,
    ``calendar_days``.
    """
    cutoff = pd.Timestamp(cutoff_date) if cutoff_date else None
    overrides = overrides or {}
    frames = []
    for folder, _city in CITY_DIRS.items():
        overridden = folder in overrides
        path = Path(overrides[folder]) if overridden else calendar_dir / folder / "calendar.csv.gz"
        if not path.exists():
            raise FileNotFoundError(
                f"Calendar not found: {path}. Pass --calendar-dir to point at the "
                "folder that contains Madrid/, Barcelona/, Malaga/ calendar dumps, "
                "or --calendar-override <Folder>=<path> for a single city."
            )
        cal = pd.read_csv(path, usecols=["listing_id", "date", "available"])
        cal["date"] = pd.to_datetime(cal["date"])
        if horizon_days is not None:
            start = cal["date"].min()  # city scrape date — common anchor
            cal = cal[cal["date"] < start + pd.Timedelta(days=horizon_days)]
        elif cutoff is not None:
            cal = cal[cal["date"] <= cutoff]
        win = f"{cal['date'].min().date()}..{cal['date'].max().date()}"
        cal["booked"] = (cal["available"].astype(str).str.strip() == "f").astype("int8")
        g = cal.groupby("listing_id")["booked"]
        occ = pd.DataFrame({
            "id": g.size().index,
            "calendar_days": g.size().to_numpy(),
            "occupancy_rate": g.mean().to_numpy(),
        })
        occ["occupancy_nights_l365d"] = (occ["occupancy_rate"] * 365).round().astype(int)
        frames.append(occ)
        log.info("  calendar %-10s %6d listings  window %s (%d days median)  mean occ %.3f%s",
                 folder, len(occ), win, int(occ["calendar_days"].median()),
                 float(occ["occupancy_rate"].mean()),
                 f"  [override: {path}]" if overridden else "")
    return pd.concat(frames, ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# 2. + 3. Build the modelling frame
# ─────────────────────────────────────────────────────────────────────────────

def build_frame(
    calendar_dir: Path,
    cutoff_date: str | None = None,
    horizon_days: int | None = None,
    overrides: dict[str, str] | None = None,
) -> pd.DataFrame:
    if horizon_days is not None:
        window_desc = f" (first {horizon_days} days per city)"
    elif cutoff_date:
        window_desc = f" (cutoff ≤ {cutoff_date})"
    else:
        window_desc = ""
    log.info("Loading calendars from %s%s ...", calendar_dir, window_desc)
    target = occupancy_target(
        calendar_dir, cutoff_date=cutoff_date, horizon_days=horizon_days,
        overrides=overrides,
    )

    cols = list({
        "id", "city", NEIGHBOURHOOD_SOURCE, "neighbourhood_group_cleansed",
        "room_type", "property_type_std", "has_availability",
        *NUMERIC, *AMENITY_COLUMNS, "instant_bookable",
    } - {"amenity_count"})  # amenity_count is derived below
    listings = pd.read_parquet(LISTINGS_PARQUET, columns=cols)

    df = listings.merge(target, on="id", how="inner")
    log.info("Joined: %d listings matched calendar of %d (%.0f%%)",
             len(df), len(target), 100 * len(df) / len(target))

    # Active listings only, with a trustworthy window, excluding fully-blocked.
    before = len(df)
    df = df[df["has_availability"] == True]                    # noqa: E712
    df = df[df["calendar_days"] >= MIN_CALENDAR_DAYS]
    df = df[df["occupancy_rate"] < FULLY_BLOCKED]
    log.info("Active filter: %d → %d rows (dropped %d inactive/blocked)",
             before, len(df), before - len(df))

    # amenity_count = number of the 22 modelled amenity flags present.
    df["amenity_count"] = df[list(AMENITY_COLUMNS)].fillna(0).astype(int).sum(axis=1)
    df["instant_bookable"] = df["instant_bookable"].astype(float)
    for c in AMENITY_COLUMNS:
        df[c] = df[c].fillna(0).astype(float)

    return df.reset_index(drop=True)


def _fit_label_maps(df: pd.DataFrame, cols: list[str]) -> dict[str, dict]:
    """Deterministic label-encode maps (sorted categories → 0..n-1)."""
    maps: dict[str, dict] = {}
    for c in cols:
        cats = sorted(df[c].dropna().astype(str).unique())
        maps[c] = {cat: i for i, cat in enumerate(cats)}
    return maps


def _apply_features(
    df: pd.DataFrame,
    *,
    neigh_map: dict[str, float],
    neigh_default: float,
    label_maps: dict[str, dict],
    features: list[str],
) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for feat in features:
        if feat == NEIGHBOURHOOD_FEATURE:
            out[feat] = df[NEIGHBOURHOOD_SOURCE].astype(str).map(neigh_map).fillna(neigh_default)
        elif feat in label_maps:
            out[feat] = df[feat].astype(str).map(label_maps[feat]).fillna(-1).astype(float)
        else:
            out[feat] = pd.to_numeric(df[feat], errors="coerce")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 4. Train / evaluate
# ─────────────────────────────────────────────────────────────────────────────

def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    y_pred = np.clip(y_pred, 0.0, 1.0)
    return {
        "r2": round(float(r2_score(y_true, y_pred)), 4),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "mae_nights": round(float(mean_absolute_error(y_true, y_pred) * 365), 1),
    }


def _new_model():
    from lightgbm import LGBMRegressor
    return LGBMRegressor(
        objective="regression",
        n_estimators=700,
        learning_rate=0.03,
        num_leaves=63,
        min_child_samples=40,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
    )


def train_and_select(df: pd.DataFrame) -> dict:
    from sklearn.model_selection import train_test_split

    target = df["occupancy_rate"].to_numpy()
    train_idx, test_idx = train_test_split(
        df.index, test_size=0.2, random_state=SEED, stratify=df["city"]
    )
    tr, te = df.loc[train_idx], df.loc[test_idx]

    # Neighbourhood target encoding — fit on TRAIN only (leakage-free metric).
    tr_neigh_map = tr.groupby(NEIGHBOURHOOD_SOURCE)["occupancy_rate"].mean().to_dict()
    tr_neigh_default = float(tr["occupancy_rate"].mean())
    label_maps = _fit_label_maps(tr, LABEL_ENCODED)

    global_features = [NEIGHBOURHOOD_FEATURE] + NUMERIC + BINARY + LABEL_ENCODED
    city_features = [f for f in global_features if f != "city"]

    def Xy(frame, feats):
        X = _apply_features(
            frame, neigh_map=tr_neigh_map, neigh_default=tr_neigh_default,
            label_maps=label_maps, features=feats,
        )
        return X, frame["occupancy_rate"].to_numpy()

    # ── Global model ────────────────────────────────────────────────────────
    Xtr, ytr = Xy(tr, global_features)
    Xte, yte = Xy(te, global_features)
    gmodel = _new_model().fit(Xtr, ytr)
    global_metrics = _metrics(yte, gmodel.predict(Xte))
    log.info("GLOBAL  R²=%.4f  MAE=%.4f (%.1f nights)  RMSE=%.4f",
             global_metrics["r2"], global_metrics["mae"],
             global_metrics["mae_nights"], global_metrics["rmse"])

    # ── By-city models ────────────────────────────────────────────────────────
    city_models, per_city, by_city_pred = {}, {}, np.empty(len(te))
    te_reset = te.reset_index(drop=True)
    for city in df["city"].unique():
        ctr = tr[tr["city"] == city]
        cte = te[te["city"] == city]
        cxtr, cytr = Xy(ctr, city_features)
        cmodel = _new_model().fit(cxtr, cytr)
        city_models[city] = cmodel
        cxte, cyte = Xy(cte, city_features)
        preds = cmodel.predict(cxte)
        per_city[city] = {"n_test": int(len(cte)), **_metrics(cyte, preds)}
        by_city_pred[te.index.get_indexer(cte.index)] = preds
    by_city_metrics = _metrics(yte, by_city_pred)
    log.info("BY-CITY R²=%.4f  MAE=%.4f (%.1f nights)  RMSE=%.4f",
             by_city_metrics["r2"], by_city_metrics["mae"],
             by_city_metrics["mae_nights"], by_city_metrics["rmse"])
    for c, m in per_city.items():
        log.info("   %-10s n=%4d  R²=%.4f  MAE=%.4f (%.1f nights)",
                 c, m["n_test"], m["r2"], m["mae"], m["mae_nights"])

    # Naive city-mean baseline (what "no features" would score).
    city_mean = tr.groupby("city")["occupancy_rate"].mean()
    baseline = _metrics(yte, te["city"].map(city_mean).to_numpy())
    log.info("BASELINE (city mean) R²=%.4f  MAE=%.4f", baseline["r2"], baseline["mae"])

    approach = "by_city" if by_city_metrics["r2"] >= global_metrics["r2"] else "global"
    log.info("→ Selected approach: %s", approach.upper())

    return {
        "approach": approach,
        "global_features": global_features,
        "city_features": city_features,
        "label_maps": label_maps,
        "train_idx": train_idx,
        "metrics": {
            "global": global_metrics,
            "by_city": by_city_metrics,
            "per_city": per_city,
            "baseline": baseline,
            "chosen": by_city_metrics if approach == "by_city" else global_metrics,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. Refit on all data + save serving artefacts
# ─────────────────────────────────────────────────────────────────────────────

def _backup_legacy() -> None:
    for f in (GLOBAL_MODEL_PKL, ENCODERS_PKL, ARTEFACTS_JSON):
        legacy = f.with_name(f.stem + "_legacy" + f.suffix)
        if f.exists() and not legacy.exists():
            shutil.copy2(f, legacy)
            log.info("Backed up %s → %s", f.name, legacy.name)


def save_artefacts(
    df: pd.DataFrame,
    sel: dict,
    cutoff_date: str | None = None,
    horizon_days: int | None = None,
    overrides: dict[str, str] | None = None,
) -> None:
    _backup_legacy()
    MODELS.mkdir(exist_ok=True)

    approach = sel["approach"]
    # Refit encoders + model(s) on the FULL active dataset for serving.
    neigh_map = df.groupby(NEIGHBOURHOOD_SOURCE)["occupancy_rate"].mean().to_dict()
    neigh_default = float(df["occupancy_rate"].mean())
    label_maps = _fit_label_maps(df, LABEL_ENCODED)

    features = sel["global_features"] if approach == "global" else sel["city_features"]

    def Xy(frame, feats):
        X = _apply_features(
            frame, neigh_map=neigh_map, neigh_default=neigh_default,
            label_maps=label_maps, features=feats,
        )
        return X, frame["occupancy_rate"].to_numpy()

    importance: dict[str, float] = {}
    city_model_files: dict[str, str] = {}
    if approach == "global":
        X, y = Xy(df, features)
        model = _new_model().fit(X, y)
        joblib.dump(model, GLOBAL_MODEL_PKL)
        importance = dict(sorted(
            zip(features, (float(v) for v in model.feature_importances_)),
            key=lambda kv: -kv[1],
        ))
        log.info("Saved global model → %s", GLOBAL_MODEL_PKL.name)
    else:
        for city in df["city"].unique():
            sub = df[df["city"] == city]
            X, y = Xy(sub, features)
            model = _new_model().fit(X, y)
            fname = f"occupancy_city_{_slug(city)}_model.pkl"
            joblib.dump(model, MODELS / fname)
            city_model_files[city] = fname
            log.info("Saved %s model → %s (n=%d)", city, fname, len(sub))
        # Feature importance from a full-data global fit (for the model card).
        gmodel = _new_model().fit(*Xy(df, sel["global_features"]))
        importance = dict(sorted(
            zip(sel["global_features"], (float(v) for v in gmodel.feature_importances_)),
            key=lambda kv: -kv[1],
        ))

    encoders = {
        NEIGHBOURHOOD_SOURCE: neigh_map,
        "_neighbourhood_default": neigh_default,
        "_medians": {f: float(pd.to_numeric(df[f], errors="coerce").median())
                     for f in NUMERIC},
        **label_maps,
    }
    joblib.dump(encoders, ENCODERS_PKL)
    log.info("Saved encoders → %s", ENCODERS_PKL.name)

    artefacts = {
        "model": "LightGBM",
        "target": "occupancy_rate",
        "target_definition": "share of forward-calendar nights with available == 'f'",
        "approach": approach,
        "features": features,
        "global_features": sel["global_features"],
        "city_features": sel["city_features"],
        "neighbourhood_feature": NEIGHBOURHOOD_FEATURE,
        "neighbourhood_source": NEIGHBOURHOOD_SOURCE,
        "label_encoded": LABEL_ENCODED,
        "model_file": GLOBAL_MODEL_PKL.name if approach == "global" else None,
        "city_model_files": city_model_files,
        "occupancy_clip": [0.02, 0.95],
        "metrics": sel["metrics"],
        "feature_importance": importance,
        "n_rows": int(len(df)),
        "n_train": int(len(sel["train_idx"])),
        "n_test": int(len(df) - len(sel["train_idx"])),
        "activeness_filter": "has_availability == True & calendar_days >= 60 & rate < 0.999",
        "mean_occupancy": round(float(df["occupancy_rate"].mean()), 4),
        "calendar_cutoff": cutoff_date,
        "occupancy_horizon_days": horizon_days,
        "calendar_overrides": overrides or {},
    }
    ARTEFACTS_JSON.write_text(json.dumps(artefacts, indent=2, ensure_ascii=False))
    log.info("Saved artefacts → %s", ARTEFACTS_JSON.name)


def _slug(city: str) -> str:
    import unicodedata
    return (
        unicodedata.normalize("NFKD", city).encode("ascii", "ignore").decode().lower()
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--calendar-dir", type=Path, default=Path.home() / "Desktop",
        help="Folder containing Madrid/ Barcelona/ Malaga/ calendar.csv.gz dumps.",
    )
    ap.add_argument(
        "--cutoff-date", type=str, default=None,
        help="Restrict the target to calendar nights on or before this absolute "
             "date (e.g. 2026-05-31). Not cross-city comparable (window length "
             "depends on scrape date). Ignored if --horizon-days is set.",
    )
    ap.add_argument(
        "--horizon-days", type=int, default=None,
        help="Use each city's first N calendar days from its scrape date (e.g. "
             "120) — an equal-length, equal-horizon window so occupancy is "
             "cross-city comparable. Recommended over --cutoff-date.",
    )
    ap.add_argument(
        "--calendar-override", action="append", default=[], metavar="Folder=PATH",
        help="Override one city's calendar with an explicit file, e.g. "
             "'Malaga=/path/to/calendar.csv.gz' (folder key matches Madrid/"
             "Barcelona/Malaga). Repeatable. Leaves --calendar-dir for the rest.",
    )
    args = ap.parse_args()

    overrides: dict[str, str] = {}
    for item in args.calendar_override:
        folder, _, path = item.partition("=")
        if not path:
            ap.error(f"--calendar-override must be Folder=PATH, got {item!r}")
        overrides[folder.strip()] = path.strip()

    df = build_frame(
        args.calendar_dir, cutoff_date=args.cutoff_date,
        horizon_days=args.horizon_days, overrides=overrides,
    )
    log.info("Modelling frame: %d rows, mean occupancy %.3f",
             len(df), float(df["occupancy_rate"].mean()))
    sel = train_and_select(df)
    save_artefacts(
        df, sel, cutoff_date=args.cutoff_date,
        horizon_days=args.horizon_days, overrides=overrides,
    )
    log.info("Done.")


if __name__ == "__main__":
    main()
