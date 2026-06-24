"""Run multilingual review sentiment per city → data/processed/<city>_sentiment.parquet (UC3).

Usage:
    python scripts/run_sentiment.py            # all cities, default sample
    python scripts/run_sentiment.py --full     # no sampling (slow)
"""
import argparse
from pathlib import Path

import pandas as pd

from airbnb_iip.models import nlp

# Update these to your local gitignored raw paths.
CITIES = {
    "madrid":    ("data/raw/reviews_madrid.csv.gz",    "data/raw/listings_madrid.csv.gz"),
    "barcelona": ("data/raw/reviews_barcelona.csv.gz", "data/raw/listings_barcelona.csv.gz"),
    "malaga":    ("data/raw/reviews_malaga.csv.gz",    "data/raw/listings_malaga.csv.gz"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="score all reviews (no sampling)")
    ap.add_argument("--sample", type=int, default=40000)
    args = ap.parse_args()
    sample = None if args.full else args.sample

    out = Path("data/processed"); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for city, (rp, lp) in CITIES.items():
        r = nlp.run_city(rp, lp, city, sample=sample)
        r["listing_sentiment"].to_parquet(out / f"{city}_sentiment.parquet", index=False)
        rows.append({k: v for k, v in r.items()
                     if k not in ("listing_sentiment", "model")})
        print(f"{city}: {r['reviews_scored']:,} scored | NB acc {r['nb_accuracy']} | "
              f"corr {r['corr_sentiment_vs_rating']}")
    pd.DataFrame(rows).to_csv(out / "sentiment_summary.csv", index=False)
    print("written data/processed/*_sentiment.parquet + sentiment_summary.csv")


if __name__ == "__main__":
    main()
