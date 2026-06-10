"""Quick structural inventory of every CSV in data/raw/."""
import pandas as pd
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"

files = sorted(RAW.rglob("*.csv"))
for f in files:
    rel = f.relative_to(RAW)
    print("=" * 90)
    print(f"FILE: {rel}")
    # robust read: sniff separator, tolerate bad lines
    try:
        df = pd.read_csv(f, sep=None, engine="python", on_bad_lines="skip", nrows=200000)
    except Exception as e:
        print(f"  read error with python engine: {e}")
        df = pd.read_csv(f, on_bad_lines="skip", nrows=200000)
    print(f"  shape (capped 200k): {df.shape}")
    print(f"  columns ({len(df.columns)}):")
    for c in df.columns:
        nn = df[c].notna().sum()
        pct = 100 * nn / len(df)
        sample = df[c].dropna().head(2).tolist()
        print(f"    - {c!r:40s} {str(df[c].dtype):10s} non-null {pct:5.1f}%  e.g. {sample}")
    print()
