"""Download the three Idealista Kaggle datasets into data/raw/.

Requires Kaggle API credentials at ~/.kaggle/kaggle.json
(see https://www.kaggle.com/settings/account -> API -> Create New Token).

Usage:
    python src/download_data.py
"""

import subprocess
import sys
from pathlib import Path

# dataset slug -> subfolder under data/raw/
DATASETS = {
    "fjcob1/idealista-madrid": "madrid",
    "victorianomh/idealista-barcelona-raw-scraped-data": "barcelona",
    "laurabarreda/rental-listins-in-idealista-spain": "spain",
}

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def main() -> None:
    for slug, subdir in DATASETS.items():
        dest = RAW_DIR / subdir
        dest.mkdir(parents=True, exist_ok=True)
        print(f"\n==> Downloading {slug} -> {dest}")
        subprocess.run(
            [
                sys.executable, "-m", "kaggle", "datasets", "download",
                "-d", slug, "-p", str(dest), "--unzip",
            ],
            check=True,
        )
    print("\nAll datasets downloaded.")


if __name__ == "__main__":
    main()
