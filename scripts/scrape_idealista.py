"""Run the Idealista scraper for all (city × operation) pairs.

Thin entry point — all logic lives in
``airbnb_iip.data.scrapers.idealista``. Reads ``APIFY_API_TOKEN`` from
the environment (``.env`` is loaded if python-dotenv is installed).

Usage:
    python scripts/scrape_idealista.py                  # all defaults
    python scripts/scrape_idealista.py --cities madrid --operations rent
    python scripts/scrape_idealista.py --max-items 5000 --fetch-details
"""
import sys

from airbnb_iip.data.scrapers.idealista import main

if __name__ == "__main__":
    sys.exit(main())
