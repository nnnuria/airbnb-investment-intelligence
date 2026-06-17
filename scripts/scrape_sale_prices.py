"""Run the district-level Idealista sale price scraper.

Thin entry point — all logic lives in
``airbnb_iip.data.scrapers.sale_prices``. Reads ``APIFY_API_TOKEN`` from
the environment (``.env`` is loaded if python-dotenv is installed).

Usage:
    python scripts/scrape_sale_prices.py                          # all cities
    python scripts/scrape_sale_prices.py --cities malaga          # one city
    python scripts/scrape_sale_prices.py --cities madrid \
        --districts centro,salamanca                              # one slice
    python scripts/scrape_sale_prices.py --aggregate-only         # CSV only
"""
import sys

from airbnb_iip.data.scrapers.sale_prices import main

if __name__ == "__main__":
    sys.exit(main())
