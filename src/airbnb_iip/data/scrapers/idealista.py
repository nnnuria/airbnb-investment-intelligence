"""Idealista scraper agent.

Drives the `igolaizola/idealista-scraper`_ Apify actor through the Apify HTTP
API to collect apartment listings for Madrid, Barcelona, and Málaga (both
sale and rent). Output is one JSON Lines file per (city, operation) under
``data/raw/idealista/<city>/<operation>_<YYYY-MM-DD>.jsonl`` — lossless raw
records plus a tidy normalised view available via :func:`normalize_listing`
or :func:`load_jsonl_as_dataframe`.

.. _`igolaizola/idealista-scraper`: https://apify.com/igolaizola/idealista-scraper

The Apify API token MUST come from the ``APIFY_API_TOKEN`` environment
variable. The module never accepts a hard-coded token.

Idealista caps search results at ~1,800 per query. To go beyond that, the
actor automatically splits the location into sub-areas when ``max_items >
2500`` — set the cap accordingly when you need full coverage.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Sequence

logger = logging.getLogger(__name__)

ACTOR_ID = "igolaizola/idealista-scraper"

# City → default Idealista `location` value. The actor accepts either a city
# name or a precise Idealista Location ID. We default to the city name (works
# out of the box) and let callers pass a Location ID to narrow the search.
CITY_LOCATIONS: dict[str, str] = {
    "madrid": "Madrid",
    "barcelona": "Barcelona",
    "malaga": "Málaga",
}

OPERATIONS: tuple[str, ...] = ("sale", "rent")

# Idealista's site-level cap (search pages stop returning new results past this).
SITE_RESULT_CAP = 1800
# Above this, the actor automatically splits the query into sub-locations.
SUBLOCATION_SPLIT_THRESHOLD = 2500

DEFAULT_OUTPUT_DIR = Path("data/raw/idealista")


# ── Job definition ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ScrapeJob:
    """One actor invocation: a (city, operation) pair plus filters."""

    city: str
    operation: str               # "sale" | "rent"
    location: str                # city name or Idealista Location ID
    max_items: int = 2500
    fetch_details: bool = False
    fetch_stats: bool = False
    property_type: str = "homes"
    country: str = "es"
    extra_input: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.operation not in OPERATIONS:
            raise ValueError(
                f"operation must be one of {OPERATIONS}, got {self.operation!r}"
            )
        if self.max_items < 0:
            raise ValueError(f"max_items must be >= 0, got {self.max_items}")

    def actor_input(self) -> dict[str, Any]:
        """Build the actor input payload."""
        payload: dict[str, Any] = {
            "operation": self.operation,
            "propertyType": self.property_type,
            "country": self.country,
            "location": self.location,
            "maxItems": self.max_items,
            "fetchDetails": self.fetch_details,
            "fetchStats": self.fetch_stats,
            "proxyConfiguration": {
                "useApifyProxy": True,
                "apifyProxyGroups": ["RESIDENTIAL"],
            },
        }
        payload.update(self.extra_input)
        return payload


# ── Scraper agent ─────────────────────────────────────────────────────────────

def _run_field(run: Any, dict_key: str, attr: str) -> Any:
    """Read one field from an Apify actor-run result.

    apify-client < 2 returns the run as a plain dict with camelCase keys
    (``run["defaultDatasetId"]``); apify-client >= 3 returns a typed model
    with snake_case attributes (``run.default_dataset_id``). Support both.
    """
    if isinstance(run, dict):
        return run.get(dict_key)
    return getattr(run, attr, None)


class IdealistaScraper:
    """Drive the Idealista Apify actor and stream results to disk."""

    def __init__(
        self,
        token: str | None = None,
        client: Any = None,
        output_dir: Path | str = DEFAULT_OUTPUT_DIR,
        snapshot_date: str | None = None,
    ):
        if client is None:
            token = token or os.environ.get("APIFY_API_TOKEN")
            if not token:
                raise RuntimeError(
                    "APIFY_API_TOKEN env var is not set. Create a token at "
                    "https://console.apify.com/account/integrations and add "
                    "it to your .env file."
                )
            from apify_client import ApifyClient  # local import keeps the dep optional
            client = ApifyClient(token)
        self.client = client
        self.output_dir = Path(output_dir)
        self.snapshot_date = snapshot_date or date.today().isoformat()

    # ─ public API ────────────────────────────────────────────────────────────

    def run_job(self, job: ScrapeJob) -> Path:
        """Run one scrape job and write the dataset to a JSONL file.

        Returns the output path. Raises on actor failure.
        """
        self._warn_on_limits(job)
        out_path = self.output_path(job)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Apify run starting: actor=%s city=%s operation=%s location=%r maxItems=%d",
            ACTOR_ID, job.city, job.operation, job.location, job.max_items,
        )
        try:
            run = self.client.actor(ACTOR_ID).call(run_input=job.actor_input())
        except Exception:
            logger.exception(
                "Apify actor call failed: city=%s operation=%s",
                job.city, job.operation,
            )
            raise

        dataset_id = _run_field(run, "defaultDatasetId", "default_dataset_id")
        if not run or not dataset_id:
            raise RuntimeError(
                f"Actor run returned no defaultDatasetId. Run object: {run!r}"
            )

        status = _run_field(run, "status", "status")
        if status and status != "SUCCEEDED":
            logger.warning(
                "Actor run finished with status=%s for city=%s operation=%s — "
                "writing whatever made it into the dataset.",
                status, job.city, job.operation,
            )

        n_items = self._write_dataset(dataset_id, out_path, job)
        logger.info(
            "Wrote %d listings to %s (status=%s)",
            n_items, out_path, status or "unknown",
        )
        return out_path

    def run_all(self, jobs: Iterable[ScrapeJob]) -> dict[tuple[str, str], Path]:
        """Run every job; jobs that fail are logged and skipped.

        Returns a dict keyed by (city, operation) for jobs that succeeded.
        """
        results: dict[tuple[str, str], Path] = {}
        for job in jobs:
            try:
                results[(job.city, job.operation)] = self.run_job(job)
            except Exception:
                # already logged inside run_job; keep going so a Madrid
                # failure doesn't kill Barcelona + Málaga.
                logger.error(
                    "Job failed (skipping): city=%s operation=%s",
                    job.city, job.operation,
                )
        return results

    def output_path(self, job: ScrapeJob) -> Path:
        return (
            self.output_dir
            / job.city
            / f"{job.operation}_{self.snapshot_date}.jsonl"
        )

    # ─ internals ─────────────────────────────────────────────────────────────

    def _write_dataset(
        self, dataset_id: str, out_path: Path, job: ScrapeJob
    ) -> int:
        """Stream the dataset to a JSONL file. Returns item count."""
        dataset = self.client.dataset(dataset_id)
        n = 0
        with out_path.open("w", encoding="utf-8") as fh:
            for item in dataset.iterate_items():
                # Tag every record with collection metadata so downstream code
                # never has to guess where a record came from.
                item.setdefault("_city", job.city)
                item.setdefault("_operation", job.operation)
                item.setdefault("_scraped_at", self.snapshot_date)
                fh.write(json.dumps(item, ensure_ascii=False) + "\n")
                n += 1
        return n

    @staticmethod
    def _warn_on_limits(job: ScrapeJob) -> None:
        if job.max_items == 0:
            logger.warning(
                "max_items=0 (unlimited) — runs may be long and expensive. "
                "Sub-location splitting will be used past %d items.",
                SUBLOCATION_SPLIT_THRESHOLD,
            )
        elif job.max_items > SUBLOCATION_SPLIT_THRESHOLD:
            logger.info(
                "max_items=%d > %d → actor will split into sub-locations "
                "(slower; ordering changes).",
                job.max_items, SUBLOCATION_SPLIT_THRESHOLD,
            )
        elif job.max_items > SITE_RESULT_CAP:
            logger.warning(
                "max_items=%d > Idealista's ~%d-per-query cap but ≤ %d "
                "sub-location threshold — actor will likely top out near %d. "
                "Raise to >%d to trigger sub-location splitting.",
                job.max_items, SITE_RESULT_CAP,
                SUBLOCATION_SPLIT_THRESHOLD, SITE_RESULT_CAP,
                SUBLOCATION_SPLIT_THRESHOLD,
            )


# ── Job factory ───────────────────────────────────────────────────────────────

def default_jobs(
    cities: Sequence[str] = tuple(CITY_LOCATIONS),
    operations: Sequence[str] = OPERATIONS,
    max_items: int = 2500,
    fetch_details: bool = False,
    fetch_stats: bool = False,
    location_overrides: dict[str, str] | None = None,
) -> list[ScrapeJob]:
    """Build the standard (city × operation) job grid for the Capstone."""
    unknown_cities = [c for c in cities if c not in CITY_LOCATIONS]
    if unknown_cities:
        raise ValueError(
            f"Unknown cities: {unknown_cities}. Known: {list(CITY_LOCATIONS)}"
        )
    unknown_ops = [o for o in operations if o not in OPERATIONS]
    if unknown_ops:
        raise ValueError(
            f"Unknown operations: {unknown_ops}. Known: {list(OPERATIONS)}"
        )
    overrides = location_overrides or {}
    jobs: list[ScrapeJob] = []
    for city in cities:
        location = overrides.get(city, CITY_LOCATIONS[city])
        for op in operations:
            jobs.append(
                ScrapeJob(
                    city=city,
                    operation=op,
                    location=location,
                    max_items=max_items,
                    fetch_details=fetch_details,
                    fetch_stats=fetch_stats,
                )
            )
    return jobs


# ── Normalisation ─────────────────────────────────────────────────────────────

# Map canonical (snake_case) field names → list of likely keys in the raw
# actor output. We try each candidate in order; first non-empty hit wins.
# This is best-effort: actor field names can change, so missing fields fall
# back to None rather than raising.
_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "listing_id":       ("propertyCode", "id", "code"),
    "url":              ("url", "link"),
    "title":            ("title", "headline"),
    "price_eur":        ("price", "priceValue"),
    "price_per_m2_eur": ("pricePerM2", "priceByArea"),
    "size_m2":          ("size", "area", "squareMeters"),
    "rooms":            ("rooms", "bedrooms"),
    "bathrooms":        ("bathrooms",),
    "property_type":    ("propertyType",),
    "home_type":        ("homeType", "subType", "type"),
    "floor":            ("floor",),
    "address":          ("address", "addressVisibility"),
    "district":         ("district",),
    "neighborhood":     ("neighborhood", "neighbourhood"),
    "municipality":     ("municipality", "city"),
    "province":         ("province",),
    "latitude":         ("latitude", "lat"),
    "longitude":        ("longitude", "lng", "lon"),
    "description":      ("description", "descriptionText"),
    "features":         ("features", "characteristics", "tags"),
    "images":           ("images", "multimedia"),
    "virtual_tour":     ("virtualTour", "tour"),
    "agency":           ("agency", "agencyName", "advertiser"),
    "phone":            ("phone", "contactPhone"),
    "publication_date": ("publicationDate", "publishedAt", "datePublished"),
    "status":           ("status", "propertyStatus"),
}


def _first_present(record: dict[str, Any], keys: Sequence[str]) -> Any:
    for k in keys:
        if k in record and record[k] not in (None, "", []):
            return record[k]
    return None


def normalize_listing(record: dict[str, Any]) -> dict[str, Any]:
    """Project an actor record onto the canonical schema.

    Unknown fields are silently dropped; missing canonical fields are set to
    None. Adds ``n_images`` and ``has_virtual_tour`` convenience fields plus
    the ``_city`` / ``_operation`` / ``_scraped_at`` tags written by the
    scraper.
    """
    out: dict[str, Any] = {}
    for canonical, aliases in _FIELD_ALIASES.items():
        out[canonical] = _first_present(record, aliases)

    images = out.get("images") or []
    out["n_images"] = len(images) if isinstance(images, list) else None
    out["has_virtual_tour"] = bool(out.get("virtual_tour"))

    for tag in ("_city", "_operation", "_scraped_at"):
        if tag in record:
            out[tag.lstrip("_")] = record[tag]
    return out


def load_jsonl_as_dataframe(path: Path | str, normalize: bool = True):
    """Load a JSONL output into a pandas DataFrame.

    Imports pandas lazily so the scraper module itself stays light.
    """
    import pandas as pd

    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            rows.append(normalize_listing(record) if normalize else record)
    return pd.DataFrame(rows)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scrape Idealista listings via Apify's igolaizola/idealista-scraper "
            "actor. Reads APIFY_API_TOKEN from the environment."
        ),
    )
    parser.add_argument(
        "--cities",
        default=",".join(CITY_LOCATIONS),
        help=(
            f"Comma-separated cities. Default: {','.join(CITY_LOCATIONS)}. "
            f"Known: {','.join(CITY_LOCATIONS)}."
        ),
    )
    parser.add_argument(
        "--operations",
        default=",".join(OPERATIONS),
        help=f"Comma-separated operations. Default and known: {','.join(OPERATIONS)}.",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=2500,
        help=(
            "Cap per (city, operation) job. Idealista caps results at ~1800 "
            "per query; set >2500 to trigger the actor's sub-location split. "
            "0 = unlimited. Default: 2500."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}.",
    )
    parser.add_argument(
        "--fetch-details",
        action="store_true",
        help="Fetch per-listing details (extra request per item; slow).",
    )
    parser.add_argument(
        "--fetch-stats",
        action="store_true",
        help="Fetch per-listing stats (extra request per item; slow).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # python-dotenv is optional — load .env if it's installed
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        logger.debug("python-dotenv not installed; relying on shell environment")

    cities = [c.strip().lower() for c in args.cities.split(",") if c.strip()]
    operations = [o.strip().lower() for o in args.operations.split(",") if o.strip()]
    try:
        jobs = default_jobs(
            cities=cities,
            operations=operations,
            max_items=args.max_items,
            fetch_details=args.fetch_details,
            fetch_stats=args.fetch_stats,
        )
    except ValueError as e:
        logger.error(str(e))
        return 2

    try:
        scraper = IdealistaScraper(output_dir=args.output_dir)
    except RuntimeError as e:
        logger.error(str(e))
        return 2

    results = scraper.run_all(jobs)
    n_ok = len(results)
    n_total = len(jobs)
    logger.info("Completed %d/%d jobs", n_ok, n_total)
    return 0 if n_ok == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
