"""Idealista sale price scraper — district-level €/m² index.

Drives the same `igolaizola/idealista-scraper`_ Apify actor used by the
generic Idealista listings scraper, but iterates per district inside each
target city (Madrid, Barcelona, Málaga). Raw per-listing output is preserved
as JSONL; the headline deliverable is an aggregated district-level €/m² CSV
written to ``data/external/sale_prices_by_district_<YYYY-MM-DD>.csv`` with
snapshot date and source label for downstream joins.

.. _`igolaizola/idealista-scraper`: https://apify.com/igolaizola/idealista-scraper

Why per-district: Idealista caps a single search at ~1,800 results, so a
city-level scrape under-samples outer districts. One small query per
district produces an even sample for a stable €/m² index.

Token: ``APIFY_API_TOKEN`` must be set in the environment (loaded from
``.env`` if python-dotenv is installed). It is never hard-coded.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import statistics
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from airbnb_iip.data.scrapers.idealista import ACTOR_ID, _run_field

logger = logging.getLogger(__name__)

# ── District catalogue ───────────────────────────────────────────────────────
# Keys are URL-safe slugs used for storage and aggregation; values are the
# search strings passed to the actor's ``location`` field. Callers can pass
# Idealista Location IDs as overrides for tighter targeting.
CITY_DISTRICTS: dict[str, dict[str, str]] = {
    "madrid": {
        "centro":               "Centro, Madrid",
        "arganzuela":           "Arganzuela, Madrid",
        "retiro":               "Retiro, Madrid",
        "salamanca":            "Salamanca, Madrid",
        "chamartin":            "Chamartín, Madrid",
        "tetuan":               "Tetuán, Madrid",
        "chamberi":             "Chamberí, Madrid",
        "fuencarral-el-pardo":  "Fuencarral-El Pardo, Madrid",
        "moncloa-aravaca":      "Moncloa-Aravaca, Madrid",
        "latina":               "Latina, Madrid",
        "carabanchel":          "Carabanchel, Madrid",
        "usera":                "Usera, Madrid",
        "puente-de-vallecas":   "Puente de Vallecas, Madrid",
        "moratalaz":            "Moratalaz, Madrid",
        "ciudad-lineal":        "Ciudad Lineal, Madrid",
        "hortaleza":            "Hortaleza, Madrid",
        "villaverde":           "Villaverde, Madrid",
        "villa-de-vallecas":    "Villa de Vallecas, Madrid",
        "vicalvaro":            "Vicálvaro, Madrid",
        "san-blas-canillejas":  "San Blas-Canillejas, Madrid",
        "barajas":              "Barajas, Madrid",
    },
    "barcelona": {
        "ciutat-vella":         "Ciutat Vella, Barcelona",
        "eixample":             "Eixample, Barcelona",
        "sants-montjuic":       "Sants-Montjuïc, Barcelona",
        "les-corts":            "Les Corts, Barcelona",
        "sarria-sant-gervasi":  "Sarrià-Sant Gervasi, Barcelona",
        "gracia":               "Gràcia, Barcelona",
        "horta-guinardo":       "Horta-Guinardó, Barcelona",
        "nou-barris":           "Nou Barris, Barcelona",
        "sant-andreu":          "Sant Andreu, Barcelona",
        "sant-marti":           "Sant Martí, Barcelona",
    },
    "malaga": {
        "centro":               "Centro, Málaga",
        "este":                 "Este, Málaga",
        "ciudad-jardin":        "Ciudad Jardín, Málaga",
        "bailen-miraflores":    "Bailén-Miraflores, Málaga",
        "palma-palmilla":       "Palma-Palmilla, Málaga",
        "cruz-de-humilladero":  "Cruz de Humilladero, Málaga",
        "carretera-de-cadiz":   "Carretera de Cádiz, Málaga",
        "churriana":            "Churriana, Málaga",
        "campanillas":          "Campanillas, Málaga",
        "puerto-de-la-torre":   "Puerto de la Torre, Málaga",
        "teatinos-universidad": "Teatinos-Universidad, Málaga",
    },
}

DEFAULT_RAW_DIR = Path("data/raw/idealista_sale_prices")
DEFAULT_INDEX_PATH = Path("data/external/sale_prices_by_district.csv")

SOURCE_LABEL = "idealista (apify igolaizola/idealista-scraper)"

INDEX_FIELDS: tuple[str, ...] = (
    "city",
    "district",
    "snapshot_date",
    "n_listings",
    "median_eur_per_m2",
    "mean_eur_per_m2",
    "p25_eur_per_m2",
    "p75_eur_per_m2",
    "min_eur_per_m2",
    "max_eur_per_m2",
    "source",
)


# ── Job ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SalePriceScrapeJob:
    """One actor invocation: sale listings for a single (city, district)."""

    city: str
    district: str
    location: str               # district search string or Idealista Location ID
    max_items: int = 500
    property_type: str = "homes"
    country: str = "es"
    extra_input: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.max_items < 0:
            raise ValueError(f"max_items must be >= 0, got {self.max_items}")
        if not self.district:
            raise ValueError("district slug is required")

    def actor_input(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "operation": "sale",
            "propertyType": self.property_type,
            "country": self.country,
            "location": self.location,
            "maxItems": self.max_items,
            "fetchDetails": False,
            "fetchStats": False,
            "proxyConfiguration": {
                "useApifyProxy": True,
                "apifyProxyGroups": ["RESIDENTIAL"],
            },
        }
        payload.update(self.extra_input)
        return payload


# ── Scraper ──────────────────────────────────────────────────────────────────

class SalePriceScraper:
    """Drive the Idealista actor per district and aggregate to a €/m² index.

    The Apify client is created lazily — pass ``client=`` to inject a fake in
    tests, or leave it for the constructor to read ``APIFY_API_TOKEN`` from the
    environment when a job is first run. Aggregation and CSV writing never
    touch the client, so ``--aggregate-only`` runs work with no token.
    """

    def __init__(
        self,
        token: str | None = None,
        client: Any = None,
        raw_output_dir: Path | str = DEFAULT_RAW_DIR,
        index_path: Path | str = DEFAULT_INDEX_PATH,
        snapshot_date: str | None = None,
    ):
        self._token = token
        self._client = client
        self.raw_output_dir = Path(raw_output_dir)
        self.index_path = Path(index_path)
        self.snapshot_date = snapshot_date or date.today().isoformat()

    # ─ Apify client (lazy) ───────────────────────────────────────────────────

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = self._make_client()
        return self._client

    def _make_client(self) -> Any:
        token = self._token or os.environ.get("APIFY_API_TOKEN")
        if not token:
            raise RuntimeError(
                "APIFY_API_TOKEN env var is not set. Create a token at "
                "https://console.apify.com/account/integrations and add it "
                "to your .env file."
            )
        from apify_client import ApifyClient  # local import keeps the dep optional
        return ApifyClient(token)

    # ─ public API ────────────────────────────────────────────────────────────

    def run_job(self, job: SalePriceScrapeJob) -> Path:
        """Run one scrape job; returns the JSONL path."""
        out_path = self.output_path(job)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Apify run starting: actor=%s city=%s district=%s location=%r maxItems=%d",
            ACTOR_ID, job.city, job.district, job.location, job.max_items,
        )
        try:
            run = self.client.actor(ACTOR_ID).call(run_input=job.actor_input())
        except Exception:
            logger.exception(
                "Apify actor call failed: city=%s district=%s",
                job.city, job.district,
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
                "Actor run finished with status=%s for city=%s district=%s — "
                "writing whatever made it into the dataset.",
                status, job.city, job.district,
            )

        n_items = self._write_dataset(dataset_id, out_path, job)
        logger.info("Wrote %d listings to %s (status=%s)",
                    n_items, out_path, status or "unknown")
        return out_path

    def run_all(
        self, jobs: Iterable[SalePriceScrapeJob]
    ) -> dict[tuple[str, str], Path]:
        """Run every job; failed jobs are logged and skipped."""
        results: dict[tuple[str, str], Path] = {}
        for job in jobs:
            try:
                results[(job.city, job.district)] = self.run_job(job)
            except Exception:
                logger.error(
                    "Job failed (skipping): city=%s district=%s",
                    job.city, job.district,
                )
        return results

    def output_path(self, job: SalePriceScrapeJob) -> Path:
        return (
            self.raw_output_dir
            / job.city
            / f"{job.district}_{self.snapshot_date}.jsonl"
        )

    # ─ aggregation ───────────────────────────────────────────────────────────

    def aggregate_index(
        self, jsonl_paths: Iterable[Path] | None = None
    ) -> list[dict[str, Any]]:
        """Aggregate scraped JSONL files into district-level €/m² rows."""
        if jsonl_paths is None:
            jsonl_paths = sorted(self.raw_output_dir.rglob("*.jsonl"))
        rows: list[dict[str, Any]] = []
        for path in jsonl_paths:
            city = path.parent.name
            district = path.stem.rsplit("_", 1)[0]
            agg = _aggregate_records(
                _iter_jsonl(path),
                city=city,
                district=district,
                snapshot_date=self.snapshot_date,
            )
            if agg is not None:
                rows.append(agg)
        return rows

    def write_index_csv(
        self,
        rows: list[dict[str, Any]] | None = None,
        path: Path | None = None,
    ) -> Path:
        """Write the district-level €/m² index to CSV (date in the filename)."""
        if rows is None:
            rows = self.aggregate_index()
        path = path or self._dated_index_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_csv(path, rows)
        logger.info("Wrote €/m² district index: %s (%d rows)", path, len(rows))
        return path

    def _dated_index_path(self) -> Path:
        stem = self.index_path.stem
        suffix = self.index_path.suffix or ".csv"
        return self.index_path.with_name(f"{stem}_{self.snapshot_date}{suffix}")

    # ─ internals ─────────────────────────────────────────────────────────────

    def _write_dataset(
        self, dataset_id: str, out_path: Path, job: SalePriceScrapeJob
    ) -> int:
        dataset = self.client.dataset(dataset_id)
        n = 0
        with out_path.open("w", encoding="utf-8") as fh:
            for item in dataset.iterate_items():
                item.setdefault("_city", job.city)
                item.setdefault("_district", job.district)
                item.setdefault("_operation", "sale")
                item.setdefault("_scraped_at", self.snapshot_date)
                fh.write(json.dumps(item, ensure_ascii=False) + "\n")
                n += 1
        return n


# ── Aggregation helpers ──────────────────────────────────────────────────────

def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _extract_price_per_m2(record: Mapping[str, Any]) -> float | None:
    """€/m² from a raw record. Falls back to price/size when missing."""
    val: Any = record.get("pricePerM2") or record.get("priceByArea")
    if val in (None, "", 0):
        price = record.get("price")
        size = record.get("size") or record.get("area")
        try:
            if price and size and float(size) > 0:
                val = float(price) / float(size)
        except (TypeError, ValueError):
            val = None
    try:
        v = float(val) if val is not None else None
    except (TypeError, ValueError):
        return None
    if v is None or v <= 0:
        return None
    return v


def _percentile(sorted_values: Sequence[float], pct: float) -> float:
    """Linear-interpolated percentile on a pre-sorted sequence."""
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    k = (len(sorted_values) - 1) * pct
    lo = int(k)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = k - lo
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac


def _aggregate_records(
    records: Iterable[Mapping[str, Any]],
    *,
    city: str,
    district: str,
    snapshot_date: str,
) -> dict[str, Any] | None:
    """Reduce a list of listings to a single district-level €/m² row."""
    prices: list[float] = []
    for r in records:
        p = _extract_price_per_m2(r)
        if p is not None:
            prices.append(p)
    if not prices:
        logger.warning("No usable €/m² values for %s/%s", city, district)
        return None
    prices.sort()
    return {
        "city": city,
        "district": district,
        "snapshot_date": snapshot_date,
        "n_listings": len(prices),
        "median_eur_per_m2": round(statistics.median(prices), 2),
        "mean_eur_per_m2": round(statistics.fmean(prices), 2),
        "p25_eur_per_m2": round(_percentile(prices, 0.25), 2),
        "p75_eur_per_m2": round(_percentile(prices, 0.75), 2),
        "min_eur_per_m2": round(min(prices), 2),
        "max_eur_per_m2": round(max(prices), 2),
        "source": SOURCE_LABEL,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(INDEX_FIELDS))
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in INDEX_FIELDS})


# ── Job factory ──────────────────────────────────────────────────────────────

def default_jobs(
    cities: Sequence[str] = tuple(CITY_DISTRICTS),
    districts: Mapping[str, Sequence[str]] | None = None,
    max_items: int = 500,
    location_overrides: Mapping[str, Mapping[str, str]] | None = None,
) -> list[SalePriceScrapeJob]:
    """Build one job per (city, district) for the standard grid.

    Pass ``districts={"madrid": ["centro"]}`` to narrow the run. Pass
    ``location_overrides={"madrid": {"centro": "<Location ID>"}}`` to swap
    a search string for an Idealista Location ID where precision matters.
    """
    unknown_cities = [c for c in cities if c not in CITY_DISTRICTS]
    if unknown_cities:
        raise ValueError(
            f"Unknown cities: {unknown_cities}. Known: {list(CITY_DISTRICTS)}"
        )
    overrides = location_overrides or {}
    jobs: list[SalePriceScrapeJob] = []
    for city in cities:
        catalog = CITY_DISTRICTS[city]
        if districts is not None and city in districts:
            requested = list(districts[city])
            unknown = [d for d in requested if d not in catalog]
            if unknown:
                raise ValueError(
                    f"Unknown {city} districts: {unknown}. "
                    f"Known: {list(catalog)}"
                )
            slugs = requested
        else:
            slugs = list(catalog)
        city_overrides = overrides.get(city, {})
        for slug in slugs:
            location = city_overrides.get(slug, catalog[slug])
            jobs.append(
                SalePriceScrapeJob(
                    city=city,
                    district=slug,
                    location=location,
                    max_items=max_items,
                )
            )
    return jobs


# ── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scrape Idealista sale listings per district for Madrid, "
            "Barcelona, and Málaga via the igolaizola/idealista-scraper "
            "Apify actor, then aggregate to a district-level €/m² CSV. "
            "Reads APIFY_API_TOKEN from the environment."
        ),
    )
    parser.add_argument(
        "--cities",
        default=",".join(CITY_DISTRICTS),
        help=f"Comma-separated cities. Default and known: {','.join(CITY_DISTRICTS)}.",
    )
    parser.add_argument(
        "--districts",
        default=None,
        help=(
            "Optional comma-separated district slugs to limit the run. "
            "Requires --cities to name a single city. Example: "
            "`--cities malaga --districts centro,este`."
        ),
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=500,
        help=(
            "Cap per (city, district) job. 500 is plenty to estimate a "
            "stable median €/m² without burning Apify budget. Default: 500."
        ),
    )
    parser.add_argument(
        "--raw-output-dir",
        default=str(DEFAULT_RAW_DIR),
        help=f"Where to write raw JSONL per district. Default: {DEFAULT_RAW_DIR}.",
    )
    parser.add_argument(
        "--index-path",
        default=str(DEFAULT_INDEX_PATH),
        help=(
            f"Output CSV path; the snapshot date is appended before the "
            f"extension. Default: {DEFAULT_INDEX_PATH}."
        ),
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Skip scraping; just aggregate already-downloaded JSONL files.",
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

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        logger.debug("python-dotenv not installed; relying on shell environment")

    cities = [c.strip().lower() for c in args.cities.split(",") if c.strip()]
    districts: dict[str, list[str]] | None = None
    if args.districts:
        if len(cities) != 1:
            logger.error("--districts requires exactly one --cities value")
            return 2
        slugs = [d.strip().lower() for d in args.districts.split(",") if d.strip()]
        districts = {cities[0]: slugs}

    try:
        jobs = default_jobs(
            cities=cities,
            districts=districts,
            max_items=args.max_items,
        )
    except ValueError as e:
        logger.error(str(e))
        return 2

    scraper = SalePriceScraper(
        raw_output_dir=args.raw_output_dir,
        index_path=args.index_path,
    )

    if not args.aggregate_only:
        try:
            results = scraper.run_all(jobs)
        except RuntimeError as e:
            logger.error(str(e))
            return 2
        logger.info("Completed %d/%d scrape jobs", len(results), len(jobs))

    scraper.write_index_csv()
    return 0


if __name__ == "__main__":
    sys.exit(main())
