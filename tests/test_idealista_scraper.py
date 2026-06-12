"""Tests for the Idealista scraper agent.

These tests stub out the Apify client so no network calls happen and no
APIFY_API_TOKEN is needed.
"""
import json
from pathlib import Path

import pytest

from airbnb_iip.data.scrapers.idealista import (
    CITY_LOCATIONS,
    OPERATIONS,
    IdealistaScraper,
    ScrapeJob,
    default_jobs,
    load_jsonl_as_dataframe,
    normalize_listing,
)


# ── Fake Apify client ─────────────────────────────────────────────────────────

class _FakeDataset:
    def __init__(self, items):
        self._items = items

    def iterate_items(self):
        yield from self._items


class _FakeActor:
    def __init__(self, items, run_status="SUCCEEDED"):
        self._items = items
        self._run_status = run_status
        self.last_input = None

    def call(self, run_input):
        self.last_input = run_input
        return {"defaultDatasetId": "fake-dataset-id", "status": self._run_status}


class FakeApifyClient:
    def __init__(self, items=None, run_status="SUCCEEDED"):
        self.items = items or []
        self.run_status = run_status
        self._actor = _FakeActor(self.items, run_status)
        self.last_actor_id = None

    def actor(self, actor_id):
        self.last_actor_id = actor_id
        return self._actor

    def dataset(self, dataset_id):
        return _FakeDataset(self.items)


# ── ScrapeJob ─────────────────────────────────────────────────────────────────

def test_scrape_job_actor_input_includes_required_fields():
    job = ScrapeJob(city="madrid", operation="rent", location="Madrid", max_items=100)
    payload = job.actor_input()
    assert payload["operation"] == "rent"
    assert payload["propertyType"] == "homes"
    assert payload["country"] == "es"
    assert payload["location"] == "Madrid"
    assert payload["maxItems"] == 100
    assert payload["fetchDetails"] is False
    assert payload["fetchStats"] is False
    assert payload["proxyConfiguration"]["useApifyProxy"] is True
    assert payload["proxyConfiguration"]["apifyProxyGroups"] == ["RESIDENTIAL"]


def test_scrape_job_rejects_unknown_operation():
    with pytest.raises(ValueError):
        ScrapeJob(city="madrid", operation="lease", location="Madrid")


def test_scrape_job_rejects_negative_max_items():
    with pytest.raises(ValueError):
        ScrapeJob(city="madrid", operation="sale", location="Madrid", max_items=-1)


def test_scrape_job_extra_input_overrides_defaults():
    job = ScrapeJob(
        city="madrid",
        operation="sale",
        location="Madrid",
        extra_input={"sortBy": "lowestPrice", "minPrice": "200000"},
    )
    payload = job.actor_input()
    assert payload["sortBy"] == "lowestPrice"
    assert payload["minPrice"] == "200000"


# ── default_jobs ──────────────────────────────────────────────────────────────

def test_default_jobs_builds_grid_for_all_cities_and_operations():
    jobs = default_jobs()
    assert len(jobs) == len(CITY_LOCATIONS) * len(OPERATIONS)
    pairs = {(j.city, j.operation) for j in jobs}
    expected = {(c, o) for c in CITY_LOCATIONS for o in OPERATIONS}
    assert pairs == expected


def test_default_jobs_rejects_unknown_city():
    with pytest.raises(ValueError, match="Unknown cities"):
        default_jobs(cities=["valencia"])


def test_default_jobs_rejects_unknown_operation():
    with pytest.raises(ValueError, match="Unknown operations"):
        default_jobs(operations=["lease"])


def test_default_jobs_location_override():
    jobs = default_jobs(
        cities=["madrid"],
        operations=["sale"],
        location_overrides={"madrid": "0-EU-ES-28-07-001-079"},
    )
    assert jobs[0].location == "0-EU-ES-28-07-001-079"


# ── IdealistaScraper.run_job ──────────────────────────────────────────────────

@pytest.fixture
def items():
    return [
        {"propertyCode": "1", "price": 300000, "size": 60,
         "images": ["a.jpg", "b.jpg"]},
        {"propertyCode": "2", "price": 450000, "size": 90,
         "images": ["c.jpg"]},
    ]


def test_run_job_writes_jsonl_and_tags_records(tmp_path, items):
    client = FakeApifyClient(items=items)
    scraper = IdealistaScraper(
        client=client,
        output_dir=tmp_path,
        snapshot_date="2026-06-10",
    )
    job = ScrapeJob(city="madrid", operation="sale", location="Madrid", max_items=50)

    out_path = scraper.run_job(job)

    assert out_path == tmp_path / "madrid" / "sale_2026-06-10.jsonl"
    assert out_path.exists()

    lines = out_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    record = json.loads(lines[0])
    assert record["propertyCode"] == "1"
    assert record["_city"] == "madrid"
    assert record["_operation"] == "sale"
    assert record["_scraped_at"] == "2026-06-10"

    # Actor id and payload made it through correctly
    assert client.last_actor_id == "igolaizola/idealista-scraper"
    assert client._actor.last_input["operation"] == "sale"
    assert client._actor.last_input["location"] == "Madrid"
    assert client._actor.last_input["maxItems"] == 50


def test_run_job_raises_when_no_dataset_id(tmp_path):
    client = FakeApifyClient()
    client._actor.call = lambda run_input: {"status": "FAILED"}  # type: ignore[method-assign]
    scraper = IdealistaScraper(client=client, output_dir=tmp_path)
    job = ScrapeJob(city="madrid", operation="sale", location="Madrid")
    with pytest.raises(RuntimeError, match="defaultDatasetId"):
        scraper.run_job(job)


def test_run_all_continues_after_a_failed_job(tmp_path, items, caplog):
    client = FakeApifyClient(items=items)

    # First call raises, second call succeeds
    calls = {"n": 0}
    original_call = client._actor.call

    def flaky_call(run_input):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("simulated transient failure")
        return original_call(run_input)

    client._actor.call = flaky_call  # type: ignore[method-assign]

    scraper = IdealistaScraper(
        client=client, output_dir=tmp_path, snapshot_date="2026-06-10"
    )
    jobs = [
        ScrapeJob(city="madrid", operation="sale", location="Madrid"),
        ScrapeJob(city="madrid", operation="rent", location="Madrid"),
    ]

    with caplog.at_level("ERROR"):
        results = scraper.run_all(jobs)

    assert (("madrid", "rent") in results) and (("madrid", "sale") not in results)
    assert any("simulated transient failure" in r.message
               or "Job failed" in r.message for r in caplog.records)


def test_missing_token_raises():
    with pytest.raises(RuntimeError, match="APIFY_API_TOKEN"):
        IdealistaScraper(token=None, client=None)


# ── Normalisation ─────────────────────────────────────────────────────────────

def test_normalize_listing_maps_canonical_fields():
    raw = {
        "propertyCode": "104578321",
        "url": "https://www.idealista.com/inmueble/104578321/",
        "title": "Piso en Madrid",
        "price": 485000,
        "pricePerM2": 6500,
        "size": 75,
        "rooms": 2,
        "bathrooms": 1,
        "homeType": "flat",
        "latitude": 40.41,
        "longitude": -3.7,
        "images": ["a.jpg", "b.jpg", "c.jpg"],
        "virtualTour": "https://www.idealista.com/tour/104578321/",
        "features": ["lift", "exterior"],
        "_city": "madrid",
        "_operation": "sale",
        "_scraped_at": "2026-06-10",
    }
    norm = normalize_listing(raw)
    assert norm["listing_id"] == "104578321"
    assert norm["price_eur"] == 485000
    assert norm["price_per_m2_eur"] == 6500
    assert norm["size_m2"] == 75
    assert norm["rooms"] == 2
    assert norm["home_type"] == "flat"
    assert norm["latitude"] == 40.41
    assert norm["n_images"] == 3
    assert norm["has_virtual_tour"] is True
    assert norm["city"] == "madrid"
    assert norm["operation"] == "sale"
    assert norm["scraped_at"] == "2026-06-10"


def test_normalize_listing_missing_fields_become_none():
    norm = normalize_listing({"propertyCode": "x"})
    assert norm["listing_id"] == "x"
    assert norm["price_eur"] is None
    assert norm["latitude"] is None
    assert norm["n_images"] == 0
    assert norm["has_virtual_tour"] is False


def test_load_jsonl_as_dataframe_loads_sample():
    pd = pytest.importorskip("pandas")
    sample = Path(__file__).resolve().parents[1] / "Data" / "sample" / "idealista_sample.jsonl"
    if not sample.exists():
        pytest.skip("sample file not present")
    df = load_jsonl_as_dataframe(sample)
    assert len(df) == 3
    assert set(df["city"]) == {"madrid", "barcelona", "malaga"}
    assert set(df["operation"]) == {"sale", "rent"}
    assert df["price_eur"].notna().all()
    assert isinstance(df, pd.DataFrame)
