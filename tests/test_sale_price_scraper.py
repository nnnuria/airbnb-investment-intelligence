"""Tests for the district-level Idealista sale price scraper.

The Apify client is stubbed out so no network calls happen and no
APIFY_API_TOKEN is needed.
"""
import csv
import json
from pathlib import Path

import pytest

from airbnb_iip.data.scrapers.sale_prices import (
    CITY_DISTRICTS,
    INDEX_FIELDS,
    SOURCE_LABEL,
    SalePriceScrapeJob,
    SalePriceScraper,
    _aggregate_records,
    _extract_price_per_m2,
    _percentile,
    default_jobs,
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
        self._actor = _FakeActor(self.items, run_status)
        self.last_actor_id = None

    def actor(self, actor_id):
        self.last_actor_id = actor_id
        return self._actor

    def dataset(self, dataset_id):
        return _FakeDataset(self.items)


# ── SalePriceScrapeJob ────────────────────────────────────────────────────────

def test_job_actor_input_includes_required_fields():
    job = SalePriceScrapeJob(
        city="madrid", district="centro", location="Centro, Madrid", max_items=100,
    )
    payload = job.actor_input()
    assert payload["operation"] == "sale"
    assert payload["propertyType"] == "homes"
    assert payload["country"] == "es"
    assert payload["location"] == "Centro, Madrid"
    assert payload["maxItems"] == 100
    assert payload["fetchDetails"] is False
    assert payload["proxyConfiguration"]["useApifyProxy"] is True


def test_job_rejects_negative_max_items():
    with pytest.raises(ValueError):
        SalePriceScrapeJob(
            city="madrid", district="centro", location="x", max_items=-1,
        )


def test_job_rejects_empty_district():
    with pytest.raises(ValueError):
        SalePriceScrapeJob(city="madrid", district="", location="x")


def test_job_extra_input_overrides_defaults():
    job = SalePriceScrapeJob(
        city="madrid",
        district="centro",
        location="Centro, Madrid",
        extra_input={"sortBy": "lowestPrice", "minPrice": "200000"},
    )
    payload = job.actor_input()
    assert payload["sortBy"] == "lowestPrice"
    assert payload["minPrice"] == "200000"


# ── default_jobs ──────────────────────────────────────────────────────────────

def test_default_jobs_covers_all_cities_and_districts():
    jobs = default_jobs()
    expected = sum(len(d) for d in CITY_DISTRICTS.values())
    assert len(jobs) == expected
    pairs = {(j.city, j.district) for j in jobs}
    expected_pairs = {
        (c, d) for c, ds in CITY_DISTRICTS.items() for d in ds
    }
    assert pairs == expected_pairs


def test_default_jobs_rejects_unknown_city():
    with pytest.raises(ValueError, match="Unknown cities"):
        default_jobs(cities=["valencia"])


def test_default_jobs_rejects_unknown_district():
    with pytest.raises(ValueError, match="Unknown madrid districts"):
        default_jobs(cities=["madrid"], districts={"madrid": ["not-a-district"]})


def test_default_jobs_district_filter_limits_jobs():
    jobs = default_jobs(
        cities=["malaga"],
        districts={"malaga": ["centro", "este"]},
    )
    assert {j.district for j in jobs} == {"centro", "este"}
    assert {j.city for j in jobs} == {"malaga"}


def test_default_jobs_location_overrides_swap_in_location_ids():
    jobs = default_jobs(
        cities=["madrid"],
        districts={"madrid": ["centro"]},
        location_overrides={"madrid": {"centro": "0-EU-ES-28-07-001-079-XYZ"}},
    )
    assert jobs[0].location == "0-EU-ES-28-07-001-079-XYZ"


# ── SalePriceScraper.run_job ──────────────────────────────────────────────────

@pytest.fixture
def items():
    return [
        {"propertyCode": "1", "price": 300000, "size": 60, "pricePerM2": 5000},
        {"propertyCode": "2", "price": 450000, "size": 90, "pricePerM2": 5000},
    ]


def test_run_job_writes_jsonl_and_tags_records(tmp_path, items):
    client = FakeApifyClient(items=items)
    scraper = SalePriceScraper(
        client=client,
        raw_output_dir=tmp_path / "raw",
        index_path=tmp_path / "ext" / "sale_prices.csv",
        snapshot_date="2026-06-15",
    )
    job = SalePriceScrapeJob(
        city="madrid", district="centro",
        location="Centro, Madrid", max_items=50,
    )

    out_path = scraper.run_job(job)

    assert out_path == tmp_path / "raw" / "madrid" / "centro_2026-06-15.jsonl"
    assert out_path.exists()

    lines = out_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    record = json.loads(lines[0])
    assert record["propertyCode"] == "1"
    assert record["_city"] == "madrid"
    assert record["_district"] == "centro"
    assert record["_operation"] == "sale"
    assert record["_scraped_at"] == "2026-06-15"

    assert client.last_actor_id == "igolaizola/idealista-scraper"
    assert client._actor.last_input["operation"] == "sale"
    assert client._actor.last_input["location"] == "Centro, Madrid"
    assert client._actor.last_input["maxItems"] == 50


def test_run_job_raises_when_no_dataset_id(tmp_path):
    client = FakeApifyClient()
    client._actor.call = lambda run_input: {"status": "FAILED"}  # type: ignore[method-assign]
    scraper = SalePriceScraper(
        client=client,
        raw_output_dir=tmp_path / "raw",
        index_path=tmp_path / "ext" / "x.csv",
    )
    job = SalePriceScrapeJob(city="madrid", district="centro", location="x")
    with pytest.raises(RuntimeError, match="defaultDatasetId"):
        scraper.run_job(job)


def test_run_all_continues_after_a_failed_job(tmp_path, items, caplog):
    client = FakeApifyClient(items=items)

    calls = {"n": 0}
    original_call = client._actor.call

    def flaky_call(run_input):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("simulated transient failure")
        return original_call(run_input)

    client._actor.call = flaky_call  # type: ignore[method-assign]

    scraper = SalePriceScraper(
        client=client,
        raw_output_dir=tmp_path / "raw",
        index_path=tmp_path / "ext" / "x.csv",
        snapshot_date="2026-06-15",
    )
    jobs = [
        SalePriceScrapeJob(city="madrid", district="centro", location="Centro"),
        SalePriceScrapeJob(city="madrid", district="retiro", location="Retiro"),
    ]

    with caplog.at_level("ERROR"):
        results = scraper.run_all(jobs)

    assert ("madrid", "retiro") in results
    assert ("madrid", "centro") not in results


def test_missing_token_raises_only_when_client_used(monkeypatch, tmp_path):
    monkeypatch.delenv("APIFY_API_TOKEN", raising=False)
    scraper = SalePriceScraper(
        raw_output_dir=tmp_path, index_path=tmp_path / "x.csv",
    )
    # No network access yet — no error.
    assert scraper.snapshot_date is not None
    with pytest.raises(RuntimeError, match="APIFY_API_TOKEN"):
        _ = scraper.client


# ── Aggregation primitives ────────────────────────────────────────────────────

def test_extract_price_per_m2_prefers_explicit_field():
    assert _extract_price_per_m2(
        {"pricePerM2": 5000, "price": 300000, "size": 30}
    ) == 5000.0


def test_extract_price_per_m2_falls_back_to_price_and_size():
    assert _extract_price_per_m2(
        {"price": 300000, "size": 60}
    ) == 5000.0


def test_extract_price_per_m2_handles_missing_or_zero():
    assert _extract_price_per_m2({}) is None
    assert _extract_price_per_m2({"price": 0, "size": 60}) is None
    assert _extract_price_per_m2({"price": 100000, "size": 0}) is None
    assert _extract_price_per_m2({"price": "garbage", "size": 60}) is None


def test_percentile_known_values():
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert _percentile(xs, 0.0) == 1.0
    assert _percentile(xs, 0.5) == 3.0
    assert _percentile(xs, 1.0) == 5.0
    assert _percentile(xs, 0.25) == 2.0
    assert _percentile(xs, 0.75) == 4.0


def test_aggregate_records_returns_none_for_empty_input():
    assert _aggregate_records(
        [], city="madrid", district="centro", snapshot_date="2026-06-15",
    ) is None


def test_aggregate_records_computes_expected_stats():
    records = [
        {"pricePerM2": 4000},
        {"pricePerM2": 5000},
        {"pricePerM2": 6000},
        {"price": 350000, "size": 50},   # falls back to 7000
    ]
    agg = _aggregate_records(
        records, city="madrid", district="centro", snapshot_date="2026-06-15",
    )
    assert agg["city"] == "madrid"
    assert agg["district"] == "centro"
    assert agg["snapshot_date"] == "2026-06-15"
    assert agg["n_listings"] == 4
    assert agg["median_eur_per_m2"] == 5500.0
    assert agg["min_eur_per_m2"] == 4000.0
    assert agg["max_eur_per_m2"] == 7000.0
    assert agg["source"] == SOURCE_LABEL


# ── Index aggregation + CSV ───────────────────────────────────────────────────

def _write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


def test_aggregate_index_reads_jsonl_tree(tmp_path):
    scraper = SalePriceScraper(
        client=object(),
        raw_output_dir=tmp_path,
        index_path=tmp_path / "out.csv",
        snapshot_date="2026-06-15",
    )
    _write_jsonl(
        tmp_path / "madrid" / "centro_2026-06-15.jsonl",
        [{"pricePerM2": 5000}, {"pricePerM2": 6000}],
    )
    _write_jsonl(
        tmp_path / "barcelona" / "eixample_2026-06-15.jsonl",
        [{"pricePerM2": 7000}, {"pricePerM2": 8000}, {"pricePerM2": 9000}],
    )

    rows = scraper.aggregate_index()
    by_key = {(r["city"], r["district"]): r for r in rows}
    assert set(by_key) == {("madrid", "centro"), ("barcelona", "eixample")}
    assert by_key[("madrid", "centro")]["n_listings"] == 2
    assert by_key[("madrid", "centro")]["median_eur_per_m2"] == 5500.0
    assert by_key[("barcelona", "eixample")]["median_eur_per_m2"] == 8000.0


def test_write_index_csv_uses_dated_path_and_canonical_header(tmp_path):
    scraper = SalePriceScraper(
        client=object(),
        raw_output_dir=tmp_path / "raw",
        index_path=tmp_path / "ext" / "sale_prices_by_district.csv",
        snapshot_date="2026-06-15",
    )
    _write_jsonl(
        tmp_path / "raw" / "malaga" / "centro_2026-06-15.jsonl",
        [{"pricePerM2": 3500}, {"pricePerM2": 4500}],
    )

    path = scraper.write_index_csv()

    assert path == tmp_path / "ext" / "sale_prices_by_district_2026-06-15.csv"
    assert path.exists()

    with path.open() as fh:
        reader = csv.DictReader(fh)
        assert tuple(reader.fieldnames or ()) == INDEX_FIELDS
        rows = list(reader)
    assert len(rows) == 1
    assert rows[0]["city"] == "malaga"
    assert rows[0]["district"] == "centro"
    assert rows[0]["snapshot_date"] == "2026-06-15"
    assert float(rows[0]["median_eur_per_m2"]) == 4000.0
    assert rows[0]["source"] == SOURCE_LABEL


def test_write_index_csv_writes_header_only_when_no_data(tmp_path):
    scraper = SalePriceScraper(
        client=object(),
        raw_output_dir=tmp_path / "raw",
        index_path=tmp_path / "ext" / "sale_prices.csv",
        snapshot_date="2026-06-15",
    )
    (tmp_path / "raw").mkdir(parents=True, exist_ok=True)
    path = scraper.write_index_csv()
    text = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(text) == 1
    assert text[0].split(",") == list(INDEX_FIELDS)
