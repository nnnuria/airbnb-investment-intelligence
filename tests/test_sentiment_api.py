"""Tests for GET /sentiment.

Skips gracefully if the precomputed artifacts aren't present (they live in
Data/processed/ and are produced by scripts/run_sentiment.py).
"""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from api.main import app  # noqa: E402
from airbnb_iip.sentiment import DATA  # noqa: E402

_HAVE_DATA = (DATA / "sentiment_summary.csv").exists()
needs_data = pytest.mark.skipif(not _HAVE_DATA, reason="sentiment artifacts not on disk")


@needs_data
def test_sentiment_madrid_payload():
    with TestClient(app) as client:
        resp = client.get("/sentiment", params={"city": "madrid"})
    assert resp.status_code == 200
    data = resp.json()

    assert data["city"] == "madrid"
    assert data["reviews_scored"] > 0
    # donut shares present and roughly normalised
    assert set(data["label_split"]) >= {"positive", "negative"}
    assert abs(sum(data["label_split"].values()) - 1.0) < 0.05
    assert data["lang_split"]  # language breakdown present
    assert data["examples"]["positive"] and data["examples"]["negative"]
    assert len(data["listing_histogram"]) == 10
    assert sum(b["count"] for b in data["listing_histogram"]) == data["n_listings"]


@needs_data
def test_sentiment_city_is_case_insensitive():
    with TestClient(app) as client:
        assert client.get("/sentiment", params={"city": "Barcelona"}).status_code == 200


def test_sentiment_unknown_city_404():
    with TestClient(app) as client:
        resp = client.get("/sentiment", params={"city": "lisbon"})
    assert resp.status_code == 404
