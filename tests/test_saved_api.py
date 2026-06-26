"""Tests for the /saved CRUD router.

Points the store at a tmp file via SAVED_STORE_PATH so the real
app/data/saved_properties.json is never touched.
"""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from api.main import app  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("SAVED_STORE_PATH", str(tmp_path / "saved.json"))
    with TestClient(app) as c:
        yield c


SAMPLE = {
    "property": {"city": "madrid", "district": "Salamanca", "bedrooms": 2},
    "scenario": {"recommendation": "airbnb", "net_revenue_year_eur": 21000},
    "nickname": "My flat",
}


def test_create_list_delete_roundtrip(client):
    assert client.get("/saved").json() == []

    created = client.post("/saved", json=SAMPLE)
    assert created.status_code == 201
    rec = created.json()
    assert rec["id"] and rec["saved_at"]
    assert rec["nickname"] == "My flat"
    assert rec["property"]["district"] == "Salamanca"

    listing = client.get("/saved").json()
    assert len(listing) == 1 and listing[0]["id"] == rec["id"]

    assert client.delete(f"/saved/{rec['id']}").json()["ok"] is True
    assert client.get("/saved").json() == []


def test_delete_missing_returns_404(client):
    assert client.delete("/saved/does-not-exist").status_code == 404


def test_clear_all(client):
    client.post("/saved", json=SAMPLE)
    client.post("/saved", json=SAMPLE)
    assert len(client.get("/saved").json()) == 2

    assert client.delete("/saved").json()["ok"] is True
    assert client.get("/saved").json() == []
