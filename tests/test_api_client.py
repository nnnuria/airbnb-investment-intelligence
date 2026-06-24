"""Test the UI's API client deserialisation (app/components/api_client.py).

The client rebuilds the shared Scenario dataclass from the JSON /scenario
returns. This guards the wire round-trip: tuples (cost_breakdown,
feature_drivers) survive JSON → list → tuple so the rebuilt object equals the
original and stays compatible with storage.save_record.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

# Mirror Streamlit's runtime import path (app/ is the script dir there).
APP_DIR = Path(__file__).resolve().parents[1] / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

pytest.importorskip("httpx")

from components.api_client import _to_scenario  # noqa: E402
from airbnb_iip.decision.engine import Property, Scenario, compute_scenario  # noqa: E402


def test_to_scenario_roundtrips_through_json():
    scenario = compute_scenario(Property(
        city="madrid", district="Centro", size_m2=60, bedrooms=1,
        bathrooms=1.0, accommodates=2, room_type="Entire home/apt",
    ))
    # Simulate the wire: dataclass → JSON (tuples become lists) → client rebuild.
    wire = json.loads(json.dumps(asdict(scenario)))
    rebuilt = _to_scenario(wire)

    assert isinstance(rebuilt, Scenario)
    assert rebuilt == scenario  # tuples restored, all fields equal
    assert all(isinstance(c, tuple) for c in rebuilt.cost_breakdown)
    assert all(isinstance(d, tuple) for d in rebuilt.feature_drivers)


def test_to_scenario_ignores_unknown_keys():
    scenario = compute_scenario(Property(
        city="malaga", district="Centro", size_m2=70, bedrooms=2,
        bathrooms=1.0, accommodates=3, room_type="Entire home/apt",
    ))
    payload = asdict(scenario)
    payload["some_future_api_field"] = 123  # newer API shouldn't break the UI
    rebuilt = _to_scenario(payload)
    assert isinstance(rebuilt, Scenario)
