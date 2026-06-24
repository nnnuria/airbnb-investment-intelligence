"""HTTP client the Streamlit UI uses to reach the FastAPI decision engine.

The app no longer imports the models/finance in-process; it POSTs the property
to ``/scenario`` and gets the full decision back. The JSON is rebuilt into the
shared ``airbnb_iip.decision.engine.Scenario`` dataclass so every page keeps
using ``scen.net_revenue_year_eur`` etc. unchanged — only the source of the
object moves from a local call to the API.

Config:
    API_BASE_URL  (env / .env)  default http://127.0.0.1:8000
"""

from __future__ import annotations

import os
from dataclasses import asdict, fields, is_dataclass
from functools import lru_cache

import httpx
from dotenv import load_dotenv

from airbnb_iip.decision.engine import Property, Scenario

load_dotenv()

DEFAULT_API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

# compute_scenario runs SHAP + two Monte-Carlo passes; give it room.
_TIMEOUT = httpx.Timeout(45.0, connect=5.0)

_SCENARIO_FIELDS = {f.name for f in fields(Scenario)}


class APIError(RuntimeError):
    """The decision API could not be reached or returned an error.

    Carries a UI-ready message; the page shows it via ``st.error``.
    """


@lru_cache(maxsize=1)
def _client() -> httpx.Client:
    return httpx.Client(base_url=DEFAULT_API_BASE_URL, timeout=_TIMEOUT)


def _to_scenario(data: dict) -> Scenario:
    """Rebuild the Scenario dataclass from the /scenario JSON response.

    cost_breakdown / feature_drivers arrive as JSON arrays; restore them to
    tuples so the object round-trips through storage.save_record unchanged.
    Unknown keys are dropped defensively so a newer API can't break the UI.
    """
    clean = {k: v for k, v in data.items() if k in _SCENARIO_FIELDS}
    clean["cost_breakdown"] = [tuple(x) for x in clean.get("cost_breakdown", [])]
    clean["feature_drivers"] = [tuple(x) for x in clean.get("feature_drivers", [])]
    return Scenario(**clean)


def get_scenario(prop: Property, *, base_url: str | None = None) -> Scenario:
    """POST a property to ``/scenario`` and return the full Scenario.

    Raises :class:`APIError` (with a friendly message) if the API is down or
    responds with an error, so the caller can surface it cleanly.
    """
    client = httpx.Client(base_url=base_url, timeout=_TIMEOUT) if base_url else _client()
    try:
        resp = client.post("/scenario", json=asdict(prop))
        resp.raise_for_status()
    except httpx.ConnectError as exc:
        raise APIError(
            "Can't reach the analysis API. Start it with "
            "`uvicorn api.main:app` and try again."
        ) from exc
    except httpx.HTTPStatusError as exc:
        raise APIError(
            f"The analysis API returned an error ({exc.response.status_code}). "
            "Check the API logs."
        ) from exc
    except httpx.HTTPError as exc:
        raise APIError(f"Analysis API request failed: {exc}") from exc
    finally:
        if base_url:
            client.close()

    return _to_scenario(resp.json())


def _as_dict(obj):
    """Dataclass (Property/Scenario) → dict; pass dicts/None through."""
    if obj is None:
        return None
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    return obj


def chat(
    message: str,
    *,
    property=None,
    scenario=None,
    use_llm: bool = True,
    base_url: str | None = None,
) -> dict:
    """Send a chat message to ``/chat`` (the LangGraph coordinator).

    ``property`` / ``scenario`` may be the dataclasses the UI holds in session
    or plain dicts. Returns ``{answer, intent, sources, meta, disclaimer}``.
    Raises :class:`APIError` if the API is unreachable.
    """
    client = httpx.Client(base_url=base_url, timeout=_TIMEOUT) if base_url else _client()
    try:
        resp = client.post(
            "/chat",
            json={
                "message": message,
                "property": _as_dict(property),
                "scenario": _as_dict(scenario),
                "use_llm": use_llm,
            },
        )
        resp.raise_for_status()
    except httpx.ConnectError as exc:
        raise APIError(
            "Can't reach the analysis API. Start it with "
            "`uvicorn api.main:app` and try again."
        ) from exc
    except httpx.HTTPStatusError as exc:
        raise APIError(
            f"The chat service returned an error ({exc.response.status_code})."
        ) from exc
    except httpx.HTTPError as exc:
        raise APIError(f"Chat request failed: {exc}") from exc
    finally:
        if base_url:
            client.close()

    return resp.json()


def api_healthy(*, base_url: str | None = None) -> bool:
    """Quick liveness check for a status indicator in the UI."""
    client = httpx.Client(base_url=base_url, timeout=httpx.Timeout(3.0)) if base_url else _client()
    try:
        return client.get("/health").status_code == 200
    except httpx.HTTPError:
        return False
    finally:
        if base_url:
            client.close()


__all__ = ["get_scenario", "chat", "api_healthy", "APIError", "DEFAULT_API_BASE_URL"]
