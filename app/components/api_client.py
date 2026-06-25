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
from types import SimpleNamespace

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


def get_scenario(
    prop: Property,
    *,
    managed: bool = True,
    ibi_eur: float | None = None,
    cadastral_value: float | None = None,
    basuras_eur: float | None = None,
    setup_cost_eur: float = 1500.0,
    noi_growth_rate: float = 0.025,
    property_appreciation_rate: float = 0.03,
    include_income_tax: bool = True,
    purchase_price: float | None = None,
    base_url: str | None = None,
) -> Scenario:
    """POST a property to ``/scenario`` and return the full Scenario.

    ``managed`` selects professional management (``True``) vs self-managed
    (``False``, which drops the 20% management fee from the decision).
    The remaining keyword arguments are financial assumption overrides forwarded
    to ``compute_scenario``; see ``api/schemas.py::ScenarioRequest`` for details.

    Raises :class:`APIError` (with a friendly message) if the API is down or
    responds with an error, so the caller can surface it cleanly.
    """
    client = httpx.Client(base_url=base_url, timeout=_TIMEOUT) if base_url else _client()
    payload = {
        **asdict(prop),
        "managed": managed,
        "setup_cost_eur": setup_cost_eur,
        "noi_growth_rate": noi_growth_rate,
        "property_appreciation_rate": property_appreciation_rate,
        "include_income_tax": include_income_tax,
    }
    if ibi_eur is not None:
        payload["ibi_eur"] = ibi_eur
    if cadastral_value is not None:
        payload["cadastral_value"] = cadastral_value
    if basuras_eur is not None:
        payload["basuras_eur"] = basuras_eur
    if purchase_price is not None:
        payload["purchase_price"] = purchase_price
    try:
        resp = client.post("/scenario", json=payload)
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


def market_report(
    property,
    scenario,
    *,
    use_llm: bool = True,
    base_url: str | None = None,
) -> dict:
    """Call ``/market_report`` and return the structured market-analysis report.

    ``property`` / ``scenario`` are the active analysis the UI holds in session
    (dataclasses or dicts). The report reuses ``scenario`` as-is and adds peer
    positioning — see ``api/schemas.py::MarketReportResponse`` for the shape.
    Raises :class:`APIError` if the API is unreachable.
    """
    client = httpx.Client(base_url=base_url, timeout=_TIMEOUT) if base_url else _client()
    try:
        resp = client.post(
            "/market_report",
            json={
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
            f"The market-report service returned an error ({exc.response.status_code})."
        ) from exc
    except httpx.HTTPError as exc:
        raise APIError(f"Market-report request failed: {exc}") from exc
    finally:
        if base_url:
            client.close()

    return resp.json()


def optimise(property, *, projected_annual_nights=None, base_url: str | None = None):
    """Call ``/optimise`` and return a plan mirroring ``OptimisationPlan``.

    The result is a namespace with ``.recommendations`` (each a namespace with
    ``.name``, ``.annual_uplift_eur``, ``.investment_eur``, ``.payback_months``,
    ``.confidence``, ``.method``, ``.lift``, ``.rationale``) plus the peer-gap
    fields, so the Optimisation page reads it exactly like the in-process plan.
    Raises :class:`APIError` if the API is unreachable.
    """
    client = httpx.Client(base_url=base_url, timeout=_TIMEOUT) if base_url else _client()
    payload = dict(_as_dict(property) or {})
    if projected_annual_nights is not None:
        payload["projected_annual_nights"] = projected_annual_nights
    try:
        resp = client.post("/optimise", json=payload)
        resp.raise_for_status()
    except httpx.ConnectError as exc:
        raise APIError(
            "Can't reach the analysis API. Start it with "
            "`uvicorn api.main:app` and try again."
        ) from exc
    except httpx.HTTPStatusError as exc:
        raise APIError(
            f"The optimisation service returned an error ({exc.response.status_code})."
        ) from exc
    except httpx.HTTPError as exc:
        raise APIError(f"Optimisation request failed: {exc}") from exc
    finally:
        if base_url:
            client.close()

    data = resp.json()
    recommendations = [
        SimpleNamespace(
            name=a["action"],
            annual_uplift_eur=a["annual_uplift_eur"],
            investment_eur=a["investment_eur"],
            payback_months=a["payback_months"],
            confidence=a["confidence"],
            method=a["method"],
            lift=a.get("lift"),
            rationale=a["rationale"],
        )
        for a in data.get("actions", [])
    ]
    return SimpleNamespace(
        recommendations=recommendations,
        peer_n=data.get("peer_n", 0),
        peer_median_revenue_eur=data.get("peer_median_revenue_eur", 0.0),
        peer_target_revenue_eur=data.get("peer_target_revenue_eur", 0.0),
        gap_to_top_quartile_eur=data.get("gap_to_top_quartile_eur", 0.0),
        projected_annual_nights=data.get("projected_annual_nights", 0),
        disclaimer=data.get("disclaimer", ""),
    )


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


__all__ = ["get_scenario", "chat", "market_report", "optimise", "api_healthy", "APIError", "DEFAULT_API_BASE_URL"]
