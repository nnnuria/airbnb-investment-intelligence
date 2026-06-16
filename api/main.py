"""FastAPI application for the Airbnb Investment Intelligence Platform.

Run locally:
    uvicorn api.main:app --reload
Docs:
    http://127.0.0.1:8000/docs

Endpoint status:
    LIVE   /predict_price, /estimate_occupancy
    STUB   /estimate_revenue, /airbnb_vs_sell, /optimise  (flagged "_stub")
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import optimise, predict, revenue
from api.schemas import HealthResponse

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm the price model once at startup so first request isn't slow."""
    try:
        from airbnb_iip.models.price import get_price_predictor

        get_price_predictor()
        app.state.price_model_loaded = True
        logger.info("Price model warmed at startup.")
    except Exception:  # pragma: no cover - degraded mode still serves stubs
        app.state.price_model_loaded = False
        logger.exception("Price model failed to load — /predict_price will error.")
    yield


app = FastAPI(
    title="Airbnb Investment Intelligence Platform API",
    version="0.1.0",
    description="Services behind the Airbnb-vs-sell decision and optimisation flows.",
    lifespan=lifespan,
)

# Open CORS for local Streamlit/agent development. Tighten before any deploy.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router)
app.include_router(revenue.router)
app.include_router(optimise.router)


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    return HealthResponse(status="ok", price_model_loaded=getattr(app.state, "price_model_loaded", False))


@app.get("/", tags=["meta"])
def root() -> dict:
    return {"service": "airbnb-iip-api", "docs": "/docs", "health": "/health"}
