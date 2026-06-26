"""CRUD for saved property analyses → /saved.

Thin HTTP wrapper over ``airbnb_iip.storage`` (a local JSON store). Lets the
React frontend persist a portfolio of analyses server-side so they survive
across machines/reloads.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from airbnb_iip import storage
from api.schemas import SavedCreateRequest, SavedRecord

router = APIRouter(tags=["saved"])


@router.get("/saved", response_model=list[SavedRecord])
def list_saved() -> list[dict]:
    """All saved analyses, oldest first."""
    return storage.load_all()


@router.post("/saved", response_model=SavedRecord, status_code=201)
def create_saved(req: SavedCreateRequest) -> dict:
    """Persist a property + scenario; returns the record with a generated id."""
    return storage.save_record(req.property, req.scenario, nickname=req.nickname)


@router.delete("/saved/{record_id}")
def delete_saved(record_id: str) -> dict:
    """Delete one saved analysis by id."""
    if not storage.delete_record(record_id):
        raise HTTPException(status_code=404, detail=f"No saved analysis '{record_id}'.")
    return {"ok": True, "deleted": record_id}


@router.delete("/saved")
def clear_saved() -> dict:
    """Delete every saved analysis (the Saved page's 'Clear all')."""
    storage.clear_all()
    return {"ok": True}
