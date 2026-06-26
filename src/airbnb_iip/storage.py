"""Local JSON persistence for saved property analyses.

Relocated here from ``app/components/storage.py`` so both the FastAPI service
(``api/routers/saved.py``) and the Streamlit app can share one implementation
without the API depending on the (soon-retired) Streamlit package. The old
location now re-exports from here for back-compat.

Schema (one record):

    {
      "id": "uuid",
      "saved_at": "ISO timestamp",
      "nickname": str | None,
      "property": { ... Property fields ... },
      "scenario": { ... Scenario fields ... }
    }

The store file defaults to ``app/data/saved_properties.json`` (preserving the
existing records); override with the ``SAVED_STORE_PATH`` env var (used by tests
to point at a tmp file).
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STORE = ROOT / "app" / "data" / "saved_properties.json"


def _resolve(path: Path | str | None) -> Path:
    """Pick the store path. Resolved at call-time so tests can point
    ``SAVED_STORE_PATH`` at a tmp file via monkeypatch after import."""
    if path is not None:
        return Path(path)
    env = os.environ.get("SAVED_STORE_PATH")
    return Path(env) if env else DEFAULT_STORE


def _serialise(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _serialise(v) for k, v in asdict(obj).items()}
    if isinstance(obj, (list, tuple)):
        return [_serialise(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _serialise(v) for k, v in obj.items()}
    return obj


def load_all(path: Path | str | None = None) -> list[dict[str, Any]]:
    p = _resolve(path)
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def save_record(
    property_obj: Any,
    scenario_obj: Any,
    *,
    nickname: str | None = None,
    path: Path | str | None = None,
) -> dict[str, Any]:
    """Append one record. Returns the record (with a generated id)."""
    p = _resolve(path)
    records = load_all(p)
    record = {
        "id": str(uuid.uuid4()),
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "nickname": nickname,
        "property": _serialise(property_obj),
        "scenario": _serialise(scenario_obj),
    }
    records.append(record)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    return record


def delete_record(record_id: str, path: Path | str | None = None) -> bool:
    p = _resolve(path)
    records = load_all(p)
    new = [r for r in records if r.get("id") != record_id]
    if len(new) == len(records):
        return False
    p.write_text(json.dumps(new, indent=2, ensure_ascii=False), encoding="utf-8")
    return True


def clear_all(path: Path | str | None = None) -> None:
    p = _resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("[]", encoding="utf-8")
