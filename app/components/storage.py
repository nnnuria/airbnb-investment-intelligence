"""Local JSON persistence for saved property analyses.

Streamlit's ``session_state`` is in-memory only — refresh and it's gone.
This module persists saved analyses to ``app/data/saved_properties.json``
so the demo survives a restart.

Schema (one record):

    {
      "id": "uuid",
      "saved_at": "ISO timestamp",
      "property": { ... Property fields ... },
      "scenario": { ... Scenario fields ... }
    }
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

DEFAULT_STORE = Path(__file__).resolve().parents[1] / "data" / "saved_properties.json"


def _serialise(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _serialise(v) for k, v in asdict(obj).items()}
    if isinstance(obj, (list, tuple)):
        return [_serialise(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _serialise(v) for k, v in obj.items()}
    return obj


def load_all(path: Path | str = DEFAULT_STORE) -> list[dict[str, Any]]:
    p = Path(path)
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
    path: Path | str = DEFAULT_STORE,
) -> dict[str, Any]:
    """Append one record. Returns the record (with a generated id)."""
    records = load_all(path)
    record = {
        "id": str(uuid.uuid4()),
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "nickname": nickname,
        "property": _serialise(property_obj),
        "scenario": _serialise(scenario_obj),
    }
    records.append(record)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    return record


def delete_record(record_id: str, path: Path | str = DEFAULT_STORE) -> bool:
    records = load_all(path)
    new = [r for r in records if r.get("id") != record_id]
    if len(new) == len(records):
        return False
    Path(path).write_text(json.dumps(new, indent=2, ensure_ascii=False), encoding="utf-8")
    return True


def clear_all(path: Path | str = DEFAULT_STORE) -> None:
    Path(path).write_text("[]", encoding="utf-8")
