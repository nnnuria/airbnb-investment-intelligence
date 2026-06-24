"""Demo-safe cache for coordinator chat answers.

Demo rule: never call a live LLM during the KPMG demo. This module backs the
coordinator's ``run_chat`` so that:

* **dev** (``DEMO_MODE`` unset): live answers are computed and *written through*
  to the cache, building the demo set as you use the app.
* **demo** (``DEMO_MODE=1``): answers are served from the cache only; on a miss
  the coordinator drops to its deterministic (LLM-free) path — so no network /
  Gemini call ever happens mid-presentation.

The cache key is the normalised message + a signature of the property context
(which deterministically fixes the scenario), so the same question about the
same property is stable across runs.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping, Optional

ROOT = Path(__file__).resolve().parents[3]
CACHE_PATH = Path(
    os.getenv("CHAT_CACHE_PATH", str(ROOT / "Data" / "processed" / "chat_cache.json"))
)

# Property fields that define "the same property" for caching purposes.
_SIG_FIELDS = (
    "city", "district", "size_m2", "bedrooms", "bathrooms", "accommodates",
    "room_type", "has_ac", "has_balcony", "has_elevator", "has_pool",
    "has_parking", "has_workspace",
)

_cache: Optional[dict[str, Any]] = None


def demo_mode() -> bool:
    return os.getenv("DEMO_MODE", "").strip().lower() in ("1", "true", "yes", "on")


def _norm_msg(message: str) -> str:
    return re.sub(r"\s+", " ", (message or "").strip().lower())


def _property_sig(property: Optional[Mapping[str, Any]]) -> str:
    if not property:
        return "none"
    parts = [f"{k}={property.get(k)}" for k in _SIG_FIELDS]
    return "|".join(parts)


def cache_key(message: str, property: Optional[Mapping[str, Any]] = None) -> str:
    raw = f"{_norm_msg(message)}::{_property_sig(property)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _load() -> dict[str, Any]:
    global _cache
    if _cache is None:
        try:
            _cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            _cache = {}
    return _cache


def get(key: str) -> Optional[dict[str, Any]]:
    return _load().get(key)


def set(key: str, payload: Mapping[str, Any]) -> None:
    cache = _load()
    cache[key] = dict(payload)
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(
        json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def reload() -> None:
    """Drop the in-memory copy (e.g. after build_demo_cache writes a new file)."""
    global _cache
    _cache = None


__all__ = ["demo_mode", "cache_key", "get", "set", "reload", "CACHE_PATH"]
