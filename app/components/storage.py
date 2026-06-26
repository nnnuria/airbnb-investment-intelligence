"""Back-compat shim. The implementation moved to ``airbnb_iip.storage`` so the
FastAPI service and the Streamlit app share one persistence layer. Existing
imports (``from app.components.storage import save_record`` …) keep working.
"""

from __future__ import annotations

from airbnb_iip.storage import (  # noqa: F401
    DEFAULT_STORE,
    clear_all,
    delete_record,
    load_all,
    save_record,
)

__all__ = ["DEFAULT_STORE", "clear_all", "delete_record", "load_all", "save_record"]
