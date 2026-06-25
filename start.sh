#!/usr/bin/env bash
# Launch the FastAPI backend (internal) + Streamlit UI (public) in one container.
set -euo pipefail

# FastAPI backend — the Streamlit UI reaches it via API_BASE_URL (127.0.0.1:8000).
uvicorn api.main:app --host 0.0.0.0 --port 8000 &

# Streamlit UI on the public port.
#   Hugging Face Spaces => no $PORT set, defaults to 7860 (the exposed port).
#   Render / other PaaS  => $PORT is injected and used.
exec streamlit run app/Home.py \
    --server.port "${PORT:-7860}" \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false
