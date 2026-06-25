# Single-container demo image: FastAPI backend + Streamlit UI in one container.
# Works on Hugging Face Spaces (Docker SDK, port 7860) and Render (uses $PORT).
FROM python:3.11-slim

# libgomp1 is required by lightgbm / xgboost at import time.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/app/hf_cache \
    HUGGINGFACE_HUB_CACHE=/app/hf_cache \
    API_BASE_URL=http://127.0.0.1:8000

WORKDIR /app

# Dependency layer (cached unless requirements/pyproject change).
COPY requirements.txt pyproject.toml ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Application code + install the airbnb_iip package (src layout).
COPY . .
RUN pip install -e .

# Bake the e5 embedding model into the image so the first regulatory query isn't
# slow. Remove this line to shrink the image / speed the build, at the cost of a
# ~30s cold download on the first request.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-small')"

EXPOSE 7860
CMD ["bash", "start.sh"]
