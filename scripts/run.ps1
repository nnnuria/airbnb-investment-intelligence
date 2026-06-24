# Launch the API and the Streamlit app together.
#
# The UI now consumes the FastAPI decision engine, so both must run. This
# starts uvicorn in the background, then Streamlit in the foreground; closing
# Streamlit (Ctrl+C) stops the API too.
#
# Run from the repo root, with the project conda env active:
#     conda activate airbnb-iip
#     ./scripts/run.ps1
#
# Demo day — serve chat from the frozen cache, never a live LLM:
#     $env:DEMO_MODE = "1"; ./scripts/run.ps1

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "Starting API on http://127.0.0.1:8000 ..." -ForegroundColor Cyan
$api = Start-Process -PassThru -NoNewWindow python `
    -ArgumentList "-m", "uvicorn", "api.main:app", "--port", "8000"

try {
    # Give the models a moment to warm before Streamlit's first request.
    Start-Sleep -Seconds 3
    Write-Host "Starting Streamlit ..." -ForegroundColor Cyan
    streamlit run app/Home.py
}
finally {
    Write-Host "Stopping API (PID $($api.Id)) ..." -ForegroundColor Yellow
    Stop-Process -Id $api.Id -Force -ErrorAction SilentlyContinue
}
