"""FastAPI application entry point for the Voice Interview Analysis System.

Routes
------
GET  /                      Local microphone-test prototype (static/index.html)
GET  /app/                  In-app call UI + interviewer dashboard (frontend/)
GET  /health                Basic health check
/api/sessions/...           Session, join-token, event and report APIs (app/api)
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api import events, reports, sessions
from app.config import settings

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI(
    title="Voice Interview Analyzer",
    version="0.1.0",
    description=(
        "Live English interview assistance. Flags are review cues for a human, "
        "never automated hiring decisions."
    ),
)

# API routers
app.include_router(sessions.router)
app.include_router(events.router)
app.include_router(reports.router)

# Static UIs
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/app", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    """Serve the M0 local microphone-test prototype."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health_check() -> dict:
    """Basic backend health response (plus non-secret feature flags for the UI)."""
    return {
        "status": "ok",
        "livekit_configured": settings.livekit_configured,
        "transcription_provider": settings.transcription_provider,
        "dev_mode": settings.dev_mode,
    }
