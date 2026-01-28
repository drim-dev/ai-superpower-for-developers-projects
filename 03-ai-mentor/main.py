"""AI Mentor API entry point."""

from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import database
from app.routes import router

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="AI Mentor API",
    description="API for AI-powered mentoring sessions",
    version="1.0.0",
)

app.include_router(router)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def serve_frontend():
    """Serve the chat interface."""
    return FileResponse(STATIC_DIR / "index.html")


@app.on_event("startup")
def startup():
    """Initialize database on startup."""
    database.init_db()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
