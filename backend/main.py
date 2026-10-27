"""FastAPI application serving the API and browser frontend."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from backend.routers.health import router as health_router
from backend.routers.rag import router as rag_router
from backend.routers.realtime import router as realtime_router
from backend.models.rag import warm_up_retrieval_in_background

ROOT_DIR = Path(__file__).resolve().parent.parent
# Resolve the frontend from this file rather than from the process working directory.
FRONTEND_DIR = ROOT_DIR / "frontend"

app = FastAPI(title="MOT Research Assistant", version="1.0.0")


@app.on_event("startup")
async def warm_up_rag() -> None:
    warm_up_retrieval_in_background()


# Routers keep health and RAG endpoints separate as the API grows.
app.include_router(health_router)
app.include_router(rag_router)
app.include_router(realtime_router)
# Serving the frontend from the same origin avoids CORS configuration for now.
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
