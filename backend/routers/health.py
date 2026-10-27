"""Health-check route."""

from fastapi import APIRouter

from backend.schemas.chat import HealthResponse

# Keep health checks under the same API namespace as the chat endpoint.
router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    # This is a lightweight liveness check; it does not load the RAG models.
    return HealthResponse(status="ok")
