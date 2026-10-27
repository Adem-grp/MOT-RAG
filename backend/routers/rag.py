"""Routes for the baseline RAG chat flow."""

from fastapi import APIRouter, HTTPException

from backend.models.rag import rag_model
from backend.schemas.chat import ChatRequest, ChatResponse

# All RAG endpoints share the /api prefix and appear under the "rag" docs tag.
router = APIRouter(prefix="/api", tags=["rag"])


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Retrieve paper context and generate an answer using the existing pipeline."""
    try:
        # The synchronous handler matches the synchronous embedding, Chroma, and Ollama calls.
        answer, chunks = rag_model.answer(request.question.strip())
    except Exception as error:
        # Keep the traceback in the server log and expose the actionable local-service cause.
        import traceback

        traceback.print_exc()
        raise HTTPException(
            status_code=503,
            detail=(
                "The RAG service could not answer the request. "
                f"Underlying error: {error}"
            ),
        ) from error

    # Pydantic validates and serializes the RAG dictionaries into the API contract.
    return ChatResponse(answer=answer, sources=chunks)
