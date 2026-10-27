"""Application models that connect the API to the existing RAG pipeline."""

from __future__ import annotations

import threading
from typing import Any

from backend.RAG_Structure.Ask import generate, retrieve


class RagModel:
    """Thin application-facing adapter around the unchanged baseline RAG logic."""

    def answer(self, question: str) -> tuple[str, list[dict[str, Any]]]:
        # Reuse the exact baseline retrieval and generation functions for comparison.
        chunks = retrieve(question)
        return generate(question, chunks), chunks

    def warm_up_retrieval(self) -> None:
        """Load the embedding model without making an LLM request."""
        retrieve("multi-object tracking")


# A shared adapter keeps route code focused on HTTP concerns.
rag_model = RagModel()


def warm_up_retrieval_in_background() -> None:
    threading.Thread(
        target=rag_model.warm_up_retrieval,
        daemon=True,
        name="rag-embedding-warmup",
    ).start()
