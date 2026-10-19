"""STEP 1c: BASELINE RAG. One retrieval, one LLM call, no agents.

    python ask.py "How does ByteTrack use low-confidence detections?"
    python ask.py                      (interactive mode)

Why keep this file even after you build the agents?
  1. It is your BASELINE. The README table needs "baseline vs agentic" numbers.
  2. agentic_rag.py imports retrieve() from here, so the agents search exactly the way the baseline does.

The two halves of RAG, and where they live in this file:
  RETRIEVAL   -> retrieve():  find the chunks most similar to the question (no LLM involved at all)
  GENERATION  -> generate():  give those chunks to the LLM and ask it to answer from them only
"""
import sys

import chromadb
import ollama
from sentence_transformers import SentenceTransformer

from Config import (COLLECTION, DB_DIR, EMBED_DEVICE, EMBED_MODEL, LLM_MODEL, NUM_CTX,
                    QUERY_PREFIX, TOP_K)

# The system prompt is how you control the model's behaviour. Three jobs here:
#   "ONLY the context"  -> reduces made-up facts (hallucination)
#   "say you could not find it" -> gives the model an honest way out instead of guessing
#   "cite like [ByteTrack p.4]" -> makes answers checkable, which is what separates RAG from a plain chatbot
SYSTEM_PROMPT = (
    "You answer questions about multi-object tracking research papers. "
    "Use ONLY the context provided. If the context does not contain the answer, "
    "say you could not find it in the papers. "
    "Cite sources after each claim like [ByteTrack p.4]."
)

# Loaded lazily (on first use) so that importing this file is instant.
_embedder = None
_collection = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL, device=EMBED_DEVICE)
    return _embedder


def _get_collection():
    global _collection
    if _collection is None:
        _collection = chromadb.PersistentClient(path=DB_DIR).get_collection(COLLECTION)
    return _collection


# ---------------------------------------------------------------------------
# RETRIEVAL (no LLM here)
# ---------------------------------------------------------------------------
def retrieve(question: str, k: int = TOP_K) -> list[dict]:
    """Return the k chunks closest in meaning to the question.

    'distance' is cosine distance: 0 means identical meaning, bigger means less related.
    Watch these numbers. If the best chunk has a big distance, retrieval failed, and no LLM
    can fix that. Most "bad RAG answers" are really bad retrieval.
    """
    q_emb = _get_embedder().encode(QUERY_PREFIX + question, normalize_embeddings=True).tolist()
    res = _get_collection().query(query_embeddings=[q_emb], n_results=k)
    return [
        {"text": doc, "source": meta["source"], "page": meta["page"], "distance": dist}
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]


# ---------------------------------------------------------------------------
# GENERATION (the LLM call)
# ---------------------------------------------------------------------------
def generate(question: str, chunks: list[dict]) -> str:
    context = "\n\n".join(f"[{c['source']} p.{c['page']}]\n{c['text']}" for c in chunks)
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
        # num_ctx: see config.py. Without it, long prompts get silently truncated.
        # temperature 0: same question gives (nearly) the same answer, which you need for fair testing.
        options={"num_ctx": NUM_CTX, "temperature": 0},
        # Qwen3 normally "thinks" for a long time before answering. For a baseline we turn that off
        # to keep it fast. (Needs ollama python >= 0.5.0 and a recent Ollama app.)
        think=False,
    )
    return response["message"]["content"]


def answer(question: str) -> None:
    chunks = retrieve(question)
    print("\n=== ANSWER ===")
    print(generate(question, chunks))
    print("\n=== RETRIEVED CHUNKS (lower distance = closer match) ===")
    for c in chunks:
        print(f"- {c['source']} p.{c['page']}  distance={c['distance']:.3f}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        answer(" ".join(sys.argv[1:]))
    else:
        while True:
            q = input("\nQuestion (or 'quit'): ").strip()
            if q.lower() in {"quit", "exit", ""}:
                break
            answer(q)