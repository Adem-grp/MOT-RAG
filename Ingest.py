"""STEP 1b: INGESTION. Turn PDFs into a searchable vector database.

Run this once (and again whenever you add papers or change chunk settings):
    python ingest.py

What happens, in order:
    PDF -> text per page -> chunks -> embeddings (vectors) -> stored in Chroma with metadata
"""
from pathlib import Path

import chromadb
import fitz  # pymupdf
from sentence_transformers import SentenceTransformer

from config import COLLECTION, DB_DIR, EMBED_DEVICE, EMBED_MODEL, PDF_DIR

# Chunking is the most underrated part of RAG. Too small: a chunk lacks the context to be useful.
# Too big: the chunk contains lots of unrelated text, so its vector is "blurry" and the prompt fills up.
# Experiment with these two numbers and re-run your test questions. It is a real tuning knob.
CHUNK_SIZE = 1000    # characters per chunk (roughly 250 tokens)
CHUNK_OVERLAP = 200  # characters shared between neighbouring chunks, so sentences are not cut in half


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Slide a window over the text. Simple on purpose, so you understand it before using a library."""
    chunks = []
    start = 0
    while start < len(text):
        piece = text[start : start + size].strip()
        if len(piece) > 100:  # skip tiny scraps like page numbers
            chunks.append(piece)
        start += size - overlap
    return chunks


def load_chunks() -> tuple[list[str], list[dict], list[str]]:
    texts, metadatas, ids = [], [], []
    for pdf_path in sorted(Path(PDF_DIR).glob("*.pdf")):
        doc = fitz.open(pdf_path)
        for page_num, page in enumerate(doc, start=1):
            for i, chunk in enumerate(chunk_text(page.get_text(), CHUNK_SIZE, CHUNK_OVERLAP)):
                texts.append(chunk)
                # Metadata is what lets the answer say "[ByteTrack p.4]". Never skip it.
                metadatas.append({"source": pdf_path.stem, "page": page_num})
                ids.append(f"{pdf_path.stem}-p{page_num}-c{i}")
        print(f"read {pdf_path.name} ({len(doc)} pages)")
    return texts, metadatas, ids


def main() -> None:
    texts, metadatas, ids = load_chunks()
    print(f"{len(texts)} chunks total")

    # Embedding: text -> list of 384 numbers. Similar meaning = nearby vectors.
    # normalize_embeddings=True makes cosine similarity work properly.
    model = SentenceTransformer(EMBED_MODEL, device=EMBED_DEVICE)
    embeddings = model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=True)

    client = chromadb.PersistentClient(path=DB_DIR)
    try:
        client.delete_collection(COLLECTION)  # rebuild from scratch each run, so no duplicates
    except Exception:
        pass
    collection = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    batch = 500  # Chroma limits how many items you can add in one call
    for i in range(0, len(texts), batch):
        collection.add(
            ids=ids[i : i + batch],
            documents=texts[i : i + batch],
            metadatas=metadatas[i : i + batch],
            embeddings=embeddings[i : i + batch].tolist(),
        )
    print(f"stored {collection.count()} chunks in {DB_DIR}")


if __name__ == "__main__":
    main()