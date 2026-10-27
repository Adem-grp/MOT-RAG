"""Shared settings for the RAG pipeline.
so the embedding model used to BUILD the database is always the same one used to SEARCH it.
(If those two ever differ, retrieval returns nonsense and nothing crashes. A nasty silent bug.)

You can override the LLM without editing code:
    Windows (PowerShell):  $env:RAG_LLM="qwen3:4b"
    Linux/macOS:           export RAG_LLM=qwen3:4b
"""
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

# ---------- LLM (runs in Ollama) ----------
# Hardware this is tuned for: 8 GB GPU memory + 32 GB system RAM.
#
# THE KEY RULE: speed depends on whether the model fits in GPU memory.
#   - Fits in 8 GB GPU  -> fast (tens of tokens per second).
#   - Does not fit      -> Ollama puts some layers in system RAM and runs them on the CPU.
#                          It still works (you have 32 GB RAM), but can drop to single-digit tokens/s.
#   One agent run makes roughly 5 to 12 LLM calls, so slow generation adds up quickly.
#
# TIER 1, DEFAULT. Dense ~8B model, fully on the GPU (~5 GB at Q4 plus context):
#   qwen3:8b       Good at instructions and JSON output, which agents depend on.
#   qwen3:4b       Fallback if anything feels slow or unstable.
#
# TIER 2, TRY LATER. Bigger models that use your 32 GB RAM:
#   - Mixture-of-experts (MoE) models, e.g. qwen3:30b-a3b (~19 GB). They hold 30B parameters
#     but only use ~3B per token, so they stay usable when part of the model sits in system RAM.
#     This is the best way to use your 32 GB. Expect better answers than 8b, at some cost in speed.
#   - Newer Qwen / Gemma releases may exist. Check ollama.com/library and look for models around
#     8B to 14B, or MoE with ~3B active. I could not verify exact tags or speeds for you.
#   - Dense 14B (~9 GB): just over what your GPU holds, so it will partly run on CPU. Usually
#     the worst of both worlds. Prefer an MoE over this.
#
# HOW TO TEST ANY MODEL BEFORE TRUSTING IT:
#   ollama run <model> --verbose     then ask anything. Read "eval rate" (tokens/s). Under ~10 is too slow here.
#   ollama ps                        shows how much of the loaded model sits on GPU vs CPU.
#   Then run your 20 test questions and compare answer quality AND time against qwen3:8b.
#
# Pull with:  ollama pull qwen3:8b
# Switch without editing code:  $env:RAG_LLM="qwen3:30b-a3b"   (PowerShell)
LLM_MODEL = os.getenv("RAG_LLM", "qwen3:8b")

# Context window in tokens. Ollama's default is small (2048 to 4096 depending on version) and it
# silently cuts off the START of long prompts, so your retrieved chunks would just vanish.
# 8192 is enough for 3 sub-questions x 5 chunks. Bigger values cost memory, so raise it carefully.
NUM_CTX = int(os.getenv("RAG_NUM_CTX", "4096"))

# ---------- Embeddings (run in Python, sentence-transformers) ----------
EMBED_MODEL = "BAAI/bge-small-en-v1.5"   # small (~130 MB), strong for its size
# Kept on CPU on purpose: the LLM gets all the GPU memory. bge-small is fast enough on CPU here.
EMBED_DEVICE = "cpu"
# bge models retrieve better if the QUESTION (not the documents) starts with this sentence.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# ---------- Data locations ----------
PDF_DIR = str(ROOT_DIR / "data" / "papers")
DB_DIR = str(ROOT_DIR / "chroma_db")
COLLECTION = "mot_papers"

# ---------- Retrieval ----------
TOP_K = 5  # how many chunks the baseline retrieves per question