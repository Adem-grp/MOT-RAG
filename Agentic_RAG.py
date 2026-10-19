"""
STEP 3: MULTI-AGENT RAG WITH LANGGRAPH
======================================

Run:
    python agentic_rag.py "Compare ByteTrack and BoT-SORT on how they handle occlusion"
    python agentic_rag.py --diagram          print a Mermaid diagram for your README

Switches (use them to build the project in stages AND to produce your README comparison table):
    --no-fanout     no planner: the whole question goes to ONE researcher
    --no-grader     researcher retrieves once and trusts the result (no grading, no query rewriting)
    --no-verifier   writer's first draft is final (no fact checking)
    --think         let the writer use Qwen3's slow "thinking" mode (better answers, much slower)

    All three "--no-" switches together = plain baseline RAG written as a graph.

THE AGENTS (each one is the SAME local model with a different role prompt and one narrow job)
---------------------------------------------------------------------------------------------
  PLANNER     in: user question      out: 1 to 3 self-contained sub-questions
              Why: "Compare A and B on X" needs separate searches for A and for B. One search
              would retrieve a blurry mix of both.

  RESEARCHER  in: one sub-question   out: the best chunks it could find
              (one researcher is launched PER sub-question, so they are "agents started by an agent")
              Inside it runs a small loop:  retrieve -> GRADER judges -> if not enough,
              REWRITER makes a new query -> retrieve again (max MAX_REWRITES times).

  GRADER      in: question + chunks  out: {sufficient: bool, missing: str}
              Why: catches "retrieved text is related but does not actually answer".
              This is what makes the system "corrective" RAG.

  REWRITER    in: question + what was missing   out: a better search query

  WRITER      in: question + all evidence       out: cited answer
              Why: only this agent writes prose. Everyone else produces small structured data.

  VERIFIER    in: evidence + draft answer       out: {supported: bool, unsupported_claims: [...]}
              Why: LLMs add plausible-sounding claims. A second pass, whose only job is to be
              suspicious, catches many of them. If it complains, the writer revises (max 1 revision).

BIG PICTURE OF THE GRAPH
------------------------
    START -> planner --fan-out--> researcher (x N, one per sub-question) -> writer -> verifier
                                                                              ^          |
                                                                              +-- fix ---+  (or END)

WHERE EACH "WHAT TO LEARN" ITEM LIVES IN THIS FILE (search for the LEARNING STEP headers)
-----------------------------------------------------------------------------------------
  LEARNING STEP 1  structured outputs (forcing JSON)      -> call_llm() and the Pydantic schemas
  LEARNING STEP 2  LangGraph basics: state/nodes/edges    -> RAGState, the node functions, build_graph()
  LEARNING STEP 3  build the linear version first         -> the --no-... switches
  LEARNING STEP 4  loops with hard limits                 -> researcher() loop and after_verifier()
  LEARNING STEP 5  fan-out with Send                      -> fan_out()
  LEARNING STEP 6  tracing, so you can see what happened  -> trace() and the traces/ folder
"""
import json
import operator
import os
import sys
import time
from pathlib import Path
from typing import Annotated, TypedDict

import ollama
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field

from Ask import retrieve  # the step 1 retriever, reused as-is
from Config import LLM_MODEL, NUM_CTX

# ---------------------------------------------------------------------------
# Settings. With 8 GB of RAM, these limits are what keep the run fast and inside the context window.
# ---------------------------------------------------------------------------
MAX_SUBQUESTIONS = 3     # more sub-questions = more LLM calls = slower, and a longer writer prompt
MAX_REWRITES = 2         # extra retrieval attempts per sub-question
MAX_WRITER_ROUNDS = 2    # first draft + at most one revision
CHUNKS_PER_SUBQ = 5      # 3 sub-questions x 5 chunks x ~250 tokens = ~3.7k tokens, fits in NUM_CTX

# Feature switches, set from the command line in __main__. A module-level dict is a deliberate
# simplification for a learning project. In a bigger system you would pass config in explicitly.
FEATURES = {"fanout": True, "grader": True, "verifier": True, "writer_think": False}


# ===========================================================================
# LEARNING STEP 1: STRUCTURED OUTPUTS
# ---------------------------------------------------------------------------
# Problem: agents make DECISIONS that code must act on ("is this enough? yes/no").
# If an LLM answers in prose ("Well, it seems the chunks mostly cover..."), your code has to
# guess what that means. Fragile.
#
# Solution: describe the answer shape as a Pydantic class. Ollama then constrains the model so
# it can ONLY produce JSON matching that shape (see the `format=` argument in call_llm below).
# Pydantic validates it, and you get a real Python object: grade.sufficient is a real bool.
#
# Rule of thumb: every agent except the writer returns a schema like these.
# The Field(description=...) text is shown to the model, so it works as extra instructions.
# ===========================================================================
class Plan(BaseModel):
    sub_questions: list[str] = Field(description="1 to 3 self-contained search questions")


class Grade(BaseModel):
    sufficient: bool = Field(description="True only if the chunks contain enough to answer the question")
    missing: str = Field(description="What information is still missing. Empty string if sufficient")


class Rewrite(BaseModel):
    new_query: str = Field(description="A better search query using different keywords")


class Verdict(BaseModel):
    supported: bool = Field(description="True only if EVERY claim in the answer is stated in the evidence")
    unsupported_claims: list[str] = Field(description="Claims not backed by the evidence. Empty list if none")


# ---------------------------------------------------------------------------
# ROLE PROMPTS. One narrow job per agent.
# Small local models (8B and under) get confused by long prompts that do several jobs at once.
# If an agent behaves badly, make its prompt shorter and stricter before anything else.
# ---------------------------------------------------------------------------
PLANNER_PROMPT = """You plan searches over a collection of multi-object tracking research papers.
Split the user's question into 1 to 3 sub-questions that can each be answered by searching the papers.
Rules:
- If the question is simple, return exactly one sub-question.
- Each sub-question must be self-contained: name the method or paper explicitly, never use "it" or "they".
- Do not invent topics the user did not ask about."""

GRADER_PROMPT = """You judge search results. Given a question and retrieved text chunks, decide if the chunks
together contain enough information to answer the question. Be strict: related but not answering means insufficient.
If insufficient, say briefly what is missing."""

REWRITER_PROMPT = """You rewrite search queries. The previous query did not retrieve enough information.
Write one new query that targets the missing information, using different keywords than before."""

WRITER_PROMPT = """You write answers about multi-object tracking research using ONLY the evidence provided.
Rules:
- Cite every claim like [ByteTrack p.4], using the labels shown in the evidence.
- If the evidence is incomplete, say clearly what you could not find. Do not fill gaps from memory.
- Be concise and structured. Compare directly when the question asks for a comparison."""

VERIFIER_PROMPT = """You are a strict fact checker. Given evidence and a written answer, check every claim in the
answer against the evidence. A claim is unsupported if the evidence does not state it, even if it sounds plausible.
List unsupported claims exactly as they appear in the answer."""


# ===========================================================================
# LEARNING STEP 6: TRACING
# ---------------------------------------------------------------------------
# With agents, "it gave a bad answer" is useless information. You need to know WHICH agent got
# WHAT input and returned WHAT. trace() appends one JSON line per event to traces/run-<time>.jsonl.
# Open that file after a run and read it top to bottom: that is the whole story of the run.
#
# This is the do-it-yourself version, so you understand what tracing is. Langfuse (optional, see
# get_langfuse_callbacks below) is the same idea with a nice web UI.
# ===========================================================================
TRACE_FILE = Path("traces") / f"run-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"


def trace(event: str, **data) -> None:
    TRACE_FILE.parent.mkdir(exist_ok=True)
    with TRACE_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"t": round(time.time(), 2), "event": event, **data}, ensure_ascii=False) + "\n")


def log(msg: str) -> None:
    print(f"  {msg}", flush=True)


# ===========================================================================
# LEARNING STEP 1 (continued): THE ONE FUNCTION EVERY AGENT USES
# ===========================================================================
def call_llm(agent: str, system: str, user: str, schema: type[BaseModel] | None = None):
    """One LLM call. 'agent' is just a label for the trace.

    With a schema: returns a validated Pydantic object (the model cannot answer in any other shape).
    Without a schema: returns plain text (used only by the writer).
    """
    kwargs = {}
    if schema is not None:
        kwargs["format"] = schema.model_json_schema()  # this is the line that forces JSON

    # Qwen3 "thinking" = it writes a long hidden reasoning before answering. Great for hard problems,
    # but slow. Agents with tiny jobs (grade, rewrite, verify) do not need it. Only the writer may.
    think = FEATURES["writer_think"] if agent == "writer" else False

    t0 = time.time()
    resp = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        options={"num_ctx": NUM_CTX, "temperature": 0},  # temperature 0 = repeatable runs for fair testing
        think=think,
        **kwargs,
    )
    content = resp["message"]["content"]
    result = schema.model_validate_json(content) if schema else content

    trace("llm_call", agent=agent, seconds=round(time.time() - t0, 1),
          input=user[:4000], output=content[:4000])
    return result


def format_chunks(chunks: list[dict]) -> str:
    return "\n\n".join(f"[{c['source']} p.{c['page']}]\n{c['text']}" for c in chunks)


def format_evidence(evidence: list[dict]) -> str:
    return "\n\n".join(f"### Sub-question: {e['sub_question']}\n{format_chunks(e['chunks'])}" for e in evidence)


# ===========================================================================
# LEARNING STEP 2: LANGGRAPH BASICS
# ---------------------------------------------------------------------------
# LangGraph has four ideas. That is all of it:
#
#   STATE   A shared dictionary that flows through the graph. Every node reads from it and
#           returns a PARTIAL update ("here are the keys I want to change").
#
#   NODE    A plain Python function: state in, partial update out. Each of our agents is a node.
#
#   EDGE    A fixed arrow: "after node A, always run node B".
#
#   CONDITIONAL EDGE   An arrow decided at runtime by a function that looks at the state and
#           returns the NAME of the next node. This is how loops and branching happen.
#
# REDUCERS: when several nodes write to the same key at the same time (our parallel researchers
# all write to 'evidence'), LangGraph needs to know how to combine them. Annotated[list, operator.add]
# means "concatenate the lists". Without it, the last writer would overwrite the others.
# ===========================================================================
class RAGState(TypedDict, total=False):
    question: str
    sub_questions: list[str]
    evidence: Annotated[list[dict], operator.add]  # reducer: researchers' results get concatenated
    answer: str
    verdict: dict
    writer_rounds: int


class ResearchTask(TypedDict):
    """What each researcher receives. NOT the full state, only its own sub-question (see fan_out)."""
    sub_question: str


# ---------------------------------------------------------------------------
# NODE: PLANNER
# ---------------------------------------------------------------------------
def planner(state: RAGState) -> dict:
    # LEARNING STEP 3: with --no-fanout we skip the planner's LLM call entirely, and the original
    # question becomes the single sub-question. That is the "linear" version of the pipeline.
    if not FEATURES["fanout"]:
        return {"sub_questions": [state["question"]]}

    plan = call_llm("planner", PLANNER_PROMPT, state["question"], Plan)
    subs = [q for q in plan.sub_questions if q.strip()][:MAX_SUBQUESTIONS] or [state["question"]]
    print(f"[planner] {len(subs)} sub-question(s)")
    for q in subs:
        log(f"- {q}")
    return {"sub_questions": subs}


# ===========================================================================
# LEARNING STEP 5: FAN-OUT WITH Send   (the "one agent starts other agents" part)
# ---------------------------------------------------------------------------
# This function is used as a CONDITIONAL EDGE out of the planner. Instead of returning one node
# name, it returns a LIST of Send objects. Each Send means: "start a copy of the 'researcher' node
# with THIS input". If the planner made 3 sub-questions, 3 researchers run.
#
# The number of agents is decided at RUNTIME by an LLM's output. That is what makes the workflow
# agentic and not just a fixed pipeline. Each researcher gets only its own small input, and the
# 'evidence' reducer in RAGState merges all their results afterwards.
#
# On one GPU these researchers effectively run one after another. Same logic, just not faster.
# ===========================================================================
def fan_out(state: RAGState) -> list[Send]:
    return [Send("researcher", {"sub_question": q}) for q in state["sub_questions"]]


# ---------------------------------------------------------------------------
# NODE: RESEARCHER (contains GRADER and REWRITER as sub-steps)
# ---------------------------------------------------------------------------
def researcher(task: ResearchTask) -> dict:
    """LEARNING STEP 4: a loop with a hard limit.

    retrieve -> grade -> (if not enough) rewrite the query -> retrieve again ...

    Every loop in an agent system needs a maximum, because an LLM judge can say "not enough"
    forever. After MAX_REWRITES we give up and pass on the best chunks we found, flagged as
    insufficient, so the writer can be honest about it.
    """
    original = task["sub_question"]
    query = original
    seen: dict[tuple, dict] = {}   # dedupe chunks found by different query attempts
    grade = None

    for attempt in range(MAX_REWRITES + 1):
        for c in retrieve(query):
            seen[(c["source"], c["page"], c["text"][:80])] = c
        # Keep only the closest chunks overall, so the writer prompt cannot outgrow the context window.
        chunks = sorted(seen.values(), key=lambda c: c["distance"])[:CHUNKS_PER_SUBQ]

        if not FEATURES["grader"]:   # LEARNING STEP 3: linear version trusts the first retrieval
            break

        # GRADER agent
        grade = call_llm(
            "grader", GRADER_PROMPT,
            f"Question: {original}\n\nChunks:\n{format_chunks(chunks)}", Grade,
        )
        log(f"[researcher] '{original[:50]}' attempt {attempt + 1}: sufficient={grade.sufficient}")
        if grade.sufficient:
            break

        # REWRITER agent (skipped on the last attempt, since there would be nothing to do with the result)
        if attempt < MAX_REWRITES:
            query = call_llm(
                "rewriter", REWRITER_PROMPT,
                f"Question: {original}\nPrevious query: {query}\nMissing: {grade.missing}", Rewrite,
            ).new_query
            log(f"[researcher] rewrote query -> {query}")

    sufficient = grade.sufficient if grade else None
    trace("researcher_done", sub_question=original, sufficient=sufficient,
          sources=[f"{c['source']} p.{c['page']}" for c in chunks])
    return {"evidence": [{"sub_question": original, "chunks": chunks, "sufficient": sufficient}]}


# ---------------------------------------------------------------------------
# NODE: WRITER
# ---------------------------------------------------------------------------
def writer(state: RAGState) -> dict:
    rounds = state.get("writer_rounds", 0)

    # On a revision round, the verifier's complaints are fed back in. This feedback loop is the
    # reason the verifier is worth having: it does not just reject, it tells the writer what to fix.
    feedback = ""
    verdict = state.get("verdict")
    if verdict and not verdict["supported"]:
        issues = "\n- ".join(verdict["unsupported_claims"])
        feedback = f"\n\nA fact checker flagged these claims in your previous draft. Fix or remove them:\n- {issues}"

    user = f"Question: {state['question']}\n\nEvidence:\n{format_evidence(state['evidence'])}{feedback}"
    answer = call_llm("writer", WRITER_PROMPT, user)
    print(f"[writer] draft {rounds + 1} done")
    return {"answer": answer, "writer_rounds": rounds + 1}


# ---------------------------------------------------------------------------
# NODE: VERIFIER
# ---------------------------------------------------------------------------
def verifier(state: RAGState) -> dict:
    verdict = call_llm(
        "verifier", VERIFIER_PROMPT,
        f"Evidence:\n{format_evidence(state['evidence'])}\n\nAnswer to check:\n{state['answer']}", Verdict,
    )
    print(f"[verifier] supported={verdict.supported}, flagged={len(verdict.unsupported_claims)}")
    return {"verdict": verdict.model_dump()}


def after_verifier(state: RAGState) -> str:
    """CONDITIONAL EDGE (LEARNING STEP 4 again): decides whether the writer<->verifier loop continues.

    Stop if the draft is supported OR we hit the round limit. The limit matters: without it, a
    verifier that always finds something would loop forever.
    """
    if state["verdict"]["supported"] or state["writer_rounds"] >= MAX_WRITER_ROUNDS:
        return END
    return "writer"


# ===========================================================================
# WIRING: this is the whole workflow, in one place. Read this function to understand the system.
# ===========================================================================
def build_graph():
    g = StateGraph(RAGState)

    g.add_node("planner", planner)
    g.add_node("researcher", researcher)
    g.add_node("writer", writer)
    g.add_node("verifier", verifier)

    g.add_edge(START, "planner")
    g.add_conditional_edges("planner", fan_out, ["researcher"])  # fan-out (LEARNING STEP 5)
    g.add_edge("researcher", "writer")   # fixed edge. Waits until ALL launched researchers are done.

    if FEATURES["verifier"]:
        g.add_edge("writer", "verifier")
        g.add_conditional_edges("verifier", after_verifier, ["writer", END])  # the loop
    else:
        g.add_edge("writer", END)        # LEARNING STEP 3: linear version, no checking

    return g.compile()


# Optional: Langfuse gives you a web UI for traces. Untested here. Check the Langfuse docs for the
# version you install, because the import path changed between v2 and v3. It only turns on if you
# set LANGFUSE_PUBLIC_KEY (and the other LANGFUSE_* variables). Because our agents call Ollama
# directly (not through LangChain), Langfuse will show graph NODES, not every LLM call. The local
# traces/ files above cover the LLM calls.
def get_langfuse_callbacks() -> list:
    if not os.getenv("LANGFUSE_PUBLIC_KEY"):
        return []
    try:
        from langfuse.langchain import CallbackHandler  # langfuse v3
        return [CallbackHandler()]
    except ImportError:
        print("LANGFUSE_PUBLIC_KEY is set but langfuse is not installed (pip install langfuse). Skipping.")
        return []


if __name__ == "__main__":
    args = sys.argv[1:]
    FEATURES["fanout"] = "--no-fanout" not in args
    FEATURES["grader"] = "--no-grader" not in args
    FEATURES["verifier"] = "--no-verifier" not in args
    FEATURES["writer_think"] = "--think" in args
    question = " ".join(a for a in args if not a.startswith("--"))

    graph = build_graph()

    if "--diagram" in args:
        print(graph.get_graph().draw_mermaid())
        sys.exit(0)

    question = question or input("Question: ").strip()
    trace("run_start", question=question, model=LLM_MODEL, features=FEATURES)
    start = time.time()

    result = graph.invoke(
        {"question": question, "writer_rounds": 0, "evidence": []},
        config={"callbacks": get_langfuse_callbacks()},
    )

    elapsed = time.time() - start
    print("\n=== ANSWER ===")
    print(result["answer"])
    verdict = result.get("verdict")
    if verdict and not verdict["supported"]:
        print("\n(warning: verifier still flagged unsupported claims)")
        for claim in verdict["unsupported_claims"]:
            print(f"  - {claim}")
    print(f"\n[{elapsed:.0f}s, {result['writer_rounds']} writer round(s), trace: {TRACE_FILE}]")
    trace("run_end", seconds=round(elapsed, 1), writer_rounds=result["writer_rounds"])