# src/api/main.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Dict
import uuid

from src.ingestion.document_loader import load_documents, split_documents
from src.retrieval.retriever import HybridRetriever
from src.agent.graph import build_agent, run_agent, AgentState

app = FastAPI(title="Agentic RAG API")

# ─────────────────────────────────────────────
# GLOBAL STATE (loaded once on startup)
# ─────────────────────────────────────────────
retriever = None
agent = None

# Store chat history per session (in-memory — resets if server restarts)
session_histories: Dict[str, List[dict]] = {}


@app.on_event("startup")
def startup_event():
    """Runs once when the API server starts."""
    global retriever, agent

    print("🔧 Initializing RAG pipeline...")
    documents = load_documents("./data/documents")
    chunks = split_documents(documents)
    retriever = HybridRetriever(chunks=chunks, k=4)
    agent = build_agent(retriever)
    print("✅ API ready!")


# ─────────────────────────────────────────────
# REQUEST / RESPONSE MODELS
# ─────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    session_id: Optional[str] = None   # If not provided, a new session is created


class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    used_web_search: bool
    retry_count: int
    session_id: str


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "Agentic RAG API is running", "docs": "/docs"}


@app.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    """
    Main endpoint — ask a question, get an agentic RAG answer.
    Supports multi-turn via session_id.
    """
    global retriever, agent, session_histories

    # Create a new session ID if none provided
    session_id = request.session_id or str(uuid.uuid4())

    # Get this session's chat history (or empty list if new)
    chat_history = session_histories.get(session_id, [])

    # Build initial state
    initial_state = AgentState(
        question=request.question,
        search_query="",
        chat_history=chat_history,
        documents=[],
        answer="",
        sources=[],
        used_web_search=False,
        retry_count=0,
        generation_faithful=False
    )

    config = {"configurable": {"thread_id": session_id}}
    final_state = agent.invoke(initial_state, config=config)

    # Update chat history for this session
    chat_history.append({
        "question": request.question,
        "answer": final_state["answer"]
    })
    session_histories[session_id] = chat_history

    return QueryResponse(
        answer=final_state["answer"],
        sources=final_state["sources"],
        used_web_search=final_state["used_web_search"],
        retry_count=final_state["retry_count"],
        session_id=session_id
    )


@app.get("/health")
def health_check():
    return {"status": "ok", "retriever_ready": retriever is not None}