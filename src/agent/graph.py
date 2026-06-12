# src/agent/graph.py

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from typing import List, TypedDict, Annotated
import operator
from dotenv import load_dotenv

from langchain.schema import Document
from langgraph.graph import StateGraph, END

from src.retrieval.retriever import HybridRetriever
from src.retrieval.rag_chain import generate_answer
from src.agent.graders import filter_relevant_docs, grade_answer_faithfulness
from src.agent.web_search import web_search

load_dotenv()

# ─────────────────────────────────────────────
# STATE — this is the "memory" passed between nodes
# Every node reads from and writes to this dict
# ─────────────────────────────────────────────
class AgentState(TypedDict):
    question: str                          # The user's question
    documents: List[Document]              # Retrieved/filtered documents
    answer: str                            # Generated answer
    sources: List[str]                     # Source citations
    used_web_search: bool                  # Did we fall back to web?
    retry_count: int                       # How many self-correction retries
    generation_faithful: bool              # Did answer pass faithfulness check?


# ─────────────────────────────────────────────
# NODE FUNCTIONS — each does one job
# ─────────────────────────────────────────────

def retrieve_node(state: AgentState, retriever: HybridRetriever) -> AgentState:
    """Node 1: Retrieve documents from ChromaDB."""
    print("\n📥 [Node: Retrieve]")
    question = state["question"]
    docs = retriever.retrieve(question)
    return {**state, "documents": docs}


def grade_documents_node(state: AgentState) -> AgentState:
    """Node 2: Grade each retrieved document for relevance."""
    print("\n⚖️  [Node: Grade Documents]")
    question = state["question"]
    docs = state["documents"]

    relevant_docs, filtered_count = filter_relevant_docs(question, docs)

    return {**state, "documents": relevant_docs}


def web_search_node(state: AgentState) -> AgentState:
    """Node 3: Fall back to web search (only triggered if docs are bad)."""
    print("\n🌐 [Node: Web Search Fallback]")
    question = state["question"]
    web_docs = web_search(question, max_results=3)

    # Combine with any surviving local docs
    existing_docs = state.get("documents", [])
    all_docs = existing_docs + web_docs

    return {**state, "documents": all_docs, "used_web_search": True}


def generate_node(state: AgentState) -> AgentState:
    """Node 4: Generate answer using LLM + retrieved context."""
    print("\n🤖 [Node: Generate Answer]")
    question = state["question"]
    docs = state["documents"]

    result = generate_answer(question, docs)

    return {
        **state,
        "answer": result["answer"],
        "sources": result["sources"],
    }


def grade_answer_node(state: AgentState) -> AgentState:
    """Node 5: Check if the answer is faithful to the documents."""
    print("\n🔍 [Node: Grade Answer]")
    answer = state["answer"]
    docs = state["documents"]

    is_faithful = grade_answer_faithfulness(answer, docs)

    if is_faithful:
        print("✅ Answer is faithful to sources")
    else:
        print("⚠️  Answer may not be fully grounded — flagging for retry")

    return {
        **state,
        "generation_faithful": is_faithful,
        "retry_count": state.get("retry_count", 0) + 1
    }


# ─────────────────────────────────────────────
# CONDITIONAL EDGE FUNCTIONS
# These decide which node to go to next
# ─────────────────────────────────────────────

def should_use_web_search(state: AgentState) -> str:
    """
    After grading docs: if too few relevant docs remain, 
    trigger web search. Otherwise go straight to generation.
    """
    docs = state.get("documents", [])

    if len(docs) == 0:
        print("🔀 Decision: No relevant docs → Web Search")
        return "web_search"
    else:
        print(f"🔀 Decision: {len(docs)} relevant docs found → Generate")
        return "generate"


def should_retry(state: AgentState) -> str:
    """
    After grading the answer:
    - If faithful → return final answer
    - If not faithful AND retry count < 2 → re-retrieve
    - If not faithful AND too many retries → return anyway
    """
    is_faithful = state.get("generation_faithful", False)
    retry_count = state.get("retry_count", 0)

    if is_faithful:
        print("🔀 Decision: Answer is good → END")
        return "end"
    elif retry_count < 2:
        print(f"🔀 Decision: Answer not faithful (attempt {retry_count}) → Retry")
        return "retry"
    else:
        print("🔀 Decision: Max retries reached → END anyway")
        return "end"


# ─────────────────────────────────────────────
# BUILD THE GRAPH
# ─────────────────────────────────────────────

def build_agent(retriever: HybridRetriever):
    """
    Assembles all nodes and edges into a LangGraph StateGraph.
    """

    # Wrap retrieve_node to inject the retriever (LangGraph nodes take only state)
    def retrieve(state):
        return retrieve_node(state, retriever)

    # Create the graph
    workflow = StateGraph(AgentState)

    # Add all nodes
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("grade_answer", grade_answer_node)

    # Set the entry point
    workflow.set_entry_point("retrieve")

    # Add edges (fixed paths)
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", "grade_answer")

    # Add conditional edges (decision points)
    workflow.add_conditional_edges(
        "grade_documents",
        should_use_web_search,
        {
            "web_search": "web_search",
            "generate": "generate"
        }
    )

    workflow.add_conditional_edges(
        "grade_answer",
        should_retry,
        {
            "end": END,
            "retry": "retrieve"   # Loop back to retrieve and try again
        }
    )

    # Compile the graph
    agent = workflow.compile()
    print("✅ LangGraph agent compiled successfully")
    return agent


# ─────────────────────────────────────────────
# MAIN RUN FUNCTION
# ─────────────────────────────────────────────

def run_agent(question: str, retriever: HybridRetriever) -> dict:
    """
    Runs the full agentic RAG pipeline for a question.
    """
    agent = build_agent(retriever)

    # Initial state
    initial_state = AgentState(
        question=question,
        documents=[],
        answer="",
        sources=[],
        used_web_search=False,
        retry_count=0,
        generation_faithful=False
    )

    print(f"\n{'='*50}")
    print(f"🚀 Agent starting for: '{question}'")
    print(f"{'='*50}")

    # Run the graph
    final_state = agent.invoke(initial_state)

    # Print final result
    print(f"\n{'='*50}")
    print(f"✅ FINAL ANSWER:")
    print(f"{'='*50}")
    print(final_state["answer"])
    print(f"\n📚 Sources: {', '.join(final_state['sources'])}")
    print(f"🌐 Used web search: {final_state['used_web_search']}")
    print(f"🔁 Retry count: {final_state['retry_count']}")

    return final_state