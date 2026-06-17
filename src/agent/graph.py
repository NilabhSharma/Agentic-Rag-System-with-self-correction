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
from src.agent.graders import filter_relevant_docs, grade_answer_faithfulness, rewrite_query_with_history
from src.agent.web_search import web_search

load_dotenv()

class AgentState(TypedDict):
    question: str                          
    search_query: str                    
    chat_history: List[dict]            
    documents: List[Document]     
    answer: str       
    sources: List[str]
    used_web_search: bool     
    retry_count: int                  
    generation_faithful: bool            


def retrieve_node(state: AgentState, retriever: HybridRetriever) -> AgentState:
    
    print("\n-- [Node: Retrieve]")
    question = state["question"]
    chat_history = state.get("chat_history", [])

    search_query = rewrite_query_with_history(question, chat_history)

    docs = retriever.retrieve(search_query)
    return {**state, "documents": docs, "search_query": search_query}


def grade_documents_node(state: AgentState) -> AgentState:
    print("\n--  [Node: Grade Documents]")
    question = state["question"]
    docs = state["documents"]

    relevant_docs, filtered_count = filter_relevant_docs(question, docs)

    return {**state, "documents": relevant_docs}

def web_search_node(state: AgentState) -> AgentState:
    print("\n-- [Node: Web Search Fallback]")
    question = state.get("search_query") or state["question"]
    web_docs = web_search(question, max_results=3)

    existing_docs = state.get("documents", [])
    all_docs = existing_docs + web_docs

    return {**state, "documents": all_docs, "used_web_search": True}

def generate_node(state: AgentState) -> AgentState:
    print("\n-- [Node: Generate Answer]")
    question = state["question"]
    docs = state["documents"]
    chat_history = state.get("chat_history", [])

    result = generate_answer(question, docs, chat_history)

    return {
        **state,
        "answer": result["answer"],
        "sources": result["sources"],
    }


def grade_answer_node(state: AgentState) -> AgentState:
    print("\n-- [Node: Grade Answer]")
    answer = state["answer"]
    docs = state["documents"]

    is_faithful = grade_answer_faithfulness(answer, docs)

    if is_faithful:
        print("-- Answer is faithful to sources")
    else:
        print("--  Answer may not be fully grounded — flagging for retry")

    return {
        **state,
        "generation_faithful": is_faithful,
        "retry_count": state.get("retry_count", 0) + 1
    }


def should_use_web_search(state: AgentState) -> str:
    docs = state.get("documents", [])

    if len(docs) == 0:
        print("-- Decision: No relevant docs → Web Search")
        return "web_search"
    else:
        print(f"-- Decision: {len(docs)} relevant docs found → Generate")
        return "generate"


def should_retry(state: AgentState) -> str:
    is_faithful = state.get("generation_faithful", False)
    retry_count = state.get("retry_count", 0)

    if is_faithful:
        print("-- Decision: Answer is good → END")
        return "end"
    elif retry_count < 2:
        print(f"-- Decision: Answer not faithful (attempt {retry_count}) → Retry")
        return "retry"
    else:
        print("-- Decision: Max retries reached → END anyway")
        return "end"


def build_agent(retriever: HybridRetriever):
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

    workflow.set_entry_point("retrieve")

    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", "grade_answer")

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
            "retry": "retrieve"   
        }
    )

    from langgraph.checkpoint.memory import MemorySaver
    memory = MemorySaver()
    agent = workflow.compile(checkpointer=memory)
    print("-- LangGraph agent compiled successfully (with memory)")
    return agent

def run_agent(question: str, retriever: HybridRetriever, agent=None, thread_id: str = "default") -> dict:
    if agent is None:
        agent = build_agent(retriever)

    initial_state = AgentState(
        question=question,
        chat_history=[],
        documents=[],
        answer="",
        sources=[],
        used_web_search=False,
        retry_count=0,
        generation_faithful=False
    )

    config = {"configurable": {"thread_id": thread_id}}

    print(f"\n{'='*50}")
    print(f"-- Agent starting for: '{question}'")
    print(f"{'='*50}")

    final_state = agent.invoke(initial_state, config=config)

    print(f"\n{'='*50}")
    print(f"-- FINAL ANSWER:")
    print(f"{'='*50}")
    print(final_state["answer"])
    print(f"\n-- Sources: {', '.join(final_state['sources'])}")
    print(f"-- Used web search: {final_state['used_web_search']}")
    print(f"-- Retry count: {final_state['retry_count']}")

    return final_state, agent