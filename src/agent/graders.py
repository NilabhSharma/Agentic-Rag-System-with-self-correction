# src/agent/graders.py

import os
from typing import List
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate

load_dotenv()

QUERY_REWRITE_PROMPT = ChatPromptTemplate.from_template("""
Given the conversation history and a follow-up question, rewrite the follow-up question 
into a standalone question that includes all necessary context.

If the follow-up question is already standalone (doesn't depend on history), return it unchanged.

CONVERSATION HISTORY:
{chat_history}

FOLLOW-UP QUESTION:
{question}

STANDALONE QUESTION (reply with ONLY the rewritten question, nothing else):
""")


def rewrite_query_with_history(question: str, chat_history: list) -> str:
    """
    Rewrites a follow-up question (like "explain more") into a 
    standalone question using conversation history.
    """
    if not chat_history:
        return question

    from src.retrieval.rag_chain import format_chat_history

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, max_tokens=100)
    history_text = format_chat_history(chat_history)

    prompt = QUERY_REWRITE_PROMPT.format(chat_history=history_text, question=question)
    response = llm.invoke(prompt)
    rewritten = response.content.strip()

    print(f"🔄 Rewrote query: '{question}' → '{rewritten}'")
    return rewritten

def get_grader_llm():
    """Small, fast model just for grading tasks."""
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=10,  # We only need "yes" or "no"
    )


# --- Prompt 1: Document Relevance Grader ---
DOC_GRADER_PROMPT = ChatPromptTemplate.from_template("""
You are a grader checking if a document is relevant to a question.

DOCUMENT:
{document}

QUESTION:
{question}

Is this document relevant to answering the question?
Reply with ONLY one word: "yes" or "no"
""")


# --- Prompt 2: Answer Faithfulness Grader ---
ANSWER_GRADER_PROMPT = ChatPromptTemplate.from_template("""
You are a grader checking if an answer is grounded in the provided documents.

DOCUMENTS:
{documents}

GENERATED ANSWER:
{answer}

Is the answer fully supported by the documents? 
Reply with ONLY one word: "yes" or "no"
""")


def grade_document_relevance(question: str, doc: Document) -> bool:
    """
    Returns True if the document is relevant to the question.
    Uses LLM to make the judgment.
    """
    llm = get_grader_llm()
    prompt = DOC_GRADER_PROMPT.format(
        document=doc.page_content,
        question=question
    )
    response = llm.invoke(prompt)
    result = response.content.strip().lower()
    return result == "yes"


def grade_answer_faithfulness(answer: str, docs: List[Document]) -> bool:
    """
    Returns True if the answer is grounded in the retrieved documents.
    Catches hallucinations — when the LLM makes things up.
    """
    llm = get_grader_llm()
    docs_text = "\n\n".join([doc.page_content for doc in docs])
    prompt = ANSWER_GRADER_PROMPT.format(
        documents=docs_text,
        answer=answer
    )
    response = llm.invoke(prompt)
    result = response.content.strip().lower()
    return result == "yes"


def filter_relevant_docs(question: str, docs: List[Document]) -> tuple[List[Document], int]:
    """
    Grades all retrieved docs and returns only the relevant ones.
    Also returns count of how many were filtered out.
    """
    print(f"⚖️  Grading {len(docs)} retrieved documents...")
    relevant = []
    filtered = 0

    for i, doc in enumerate(docs):
        is_relevant = grade_document_relevance(question, doc)
        if is_relevant:
            relevant.append(doc)
            print(f"  ✅ Chunk {i+1}: Relevant")
        else:
            filtered += 1
            print(f"  ❌ Chunk {i+1}: Not relevant — filtered out")

    print(f"📊 Kept {len(relevant)}/{len(docs)} chunks")
    return relevant, filtered