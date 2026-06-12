# src/retrieval/rag_chain.py

import os
from typing import List, Dict
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate

load_dotenv()


def get_llm():
    """
    Returns the Groq LLM (free, fast Llama 3.3).
    """
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,         # 0 = more factual, less creative
        max_tokens=1024,
    )


# This is the prompt template — it tells the LLM exactly how to behave
RAG_PROMPT = ChatPromptTemplate.from_template("""
You are a helpful assistant that answers questions based on the provided context.

INSTRUCTIONS:
- Answer the question using ONLY the information in the context below
- If the context doesn't contain enough information, say "I don't have enough information to answer this"
- Be concise and accurate
- At the end of your answer, list the sources you used

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
""")


def format_docs(docs: List[Document]) -> str:
    """
    Formats retrieved document chunks into a single string
    that gets inserted into the prompt as context.
    """
    formatted = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "Unknown")
        # Just get the filename, not the full path
        source_name = os.path.basename(source)
        formatted.append(f"[Source {i+1}: {source_name}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


def generate_answer(question: str, docs: List[Document]) -> Dict:
    """
    Takes a question and retrieved docs, sends them to the LLM,
    and returns the answer with source information.
    """
    llm = get_llm()

    # Format docs into context string
    context = format_docs(docs)

    # Build the prompt with our question and context
    prompt = RAG_PROMPT.format(context=context, question=question)

    # Call the LLM
    print(f"🤖 Generating answer with Groq LLM...")
    response = llm.invoke(prompt)

    # Extract source file names for citation
    sources = list(set([
        os.path.basename(doc.metadata.get("source", "Unknown"))
        for doc in docs
    ]))

    return {
        "answer": response.content,
        "sources": sources,
        "retrieved_docs": docs,
        "num_docs_used": len(docs)
    }