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
- Use the conversation history to understand follow-up questions (e.g. "explain more", "what about X")
- If the context doesn't contain enough information, say "I don't have enough information to answer this"
- Be concise and accurate
- At the end of your answer, list the sources you used

CONVERSATION HISTORY:
{chat_history}

CONTEXT:
{context}

CURRENT QUESTION:
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


def format_chat_history(chat_history: list) -> str:
    """
    Formats previous Q&A pairs into a string for the prompt.
    """
    if not chat_history:
        return "No previous conversation."
    
    formatted = []
    for turn in chat_history[-3:]:  # only last 3 turns to keep prompt small
        formatted.append(f"User: {turn['question']}\nAssistant: {turn['answer']}")
    return "\n\n".join(formatted)


def generate_answer(question: str, docs: List[Document], chat_history: list = None) -> Dict:
    """
    Takes a question and retrieved docs, sends them to the LLM,
    and returns the answer with source information.
    """
    llm = get_llm()

    # Format docs into context string
    context = format_docs(docs)

    # Format chat history
    history_text = format_chat_history(chat_history or [])

    # Build the prompt with our question, context, and history
    prompt = RAG_PROMPT.format(context=context, question=question, chat_history=history_text)

    # Call the LLM
    print(f"🤖 Generating answer with Groq LLM...")
    response = llm.invoke(prompt)

    # Extract source file names for citation
    # Extract source citations — handle both local files and web URLs
    sources = []
    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        title = doc.metadata.get("title", "")
        
        if source.startswith("http"):
            # Web result — use title if available, else the URL
            label = title if title else source
        else:
            # Local file — just the filename
            label = os.path.basename(source)
        
        if label and label not in sources:
            sources.append(label)
    print(f"🔎 DEBUG final sources list: {sources}")

    return {
        "answer": response.content,
        "sources": sources,
        "retrieved_docs": docs,
        "num_docs_used": len(docs)
    }