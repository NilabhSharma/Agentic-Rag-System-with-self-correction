# src/retrieval/pipeline.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.ingestion.document_loader import load_documents, split_documents
from src.retrieval.retriever import HybridRetriever
from src.retrieval.rag_chain import generate_answer


def build_pipeline():
    """
    Builds the full RAG pipeline.
    Loads chunks from disk to set up the hybrid retriever.
    """
    print("🔧 Building RAG pipeline...")

    # We reload the documents just to build the BM25 index
    # (ChromaDB handles the dense side from disk automatically)
    documents = load_documents("./data/documents")
    if not documents:
        print("❌ No documents found. Run ingestion first.")
        return None

    chunks = split_documents(documents)
    retriever = HybridRetriever(chunks=chunks, k=4)

    print("✅ Pipeline ready!\n")
    return retriever


def ask(question: str, retriever: HybridRetriever) -> dict:
    """
    Full RAG flow for a single question:
    1. Retrieve relevant chunks
    2. Generate answer using LLM
    3. Return answer + sources
    """
    print(f"\n{'='*50}")
    print(f"❓ Question: {question}")
    print(f"{'='*50}")

    # Step 1: Retrieve
    docs = retriever.retrieve(question)

    # Step 2: Generate
    result = generate_answer(question, docs)

    # Step 3: Display
    print(f"\n💬 Answer:\n{result['answer']}")
    print(f"\n📚 Sources used: {', '.join(result['sources'])}")
    print(f"📊 Chunks retrieved: {result['num_docs_used']}")

    return result


if __name__ == "__main__":
    # Build the pipeline once
    retriever = build_pipeline()

    if retriever:
        # Test with a few questions
        ask("What is RAG?", retriever)
        ask("How does ChromaDB work?", retriever)
        ask("What is the difference between deep learning and machine learning?", retriever)