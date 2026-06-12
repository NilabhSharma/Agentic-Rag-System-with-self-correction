# test_agent.py

import sys
import os
sys.path.append(os.path.dirname(__file__))

from src.ingestion.document_loader import load_documents, split_documents
from src.retrieval.retriever import HybridRetriever
from src.agent.graph import run_agent

def main():
    # Build retriever
    print("🔧 Loading documents and building retriever...")
    documents = load_documents("./data/documents")
    chunks = split_documents(documents)
    retriever = HybridRetriever(chunks=chunks, k=4)

    # Test 1: Question answerable from local docs
    run_agent("What is Retrieval Augmented Generation?", retriever)

    print("\n" + "="*60 + "\n")

    # Test 2: Question NOT in local docs — should trigger web search
    run_agent("Who won the FIFA World Cup in 2022?", retriever)

if __name__ == "__main__":
    main()