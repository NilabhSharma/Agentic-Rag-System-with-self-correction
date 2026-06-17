
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.ingestion.document_loader import load_documents, split_documents
from src.retrieval.retriever import HybridRetriever
from src.retrieval.rag_chain import generate_answer

def build_pipeline():
    print("-- Building RAG pipeline...")

    documents = load_documents("./data/documents")
    if not documents:
        print("-- No documents found. Run ingestion first.")
        return None

    chunks = split_documents(documents)
    retriever = HybridRetriever(chunks=chunks, k=4)

    print("-- Pipeline ready!\n")
    return retriever


def ask(question: str, retriever: HybridRetriever) -> dict:
    print(f"\n{'='*50}")
    print(f"-- Question: {question}")
    print(f"{'='*50}")

    docs = retriever.retrieve(question)
    result = generate_answer(question, docs)

    print(f"\n-- Answer:\n{result['answer']}")
    print(f"\n-- Sources used: {', '.join(result['sources'])}")
    print(f"-- Chunks retrieved: {result['num_docs_used']}")

    return result


if __name__ == "__main__":
    retriever = build_pipeline()

    if retriever:
        ask("What is RAG?", retriever)
        ask("How does ChromaDB work?", retriever)
        ask("What is the difference between deep learning and machine learning?", retriever)