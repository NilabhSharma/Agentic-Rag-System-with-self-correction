# src/ingestion/ingest.py

import sys
import os

# This makes sure Python can find our src/ folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.ingestion.document_loader import load_documents, split_documents
from src.ingestion.vector_store import create_vector_store

def run_ingestion(documents_folder: str = "./data/documents"):
    print("=" * 50)
    print("🚀 Starting Document Ingestion Pipeline")
    print("=" * 50)

    # Step 1: Load documents
    print("\n📥 Step 1: Loading documents...")
    documents = load_documents(documents_folder)
    
    if not documents:
        print("❌ No documents found. Add files to data/documents/ and try again.")
        return None

    # Step 2: Split into chunks
    print("\n✂️  Step 2: Splitting documents...")
    chunks = split_documents(documents)

    # Step 3: Store in ChromaDB
    print("\n🗄️  Step 3: Creating vector store...")
    vector_store = create_vector_store(chunks)

    print("\n" + "=" * 50)
    print("✅ Ingestion Complete!")
    print("=" * 50)
    return vector_store


if __name__ == "__main__":
    run_ingestion()