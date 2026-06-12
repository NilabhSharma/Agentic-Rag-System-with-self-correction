# src/ingestion/vector_store.py

import os
from typing import List
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


# Where ChromaDB will save its data on your disk
CHROMA_DB_PATH = "./chroma_db"

# The free HuggingFace embedding model we're using
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def get_embeddings():
    """
    Loads the HuggingFace embedding model.
    First run will download the model (~90MB). After that it's cached locally.
    """
    print(f"🔄 Loading embedding model: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},   # Use CPU (no GPU needed)
        encode_kwargs={"normalize_embeddings": True}
    )
    print("✅ Embedding model loaded")
    return embeddings


def create_vector_store(chunks: List[Document]):
    """
    Takes document chunks, converts them to embeddings,
    and stores them in ChromaDB.
    """
    print(f"\n🗄️  Creating vector store at: {CHROMA_DB_PATH}")
    
    embeddings = get_embeddings()
    
    # This does 3 things: embed all chunks, create ChromaDB, save to disk
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH
    )
    
    print(f"✅ Vector store created with {len(chunks)} chunks")
    return vector_store


def load_vector_store():
    """
    Loads an existing ChromaDB from disk.
    Use this after you've already created it once.
    """
    if not os.path.exists(CHROMA_DB_PATH):
        print("❌ No vector store found. Run ingestion first.")
        return None
    
    print(f"📂 Loading existing vector store from: {CHROMA_DB_PATH}")
    embeddings = get_embeddings()
    
    vector_store = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings
    )
    
    count = vector_store._collection.count()
    print(f"✅ Vector store loaded with {count} chunks")
    return vector_store