
import os
from typing import List
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_documents(folder_path: str) -> List[Document]:
    documents = []
    
    if not os.path.exists(folder_path):
        print(f"-- Folder not found: {folder_path}")
        return []

    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        
        try:
            if filename.endswith(".pdf"):
                loader = PyPDFLoader(filepath)
                docs = loader.load()
                documents.extend(docs)
                print(f"-- Loaded PDF: {filename} ({len(docs)} pages)")

            elif filename.endswith(".txt"):
                loader = TextLoader(filepath, encoding="utf-8")
                docs = loader.load()
                documents.extend(docs)
                print(f"-- Loaded TXT: {filename} ({len(docs)} sections)")

        except Exception as e:
            print(f"-- Error loading {filename}: {e}")

    print(f"\n-- Total documents loaded: {len(documents)}")
    return documents


def split_documents(documents: List[Document]) -> List[Document]:
    """
    Splits large documents into smaller chunks.
    This is important because LLMs have a limited context window.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,       
        chunk_overlap=100,    
        length_function=len,
    )

    chunks = splitter.split_documents(documents)
    print(f"--  Split into {len(chunks)} chunks")
    return chunks