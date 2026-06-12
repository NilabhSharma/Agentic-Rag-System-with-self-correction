# src/retrieval/retriever.py

from typing import List, Tuple
from langchain.schema import Document
from rank_bm25 import BM25Okapi
from src.ingestion.vector_store import load_vector_store


def get_dense_retriever(k: int = 4):
    """
    Dense retriever — uses vector embeddings (semantic search).
    Finds documents by MEANING, not exact words.
    k = number of documents to retrieve
    """
    vector_store = load_vector_store()
    if not vector_store:
        return None

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
    print(f"✅ Dense retriever ready (top {k} results)")
    return retriever


class HybridRetriever:
    """
    Hybrid Retriever = Dense (semantic) + BM25 (keyword) search combined.
    
    Why hybrid?
    - Dense search is great for meaning: "what is ML?" matches "machine learning explanation"
    - BM25 is great for exact terms: searching "ChromaDB" finds exact mentions
    - Together they cover more ground and improve accuracy
    """

    def __init__(self, chunks: List[Document], k: int = 4):
        self.k = k
        self.chunks = chunks

        # --- Dense retriever setup ---
        self.vector_store = load_vector_store()

        # --- BM25 sparse retriever setup ---
        # Tokenize each chunk's text into words for BM25
        tokenized_corpus = [doc.page_content.lower().split() for doc in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

        print(f"✅ Hybrid retriever ready (dense + BM25, top {k} results each)")

    def retrieve(self, query: str) -> List[Document]:
        """
        Retrieves documents using both methods and merges results.
        Deduplicates so the same chunk doesn't appear twice.
        """
        results = []
        seen_contents = set()

        # 1. Dense (semantic) search
        if self.vector_store:
            dense_docs = self.vector_store.similarity_search(query, k=self.k)
            for doc in dense_docs:
                if doc.page_content not in seen_contents:
                    results.append(doc)
                    seen_contents.add(doc.page_content)

        # 2. BM25 (keyword) search
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices sorted by score
        top_indices = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True
        )[:self.k]

        for idx in top_indices:
            doc = self.chunks[idx]
            if doc.page_content not in seen_contents:
                results.append(doc)
                seen_contents.add(doc.page_content)

        print(f"🔍 Retrieved {len(results)} unique chunks for query: '{query}'")
        return results