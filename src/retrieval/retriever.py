
from typing import List, Tuple
from langchain.schema import Document
from rank_bm25 import BM25Okapi
from src.ingestion.vector_store import load_vector_store


def get_dense_retriever(k: int = 4):
    vector_store = load_vector_store()
    if not vector_store:
        return None

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
    print(f"-- Dense retriever ready (top {k} results)")
    return retriever


class HybridRetriever:
    def __init__(self, chunks: List[Document], k: int = 4):
        self.k = k
        self.chunks = chunks

        self.vector_store = load_vector_store()
        tokenized_corpus = [doc.page_content.lower().split() for doc in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

        print(f"-- Hybrid retriever ready (dense + BM25, top {k} results each)")

    def retrieve(self, query: str) -> List[Document]:
        results = []
        seen_contents = set()

        if self.vector_store:
            dense_docs = self.vector_store.similarity_search(query, k=self.k)
            for doc in dense_docs:
                if doc.page_content not in seen_contents:
                    results.append(doc)
                    seen_contents.add(doc.page_content)

        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
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

        print(f"-- Retrieved {len(results)} unique chunks for query: '{query}'")
        return results