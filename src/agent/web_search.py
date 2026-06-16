# src/agent/web_search.py

import os
from typing import List
from dotenv import load_dotenv
from langchain.schema import Document
from tavily import TavilyClient

load_dotenv()


def web_search(query: str, max_results: int = 3) -> List[Document]:
    """
    Falls back to Tavily web search when ChromaDB docs
    are not relevant enough to answer the question.
    
    Returns results as LangChain Document objects
    so the rest of the pipeline can treat them the same way.
    """
    print(f"🌐 Falling back to web search for: '{query}'")

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        print("❌ TAVILY_API_KEY not found in .env")
        return []

    client = TavilyClient(api_key=api_key)

    try:
        response = client.search(
            query=query,
            max_results=max_results,
            include_answer=False,        # We want raw results, not Tavily's answer
            include_raw_content=False,   # Summaries are enough
        )

        docs = []
        for result in response.get("results", []):
            #print(f"🔎 DEBUG Tavily result → title: {result.get('title')!r}, url: {result.get('url')!r}")
            doc = Document(
                page_content=result.get("content", ""),
                metadata={
                    "source": result.get("url", "Web Search"),
                    "title": result.get("title", ""),
                    "type": "web_search"   # Tag so we know it came from web
                }
            )
            docs.append(doc)

        print(f"✅ Web search returned {len(docs)} results")
        return docs

    except Exception as e:
        print(f"❌ Web search error: {e}")
        return []