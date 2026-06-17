# 🤖 Agentic RAG System with Self-Correction

A Retrieval-Augmented Generation (RAG) system that doesn't just retrieve and answer — it **grades its own retrieved documents**, **falls back to live web search** when local knowledge is insufficient, and **checks its own answers for faithfulness** before responding, retrying if it isn't grounded in evidence.

Built entirely with **free, open-source tools** — no paid APIs required.

---

## 🧠 What makes this "agentic"

A standard RAG pipeline does: `retrieve → generate`. That's it — no quality checks, no fallback if retrieval fails.

This system instead runs a graph of decisions:

```
                    ┌─────────────┐
                    │   Question  │
                    └──────┬──────┘
                           ▼
                  ┌─────────────────┐
                  │  Rewrite Query   │  (resolves follow-ups like
                  │  (using history) │   "explain more" into a
                  └────────┬─────────┘   standalone question)
                           ▼
                  ┌─────────────────┐
                  │     Retrieve     │  Hybrid search:
                  │ (Dense + BM25)   │  dense vectors + keyword (BM25)
                  └────────┬─────────┘
                           ▼
                  ┌─────────────────┐
                  │  Grade Documents │  LLM judges relevance of
                  │   (relevance)    │  each retrieved chunk
                  └────────┬─────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
      Relevant docs found        No relevant docs
              │                         ▼
              │                 ┌───────────────┐
              │                 │  Web Search    │  Tavily fallback
              │                 │   Fallback     │
              │                 └───────┬────────┘
              │                         │
              └────────────┬────────────┘
                           ▼
                  ┌─────────────────┐
                  │    Generate      │  LLM answers using only
                  │     Answer       │  retrieved context + history
                  └────────┬─────────┘
                           ▼
                  ┌─────────────────┐
                  │  Grade Answer    │  Is the answer actually
                  │  (faithfulness)  │  grounded in the sources?
                  └────────┬─────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
        Faithful → END           Not faithful (retry < 2)
                                          │
                                          ▼
                                  Loop back to Retrieve
```

This loop — grade documents, fall back to search, grade the answer, retry if ungrounded — is what the Self-RAG and Corrective-RAG patterns are about, and it's implemented as an explicit state machine using **LangGraph**.

---

## 🏗️ Architecture

| Layer | Tool | Why |
|---|---|---|
| Orchestration | **LangGraph** | Models the agent as a graph with conditional branches and retry loops |
| LLM | **Groq (Llama 3.3 70B)** | Free tier, extremely fast inference |
| Embeddings | **HuggingFace `all-MiniLM-L6-v2`** | Runs locally, completely free, no API needed |
| Vector store | **ChromaDB** | Local, persistent, zero-cost vector database |
| Sparse retrieval | **BM25 (`rank_bm25`)** | Keyword-based search, complements dense vectors |
| Web fallback | **Tavily** | Free tier (1,000 searches/month) for out-of-scope questions |
| Backend | **FastAPI** | REST API exposing the agent, session-based multi-turn memory |
| Frontend | **Streamlit** | Chat UI with inline source citations and session management |

### Hybrid retrieval

Documents are retrieved two ways and merged:
- **Dense (semantic) search** — finds chunks by *meaning* via embeddings, e.g. a query about "ML" can match a chunk that says "machine learning"
- **BM25 (sparse) search** — finds chunks by *exact keyword* overlap, good for specific terms, names, acronyms

### Self-correction loop

1. Every retrieved chunk is graded individually for relevance by the LLM — irrelevant chunks are filtered out before generation.
2. If **zero** chunks survive grading, the agent automatically falls back to a live Tavily web search instead of answering from nothing.
3. After generating an answer, a second LLM call grades whether the answer is actually faithful to the retrieved context (catches hallucination).
4. If not faithful, the agent loops back to retrieval and tries again (capped at 2 retries to avoid infinite loops).

### Multi-turn memory

Follow-up questions like *"explain more"* or *"give me the procedure to do this"* are not standalone — they depend on what was discussed before. Before retrieval happens, the agent rewrites the raw follow-up into a fully standalone question using the last few turns of conversation history, so retrieval actually searches for something meaningful.

---

## 📁 Project Structure

```
agentic-rag/
├── data/
│   └── documents/          # Your source documents (PDF/TXT) go here
├── src/
│   ├── ingestion/
│   │   ├── document_loader.py   # Loads & chunks documents
│   │   └── vector_store.py      # ChromaDB + HuggingFace embeddings
│   ├── retrieval/
│   │   ├── retriever.py         # Hybrid (dense + BM25) retriever
│   │   └── rag_chain.py         # Prompting + answer generation
│   ├── agent/
│   │   ├── graders.py           # Relevance/faithfulness grading + query rewriting
│   │   ├── web_search.py        # Tavily web search fallback
│   │   └── graph.py             # LangGraph state machine (the agent itself)
│   └── api/
│       └── main.py              # FastAPI REST API
├── app.py                  # Streamlit chat frontend
├── requirements.txt
└── .env                     # API keys (not committed)
```

---

## ⚙️ Setup & Installation

### 1. Clone and create a virtual environment

```bash
git clone <https://github.com/NilabhSharma/Agentic-Rag-System-with-self-correction>
cd agentic-rag
python -m venv venv
```

Activate it:
- **Windows:** `venv\Scripts\activate`
- **Mac/Linux:** `source venv/bin/activate`

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Get free API keys

| Service | Link | Free tier |
|---|---|---|
| Groq | https://console.groq.com | Generous free tier, no card required |
| Tavily | https://tavily.com | 1,000 searches/month free |

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_key_here
TAVILY_API_KEY=your_tavily_key_here
```

### 5. Add documents and ingest them

Drop `.pdf` or `.txt` files into `data/documents/`, then run:

```bash
python -m src.ingestion.ingest
```

This chunks your documents, generates embeddings, and stores them in a local ChromaDB instance at `./chroma_db`.

---

## ▶️ Running the App

You need **two processes running simultaneously**, in two separate terminals.

**Terminal 1 — backend API:**
```bash
python -m uvicorn src.api.main:app --reload
```
Runs at `http://127.0.0.1:8000`. Interactive API docs available at `http://127.0.0.1:8000/docs`.

**Terminal 2 — frontend UI:**
```bash
streamlit run app.py
```
Opens at `http://localhost:8501`.

---

## 💬 Using the App

- Type a question in the chat box. The sidebar shows the current session's topic (auto-generated from your first message).
- Click **➕ New Chat** to reset memory and start a completely fresh conversation context.
- Each answer shows an expandable **📚 Sources** section, and a badge if web search was used or if the agent self-corrected.
- Follow-up questions ("explain more", "what about X") work correctly — the agent resolves them against prior conversation turns before retrieving.

---

## 🔍 Example API Usage

The backend can also be called directly (e.g. for testing or building a different frontend):

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG?", "session_id": null}'
```

Response:
```json
{
  "answer": "RAG (Retrieval Augmented Generation) is...",
  "sources": ["sample.txt"],
  "used_web_search": false,
  "retry_count": 1,
  "session_id": "a1b2c3d4-..."
}
```

Pass the returned `session_id` back in subsequent requests to continue the same conversation with memory.

---

## 🚧 Known Limitations & Future Improvements

- **Chat history is in-memory only** — restarting the FastAPI server wipes all active sessions. A production version would persist sessions in a lightweight database (e.g. SQLite or Redis).
- **No re-ranking step** — after hybrid retrieval, a cross-encoder re-ranker (e.g. `BAAI/bge-reranker-base`) could further improve which chunks reach the LLM.
- **Single web search provider** — Tavily is the only fallback; adding a second provider (e.g. DuckDuckGo via `duckduckgo-search`) would add redundancy if the free tier is exhausted.
- **Retry strategy is fixed at 2 attempts** — could be made adaptive based on how far the faithfulness score is from passing.
- **No evaluation suite** — a small benchmark set of Q&A pairs with expected sources would make it possible to measure retrieval/answer quality changes objectively over time.

---

## 🧰 Tech Stack Summary

`Python` · `LangGraph` · `LangChain` · `ChromaDB` · `Groq (Llama 3.3)` · `HuggingFace Embeddings` · `BM25` · `Tavily` · `FastAPI` · `Streamlit`