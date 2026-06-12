# test_setup.py
import os
from dotenv import load_dotenv

load_dotenv()

print("=== Testing Setup ===\n")

# Test 1: Environment variables
groq_key = os.getenv("GROQ_API_KEY")
tavily_key = os.getenv("TAVILY_API_KEY")
print(f"✅ Groq API Key loaded: {'Yes' if groq_key else 'NO - Check your .env file'}")
print(f"✅ Tavily API Key loaded: {'Yes' if tavily_key else 'NO - Check your .env file'}")

# Test 2: Core imports
try:
    import langchain
    print(f"✅ LangChain version: {langchain.__version__}")
except ImportError:
    print("❌ LangChain not installed")

try:
    import langgraph
    import importlib.metadata
    version = importlib.metadata.version("langgraph")
    print(f"✅ LangGraph version: {version}")
except ImportError:
    print("❌ LangGraph not installed")

try:
    import chromadb
    print(f"✅ ChromaDB version: {chromadb.__version__}")
except ImportError:
    print("❌ ChromaDB not installed")

try:
    from sentence_transformers import SentenceTransformer
    print("✅ Sentence Transformers imported")
except ImportError:
    print("❌ Sentence Transformers not installed")

try:
    import streamlit
    print(f"✅ Streamlit version: {streamlit.__version__}")
except ImportError:
    print("❌ Streamlit not installed")

try:
    import fastapi
    print(f"✅ FastAPI version: {fastapi.__version__}")
except ImportError:
    print("❌ FastAPI not installed")

# Test 3: Quick Groq connection test
print("\n=== Testing Groq API Connection ===\n")
try:
    from langchain_groq import ChatGroq
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    response = llm.invoke("Say hello in one sentence.")
    print(f"✅ Groq API works! Response: {response.content}")
except Exception as e:
    print(f"❌ Groq API error: {e}")

print("\n=== Setup Complete ===")