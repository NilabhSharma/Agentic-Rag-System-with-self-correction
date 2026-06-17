
import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/ask"

st.set_page_config(
    page_title="Agentic RAG Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Agentic RAG Assistant")
st.caption("Self-correcting RAG with web search fallback — powered by LangGraph + Groq")


if "messages" not in st.session_state:
    st.session_state.messages = []  

if "session_id" not in st.session_state:
    st.session_state.session_id = None   

if "session_topic" not in st.session_state:
    st.session_state.session_topic = None   


with st.sidebar:
    st.header("💬 Current Chat")

    if st.session_state.session_id:
        topic = st.session_state.session_topic or "Untitled chat"
        st.success(topic)
    else:
        st.info("Starts when you send your first message")

    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = None
        st.session_state.session_topic = None
        st.rerun()

    st.divider()

    st.header("ℹ️ About")
    st.markdown("""
    This assistant:
    - 🔍 Retrieves from your documents (ChromaDB)
    - ⚖️ Grades document relevance
    - 🌐 Falls back to web search if needed
    - 🔁 Self-corrects if the answer isn't grounded
    """)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        
        if message["role"] == "assistant" and "metadata" in message:
            meta = message["metadata"]

            if meta.get("sources"):
                with st.expander("📚 Sources"):
                    for src in meta["sources"]:
                        st.markdown(f"- `{src}`")

    
            badges = []
            if meta.get("used_web_search"):
                badges.append("🌐 Used web search")
            if meta.get("retry_count", 0) > 1:
                badges.append(f"🔁 Self-corrected ({meta['retry_count']} attempts)")

            if badges:
                st.caption(" | ".join(badges))


if prompt := st.chat_input("Ask a question..."):

    if st.session_state.session_topic is None:
        
        st.session_state.session_topic = prompt if len(prompt) <= 40 else prompt[:40] + "..."

    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Thinking... (retrieving, grading, generating)"):
            try:
                response = requests.post(
                    API_URL,
                    json={
                        "question": prompt,
                        "session_id": st.session_state.session_id
                    },
                    timeout=60
                )

                if response.status_code == 200:
                    data = response.json()

                    answer = data["answer"]
                    st.session_state.session_id = data["session_id"]

                    
                    st.markdown(answer)
                    if data.get("sources"):
                        with st.expander("📚 Sources"):
                            for src in data["sources"]:
                                st.markdown(f"- `{src}`")

                    badges = []
                    if data.get("used_web_search"):
                        badges.append("🌐 Used web search")
                    if data.get("retry_count", 0) > 1:
                        badges.append(f"🔁 Self-corrected ({data['retry_count']} attempts)")
                    if badges:
                        st.caption(" | ".join(badges))

                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "metadata": {
                            "sources": data.get("sources", []),
                            "used_web_search": data.get("used_web_search", False),
                            "retry_count": data.get("retry_count", 0)
                        }
                    })

                    
                    st.rerun()

                else:
                    error_msg = f"❌ API Error: {response.status_code}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

            except requests.exceptions.ConnectionError:
                error_msg = "❌ Cannot connect to API. Make sure it's running: `uvicorn src.api.main:app --reload`"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})