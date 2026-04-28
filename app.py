"""
app.py — RAG Chatbot (Streamlit UI)
Retrieves relevant document chunks from ChromaDB using TF-IDF similarity,
then sends them as context to Groq LLM for answer generation.

Start with:
    streamlit run app.py
"""

import os
import pickle
import streamlit as st  # pyright: ignore[reportMissingImports]
import chromadb  # pyright: ignore[reportMissingImports]
from groq import Groq  # pyright: ignore[reportMissingImports]
from dotenv import load_dotenv  # pyright: ignore[reportMissingImports]

load_dotenv()

# ─── Configuration ────────────────────────────────────────────────────────────
CHROMA_PATH     = "./chroma_db"
VECTORIZER_PATH = "./tfidf_vectorizer.pkl"
COLLECTION_NAME = "rag_collection"
GROQ_MODEL      = "llama-3.1-8b-instant"   # free Groq model
TOP_K           = 4                   # number of chunks to retrieve
MAX_TOKENS      = 600
TEMPERATURE     = 0.5
# ──────────────────────────────────────────────────────────────────────────────

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .stChatMessage { border-radius: 12px; }
    .source-box {
        background: #f0f4ff;
        border-left: 3px solid #4f8ef7;
        border-radius: 6px;
        padding: 0.6rem 0.8rem;
        margin-bottom: 0.5rem;
        font-size: 0.82rem;
        color: #333;
    }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")
    st.markdown("---")

    # API key input (env var takes priority)
    env_key = os.getenv("GROQ_API_KEY", "")
    if env_key:
        groq_api_key = env_key
        st.success("✅ API key loaded from `.env`")
    else:
        groq_api_key = st.text_input(
            "Groq API Key",
            type="password",
            placeholder="gsk_...",
            help="Get your free key at https://console.groq.com",
        )

    top_k = st.slider("Chunks to retrieve (Top-K)", 1, 8, TOP_K)
    model = st.selectbox(
        "Groq model",
        ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "gemma2-9b-it", "mixtral-8x7b-32768"],
        index=0,
    )

    st.markdown("---")
    st.markdown("**Document:** State of the Union Address")
    st.markdown("**Embeddings:** TF-IDF (local)")
    st.markdown("**Vector DB:** ChromaDB")
    st.markdown("**LLM:** Groq (free tier)")

    if st.button("🗑️ Clear chat history"):
        st.session_state.messages = []
        st.rerun()


# ─── Load RAG resources (cached across reruns) ────────────────────────────────
@st.cache_resource(show_spinner="Loading knowledge base…")
def load_resources():
    if not os.path.exists(VECTORIZER_PATH):
        return None, None
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(COLLECTION_NAME)
    return vectorizer, collection


# ─── RAG helpers ──────────────────────────────────────────────────────────────
def retrieve_chunks(query: str, vectorizer, collection, k: int) -> list[str]:
    """Transform query with TF-IDF and retrieve top-k similar chunks."""
    query_vec = vectorizer.transform([query]).toarray().tolist()
    results = collection.query(query_embeddings=query_vec, n_results=k)
    return results["documents"][0]  # list of chunk strings


SYSTEM_PROMPT = """You are a knowledgeable assistant that answers questions strictly based on the provided context excerpts.
- Be concise and direct.
- If the answer is clearly present in the context, answer confidently.
- If the context does not contain enough information, say: "I don't have enough information in the document to answer that."
- Do NOT make up facts or use outside knowledge."""


def ask_groq(question: str, chunks: list[str], client: Groq,
             model: str, max_tokens: int = MAX_TOKENS) -> str:
    """Build a RAG prompt and call the Groq LLM."""
    context = "\n\n---\n\n".join(
        f"[Excerpt {i+1}]\n{chunk}" for i, chunk in enumerate(chunks)
    )
    user_message = f"""Use the following document excerpts to answer the question.

{context}

Question: {question}"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        max_tokens=max_tokens,
        temperature=TEMPERATURE,
    )
    return response.choices[0].message.content.strip()


# ─── Main UI ──────────────────────────────────────────────────────────────────
st.title("🤖 RAG Chatbot")
st.caption("Ask questions about the **State of the Union Address** — powered by Groq + ChromaDB")

# Guard: API key missing
if not groq_api_key:
    st.warning("👈 Please enter your **Groq API key** in the sidebar to get started.")
    st.stop()

groq_client = Groq(api_key=groq_api_key)

# Guard: ingestion not run yet
vectorizer, collection = load_resources()
if vectorizer is None or collection is None:
    st.error(
        "**Knowledge base not found.**\n\n"
        "Please run the ingestion script first:\n```\npython ingest.py\n```"
    )
    st.stop()

# Chat history init
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📄 Source excerpts used"):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(
                        f'<div class="source-box"><b>Excerpt {i}:</b> {src}</div>',
                        unsafe_allow_html=True,
                    )

# Chat input
if prompt := st.chat_input("Ask a question about the document…"):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating answer…"):
            try:
                chunks = retrieve_chunks(prompt, vectorizer, collection, k=top_k)
                answer = ask_groq(prompt, chunks, groq_client, model=model)
            except Exception as e:
                answer = f"⚠️ Error: {e}"
                chunks = []

        st.markdown(answer)

        if chunks:
            with st.expander("📄 Source excerpts used"):
                for i, src in enumerate(chunks, 1):
                    st.markdown(
                        f'<div class="source-box"><b>Excerpt {i}:</b> {src}</div>',
                        unsafe_allow_html=True,
                    )

    # Save to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": chunks,
    })
