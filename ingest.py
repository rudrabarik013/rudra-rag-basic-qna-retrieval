"""
ingest.py — RAG Ingestion Pipeline
Reads the source document, chunks it, builds TF-IDF embeddings,
and stores everything in a persistent ChromaDB collection.

Run once before starting the chatbot:
    python ingest.py
"""

import os
import re
import pickle
import chromadb  # pyright: ignore[reportMissingImports]
from sklearn.feature_extraction.text import TfidfVectorizer  # pyright: ignore[reportMissingImports]

# ─── Configuration ────────────────────────────────────────────────────────────
DOCUMENT_PATH   = "rag_demo_doc.txt"
CHROMA_PATH     = "./chroma_db"
VECTORIZER_PATH = "./tfidf_vectorizer.pkl"
COLLECTION_NAME = "rag_collection"

CHUNK_SIZE    = 150   # words per chunk
CHUNK_OVERLAP = 30    # overlapping words between consecutive chunks
# ──────────────────────────────────────────────────────────────────────────────


def load_document(path: str) -> str:
    """Load raw text, auto-detecting encoding and cleaning common mojibake."""
    with open(path, "rb") as f:
        raw = f.read()

    # Try windows-1252 first (common for US govt / MS Word export)
    for enc in ("windows-1252", "utf-8", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except (UnicodeDecodeError, LookupError):
            continue
    else:
        text = raw.decode("utf-8", errors="replace")

    # Normalise punctuation to plain ASCII
    text = re.sub(r"[‘’ʼ`´]", "'", text)   # apostrophes
    text = re.sub(r"[“”«»]", '"', text)          # double quotes
    text = re.sub(r"[–—‒]", "-", text)                # dashes
    text = re.sub(r"…", "...", text)                             # ellipsis
    # Remove residual non-ASCII garbage characters
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    # Collapse multiple whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list:
    """Split text into overlapping word-window chunks."""
    words = text.split()
    chunks = []
    step = max(1, chunk_size - overlap)
    i = 0
    while i < len(words):
        chunk = " ".join(words[i: i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
        i += step
    return chunks


def ingest():
    # 1. Load document
    print(f"Loading document from '{DOCUMENT_PATH}' ...")
    if not os.path.exists(DOCUMENT_PATH):
        raise FileNotFoundError(
            f"Document not found: {DOCUMENT_PATH}\n"
            "Make sure rag_demo_doc.txt is in the same folder as ingest.py."
        )
    text = load_document(DOCUMENT_PATH)
    print(f"    Document length: {len(text.split())} words")

    # 2. Chunk
    print(f"\nChunking (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}) ...")
    chunks = chunk_text(text)
    print(f"    Created {len(chunks)} chunks")

    # 3. Build TF-IDF embeddings
    print("\nComputing TF-IDF embeddings ...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),   # unigrams + bigrams for better matching
        sublinear_tf=True,    # apply log(1 + tf) for smoother term weighting
        min_df=1,
    )
    tfidf_matrix = vectorizer.fit_transform(chunks)
    embeddings = tfidf_matrix.toarray().tolist()
    print(f"    Embedding dimension: {len(embeddings[0])}")

    # 4. Persist vectorizer
    print(f"\nSaving TF-IDF vectorizer to '{VECTORIZER_PATH}' ...")
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)

    # 5. Store in ChromaDB
    print(f"\nStoring chunks in ChromaDB at '{CHROMA_PATH}' ...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Drop old collection so re-running ingest is idempotent
    try:
        client.delete_collection(COLLECTION_NAME)
        print("    (deleted existing collection)")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    collection.add(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings,
    )

    print(f"\nIngestion complete! {len(chunks)} chunks stored.")
    print("    You can now run:  streamlit run app.py")


if __name__ == "__main__":
    ingest()
