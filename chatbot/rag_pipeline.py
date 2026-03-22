"""
rag_pipeline.py
---------------
Retrieval-Augmented Generation (RAG) pipeline for the real estate chatbot.

Steps:
  1. Load project_details.txt
  2. Split into overlapping chunks
  3. Create embeddings using OpenAI or a local fallback
  4. Store in a FAISS vector index
  5. At query time, embed the question and retrieve the top-k similar chunks
  6. Return retrieved context to be injected into the LLM prompt
"""

import os
import logging
import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PROJECT_FILE = DATA_DIR / "project_details.txt"
INDEX_CACHE = DATA_DIR / "faiss_index.pkl"   # cached index to avoid re-indexing

CHUNK_SIZE = 300        # characters per chunk
CHUNK_OVERLAP = 60      # character overlap between consecutive chunks
TOP_K = 4               # number of chunks to retrieve per query


# ---------------------------------------------------------------------------
# Text splitting helpers
# ---------------------------------------------------------------------------

def split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split a long string into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return [c for c in chunks if c]  # remove empty chunks


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def _get_openai_embeddings(texts: List[str]) -> np.ndarray:
    """Embed a list of strings using OpenAI text-embedding-3-small."""
    from openai import OpenAI  # lazy import

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    vectors = [item.embedding for item in response.data]
    return np.array(vectors, dtype=np.float32)


def _get_fallback_embeddings(texts: List[str]) -> np.ndarray:
    """
    Minimal TF-IDF based pseudo-embeddings for use when no OpenAI key is set.
    NOT suitable for production — replace with a proper embedding model.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(max_features=512)
    matrix = vectorizer.fit_transform(texts).toarray().astype(np.float32)
    # Normalise rows to unit length for cosine similarity
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
    return matrix / norms, vectorizer


def get_embeddings(texts: List[str]):
    """Return embeddings using OpenAI if available, else TF-IDF fallback."""
    if os.getenv("OPENAI_API_KEY"):
        logger.info("Using OpenAI embeddings.")
        return _get_openai_embeddings(texts), None
    else:
        logger.warning("OPENAI_API_KEY not set. Using TF-IDF fallback embeddings.")
        return _get_fallback_embeddings(texts)


# ---------------------------------------------------------------------------
# FAISS index
# ---------------------------------------------------------------------------

class FAISSRetriever:
    """Wraps a FAISS flat index for similarity search over text chunks."""

    def __init__(self):
        self.index = None
        self.chunks: List[str] = []
        self.vectorizer = None          # used only for TF-IDF fallback
        self.use_openai: bool = bool(os.getenv("OPENAI_API_KEY"))

    # ------------------------------------------------------------------
    def build(self, chunks: List[str]) -> None:
        """Embed all chunks and build the FAISS index."""
        import faiss

        logger.info(f"Building FAISS index over {len(chunks)} chunks …")
        self.chunks = chunks

        if self.use_openai:
            vectors, _ = get_embeddings(chunks)
            self.vectorizer = None
        else:
            vectors, self.vectorizer = get_embeddings(chunks)

        dim = vectors.shape[1]
        self.index = faiss.IndexFlatIP(dim)   # Inner Product ≈ cosine on unit vectors
        faiss.normalize_L2(vectors)
        self.index.add(vectors)
        logger.info("FAISS index built successfully.")

    # ------------------------------------------------------------------
    def query(self, question: str, top_k: int = TOP_K) -> List[str]:
        """Return the top_k most relevant chunks for the question."""
        import faiss

        if self.index is None:
            raise RuntimeError("Index not built. Call build() first.")

        # Embed the query
        if self.use_openai:
            q_vec = _get_openai_embeddings([question])
        else:
            q_vec = self.vectorizer.transform([question]).toarray().astype(np.float32)

        faiss.normalize_L2(q_vec)
        _, indices = self.index.search(q_vec, top_k)
        results = [self.chunks[i] for i in indices[0] if i < len(self.chunks)]
        return results

    # ------------------------------------------------------------------
    def save(self, path: Path = INDEX_CACHE) -> None:
        """Pickle the entire retriever for reuse between sessions."""
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"FAISS index saved to {path}")

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: Path = INDEX_CACHE) -> "FAISSRetriever":
        """Load a previously saved retriever from disk."""
        with open(path, "rb") as f:
            retriever = pickle.load(f)
        logger.info(f"FAISS index loaded from {path}")
        return retriever


# ---------------------------------------------------------------------------
# Public factory function
# ---------------------------------------------------------------------------

def get_retriever(force_rebuild: bool = False) -> FAISSRetriever:
    """
    Return a ready-to-use FAISSRetriever.
    Loads from cache if available and force_rebuild is False.
    """
    if not force_rebuild and INDEX_CACHE.exists():
        try:
            return FAISSRetriever.load()
        except Exception as e:
            logger.warning(f"Failed to load cached index ({e}). Rebuilding …")

    # Load and chunk project details
    if not PROJECT_FILE.exists():
        raise FileNotFoundError(f"Knowledge base not found: {PROJECT_FILE}")

    text = PROJECT_FILE.read_text(encoding="utf-8")
    chunks = split_text(text)

    retriever = FAISSRetriever()
    retriever.build(chunks)
    retriever.save()
    return retriever


def retrieve_context(question: str, retriever: FAISSRetriever) -> str:
    """
    Retrieve the top matching chunks and return them as a single context string.
    """
    chunks = retriever.query(question)
    return "\n\n".join(chunks)
