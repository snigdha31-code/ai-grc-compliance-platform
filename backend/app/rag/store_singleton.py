# Singleton pattern for vector store - ensures only one instance of FaissStore is created and shared across the app.
# This is important because FaissStore holds in-memory index and metadata, and we want to avoid loading multiple copies or having state inconsistencies.

from __future__ import annotations
import os

from app.rag.vector_store import FaissStore

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
INDEX_DIR = os.path.join(DATA_DIR, "faiss")
INDEX_PATH = os.path.join(INDEX_DIR, "policy.index")
META_PATH = os.path.join(INDEX_DIR, "policy_chunks.pkl")

os.makedirs(INDEX_DIR, exist_ok=True)

# all-MiniLM-L6-v2 => 384 dims
store = FaissStore(dim=384, index_path=INDEX_PATH, meta_path=META_PATH)

# Load existing index if present
if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
    store.load()
