from typing import List

import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from app.rag.vector_store import FaissStore

# This module contains functions to ingest policy documents (PDFs) into the FAISS vector store.
# It includes loading the PDF, chunking the text, generating embeddings, and adding them to the store.
# The main function is ingest_pdf_into_faiss, which orchestrates these steps.

# We use the "all-MiniLM-L6-v2" model from Sentence Transformers, which produces 384-dimensional embeddings.
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_pdf_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    pages = []
    for p in reader.pages:
        pages.append(p.extract_text() or "")
    return "\n".join(pages)


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    
    text = " ".join(text.split())
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start = end - overlap
        if start < 0:
            start = 0
        if end == len(text):
            break
    return [c for c in chunks if c.strip()]

# Normalize embeddings to unit length.
# This is important for cosine similarity search in Faiss.
def normalize(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / norms

# Main function to ingest a PDF into the FAISS store.
# It loads the PDF text, chunks it, generates embeddings, normalizes them, and adds them to the store. 
# Finally, it saves the updated index and metadata to disk.
def ingest_pdf_into_faiss(pdf_path: str, store: FaissStore) -> int:
    text = load_pdf_text(pdf_path)
    chunks = chunk_text(text)

    model = SentenceTransformer(EMBED_MODEL_NAME)
    embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
    embeddings = normalize(embeddings)

    store.add(embeddings, chunks)
    store.save()
    return len(chunks)
