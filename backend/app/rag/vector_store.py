import os
import pickle
from typing import List, Tuple

import faiss
import numpy as np

# FaissStore is a simple wrapper around a FAISS index and associated metadata (the text chunks). 
# It provides methods to add new embeddings, search for similar chunks, and save/load the index and metadata to disk.
class FaissStore:
    # dim is the dimensionality of the embeddings 
    # index_path and meta_path are file paths where the FAISS index and metadata will be saved/loaded from.
    # index is the in-memory FAISS index object, and 
    # chunks is a list of text chunks corresponding to the embeddings in the index (same order).
    def __init__(self, dim: int, index_path: str, meta_path: str):
        self.dim = dim
        self.index_path = index_path
        self.meta_path = meta_path
        self.index = faiss.IndexFlatIP(dim)  # cosine similarity if vectors are normalized
        self.chunks: List[str] = []

    # Add new embeddings and their corresponding text chunks to the store.
    def add(self, embeddings: np.ndarray, chunks: List[str]) -> None:
        if embeddings.ndim != 2 or embeddings.shape[1] != self.dim:
            raise ValueError(f"Embeddings must be shape (n, {self.dim})")
        self.index.add(embeddings.astype(np.float32))
        self.chunks.extend(chunks)

    # Search the index for the most similar chunks to the query embedding. 
    # Returns a list of (chunk, score) tuples.
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        scores, idxs = self.index.search(query_embedding.astype(np.float32), top_k)
        results = []
        for i, score in zip(idxs[0], scores[0]):
            if i == -1:
                continue
            results.append((self.chunks[i], float(score)))
        return results

    # Save the FAISS index and metadata to disk. 
    # The index is saved in a binary format using Faiss's built-in functions, while the metadata (the list of chunks) is saved using Python's pickle module.
    def save(self) -> None:
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.chunks, f)

    # Load the FAISS index and metadata from disk.
    # This allows the application to persist the vector store across restarts.
    def load(self) -> None:
        self.index = faiss.read_index(self.index_path)
        with open(self.meta_path, "rb") as f:
            self.chunks = pickle.load(f)
