"""FAISS helpers for cosine-similarity search over gallery embeddings."""

from __future__ import annotations

from typing import Tuple

import faiss
import numpy as np


def build_faiss(vectors: np.ndarray) -> faiss.IndexFlatIP:
    """Build an inner-product FAISS index for normalized vectors (cosine similarity)."""
    if vectors.ndim != 2:
        raise ValueError("Expected vectors with shape (n, dim)")
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-3):
        raise ValueError("Input vectors must be L2-normalized before indexing.")
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors.astype(np.float32))
    return index


def search_faiss(index: faiss.IndexFlatIP, query_vecs: np.ndarray, topk: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """Search normalized query vectors against a FAISS index."""
    if query_vecs.ndim != 2:
        raise ValueError("Query vectors must have shape (n, dim)")
    if not np.allclose(np.linalg.norm(query_vecs, axis=1), 1.0, atol=1e-3):
        raise ValueError("Query vectors must be L2-normalized.")
    scores, idxs = index.search(query_vecs.astype(np.float32), topk)
    return scores, idxs
