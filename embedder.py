"""
embedder.py
-----------
Encodes text chunks using CodeBERT (or any sentence-transformer model).
Builds and persists a FAISS index.
Supports:
  - Dense embedding (CodeBERT)
  - Hybrid search (dense + BM25 sparse, merged with Reciprocal Rank Fusion)
"""

import json
import pickle
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
import torch
from loguru import logger
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.ingestion.chunker import Chunk


class CodeEmbedder:
    """
    Wraps a SentenceTransformer (CodeBERT-based) for encoding.
    Uses mean pooling over token embeddings.
    """

    def __init__(self, model_name: str = "microsoft/codebert-base"):
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dim={self.dim}, device={self.device}")

    def encode(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        """Return L2-normalized embeddings, shape (N, dim)."""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,   # important for cosine similarity via dot product
            device=self.device,
        )
        return embeddings.astype(np.float32)


class FAISSIndex:
    """
    FAISS flat inner-product index (equivalent to cosine when embeddings are L2-normalised).
    Stores chunk metadata alongside the index.
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)   # Inner Product = cosine for unit vectors
        self.chunks: List[Chunk] = []

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def add(self, chunks: List[Chunk], embeddings: np.ndarray):
        assert len(chunks) == len(embeddings), "chunks/embeddings length mismatch"
        self.index.add(embeddings)
        self.chunks.extend(chunks)
        logger.info(f"FAISS index now contains {self.index.ntotal} vectors")

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query_vec: np.ndarray, top_k: int = 20) -> List[Tuple[Chunk, float]]:
        """
        query_vec: shape (1, dim) or (dim,)
        Returns list of (Chunk, score) sorted descending.
        """
        if query_vec.ndim == 1:
            query_vec = query_vec[np.newaxis, :]
        scores, indices = self.index.search(query_vec, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], float(score)))
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "index.faiss"))
        # Serialize chunks (without large tensors)
        chunk_data = [
            {
                "text": c.text,
                "source": c.source,
                "chunk_id": c.chunk_id,
                "language": c.language,
                "start_line": c.start_line,
                "end_line": c.end_line,
                "metadata": c.metadata,
            }
            for c in self.chunks
        ]
        with open(path / "chunks.json", "w") as f:
            json.dump(chunk_data, f, indent=2)
        logger.info(f"Saved FAISS index → {path}")

    @classmethod
    def load(cls, path: str | Path) -> "FAISSIndex":
        path = Path(path)
        index = faiss.read_index(str(path / "index.faiss"))
        with open(path / "chunks.json") as f:
            chunk_data = json.load(f)
        chunks = [Chunk(**d) for d in chunk_data]
        obj = cls.__new__(cls)
        obj.dim = index.d
        obj.index = index
        obj.chunks = chunks
        logger.info(f"Loaded FAISS index ({index.ntotal} vectors) from {path}")
        return obj


class BM25Retriever:
    """Sparse BM25 retriever over chunk texts."""

    def __init__(self, chunks: List[Chunk]):
        tokenized = [c.text.lower().split() for c in chunks]
        self.bm25 = BM25Okapi(tokenized)
        self.chunks = chunks

    def search(self, query: str, top_k: int = 20) -> List[Tuple[Chunk, float]]:
        scores = self.bm25.get_scores(query.lower().split())
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[i], float(scores[i])) for i in top_indices]


def reciprocal_rank_fusion(
    dense_results: List[Tuple[Chunk, float]],
    sparse_results: List[Tuple[Chunk, float]],
    k: int = 60,
    dense_weight: float = 0.6,
    sparse_weight: float = 0.4,
) -> List[Tuple[Chunk, float]]:
    """
    Merge two ranked lists using Reciprocal Rank Fusion.
    RRF score = dense_weight/(k+rank_dense) + sparse_weight/(k+rank_sparse)
    """
    rrf_scores: dict[str, float] = {}
    chunk_map: dict[str, Chunk] = {}

    for rank, (chunk, _) in enumerate(dense_results, start=1):
        rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0) + dense_weight / (k + rank)
        chunk_map[chunk.chunk_id] = chunk

    for rank, (chunk, _) in enumerate(sparse_results, start=1):
        rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0) + sparse_weight / (k + rank)
        chunk_map[chunk.chunk_id] = chunk

    merged = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [(chunk_map[cid], score) for cid, score in merged]
