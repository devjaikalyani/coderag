"""
reranker.py
-----------
Two-stage retrieval:
  Stage 1: Fast ANN search (FAISS) → top-k candidates
  Stage 2: Cross-encoder reranker → rerank top-k, return top-n

The cross-encoder sees (query, passage) pairs and scores relevance directly,
giving much higher precision than the bi-encoder alone.
"""

from typing import List, Tuple

import torch
from loguru import logger
from sentence_transformers import CrossEncoder

from src.ingestion.chunker import Chunk


class CrossEncoderReranker:
    """
    Wraps a HuggingFace cross-encoder for reranking.
    Default: ms-marco-MiniLM-L-6-v2 (fast, good quality)
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        logger.info(f"Loading reranker: {model_name}")
        self.model = CrossEncoder(
            model_name,
            max_length=512,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[Chunk, float]],
        top_n: int = 5,
    ) -> List[Tuple[Chunk, float]]:
        """
        Score each (query, chunk.text) pair and return top_n sorted descending.

        Args:
            query: User question string
            candidates: List of (Chunk, dense_score) from stage-1
            top_n: How many to return after reranking

        Returns:
            List of (Chunk, rerank_score), best first
        """
        if not candidates:
            return []

        pairs = [(query, c.text) for c, _ in candidates]
        scores = self.model.predict(pairs, show_progress_bar=False)

        reranked = sorted(
            zip([c for c, _ in candidates], scores.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )

        top = reranked[:top_n]
        logger.debug(
            f"Reranked {len(candidates)} → {len(top)} "
            f"(top score={top[0][1]:.3f} if top else 'n/a')"
        )
        return top
