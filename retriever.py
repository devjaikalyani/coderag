"""
retriever.py
------------
Orchestrates the full two-stage hybrid retrieval pipeline:

  Query
    ↓
  [Dense FAISS search] + [BM25 sparse search]
    ↓
  Reciprocal Rank Fusion
    ↓
  Cross-encoder reranker
    ↓
  Top-N chunks
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from loguru import logger

from src.ingestion.chunker import Chunk
from src.retrieval.embedder import (
    BM25Retriever,
    CodeEmbedder,
    FAISSIndex,
    reciprocal_rank_fusion,
)
from src.retrieval.reranker import CrossEncoderReranker


@dataclass
class RetrievalResult:
    chunk: Chunk
    dense_score: float
    rerank_score: float

    @property
    def source_link(self) -> str:
        """Return a GitHub-style source reference."""
        if "github.com" in self.chunk.source or self.chunk.source.startswith("http"):
            return self.chunk.source
        return f"{self.chunk.source}#L{self.chunk.start_line}-L{self.chunk.end_line}"


class HybridRetriever:
    """
    Full retrieval pipeline. Requires a built FAISSIndex and CodeEmbedder.
    BM25Retriever is built lazily from the index chunks.
    """

    def __init__(
        self,
        faiss_index: FAISSIndex,
        embedder: CodeEmbedder,
        reranker: CrossEncoderReranker,
        top_k_retrieve: int = 20,
        top_k_rerank: int = 5,
    ):
        self.faiss_index = faiss_index
        self.embedder = embedder
        self.reranker = reranker
        self.top_k_retrieve = top_k_retrieve
        self.top_k_rerank = top_k_rerank

        # Build BM25 from the same chunks
        logger.info("Building BM25 index from FAISS chunks…")
        self.bm25 = BM25Retriever(faiss_index.chunks)

    def retrieve(self, query: str) -> List[RetrievalResult]:
        """
        Run the full pipeline and return top-N RetrievalResult objects.
        """
        # Stage 1a: Dense retrieval
        query_vec = self.embedder.encode([query], show_progress=False)[0]
        dense_results = self.faiss_index.search(query_vec, top_k=self.top_k_retrieve)

        # Stage 1b: Sparse BM25 retrieval
        sparse_results = self.bm25.search(query, top_k=self.top_k_retrieve)

        # Merge with RRF
        merged = reciprocal_rank_fusion(dense_results, sparse_results)

        # Stage 2: Cross-encoder reranking
        reranked = self.reranker.rerank(query, merged, top_n=self.top_k_rerank)

        results = []
        dense_score_map = {c.chunk_id: s for c, s in dense_results}
        for chunk, rerank_score in reranked:
            results.append(RetrievalResult(
                chunk=chunk,
                dense_score=dense_score_map.get(chunk.chunk_id, 0.0),
                rerank_score=rerank_score,
            ))

        logger.info(
            f"Retrieved {len(results)} chunks for query: '{query[:60]}…'"
        )
        return results

    def get_context_string(self, results: List[RetrievalResult]) -> str:
        """Format retrieved chunks into a context block for the LLM prompt."""
        parts = []
        for i, r in enumerate(results, 1):
            lang = r.chunk.language or "text"
            parts.append(
                f"### Source {i}: `{r.chunk.source}` "
                f"(lines {r.chunk.start_line}–{r.chunk.end_line})\n"
                f"```{lang}\n{r.chunk.text}\n```"
            )
        return "\n\n".join(parts)
