"""
test_retriever.py
-----------------
Tests for the hybrid retriever pipeline.
Uses mock embeddings to avoid loading real models in CI.
"""

import numpy as np
import pytest

from src.ingestion.chunker import Chunk
from src.retrieval.embedder import FAISSIndex, BM25Retriever, reciprocal_rank_fusion


def make_chunks(n: int) -> list:
    return [
        Chunk(
            text=f"def function_{i}():\n    return {i}",
            source=f"module_{i}.py",
            chunk_id=f"module_{i}.py::0",
            language="python",
            start_line=0,
            end_line=5,
        )
        for i in range(n)
    ]


def make_random_embeddings(n: int, dim: int = 64) -> np.ndarray:
    rng = np.random.default_rng(42)
    emb = rng.random((n, dim)).astype(np.float32)
    # L2 normalize
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / norms


class TestFAISSIndex:

    def test_add_and_search(self):
        dim = 64
        chunks = make_chunks(20)
        embeddings = make_random_embeddings(20, dim)

        idx = FAISSIndex(dim=dim)
        idx.add(chunks, embeddings)
        assert idx.index.ntotal == 20

        query = make_random_embeddings(1, dim)[0]
        results = idx.search(query, top_k=5)
        assert len(results) == 5
        assert all(isinstance(r[0], Chunk) for r in results)
        assert all(isinstance(r[1], float) for r in results)

    def test_results_sorted_descending(self):
        dim = 64
        chunks = make_chunks(10)
        embeddings = make_random_embeddings(10, dim)
        idx = FAISSIndex(dim=dim)
        idx.add(chunks, embeddings)
        query = make_random_embeddings(1, dim)[0]
        results = idx.search(query, top_k=10)
        scores = [r[1] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_save_and_load(self, tmp_path):
        dim = 64
        chunks = make_chunks(5)
        embeddings = make_random_embeddings(5, dim)
        idx = FAISSIndex(dim=dim)
        idx.add(chunks, embeddings)
        idx.save(tmp_path / "test_index")

        loaded = FAISSIndex.load(tmp_path / "test_index")
        assert loaded.index.ntotal == 5
        assert len(loaded.chunks) == 5
        assert loaded.chunks[0].text == chunks[0].text

    def test_top_k_capped_by_index_size(self):
        dim = 64
        chunks = make_chunks(3)
        embeddings = make_random_embeddings(3, dim)
        idx = FAISSIndex(dim=dim)
        idx.add(chunks, embeddings)
        query = make_random_embeddings(1, dim)[0]
        results = idx.search(query, top_k=100)
        assert len(results) <= 3


class TestBM25Retriever:

    def test_basic_search(self):
        chunks = [
            Chunk(text="def authenticate_user(username, password):", source="auth.py",
                  chunk_id="auth.py::0", language="python", start_line=0, end_line=1),
            Chunk(text="def create_database_connection(host, port):", source="db.py",
                  chunk_id="db.py::0", language="python", start_line=0, end_line=1),
            Chunk(text="class UserSerializer(BaseSerializer):", source="serializers.py",
                  chunk_id="serializers.py::0", language="python", start_line=0, end_line=1),
        ]
        bm25 = BM25Retriever(chunks)
        results = bm25.search("authenticate user login", top_k=3)
        assert len(results) > 0
        # The auth chunk should score highest for auth-related query
        top_chunk = results[0][0]
        assert "auth" in top_chunk.source or "authenticate" in top_chunk.text.lower()

    def test_returns_scores(self):
        chunks = make_chunks(5)
        bm25 = BM25Retriever(chunks)
        results = bm25.search("function return", top_k=3)
        assert all(isinstance(r[1], float) for r in results)


class TestRRF:

    def test_fusion_merges_lists(self):
        chunks = make_chunks(10)
        dense = [(chunks[i], float(10 - i)) for i in range(5)]
        sparse = [(chunks[i + 3], float(7 - i)) for i in range(5)]
        merged = reciprocal_rank_fusion(dense, sparse)
        # Should include chunks from both lists
        ids = {c.chunk_id for c, _ in merged}
        assert len(ids) > 5

    def test_fusion_scores_sorted(self):
        chunks = make_chunks(10)
        dense = [(chunks[i], float(10 - i)) for i in range(5)]
        sparse = [(chunks[i], float(5 - i)) for i in range(5)]
        merged = reciprocal_rank_fusion(dense, sparse)
        scores = [s for _, s in merged]
        assert scores == sorted(scores, reverse=True)

    def test_overlap_boosts_score(self):
        """A chunk appearing in both lists should score higher than one in only one."""
        chunks = make_chunks(4)
        # chunk[0] appears in both dense and sparse lists
        dense = [(chunks[0], 1.0), (chunks[1], 0.5)]
        sparse = [(chunks[0], 1.0), (chunks[2], 0.5)]
        merged = reciprocal_rank_fusion(dense, sparse)
        score_map = {c.chunk_id: s for c, s in merged}
        # chunk[0] should outscore chunk[1] and chunk[2]
        assert score_map[chunks[0].chunk_id] > score_map.get(chunks[1].chunk_id, 0)
        assert score_map[chunks[0].chunk_id] > score_map.get(chunks[2].chunk_id, 0)
