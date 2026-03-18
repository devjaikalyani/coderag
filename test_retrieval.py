"""
test_retrieval.py — Tests for FAISS index, BM25, and RRF.
"""
import numpy as np
import pytest
from src.ingestion.chunker import Chunk
from src.retrieval.embedder import FAISSIndex, BM25Retriever, reciprocal_rank_fusion

def make_chunk(text, idx):
    return Chunk(text=text, source=f"f_{idx}.py", chunk_id=f"f_{idx}.py::0",
                 language="python", start_line=0, end_line=10, metadata={})

CHUNKS = [
    make_chunk("def authenticate(user, password): return check_db(user, password)", 0),
    make_chunk("class DatabaseConnection: def connect(self): pass", 1),
    make_chunk("def parse_json(data): return json.loads(data)", 2),
    make_chunk("def send_email(to, subject, body): smtp.send(to, subject, body)", 3),
    make_chunk("class AuthMiddleware: def process(self, request): validate_token(request)", 4),
]
DIM = 64

def rand_vecs(n, dim=DIM):
    v = np.random.randn(n, dim).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)

class TestFAISSIndex:
    def test_add_and_search(self):
        idx = FAISSIndex(dim=DIM)
        emb = rand_vecs(len(CHUNKS))
        idx.add(CHUNKS, emb)
        results = idx.search(rand_vecs(1)[0], top_k=3)
        assert len(results) == 3

    def test_scores_descending(self):
        idx = FAISSIndex(dim=DIM)
        emb = rand_vecs(len(CHUNKS))
        idx.add(CHUNKS, emb)
        results = idx.search(rand_vecs(1)[0], top_k=5)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_exact_match_top(self):
        idx = FAISSIndex(dim=DIM)
        emb = rand_vecs(len(CHUNKS))
        idx.add(CHUNKS, emb)
        results = idx.search(emb[2], top_k=5)
        assert results[0][0].chunk_id == CHUNKS[2].chunk_id
        assert results[0][1] > 0.99

    def test_save_load(self, tmp_path):
        idx = FAISSIndex(dim=DIM)
        emb = rand_vecs(len(CHUNKS))
        idx.add(CHUNKS, emb)
        idx.save(tmp_path)
        loaded = FAISSIndex.load(tmp_path)
        assert loaded.index.ntotal == len(CHUNKS)
        assert len(loaded.chunks) == len(CHUNKS)

class TestBM25:
    def test_relevant_chunk_top(self):
        bm25 = BM25Retriever(CHUNKS)
        results = bm25.search("authenticate user password", top_k=5)
        top_ids = [c.chunk_id for c, _ in results[:2]]
        assert CHUNKS[0].chunk_id in top_ids

    def test_email_query(self):
        bm25 = BM25Retriever(CHUNKS)
        results = bm25.search("send email smtp", top_k=3)
        top_ids = [c.chunk_id for c, _ in results[:2]]
        assert CHUNKS[3].chunk_id in top_ids

class TestRRF:
    def test_no_duplicates(self):
        dense = [(CHUNKS[0], 0.9), (CHUNKS[1], 0.8)]
        sparse = [(CHUNKS[0], 5.0), (CHUNKS[2], 3.0)]
        merged = reciprocal_rank_fusion(dense, sparse)
        ids = [c.chunk_id for c, _ in merged]
        assert len(ids) == len(set(ids))

    def test_scores_descending(self):
        dense = [(CHUNKS[0], 0.9), (CHUNKS[1], 0.8)]
        sparse = [(CHUNKS[1], 5.0), (CHUNKS[2], 3.0)]
        merged = reciprocal_rank_fusion(dense, sparse)
        scores = [s for _, s in merged]
        assert scores == sorted(scores, reverse=True)

    def test_double_rank_boosts_score(self):
        dense  = [(CHUNKS[0], 0.9), (CHUNKS[1], 0.5)]
        sparse = [(CHUNKS[0], 5.0), (CHUNKS[2], 3.0)]
        merged = reciprocal_rank_fusion(dense, sparse)
        # Chunk 0 ranks in both → should be #1
        assert merged[0][0].chunk_id == CHUNKS[0].chunk_id
