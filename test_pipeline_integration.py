"""
test_pipeline_integration.py
-----------------------------
Integration test for the full CodeRAG pipeline.
Uses mocked Groq and real local models.

Run with: pytest tests/test_pipeline_integration.py -v -s
Note: Requires GROQ_API_KEY to be set for generation tests.
      Retrieval/embedding tests work without it.
"""

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.ingestion.chunker import chunk_text
from src.retrieval.embedder import FAISSIndex, CodeEmbedder
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.retriever import HybridRetriever

SAMPLE_CODE = """
def add(a: int, b: int) -> int:
    \"\"\"Add two integers and return the result.\"\"\"
    return a + b

def subtract(a: int, b: int) -> int:
    \"\"\"Subtract b from a.\"\"\"
    return a - b

class Calculator:
    \"\"\"A simple calculator class.\"\"\"

    def multiply(self, a: int, b: int) -> int:
        return a * b

    def divide(self, a: float, b: float) -> float:
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b
"""


@pytest.fixture(scope="module")
def sample_chunks():
    return chunk_text(SAMPLE_CODE, source="calculator.py")


@pytest.fixture(scope="module")
def embedder():
    """Load real CodeBERT embedder (cached after first load)."""
    return CodeEmbedder(model_name="microsoft/codebert-base")


@pytest.fixture(scope="module")
def built_index(sample_chunks, embedder):
    """Build a real FAISS index from sample chunks."""
    texts = [c.text for c in sample_chunks]
    embeddings = embedder.encode(texts, show_progress=False)
    idx = FAISSIndex(dim=embedder.dim)
    idx.add(sample_chunks, embeddings)
    return idx


class TestEmbedderIntegration:

    def test_encode_returns_correct_shape(self, embedder):
        texts = ["def foo(): pass", "class Bar: pass"]
        emb = embedder.encode(texts, show_progress=False)
        assert emb.shape == (2, embedder.dim)

    def test_embeddings_are_normalized(self, embedder):
        texts = ["hello world"]
        emb = embedder.encode(texts, show_progress=False)
        norm = np.linalg.norm(emb[0])
        assert abs(norm - 1.0) < 1e-5, f"Expected unit norm, got {norm}"

    def test_similar_texts_have_high_similarity(self, embedder):
        texts = [
            "def authenticate_user(username, password): pass",
            "def login_user(user, pwd): pass",
            "def calculate_tax(income, rate): pass",
        ]
        embs = embedder.encode(texts, show_progress=False)
        # Cosine similarity = dot product for normalized vectors
        sim_auth = float(np.dot(embs[0], embs[1]))
        sim_diff = float(np.dot(embs[0], embs[2]))
        assert sim_auth > sim_diff, "Similar functions should have higher similarity"


class TestRetrieverIntegration:

    def test_retrieve_relevant_chunk(self, built_index, embedder):
        reranker = CrossEncoderReranker()
        retriever = HybridRetriever(
            faiss_index=built_index,
            embedder=embedder,
            reranker=reranker,
            top_k_retrieve=10,
            top_k_rerank=3,
        )
        results = retriever.retrieve("How do I add two numbers?")
        assert len(results) > 0
        # The 'add' function should be retrieved
        top_texts = " ".join([r.chunk.text for r in results])
        assert "add" in top_texts.lower() or "calculator" in top_texts.lower()

    def test_retrieve_returns_reranked_results(self, built_index, embedder):
        reranker = CrossEncoderReranker()
        retriever = HybridRetriever(
            faiss_index=built_index,
            embedder=embedder,
            reranker=reranker,
        )
        results = retriever.retrieve("divide by zero error")
        assert len(results) > 0
        # rerank_score should be set
        for r in results:
            assert isinstance(r.rerank_score, float)

    def test_context_string_format(self, built_index, embedder):
        reranker = CrossEncoderReranker()
        retriever = HybridRetriever(
            faiss_index=built_index,
            embedder=embedder,
            reranker=reranker,
        )
        results = retriever.retrieve("multiplication")
        ctx = retriever.get_context_string(results)
        assert "Source" in ctx
        assert "```" in ctx   # code blocks present


class TestGeneratorMocked:
    """Test generation with mocked Groq client (no API key needed)."""

    def test_generate_returns_string(self):
        from src.generation.generator import GroqGenerator
        with patch("src.generation.generator.Groq") as mock_groq:
            mock_client = MagicMock()
            mock_groq.return_value = mock_client
            mock_client.chat.completions.create.return_value = MagicMock(
                choices=[MagicMock(message=MagicMock(content="The add function returns a + b."))]
            )
            gen = GroqGenerator(api_key="fake_key")
            result = gen.generate("How does add work?", context="def add(a,b): return a+b")
            assert isinstance(result, str)
            assert len(result) > 0

    def test_stream_yields_tokens(self):
        from src.generation.generator import GroqGenerator
        with patch("src.generation.generator.Groq") as mock_groq:
            mock_client = MagicMock()
            mock_groq.return_value = mock_client

            chunks = [
                MagicMock(choices=[MagicMock(delta=MagicMock(content="The "))]),
                MagicMock(choices=[MagicMock(delta=MagicMock(content="answer "))]),
                MagicMock(choices=[MagicMock(delta=MagicMock(content="is 42."))]),
            ]
            mock_client.chat.completions.create.return_value = iter(chunks)

            gen = GroqGenerator(api_key="fake_key")
            tokens = list(gen.stream("question", context="context"))
            assert len(tokens) == 3
            assert "".join(tokens) == "The answer is 42."
