"""
pipeline.py
-----------
CodeRAG pipeline with PER-REPO isolated indexes.

Each ingested source gets its own FAISS index stored separately:
  data/processed/indexes/{repo_key}/index.faiss
  data/processed/indexes/{repo_key}/chunks.json

The active repo is selected via switch_repo(key).
Registry tracks all ingested repos with metadata.
"""

import json
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
from loguru import logger
from tqdm import tqdm

from src.config import settings
from src.generation.faithfulness import FaithfulnessChecker, FaithfulnessResult
from src.generation.generator import GroqGenerator
from src.ingestion.chunker import Chunk
from src.ingestion.loaders import load_github, load_local, load_text
from src.retrieval.embedder import CodeEmbedder, FAISSIndex
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.retriever import HybridRetriever, RetrievalResult

REGISTRY_PATH = Path("data/processed/registry.json")
INDEXES_DIR   = Path("data/processed/indexes")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_key(s: str) -> str:
    """Turn any string into a safe directory name."""
    s = re.sub(r"https?://", "", s)
    s = re.sub(r"[^\w\-]", "_", s)
    return s[:80]


def _load_registry() -> Dict:
    if REGISTRY_PATH.exists():
        try:
            return json.loads(REGISTRY_PATH.read_text())
        except Exception:
            pass
    return {"sources": [], "active_key": None}


def _save_registry(registry: Dict):
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2))


@dataclass
class RAGResponse:
    query: str
    answer: str
    sources: List[RetrievalResult]
    faithfulness: Optional[FaithfulnessResult] = None
    mlflow_run_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class CodeRAGPipeline:
    """CodeRAG pipeline — each repo has its own isolated FAISS index."""

    def __init__(
        self,
        embedder: CodeEmbedder,
        reranker: CrossEncoderReranker,
        generator: GroqGenerator,
        faithfulness_checker: FaithfulnessChecker,
    ):
        self.embedder = embedder
        self.reranker = reranker
        self.generator = generator
        self.faithfulness_checker = faithfulness_checker
        self._registry = _load_registry()
        self._history: List[dict] = []

        # Active repo state
        self.active_key: Optional[str] = None
        self.faiss_index: Optional[FAISSIndex] = None
        self.retriever: Optional[HybridRetriever] = None

        # Auto-load last active repo
        active = self._registry.get("active_key")
        if active:
            self._load_repo_index(active)

    @classmethod
    def from_config(cls) -> "CodeRAGPipeline":
        embedder = CodeEmbedder(model_name=settings.embedding_model)
        reranker = CrossEncoderReranker(model_name=settings.reranker_model)
        generator = GroqGenerator(
            api_key=settings.groq_api_key,
            model=settings.groq_model,
        )
        checker = FaithfulnessChecker(
            model_name="cross-encoder/nli-MiniLM2-L6-H768",
            threshold=settings.faithfulness_threshold,
        )
        return cls(embedder, reranker, generator, checker)

    # ------------------------------------------------------------------
    # Registry helpers
    # ------------------------------------------------------------------

    def get_ingested_sources(self) -> List[dict]:
        return self._registry.get("sources", [])

    def is_already_ingested(self, key: str) -> bool:
        return any(s["key"] == key for s in self._registry.get("sources", []))

    def _register_source(self, key: str, source_type: str, identifier: str,
                          display_name: str, chunk_count: int):
        sources = self._registry.get("sources", [])
        entry = {
            "key": key,
            "type": source_type,
            "identifier": identifier,
            "display_name": display_name,
            "chunk_count": chunk_count,
            "ingested_at": datetime.now().isoformat(timespec="seconds"),
        }
        # Update if exists
        for i, s in enumerate(sources):
            if s["key"] == key:
                sources[i] = entry
                self._registry["sources"] = sources
                _save_registry(self._registry)
                return
        sources.append(entry)
        self._registry["sources"] = sources
        _save_registry(self._registry)

    def _index_path(self, key: str) -> Path:
        return INDEXES_DIR / key

    def _load_repo_index(self, key: str) -> bool:
        """Load a specific repo's index. Returns True if successful."""
        path = self._index_path(key)
        if not path.exists():
            logger.warning(f"Index not found for key: {key}")
            return False
        try:
            self.faiss_index = FAISSIndex.load(path)
            self.retriever = HybridRetriever(
                faiss_index=self.faiss_index,
                embedder=self.embedder,
                reranker=self.reranker,
                top_k_retrieve=settings.top_k_retrieve,
                top_k_rerank=settings.top_k_rerank,
            )
            self.active_key = key
            self._history = []   # Clear history when switching repos
            logger.info(f"Loaded repo '{key}' ({self.faiss_index.index.ntotal} chunks)")
            return True
        except Exception as e:
            logger.error(f"Failed to load index for {key}: {e}")
            return False

    # ------------------------------------------------------------------
    # Switch active repo
    # ------------------------------------------------------------------

    def switch_repo(self, key: str) -> dict:
        """Switch the active repo by key. Returns status dict."""
        if not self.is_already_ingested(key):
            return {"status": "error", "reason": f"Repo '{key}' not found in registry"}
        if self.active_key == key:
            return {"status": "ok", "message": "Already active", "key": key}
        success = self._load_repo_index(key)
        if success:
            self._registry["active_key"] = key
            _save_registry(self._registry)
            source = next((s for s in self.get_ingested_sources() if s["key"] == key), {})
            return {
                "status": "ok",
                "key": key,
                "display_name": source.get("display_name", key),
                "total_chunks": self.faiss_index.index.ntotal,
            }
        return {"status": "error", "reason": f"Could not load index for '{key}'"}

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def _build_repo_index(self, chunks: List[Chunk], key: str):
        """Build a fresh isolated index for a repo and make it active."""
        logger.info(f"Embedding {len(chunks)} chunks for '{key}'…")
        texts = [c.text for c in chunks]
        all_embeddings = []
        for i in tqdm(range(0, len(texts), 32), desc="Embedding"):
            batch = texts[i:i + 32]
            emb = self.embedder.encode(batch, show_progress=False)
            all_embeddings.append(emb)
        embeddings = np.vstack(all_embeddings)

        index = FAISSIndex(dim=self.embedder.dim)
        index.add(chunks, embeddings)
        index.save(self._index_path(key))

        # Make it active
        self.faiss_index = index
        self.retriever = HybridRetriever(
            faiss_index=index,
            embedder=self.embedder,
            reranker=self.reranker,
            top_k_retrieve=settings.top_k_retrieve,
            top_k_rerank=settings.top_k_rerank,
        )
        self.active_key = key
        self._history = []
        self._registry["active_key"] = key
        _save_registry(self._registry)
        logger.info(f"✅ Built isolated index for '{key}' ({index.index.ntotal} chunks)")

    def ingest_github(self, url: str, branch: str = "main") -> dict:
        identifier = f"{url}@{branch}"
        key = _safe_key(identifier)
        repo_name = url.rstrip("/").split("/")[-1]
        display_name = f"{repo_name} ({branch})"

        if self.is_already_ingested(key):
            # Just switch to it
            self.switch_repo(key)
            total = self.faiss_index.index.ntotal if self.faiss_index else 0
            return {
                "status": "skipped",
                "reason": f"Already ingested. Switched to '{display_name}'.",
                "total_chunks": total,
                "key": key,
                "display_name": display_name,
            }

        chunks = load_github(url, branch=branch,
                             max_tokens=settings.chunk_size,
                             overlap=settings.chunk_overlap)
        self._build_repo_index(chunks, key)
        self._register_source(key, "github", identifier, display_name, len(chunks))
        return {
            "status": "ok",
            "new_chunks": len(chunks),
            "total_chunks": self.faiss_index.index.ntotal,
            "key": key,
            "display_name": display_name,
        }

    def ingest_directory(self, path: str) -> dict:
        key = _safe_key(f"local::{path}")
        display_name = Path(path).name

        if self.is_already_ingested(key):
            self.switch_repo(key)
            total = self.faiss_index.index.ntotal if self.faiss_index else 0
            return {
                "status": "skipped",
                "reason": f"Already ingested. Switched to '{display_name}'.",
                "total_chunks": total,
                "key": key,
                "display_name": display_name,
            }

        chunks = load_local(path, max_tokens=settings.chunk_size,
                            overlap=settings.chunk_overlap)
        self._build_repo_index(chunks, key)
        self._register_source(key, "local", path, display_name, len(chunks))
        return {
            "status": "ok",
            "new_chunks": len(chunks),
            "total_chunks": self.faiss_index.index.ntotal,
            "key": key,
            "display_name": display_name,
        }

    def ingest_text(self, text: str, source_name: str = "inline") -> dict:
        key = _safe_key(f"text::{source_name}")
        if self.is_already_ingested(key):
            self.switch_repo(key)
            total = self.faiss_index.index.ntotal if self.faiss_index else 0
            return {"status": "skipped", "reason": "Already ingested.",
                    "total_chunks": total, "key": key, "display_name": source_name}

        chunks = load_text(text, source_name=source_name,
                           max_tokens=settings.chunk_size, overlap=settings.chunk_overlap)
        self._build_repo_index(chunks, key)
        self._register_source(key, "text", source_name, source_name, len(chunks))
        return {"status": "ok", "new_chunks": len(chunks),
                "total_chunks": self.faiss_index.index.ntotal,
                "key": key, "display_name": source_name}

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def delete_repo(self, key: str):
        """Delete a specific repo's index and registry entry."""
        path = self._index_path(key)
        if path.exists():
            shutil.rmtree(path)
        sources = [s for s in self._registry.get("sources", []) if s["key"] != key]
        self._registry["sources"] = sources
        if self._registry.get("active_key") == key:
            self._registry["active_key"] = None
            self.faiss_index = None
            self.retriever = None
            self.active_key = None
        _save_registry(self._registry)
        logger.info(f"Deleted repo '{key}'")

    def clear_all(self):
        """Wipe everything."""
        if INDEXES_DIR.exists():
            shutil.rmtree(INDEXES_DIR)
        self._registry = {"sources": [], "active_key": None}
        _save_registry(self._registry)
        self.faiss_index = None
        self.retriever = None
        self.active_key = None
        self._history = []

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, question: str, use_history: bool = True,
              check_faithfulness: bool = True) -> RAGResponse:
        if self.retriever is None:
            raise RuntimeError("No repo selected. Ingest or switch to a repo first.")

        results = self.retriever.retrieve(question)
        context = self.retriever.get_context_string(results)
        history = self._history if use_history else None
        answer = self.generator.generate(question, context, history)

        faith_result = None
        if check_faithfulness:
            try:
                faith_result = self.faithfulness_checker.check(answer, context)
            except Exception as e:
                logger.warning(f"Faithfulness check failed: {e}")

        self._history.append({"role": "user", "content": question})
        self._history.append({"role": "assistant", "content": answer})
        if len(self._history) > 20:
            self._history = self._history[-20:]

        return RAGResponse(query=question, answer=answer,
                           sources=results, faithfulness=faith_result)

    def stream_query(self, question: str):
        if self.retriever is None:
            raise RuntimeError("No repo selected.")
        results = self.retriever.retrieve(question)
        context = self.retriever.get_context_string(results)
        yield from self.generator.stream(question, context, self._history or None)

    def clear_history(self):
        self._history = []