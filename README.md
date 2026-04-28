# CodeRAG

A retrieval-augmented generation (RAG) system for querying GitHub repositories in plain English. Ask questions about any codebase and receive cited answers with file paths and line numbers.

---

## How It Works

```
User Query
  ↓ QueryRewriter — expands abbreviations, adds synonyms
  ↓ Hybrid Retriever
      ├─ Dense: CodeBERT embeddings + FAISS cosine similarity
      ├─ Sparse: BM25 keyword matching
      └─ Merge: Reciprocal Rank Fusion (top 20 candidates)
  ↓ CrossEncoder Reranker — narrows to top 5
  ↓ Groq Llama 3.3 70B — generates answer with source citations
  ↓ NLI Faithfulness Checker — validates answer is grounded in context
  → Response with answer, cited sources, and confidence score
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Llama 3.3 70B via Groq API |
| Embeddings | CodeBERT (`microsoft/codebert-base`) |
| Vector Search | FAISS (flat index, cosine similarity) |
| Keyword Search | BM25 (`rank-bm25`) |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Faithfulness | `cross-encoder/nli-MiniLM2-L6-H768` |
| API | FastAPI + Uvicorn |
| UI | Streamlit |
| Experiment Tracking | MLflow |
| Evaluation | RAGAS + ROUGE |

---

## Project Structure

```
coderag/
├── agents/                          # Core RAG pipeline modules
│   ├── chunker.py                   # Semantic code chunking
│   ├── embedder.py                  # CodeBERT embeddings + FAISS index
│   ├── retriever.py                 # Hybrid retrieval pipeline
│   ├── reranker.py                  # Cross-encoder reranking
│   ├── generator.py                 # Groq LLM integration
│   ├── query_rewriter.py            # Query expansion
│   ├── loaders.py                   # GitHub / local / text loaders
│   ├── faithfulness.py              # NLI answer validation
│   ├── evaluator.py                 # RAG evaluation metrics
│   └── finetune_codebert.py         # CodeBERT fine-tuning script
│
├── tests/                           # Test suite
│   ├── test_chunker.py
│   ├── test_retrieval.py
│   ├── test_retriever.py
│   └── test_pipeline_integration.py
│
├── app.py                           # Streamlit UI
├── main.py                          # FastAPI backend
├── cli.py                           # CLI interface
├── config.py                        # Configuration (Pydantic Settings)
├── pipeline.py                      # CodeRAGPipeline orchestrator
├── start.sh                         # Launch API + UI together
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── pyproject.toml
└── env.example
```

---

## Setup

### Prerequisites

- Python 3.11+
- A free [Groq API key](https://console.groq.com)

### Install

```bash
git clone https://github.com/devjaikalyani/coderag.git
cd coderag

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp env.example .env
# Edit .env and set your GROQ_API_KEY
```

### Run

```bash
# Start API (port 8001) + UI (port 8501) together
bash start.sh
```

Or run services individually:

```bash
# API only
uvicorn main:app --host 0.0.0.0 --port 8001 --reload

# UI only
streamlit run app.py --server.port 8501
```

### Docker

```bash
docker-compose up --build
```

Services started by `docker-compose`:
- **API** — `http://localhost:8000`
- **UI** — `http://localhost:8501`
- **MLflow** — `http://localhost:5000`

---

## Configuration

All settings are read from environment variables (or `.env`):

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | **required** | Groq API key |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | LLM model |
| `EMBEDDING_MODEL` | `microsoft/codebert-base` | Embedding model |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranker model |
| `FAISS_INDEX_PATH` | `data/processed/indexes` | Index storage path |
| `CHUNK_SIZE` | `512` | Tokens per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between chunks |
| `TOP_K_RETRIEVE` | `20` | Candidates from hybrid search |
| `TOP_K_RERANK` | `5` | Final results after reranking |
| `FAITHFULNESS_THRESHOLD` | `0.5` | Minimum NLI score to flag unfaithful answers |
| `API_HOST` | `0.0.0.0` | API bind address |
| `API_PORT` | `8001` | API port |
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | MLflow server |
| `MLFLOW_EXPERIMENT_NAME` | `coderag` | MLflow experiment name |

---

## API Reference

Base URL: `http://localhost:8001`

### Ingestion

| Method | Endpoint | Body | Description |
|---|---|---|---|
| `POST` | `/ingest/github` | `{"repo_url": "...", "branch": "main"}` | Clone and index a GitHub repo |
| `POST` | `/ingest/directory` | `{"path": "/path/to/code"}` | Index a local directory |
| `POST` | `/ingest/text` | `{"text": "..."}` | Index a raw code snippet |
| `POST` | `/ingest/file` | Form upload | Index an uploaded file |

### Querying

| Method | Endpoint | Body / Params | Description |
|---|---|---|---|
| `POST` | `/query` | `{"query": "...", "check_faithfulness": true}` | Ask a question, get cited answer |
| `GET` | `/query/stream` | `?query=...&stream=true` | Streaming response (server-sent events) |
| `DELETE` | `/history` | — | Clear conversation history |

### Repository Management

| Method | Endpoint | Body | Description |
|---|---|---|---|
| `POST` | `/repo/switch` | `{"repo_key": "..."}` | Switch active repository |
| `DELETE` | `/repo/{repo_key}` | — | Delete a repository and its index |

### Status

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | `{status, num_repos, total_chunks, active_repo}` |
| `GET` | `/index/stats` | Detailed index statistics |
| `DELETE` | `/index` | Clear all indices |

### Query Response Schema

```json
{
  "answer": "...",
  "sources": [
    {
      "file": "src/pipeline.py",
      "line_start": 42,
      "line_end": 68,
      "language": "python",
      "relevance_score": 0.91
    }
  ],
  "faithfulness_score": 0.87
}
```

---

## CLI

```bash
# Ingest a GitHub repository
python cli.py ingest github https://github.com/owner/repo

# Ingest with a specific branch
python cli.py ingest github https://github.com/owner/repo --branch develop

# Ingest a local directory
python cli.py ingest directory /path/to/project

# Ask a one-off question
python cli.py query "How does the authentication middleware work?"

# Ask with faithfulness check
python cli.py query "What does the pipeline do?" --faithfulness

# Start interactive chat session
python cli.py chat
```

Interactive chat commands: `clear` to reset history, `exit` or `quit` to stop.

---

## Architecture Details

### Per-Repository Index Isolation

Each ingested source gets its own FAISS index stored at `data/processed/indexes/{repo_key}/`. A registry tracks all ingested repositories with metadata. Switching between repositories is instant — no re-indexing required.

### Semantic Code Chunking

- Python, JavaScript, Java, Go, Rust, and other languages are split at function/class boundaries to preserve semantic units.
- HTML, CSS, and JSON are kept whole or split on blank lines to preserve markup integrity.
- Default: 512-token chunks with 64-token overlap. Both are configurable.
- Chunk metadata: `source`, `language`, `start_line`, `end_line`, `token_count`.

### Hybrid Retrieval

Dense (CodeBERT) and sparse (BM25) results are merged using **Reciprocal Rank Fusion**:

```
score = Σ 1 / (k + rank)
```

No manual weight tuning required. The cross-encoder reranker then scores all candidates directly against the query for the final top-5.

### Faithfulness Checking

The NLI model breaks the generated answer into individual sentences and scores each against the retrieved context. Returns an overall score (0.0–1.0) and a per-sentence breakdown. Answers below `FAITHFULNESS_THRESHOLD` are flagged.

### Conversation Memory

Up to 20 messages (10 turns) are retained per session. History is automatically cleared when switching repositories or via `DELETE /history`.

---

## Evaluation

```bash
# Run evaluation against a JSON dataset
python cli.py eval eval_dataset.json
```

Dataset format:

```json
[
  {
    "question": "What does the ingest function do?",
    "answer": "It clones the repository and indexes the source files.",
    "context": ["...relevant chunk..."]
  }
]
```

Metrics logged to MLflow:
- **ROUGE-L** — answer similarity to reference
- **Faithfulness score** — NLI grounding of generated answers
- **Context hit rate** — retrieval accuracy

---

## Fine-tuning CodeBERT

To fine-tune the embedding model on your own code-documentation pairs:

```bash
python finetune_codebert.py
```

Uses `MultipleNegativesRankingLoss` with hard negatives from the batch. Defaults: 3 epochs, batch size 16, learning rate 2e-5, validated on 2,000 CodeSearchNet Python samples.

---

## Testing

```bash
# Run all tests
pytest -v

# Skip slow integration tests
pytest -m "not integration" -v

# Run with coverage report
pytest --cov=. --cov-fail-under=70
```

Test coverage minimum: **70%**.

| File | Coverage |
|---|---|
| `test_chunker.py` | Language detection, token limits, field validation |
| `test_retrieval.py` | FAISS add/search, BM25 ranking, RRF merging |
| `test_retriever.py` | Hybrid retriever pipeline and result formatting |
| `test_pipeline_integration.py` | End-to-end with real models (Groq mocked) |

---

## License

MIT
