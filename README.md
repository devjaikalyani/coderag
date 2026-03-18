# CodeRAG — Chat With Any Codebase

> Point it at any GitHub repo and ask questions in plain English. Get accurate, cited answers powered by CodeBERT, FAISS, and Groq Llama 3.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2-red?logo=pytorch)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-FF4B4B?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Demo

Ask questions like:
- *"What does this project do? Give me a high level overview."*
- *"How does authentication work in this codebase?"*
- *"What API endpoints are available and what do they do?"*
- *"How would I add a new feature following the existing patterns?"*

Every answer cites the exact source file and line numbers.

---

## Architecture

```
GitHub Repo / Local Code
         ↓
  Code-aware Chunker         splits at class/function/section boundaries
         ↓
  CodeBERT Embedder          microsoft/codebert-base (PyTorch, local)
         ↓
  ┌──────────────┐
  │  FAISS Index │  ←── Dense retrieval (cosine similarity)
  │  BM25 Index  │  ←── Sparse retrieval (keyword matching)
  └──────┬───────┘
         ↓
  Reciprocal Rank Fusion     merges both ranked lists
         ↓
  Cross-encoder Reranker     ms-marco-MiniLM precision boost
         ↓
  Llama 3.3 70B Versatile    cited, structured answers
         ↓
  NLI Faithfulness Check     flags hallucinated sentences
         ↓
  FastAPI + Streamlit        REST API + chat UI
         ↓
  MLflow Tracker             logs every query, score, and run
```

---

## Features

| Feature                  | Details                                                            |
| ------------------------ | ------------------------------------------------------------------ |
| **Per-repo isolation**   | Each repo gets its own FAISS index — no cross-contamination        |
| **Hybrid retrieval**     | Dense (CodeBERT) + Sparse (BM25) merged via Reciprocal Rank Fusion |
| **Two-stage reranking**  | Cross-encoder for high-precision results                           |
| **Persistent index**     | Survives restarts — no re-embedding needed                         |
| **Multi-repo switching** | Switch between ingested repos in one click                         |
| **Faithfulness scoring** | NLI model detects hallucinated answers                             |
| **All file types**       | Python, JS, TS, HTML, CSS, Go, Rust, Java, and more                |
| **Streaming UI**         | Clean formatted answers with source citations                      |
| **MLflow tracking**      | Every query logged with scores and parameters                      |
| **Fine-tuning script**   | Train CodeBERT on CodeSearchNet with contrastive loss              |

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/yourusername/coderag
cd coderag

python3.11 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install torch==2.2.2
pip install "numpy<2"
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Add your GROQ_API_KEY — free at https://console.groq.com
```

### 3. Start everything

```bash
chmod +x start.sh
./start.sh
```

Opens:
- **UI** → http://localhost:8501
- **API** → http://localhost:8001
- **API Docs** → http://localhost:8001/docs

### 4. Ingest a repo and ask

Paste any GitHub URL in the sidebar → **Ingest GitHub Repo** → start asking questions.

---

## Tech Stack

| Component           | Technology                                         |
| ------------------- | -------------------------------------------------- |
| Embeddings          | `microsoft/codebert-base` via SentenceTransformers |
| Vector store        | FAISS flat inner-product (cosine, L2-normalised)   |
| Sparse retrieval    | BM25 (`rank-bm25`)                                 |
| Fusion              | Reciprocal Rank Fusion                             |
| Reranker            | `cross-encoder/ms-marco-MiniLM-L-6-v2`             |
| Faithfulness        | `cross-encoder/nli-MiniLM2-L6-H768`                |
| LLM                 | Llama 3.3 70B Versatile via Groq API                           |
| Backend             | FastAPI + Uvicorn                                  |
| Frontend            | Streamlit                                          |
| Experiment tracking | MLflow                                             |
| Fine-tuning         | PyTorch + SentenceTransformers                     |

---

## Project Structure

```
coderag/
├── src/
│   ├── config.py                  # Pydantic settings
│   ├── pipeline.py                # Main orchestrator
│   ├── ingestion/
│   │   ├── chunker.py             # Language-aware chunker
│   │   └── loaders.py             # GitHub, local, text loaders
│   ├── retrieval/
│   │   ├── embedder.py            # CodeBERT + FAISS + BM25 + RRF
│   │   ├── reranker.py            # Cross-encoder reranker
│   │   ├── retriever.py           # Hybrid pipeline
│   │   └── query_rewriter.py      # LLM query expansion
│   ├── generation/
│   │   ├── generator.py           # Groq LLM
│   │   └── faithfulness.py        # NLI hallucination checker
│   ├── evaluation/
│   │   └── evaluator.py           # ROUGE + faithfulness metrics
│   └── api/
│       └── main.py                # FastAPI REST endpoints
├── ui/
│   └── app.py                     # Streamlit chat UI
├── scripts/
│   ├── cli.py                     # CLI tool
│   └── finetune_codebert.py       # Fine-tuning script
├── tests/
├── start.sh                       # Start all services
├── .env.example
└── requirements.txt
```

---

## Fine-tuning CodeBERT

```bash
python scripts/finetune_codebert.py \
  --output models/codebert-finetuned \
  --epochs 3 --batch-size 16

# Then in .env:
# EMBEDDING_MODEL=models/codebert-finetuned
```

---

## Note on Intel Mac / PyTorch 2.2

This project is tested on Intel Mac with PyTorch 2.2.2 and Python 3.11. The `start.sh` script sets all required environment variables automatically (`KMP_DUPLICATE_LIB_OK`, `OMP_NUM_THREADS`, etc.).

---

## License

MIT
