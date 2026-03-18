"""
main.py — FastAPI backend
"""

from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel

from src.pipeline import CodeRAGPipeline

app = FastAPI(title="CodeRAG API", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

_pipeline: Optional[CodeRAGPipeline] = None


def get_pipeline() -> CodeRAGPipeline:
    global _pipeline
    if _pipeline is None:
        logger.info("Initializing CodeRAG pipeline…")
        _pipeline = CodeRAGPipeline.from_config()
    return _pipeline


# ── Models ────────────────────────────────────────────────────────────────

class IngestGitHubRequest(BaseModel):
    url: str
    branch: str = "main"

class IngestDirectoryRequest(BaseModel):
    path: str

class IngestTextRequest(BaseModel):
    text: str
    source_name: str = "inline"

class SwitchRepoRequest(BaseModel):
    key: str

class QueryRequest(BaseModel):
    question: str
    check_faithfulness: bool = True

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[dict]
    faithfulness_score: Optional[float]
    is_faithful: Optional[bool]
    active_repo: Optional[str]


# ── Routes ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    p = get_pipeline()
    return {
        "status": "ok",
        "active_key": p.active_key,
        "total_chunks": p.faiss_index.index.ntotal if p.faiss_index else 0,
    }


@app.get("/index/stats")
def index_stats():
    p = get_pipeline()
    return {
        "total_chunks": p.faiss_index.index.ntotal if p.faiss_index else 0,
        "index_loaded": p.faiss_index is not None,
        "active_key": p.active_key,
        "ingested_sources": p.get_ingested_sources(),
    }


@app.post("/repo/switch")
def switch_repo(req: SwitchRepoRequest):
    p = get_pipeline()
    result = p.switch_repo(req.key)
    if result["status"] == "error":
        raise HTTPException(status_code=404, detail=result["reason"])
    return result


@app.delete("/repo/{key}")
def delete_repo(key: str):
    p = get_pipeline()
    p.delete_repo(key)
    return {"status": "ok", "message": f"Deleted repo '{key}'"}


@app.post("/ingest/github")
def ingest_github(req: IngestGitHubRequest):
    p = get_pipeline()
    try:
        return p.ingest_github(req.url, branch=req.branch)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/directory")
def ingest_directory(req: IngestDirectoryRequest):
    if not Path(req.path).exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {req.path}")
    p = get_pipeline()
    return p.ingest_directory(req.path)


@app.post("/ingest/text")
def ingest_text(req: IngestTextRequest):
    return get_pipeline().ingest_text(req.text, source_name=req.source_name)


@app.post("/ingest/file")
async def ingest_file(file: UploadFile = File(...)):
    content = await file.read()
    text = content.decode("utf-8", errors="ignore")
    return get_pipeline().ingest_text(text, source_name=file.filename or "upload")


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    p = get_pipeline()
    if p.retriever is None:
        raise HTTPException(status_code=400, detail="No repo selected. Ingest or switch to a repo first.")
    response = p.query(req.question, check_faithfulness=req.check_faithfulness)
    return QueryResponse(
        question=response.query,
        answer=response.answer,
        sources=[{
            "source": r.chunk.source,
            "start_line": r.chunk.start_line,
            "end_line": r.chunk.end_line,
            "language": r.chunk.language,
            "rerank_score": round(r.rerank_score, 4),
            "text_preview": r.chunk.text[:200],
        } for r in response.sources],
        faithfulness_score=response.faithfulness.score if response.faithfulness else None,
        is_faithful=response.faithfulness.is_faithful if response.faithfulness else None,
        active_repo=p.active_key,
    )


@app.get("/query/stream")
def query_stream(question: str):
    p = get_pipeline()
    if p.retriever is None:
        raise HTTPException(status_code=400, detail="No repo selected.")

    def token_generator():
        try:
            for token in p.stream_query(question):
                yield f"data: {token}\n\n"
        except Exception as e:
            yield f"data: Error: {str(e)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(token_generator(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache"})


@app.delete("/history")
def clear_history():
    get_pipeline().clear_history()
    return {"status": "ok"}


@app.delete("/index")
def clear_all():
    get_pipeline().clear_all()
    return {"status": "ok", "message": "All indexes cleared"}