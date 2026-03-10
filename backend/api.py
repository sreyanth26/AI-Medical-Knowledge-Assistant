"""
FastAPI Backend — Production-Grade v2.0
AI Medical Knowledge Assistant

Improvements over v1:
  ✅ Structured rotating file logging
  ✅ Request ID middleware for tracing
  ✅ Full Pydantic input validation
  ✅ Granular HTTP error codes
  ✅ Global exception handler
  ✅ Document list + delete endpoints
  ✅ Health / readiness probes
  ✅ File size + type validation
  ✅ Environment-based config

Run:
    uvicorn api:app --reload --host 0.0.0.0 --port 8000
"""

import logging
import logging.handlers
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

sys.path.insert(0, str(Path(__file__).parent))
from rag_pipeline import RAGPipeline, RAGResponse
from vector_store import MedicalVectorStore

# Load .env file from project root (one level above backend/)
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

# ──────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

_handlers = [
    logging.StreamHandler(sys.stdout),
    logging.handlers.RotatingFileHandler(
        LOG_DIR / "api.log", maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    ),
]
for h in _handlers:
    h.setFormatter(logging.Formatter(LOG_FORMAT))

logging.basicConfig(level=logging.INFO, handlers=_handlers)

# Disable ChromaDB and Streamlit telemetry — prevents PostHog analytics errors
os.environ["ANONYMIZED_TELEMETRY"] = "False"
logger = logging.getLogger("medrag.api")

# ──────────────────────────────────────────────────────────────────
# App
# ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title="MedRAG — AI Medical Knowledge Assistant",
    description="Production-grade RAG system for medical document Q&A with full source citations.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────────────────────────
# Middleware — Request ID + timing
# ──────────────────────────────────────────────────────────────────

@app.middleware("http")
async def request_middleware(request: Request, call_next):
    rid = str(uuid.uuid4())[:8]
    request.state.request_id = rid
    t0 = time.perf_counter()
    logger.info(f"[{rid}] → {request.method} {request.url.path}")
    response = await call_next(request)
    ms = (time.perf_counter() - t0) * 1000
    logger.info(f"[{rid}] ← {response.status_code} ({ms:.1f}ms)")
    response.headers["X-Request-ID"] = rid
    response.headers["X-Response-Time"] = f"{ms:.1f}ms"
    return response

# ──────────────────────────────────────────────────────────────────
# Global Exception Handler
# ──────────────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exc_handler(request: Request, exc: Exception):
    rid = getattr(request.state, "request_id", "?")
    logger.error(f"[{rid}] Unhandled: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "request_id": rid},
    )

# ──────────────────────────────────────────────────────────────────
# Config + Pipeline Init
# ──────────────────────────────────────────────────────────────────

CHROMA_DB_PATH  = os.getenv("CHROMA_DB_PATH", "../models/chroma_db")
LLM_MODEL       = os.getenv("LLM_MODEL", "phi3")
OLLAMA_HOST     = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MAX_FILE_MB     = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
ALLOWED_EXT     = {"pdf", "txt", "md"}

_doc_registry: Dict[str, dict] = {}  # doc_id → metadata

try:
    vector_store = MedicalVectorStore(persist_directory=CHROMA_DB_PATH)
    pipeline     = RAGPipeline(vector_store=vector_store, llm_model=LLM_MODEL, ollama_host=OLLAMA_HOST)
    logger.info(f"Pipeline ready | model={LLM_MODEL}")
except Exception as e:
    logger.critical(f"Pipeline init failed: {e}", exc_info=True)
    raise

# ──────────────────────────────────────────────────────────────────
# Schemas
# ──────────────────────────────────────────────────────────────────

class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    top_k: Optional[int] = Field(default=5, ge=1, le=20)

    @field_validator("question")
    @classmethod
    def strip_question(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Question cannot be whitespace only.")
        return v

class SourceModel(BaseModel):
    document: str
    chunk_index: int
    snippet: str

class AnswerResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    answer: str
    sources: List[SourceModel]
    retrieved_chunks: int
    model_used: str
    request_id: Optional[str] = None

class UploadResponse(BaseModel):
    filename: str
    chunks_indexed: int
    file_size_kb: float
    doc_id: str
    message: str

class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    chunks_indexed: int
    file_size_kb: float
    uploaded_at: str

class StatusResponse(BaseModel):
    status: str
    documents_indexed: bool
    document_count: int
    llm_model: str
    version: str

class SummarizeRequest(BaseModel):
    text: str = Field(..., min_length=50, max_length=10000)

# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────

def _validate_upload(file: UploadFile, content: bytes) -> None:
    ext = (file.filename or "").rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(415, f"Unsupported file type '.{ext}'. Allowed: {ALLOWED_EXT}")
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_FILE_MB:
        raise HTTPException(413, f"File {size_mb:.1f} MB exceeds {MAX_FILE_MB} MB limit.")
    if len(content) == 0:
        raise HTTPException(400, "Uploaded file is empty.")

# ──────────────────────────────────────────────────────────────────
# Health Endpoints
# ──────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {"service": "MedRAG", "version": "2.0.0", "docs": "/docs"}

@app.get("/health", tags=["Health"])
def health():
    """Liveness probe."""
    return {"status": "alive", "timestamp": time.time()}

@app.get("/ready", tags=["Health"])
def ready():
    """Readiness probe — 503 if pipeline not ready."""
    try:
        _ = pipeline.vector_store
        return {"status": "ready"}
    except Exception:
        raise HTTPException(503, "Pipeline not ready.")

@app.get("/status", response_model=StatusResponse, tags=["Health"])
def get_status():
    return StatusResponse(
        status="online",
        documents_indexed=not vector_store.is_empty,
        document_count=len(_doc_registry),
        llm_model=LLM_MODEL,
        version="2.0.0",
    )

# ──────────────────────────────────────────────────────────────────
# Document Endpoints
# ──────────────────────────────────────────────────────────────────

@app.post("/upload", response_model=UploadResponse, status_code=201, tags=["Documents"])
async def upload_document(request: Request, file: UploadFile = File(...)):
    """Upload and index a medical document (PDF / TXT / MD)."""
    rid = getattr(request.state, "request_id", "?")
    logger.info(f"[{rid}] Upload: '{file.filename}'")

    if not file.filename:
        raise HTTPException(400, "Filename is required.")

    try:
        content = await file.read()
    except Exception as e:
        logger.error(f"[{rid}] Read error: {e}")
        raise HTTPException(400, "Could not read uploaded file.")

    _validate_upload(file, content)

    size_kb = len(content) / 1024

    try:
        chunks = pipeline.ingest_bytes(content, file.filename)
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        logger.error(f"[{rid}] Ingestion error for '{file.filename}': {e}", exc_info=True)
        raise HTTPException(500, f"Document processing failed: {type(e).__name__}")

    doc_id = str(uuid.uuid4())[:12]
    _doc_registry[doc_id] = {
        "doc_id": doc_id,
        "filename": file.filename,
        "chunks_indexed": chunks,
        "file_size_kb": round(size_kb, 2),
        "uploaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    logger.info(f"[{rid}] '{file.filename}' → {chunks} chunks | id={doc_id}")
    return UploadResponse(
        filename=file.filename,
        chunks_indexed=chunks,
        file_size_kb=round(size_kb, 2),
        doc_id=doc_id,
        message=f"Indexed {chunks} chunks from '{file.filename}'.",
    )


@app.get("/documents", response_model=List[DocumentInfo], tags=["Documents"])
def list_documents():
    """List all indexed documents."""
    return [DocumentInfo(**d) for d in _doc_registry.values()]


@app.delete("/documents/{doc_id}", tags=["Documents"])
def delete_document(doc_id: str, request: Request):
    """Remove a document by ID."""
    rid = getattr(request.state, "request_id", "?")
    if doc_id not in _doc_registry:
        raise HTTPException(404, f"Document '{doc_id}' not found.")
    doc = _doc_registry.pop(doc_id)
    logger.info(f"[{rid}] Deleted doc record: '{doc['filename']}' (id={doc_id})")
    return {"message": f"'{doc['filename']}' removed.", "doc_id": doc_id}


@app.delete("/index", tags=["Documents"])
def clear_index(request: Request):
    """Clear entire FAISS index and all document records."""
    rid = getattr(request.state, "request_id", "?")
    vector_store.clear()
    _doc_registry.clear()
    logger.warning(f"[{rid}] Index cleared.")
    return {"message": "All documents and index cleared."}

# ──────────────────────────────────────────────────────────────────
# Q&A Endpoints
# ──────────────────────────────────────────────────────────────────

@app.post("/ask", response_model=AnswerResponse, tags=["Q&A"])
def ask_question(request: Request, body: QuestionRequest):
    """Ask a natural-language medical question grounded in uploaded documents."""
    rid = getattr(request.state, "request_id", "?")
    logger.info(f"[{rid}] Q: '{body.question[:80]}' | top_k={body.top_k}")

    if vector_store.is_empty:
        raise HTTPException(422, "No documents indexed. Please upload documents first.")

    prev_k, pipeline.top_k = pipeline.top_k, body.top_k or pipeline.top_k
    try:
        result: RAGResponse = pipeline.answer(body.question)
    except Exception as e:
        logger.error(f"[{rid}] Pipeline error: {e}", exc_info=True)
        raise HTTPException(500, f"Answer generation failed: {type(e).__name__}")
    finally:
        pipeline.top_k = prev_k

    logger.info(f"[{rid}] Answer ready | chunks={result.retrieved_chunks}")
    return AnswerResponse(
        answer=result.answer,
        sources=[SourceModel(document=s.document, chunk_index=s.chunk_index, snippet=s.snippet) for s in result.sources],
        retrieved_chunks=result.retrieved_chunks,
        model_used=result.model_used,
        request_id=rid,
    )


@app.post("/summarize", tags=["Documents"])
def summarize(request: Request, body: SummarizeRequest):
    """Summarize a block of medical text."""
    rid = getattr(request.state, "request_id", "?")
    logger.info(f"[{rid}] Summarize | len={len(body.text)}")
    try:
        summary = pipeline.summarize(body.text)
    except Exception as e:
        logger.error(f"[{rid}] Summarize error: {e}", exc_info=True)
        raise HTTPException(500, f"Summarization failed: {type(e).__name__}")
    return {"summary": summary, "original_length": len(body.text)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=int(os.getenv("API_PORT", "8000")), reload=True)