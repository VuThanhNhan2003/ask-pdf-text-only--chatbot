from typing import Optional, Dict, Tuple, Any, List
import threading
import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from processor import RAGProcessor, logger
from config import config

app = FastAPI(title="RAG Chatbot API", version="1.0.0")

# Add CORS middleware for widget support
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve widget static files
import os
_widget_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "widget")
if os.path.isdir(_widget_dir):
    app.mount("/widget", StaticFiles(directory=_widget_dir), name="widget")

# Cache processors by (subject, model_key) to reuse embeddings/clients
_processor_cache: Dict[Tuple[Optional[str], str], RAGProcessor] = {}
_cache_lock = threading.Lock()


class QueryRequest(BaseModel):
    question: str
    subject: Optional[str] = None  # None means all subjects
    model_key: Optional[str] = None
    use_history: bool = True
    debug: bool = False


class QueryResponse(BaseModel):
    answer: str
    debug_scores: Optional[List[Dict[str, Any]]] = None
    retrieval_meta: Optional[Dict[str, Any]] = None


class ReloadRequest(BaseModel):
    subject: Optional[str] = None
    model_key: Optional[str] = None
    clean: bool = False


@app.on_event("startup")
def _startup():
    # Ensure config folders exist (data/logs)
    config.app.__post_init__()
    logger.info("FastAPI service started")


def _get_processor(subject: Optional[str], model_key: Optional[str]) -> RAGProcessor:
    """Return cached processor for subject/model, create if missing."""
    key = (subject, model_key or config.llm.current_model)
    with _cache_lock:
        if key not in _processor_cache:
            proc = RAGProcessor(subject=subject, llm_model=key[1])
            _processor_cache[key] = proc
        return _processor_cache[key]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/models")
def list_models():
    return RAGProcessor.get_available_llm_models()


@app.get("/subjects")
def list_subjects():
    return {"subjects": RAGProcessor.get_available_subjects()}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    subject = req.subject if req.subject and req.subject != "Tat ca mon hoc" else None
    try:
        processor = _get_processor(subject, req.model_key)
        if req.debug:
            debug_result = processor.get_response_with_debug(req.question, use_history=req.use_history)
            return QueryResponse(
                answer=debug_result.get("answer", ""),
                debug_scores=debug_result.get("debug_scores"),
                retrieval_meta=debug_result.get("retrieval_meta"),
            )
        answer = processor.get_response(req.question, use_history=req.use_history)
        return QueryResponse(answer=answer)
    except Exception as exc:
        logger.error(f"Query failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/query/debug", response_model=QueryResponse)
def query_debug(req: QueryRequest):
    """Debug query endpoint that always returns retrieval score table."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    subject = req.subject if req.subject and req.subject != "Tat ca mon hoc" else None
    try:
        processor = _get_processor(subject, req.model_key)
        debug_result = processor.get_response_with_debug(req.question, use_history=req.use_history)
        return QueryResponse(
            answer=debug_result.get("answer", ""),
            debug_scores=debug_result.get("debug_scores"),
            retrieval_meta=debug_result.get("retrieval_meta"),
        )
    except Exception as exc:
        logger.error(f"Debug query failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/reload")
def reload_data(req: ReloadRequest):
    subject = req.subject if req.subject and req.subject != "Tat ca mon hoc" else None
    try:
        processor = _get_processor(subject, req.model_key)
        processor.reload_data(clean=req.clean)
        return {"status": "reloaded", "subject": subject or "all"}
    except Exception as exc:
        logger.error(f"Reload failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/query/stream")
def query_stream(req: QueryRequest):
    """Streaming endpoint for chat widget"""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    subject = req.subject if req.subject and req.subject != "Tat ca mon hoc" else None
    
    def generate():
        try:
            processor = _get_processor(subject, req.model_key)
            for chunk in processor.get_response_stream(req.question, use_history=req.use_history):
                # Send chunk as JSON
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
            # Send completion signal
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as exc:
            logger.error(f"Stream query failed: {exc}", exc_info=True)
            error_msg = json.dumps({'error': str(exc)})
            yield f"data: {error_msg}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api_service:app",
        host="0.0.0.0",
        port=9100,
        reload=True
    )