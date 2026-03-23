"""
Optimized RAG Processor with logging, better error handling, and performance improvements
"""
import hashlib
import json
import logging
import os
import re
import shutil
import time
from typing import Any, Dict, List, Optional, Generator
from datetime import datetime

import fitz  # PyMuPDF
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer
import torch

from config import config
from llm_manager import LLMManager


# =====================================================================
# LOGGING SETUP
# =====================================================================
def setup_logging() -> logging.Logger:
    """Setup logging with file and console handlers"""
    logger = logging.getLogger("RAGProcessor")
    logger.setLevel(logging.INFO)
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    # File handler
    log_file = os.path.join(
        config.app.log_folder, 
        f"rag_{datetime.now().strftime('%Y%m%d')}.log"
    )
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


logger = setup_logging()


# =====================================================================
# RETRIEVAL CONFIG (Hybrid + Rerank)
# =====================================================================
# Embedding model (dense only, no sparse/multi-vector)
EMBEDDING_MODEL_NAME = os.getenv("RAG_EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_LOCAL_PATH = os.getenv(
    "RAG_EMBEDDING_LOCAL_PATH",
    os.path.join(config.embedding.cache_folder, "bge-m3"),
)

# Hybrid retrieval knobs
HYBRID_TOP_K = int(os.getenv("RAG_HYBRID_TOP_K", "20"))
DENSE_TOP_K = int(os.getenv("RAG_DENSE_TOP_K", str(HYBRID_TOP_K)))
BM25_TOP_K = int(os.getenv("RAG_BM25_TOP_K", str(HYBRID_TOP_K)))
HYBRID_ALPHA = float(os.getenv("RAG_HYBRID_ALPHA", "0.7"))
HYBRID_BETA = float(os.getenv("RAG_HYBRID_BETA", "0.3"))
HYBRID_MODE = os.getenv("RAG_HYBRID_MODE", "score").strip().lower()
RRF_K = int(os.getenv("RAG_RRF_K", "60"))
HYBRID_MISSING_STRATEGY = os.getenv("RAG_HYBRID_MISSING_STRATEGY", "zero").strip().lower()
HYBRID_MISSING_EPSILON = float(os.getenv("RAG_HYBRID_MISSING_EPSILON", "0.01"))

# Reranking knobs
RERANKER_MODEL_NAME = os.getenv("RAG_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
RERANK_TOP_N_DEFAULT = int(os.getenv("RAG_RERANK_TOP_N", str(config.retrieval.top_k)))
RERANK_BATCH_SIZE = int(os.getenv("RAG_RERANK_BATCH_SIZE", "8"))

# Dense search threshold for Qdrant; set <=0 to disable thresholding.
DENSE_SCORE_THRESHOLD = float(os.getenv("RAG_DENSE_SCORE_THRESHOLD", str(config.retrieval.score_threshold)))


# =====================================================================
# UTILITY FUNCTIONS
# =====================================================================
def make_chunk_id(subject: str, filename: str, chunk_index: int, chunk_text: str) -> str:
    """Create a deterministic chunk ID based on file and content."""
    # Include chunk index for better uniqueness
    raw = f"{subject}/{filename}/chunk_{chunk_index}/{chunk_text[:100]}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def validate_pdf(file_path: str) -> bool:
    """Validate PDF file before processing"""
    try:
        if not os.path.exists(file_path):
            return False
        
        # Check file size (max 50MB)
        file_size = os.path.getsize(file_path) / (1024 * 1024)
        if file_size > 50:
            logger.warning(f"File too large: {file_path} ({file_size:.2f}MB)")
            return False
        
        # Try to open with PyMuPDF
        with fitz.open(file_path) as doc:
            if len(doc) == 0:
                return False
        
        return True
    except Exception as e:
        logger.error(f"PDF validation failed for {file_path}: {e}")
        return False


# =====================================================================
# EMBEDDING MODEL CACHE
# =====================================================================
_embedding_model = None

def get_embedding_model() -> SentenceTransformer:
    """Get cached embedding model"""
    global _embedding_model

    if _embedding_model is None:
        try:
            local_path = EMBEDDING_LOCAL_PATH
            embedding_device = "cuda" if torch.cuda.is_available() else "cpu"
            model_kwargs = {
                "low_cpu_mem_usage": False,
                "device_map": None,
            }
            
            if not os.path.exists(local_path):
                logger.info(f"📥 Downloading model {EMBEDDING_MODEL_NAME}...")
                os.makedirs(config.embedding.cache_folder, exist_ok=True)
                model = SentenceTransformer(
                    EMBEDDING_MODEL_NAME,
                    device=embedding_device,
                    model_kwargs=model_kwargs,
                )
                model.save(local_path)
                logger.info(f"✅ Model saved to {local_path}")
            else:
                logger.info(f"📂 Loading model from {local_path}")

            try:
                _embedding_model = SentenceTransformer(
                    local_path,
                    local_files_only=True,
                    device=embedding_device,
                    model_kwargs=model_kwargs,
                )
            except Exception as local_exc:
                if "meta tensor" in str(local_exc).lower():
                    logger.warning("⚠️ Corrupted local embedding cache detected, rebuilding: %s", local_exc)
                    if os.path.exists(local_path):
                        shutil.rmtree(local_path, ignore_errors=True)

                    model = SentenceTransformer(
                        EMBEDDING_MODEL_NAME,
                        device=embedding_device,
                        model_kwargs=model_kwargs,
                    )
                    model.save(local_path)
                    _embedding_model = SentenceTransformer(
                        local_path,
                        local_files_only=True,
                        device=embedding_device,
                        model_kwargs=model_kwargs,
                    )
                else:
                    raise

            logger.info("✅ Embedding model loaded successfully on %s", embedding_device)
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    return _embedding_model


# =====================================================================
# RAG PROCESSOR
# =====================================================================
class RAGProcessor:
    """Optimized RAG Processor with streaming support"""
    
    def __init__(self, subject: Optional[str] = None, llm_model: Optional[str] = None):
        """Initialize the RAG processor for a specific subject."""
        logger.info(f"Initializing RAGProcessor for subject: {subject}, model: {llm_model}")
        start_time = time.time()
        
        try:
            # Use cached embedding model
            self.embedding_model = get_embedding_model()
            self._reranker: Optional[CrossEncoder] = None
            self._reranker_device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"🔧 Reranker device: {self._reranker_device}")
            
            # Initialize Qdrant client
            self.qdrant_client = QdrantClient(
                host=config.qdrant.host,
                port=config.qdrant.port,
                timeout=config.qdrant.timeout
            )
            
            # Subject filter
            if subject is None or subject == "Tất cả môn học":
                self.subject = "Tất cả môn học"
                self.subject_filter: Optional[str] = None
            else:
                self.subject = subject
                self.subject_filter = subject
            
            self.collection_name = config.qdrant.collection_name
            self.data_folder = config.app.data_folder
            self.source_docs_folder = os.path.join(self.data_folder, "source_docs")
            self.processed_chunks_folder = os.path.join(self.data_folder, "processed_chunks")

            # In-memory BM25 state (built from Qdrant payload text)
            self._bm25_index: Optional[BM25Okapi] = None
            self._bm25_docs: List[Dict[str, Any]] = []
            self._bm25_tokenized_corpus: List[List[str]] = []
            
            # Initialize LLM using LLMManager
            self.llm_model_key = llm_model or config.llm.current_model
            self.llm = LLMManager.get_llm(self.llm_model_key)
            
            # Initialize text splitter
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.chunking.chunk_size,
                chunk_overlap=config.chunking.chunk_overlap,
                separators=config.chunking.separators,
            )
            
            # Initialize conversation history
            self.conversation_history: List[Dict[str, str]] = []
            self.max_history_messages = 3  # Giữ tối đa 3 cặp hội thoại
            
            # Ensure collection exists
            self._ensure_collection()

            # Build lexical index for hybrid retrieval
            self._build_bm25_index_from_qdrant()
            
            elapsed = time.time() - start_time
            logger.info(f"✅ RAGProcessor initialized in {elapsed:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAGProcessor: {e}")
            raise

    # ================================================================
    # COLLECTION MANAGEMENT
    # ================================================================
    def _collection_exists(self) -> bool:
        """Check if collection exists"""
        try:
            collections = self.qdrant_client.get_collections()
            return any(c.name == self.collection_name for c in collections.collections)
        except Exception as e:
            logger.error(f"Error checking collection existence: {e}")
            return False

    def _ensure_collection(self):
        """Ensure collection exists with proper configuration"""
        embedding_dim = self.embedding_model.get_sentence_embedding_dimension()

        if self._collection_exists():
            try:
                collection = self.qdrant_client.get_collection(self.collection_name)
                vectors_config = collection.config.params.vectors

                current_size = None
                if isinstance(vectors_config, dict):
                    default_vector = vectors_config.get("default")
                    if default_vector is not None:
                        current_size = getattr(default_vector, "size", None)
                else:
                    current_size = getattr(vectors_config, "size", None)

                if current_size is not None and int(current_size) != int(embedding_dim):
                    logger.warning(
                        "⚠️ Embedding dimension changed (%s -> %s). Recreating collection.",
                        current_size,
                        embedding_dim,
                    )
                    self.qdrant_client.delete_collection(self.collection_name)
                else:
                    logger.info(f"📂 Using existing collection: {self.collection_name}")
                    return
            except Exception as e:
                logger.warning(f"Could not validate collection vector size: {e}")
                logger.info(f"📂 Using existing collection: {self.collection_name}")
                return

        try:
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=embedding_dim, 
                    distance=Distance.COSINE
                ),
            )
            logger.info(f"✅ Created collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            raise

    # ================================================================
    # DATA PROCESSING (PDF + JSON CHUNKS)
    # ================================================================
    def _extract_text_from_pdf(self, pdf_path: str) -> List[Dict]:
        """
        Extract text from PDF with page information
        Returns list of {page_num, text}
        """
        pages_data = []
        try:
            with fitz.open(pdf_path) as doc:
                for page_num, page in enumerate(doc, start=1):
                    text = page.get_text("text").strip()
                    if text:  # Only add non-empty pages
                        pages_data.append({
                            "page_num": page_num,
                            "text": text
                        })
            logger.info(f"📄 Extracted {len(pages_data)} pages from {os.path.basename(pdf_path)}")
        except Exception as e:
            logger.error(f"Failed to extract text from {pdf_path}: {e}")
            raise
        
        return pages_data

    def _process_pdf_subject_folder(
        self, subject_name: str, subject_path: str
    ) -> List[Dict]:
        """Process all PDFs in a subject folder."""
        chunks: List[Dict] = []
        pdf_files = sorted([f for f in os.listdir(subject_path) if f.endswith(".pdf")])
        
        logger.info(f"📚 Processing {len(pdf_files)} PDFs for subject: {subject_name}")

        for filename in pdf_files:
            file_path = os.path.join(subject_path, filename)
            
            # Validate PDF first
            if not validate_pdf(file_path):
                logger.warning(f"⚠️ Skipping invalid PDF: {filename}")
                continue

            try:
                # Extract pages with text
                pages_data = self._extract_text_from_pdf(file_path)
                
                pending_chunks: List[Dict] = []
                
                # Process each page
                for page_data in pages_data:
                    page_num = page_data["page_num"]
                    page_text = page_data["text"]
                    
                    # Split page text into chunks
                    text_chunks = self.text_splitter.split_text(page_text)
                    
                    for chunk_idx, chunk_text in enumerate(text_chunks):
                        chunk_text = chunk_text.strip()
                        if not chunk_text:
                            continue

                        chunk_id = make_chunk_id(
                            subject_name, filename, chunk_idx, chunk_text
                        )

                        pending_chunks.append({
                            "id": chunk_id,
                            "text": chunk_text,
                            "metadata": {
                                "subject": subject_name,
                                "file": filename,
                                "page": page_num,
                                "chunk_id": chunk_id,
                            },
                        })

                # Filter out existing chunks
                new_chunks = self._filter_existing_chunks(pending_chunks)
                if new_chunks:
                    chunks.extend(new_chunks)
                    logger.info(f"➕ {filename}: {len(new_chunks)} new chunks")
                else:
                    logger.info(f"✅ {filename}: up-to-date")

            except Exception as e:
                logger.error(f"Error processing {filename}: {e}", exc_info=True)
                continue

        return chunks

    @staticmethod
    def _extract_json_items(payload: Any) -> List[Dict[str, Any]]:
        """Extract chunk-like dict records from flexible JSON payload shapes."""
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]

        if isinstance(payload, dict):
            for key in ("chunks", "data", "items", "documents"):
                nested = payload.get(key)
                if isinstance(nested, list):
                    return [item for item in nested if isinstance(item, dict)]
            return [payload]

        return []

    @staticmethod
    def _normalize_json_chunk(
        item: Dict[str, Any],
        subject_name: str,
        json_filename: str,
        chunk_index: int,
    ) -> Optional[Dict[str, Any]]:
        """Normalize a JSON chunk into internal chunk schema."""
        text = ""
        for key in ("text", "chunk", "content", "page_content"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                text = value.strip()
                break

        if not text:
            return None

        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        subject = str(metadata.get("subject") or item.get("subject") or subject_name).strip() or subject_name
        source_file = str(
            metadata.get("file")
            or metadata.get("file_name")
            or metadata.get("source")
            or item.get("file")
            or json_filename
        )

        page_raw = metadata.get("page") or item.get("page") or 0
        try:
            page = int(page_raw)
        except (TypeError, ValueError):
            page = 0

        raw_chunk_id = item.get("id") or item.get("id_") or metadata.get("chunk_id")
        if raw_chunk_id is not None and str(raw_chunk_id).strip():
            chunk_id = str(raw_chunk_id)
        else:
            chunk_id = make_chunk_id(subject, source_file, chunk_index, text)

        merged_metadata: Dict[str, Any] = {**metadata}
        merged_metadata["subject"] = subject
        merged_metadata["file_name"] = source_file
        merged_metadata.pop("file", None)
        merged_metadata["page"] = page
        merged_metadata["chunk_id"] = chunk_id
        merged_metadata["source_type"] = "json"
        merged_metadata["source_json_file"] = json_filename

        return {
            "id": chunk_id,
            "text": text,
            "metadata": merged_metadata,
        }

    def _process_json_subject_folder(self, subject_name: str, subject_path: str) -> List[Dict[str, Any]]:
        """Process all pre-chunked JSON files in a subject folder."""
        chunks: List[Dict[str, Any]] = []
        json_files = sorted([f for f in os.listdir(subject_path) if f.lower().endswith(".json")])

        logger.info(f"📚 Processing {len(json_files)} JSON files for subject: {subject_name}")

        for filename in json_files:
            file_path = os.path.join(subject_path, filename)

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)

                records = self._extract_json_items(payload)
                pending_chunks: List[Dict[str, Any]] = []

                for chunk_idx, item in enumerate(records):
                    normalized = self._normalize_json_chunk(item, subject_name, filename, chunk_idx)
                    if normalized is not None:
                        pending_chunks.append(normalized)

                # Keep latest occurrence if duplicate IDs appear in one JSON file.
                unique_by_id: Dict[str, Dict[str, Any]] = {}
                for chunk in pending_chunks:
                    unique_by_id[chunk["id"]] = chunk
                deduped_chunks = list(unique_by_id.values())

                # Always upsert JSON chunks so metadata changes are applied for existing IDs.
                if deduped_chunks:
                    chunks.extend(deduped_chunks)
                    logger.info(f"🔁 {filename}: {len(deduped_chunks)} chunks queued for upsert")
                else:
                    logger.info(f"✅ {filename}: up-to-date")

            except Exception as e:
                logger.error(f"Error processing JSON {filename}: {e}", exc_info=True)
                continue

        return chunks

    def _process_subject_folder(self, subject_name: str) -> List[Dict[str, Any]]:
        """Process one subject from both PDF source and pre-chunked JSON source."""
        chunks: List[Dict[str, Any]] = []
        pdf_subject_path = os.path.join(self.source_docs_folder, subject_name)
        json_subject_path = os.path.join(self.processed_chunks_folder, subject_name)

        if os.path.isdir(pdf_subject_path):
            chunks.extend(self._process_pdf_subject_folder(subject_name, pdf_subject_path))

        if os.path.isdir(json_subject_path):
            chunks.extend(self._process_json_subject_folder(subject_name, json_subject_path))

        if not os.path.isdir(pdf_subject_path) and not os.path.isdir(json_subject_path):
            logger.warning(f"⚠️ Subject folder not found in both sources: {subject_name}")

        return chunks

    def _filter_existing_chunks(self, chunk_items: List[Dict]) -> List[Dict]:
        """Filter out chunks that already exist in the database"""
        if not chunk_items:
            return []

        chunk_ids = [chunk["id"] for chunk in chunk_items]
        existing_ids = set()

        try:
            # Batch retrieve to check existence
            existing_points = self.qdrant_client.retrieve(
                collection_name=self.collection_name,
                ids=chunk_ids,
                with_payload=False,
                with_vectors=False,
            )
            existing_ids = {str(point.id) for point in existing_points}
            
        except Exception as e:
            logger.warning(f"Error retrieving existing chunks: {e}")

        new_chunks = [c for c in chunk_items if c["id"] not in existing_ids]
        logger.info(f"📊 Found {len(new_chunks)} new chunks out of {len(chunk_items)}")
        
        return new_chunks

    def _add_chunks_to_vectorstore(self, chunks: List[Dict]):
        """Add chunks to vector store with batching"""
        if not chunks:
            return

        try:
            texts = [chunk["text"] for chunk in chunks]
            
            # Batch encode for efficiency
            logger.info(f"🔢 Encoding {len(texts)} chunks...")
            embeddings = self.embedding_model.encode(
                texts,
                batch_size=config.embedding.batch_size,
                show_progress_bar=False,
            )

            # Create points
            points: List[PointStruct] = []
            for chunk, embedding in zip(chunks, embeddings):
                metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
                payload = {"text": chunk["text"], **metadata}

                payload["chunk_id"] = str(payload.get("chunk_id") or chunk["id"])
                payload["subject"] = str(payload.get("subject") or "")
                try:
                    payload["page"] = int(payload.get("page", 0))
                except (TypeError, ValueError):
                    payload["page"] = 0

                source_type = str(payload.get("source_type") or "").lower()
                resolved_file = str(payload.get("file_name") or payload.get("file") or "")
                if source_type == "json":
                    payload["file_name"] = resolved_file
                    payload.pop("file", None)
                else:
                    payload["file"] = resolved_file

                point = PointStruct(
                    id=chunk["id"],
                    vector=embedding.tolist(),
                    payload=payload,
                )
                points.append(point)

            # Batch upsert
            self.qdrant_client.upsert(
                collection_name=self.collection_name, 
                points=points
            )
            logger.info(f"📌 Added {len(points)} points to {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to add chunks to vectorstore: {e}")
            raise

    # ================================================================
    # HYBRID RETRIEVAL + RERANKING
    # ================================================================
    @staticmethod
    def _tokenize_for_bm25(text: str) -> List[str]:
        """Simple Unicode-friendly tokenizer for lexical search."""
        if not text:
            return []
        return re.findall(r"\w+", text.lower(), flags=re.UNICODE)

    @staticmethod
    def _minmax_normalize(scores: Dict[str, float]) -> Dict[str, float]:
        """Normalize score dict values into [0, 1]."""
        if not scores:
            return {}
        values = list(scores.values())
        min_v = min(values)
        max_v = max(values)
        if max_v - min_v < 1e-12:
            return {k: 1.0 for k in scores}
        return {k: (v - min_v) / (max_v - min_v) for k, v in scores.items()}

    def _load_reranker(self) -> CrossEncoder:
        """Lazy-load reranker to keep startup memory low."""
        if self._reranker is None:
            logger.info(f"📥 Loading reranker model: {RERANKER_MODEL_NAME}")
            try:
                self._reranker = CrossEncoder(
                    RERANKER_MODEL_NAME,
                    device=self._reranker_device,
                    max_length=512,
                    model_kwargs={
                        "low_cpu_mem_usage": False,
                        "device_map": None,
                    },
                )
            except Exception as rerank_exc:
                if "meta tensor" in str(rerank_exc).lower():
                    logger.warning(
                        "⚠️ Reranker meta-tensor error. Falling back to lightweight reranker on CPU: %s",
                        rerank_exc,
                    )
                    self._reranker = CrossEncoder(
                        "cross-encoder/ms-marco-MiniLM-L-6-v2",
                        device="cpu",
                        max_length=512,
                    )
                else:
                    raise
            logger.info("✅ Reranker loaded")
        return self._reranker

    def _build_bm25_index_from_qdrant(self):
        """Build in-memory BM25 index from existing Qdrant payloads."""
        logger.info("📚 Building BM25 index from Qdrant payloads...")
        start_time = time.time()

        docs: List[Dict[str, Any]] = []
        tokenized_corpus: List[List[str]] = []
        offset = None

        try:
            while True:
                points, next_offset = self.qdrant_client.scroll(
                    collection_name=self.collection_name,
                    with_payload=True,
                    with_vectors=False,
                    limit=256,
                    offset=offset,
                )

                if not points:
                    break

                for point in points:
                    payload = point.payload or {}
                    text = payload.get("text", "")
                    tokens = self._tokenize_for_bm25(text)
                    if not tokens:
                        continue

                    chunk_id = str(payload.get("chunk_id") or point.id)
                    docs.append(
                        {
                            "id": chunk_id,
                            "text": text,
                            "subject": payload.get("subject", ""),
                            "file": payload.get("file_name") or payload.get("file", ""),
                            "page": payload.get("page", 0),
                        }
                    )
                    tokenized_corpus.append(tokens)

                if next_offset is None:
                    break
                offset = next_offset

            self._bm25_docs = docs
            self._bm25_tokenized_corpus = tokenized_corpus
            self._bm25_index = BM25Okapi(tokenized_corpus) if tokenized_corpus else None

            elapsed = time.time() - start_time
            logger.info(f"✅ BM25 index ready with {len(self._bm25_docs)} docs in {elapsed:.2f}s")

        except Exception as e:
            logger.error(f"Failed to build BM25 index: {e}", exc_info=True)
            self._bm25_docs = []
            self._bm25_tokenized_corpus = []
            self._bm25_index = None

    def retrieve_dense(self, query: str, top_k: int = DENSE_TOP_K) -> List[Dict[str, Any]]:
        """Dense retrieval from Qdrant."""
        try:
            query_embedding = self.embedding_model.encode([query], normalize_embeddings=True)[0]

            query_filter = None
            if self.subject_filter:
                query_filter = Filter(
                    must=[
                        FieldCondition(
                            key="subject",
                            match=MatchValue(value=self.subject_filter),
                        )
                    ]
                )

            threshold = DENSE_SCORE_THRESHOLD if DENSE_SCORE_THRESHOLD > 0 else None
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                limit=top_k,
                query_filter=query_filter,
                score_threshold=threshold,
            )

            results: List[Dict[str, Any]] = []
            for result in search_results:
                payload = result.payload or {}
                results.append(
                    {
                        "id": str(payload.get("chunk_id") or result.id),
                        "text": payload.get("text", ""),
                        "subject": payload.get("subject", ""),
                        "file": payload.get("file_name") or payload.get("file", ""),
                        "page": payload.get("page", 0),
                        "dense_score": float(result.score),
                    }
                )

            return results
        except Exception as e:
            logger.error(f"Failed dense retrieval: {e}", exc_info=True)
            return []

    def retrieve_bm25(self, query: str, top_k: int = BM25_TOP_K) -> List[Dict[str, Any]]:
        """Lexical retrieval using in-memory BM25 index."""
        if self._bm25_index is None or not self._bm25_docs:
            return []

        query_tokens = self._tokenize_for_bm25(query)
        if not query_tokens:
            return []

        try:
            raw_scores = self._bm25_index.get_scores(query_tokens)
            candidate_indices = list(range(len(self._bm25_docs)))

            if self.subject_filter:
                candidate_indices = [
                    i
                    for i in candidate_indices
                    if self._bm25_docs[i].get("subject") == self.subject_filter
                ]

            if not candidate_indices:
                return []

            ranked = sorted(
                candidate_indices,
                key=lambda i: raw_scores[i],
                reverse=True,
            )[:top_k]

            results: List[Dict[str, Any]] = []
            for idx in ranked:
                score = float(raw_scores[idx])
                if score <= 0:
                    continue
                doc = self._bm25_docs[idx]
                results.append(
                    {
                        "id": doc["id"],
                        "text": doc["text"],
                        "subject": doc["subject"],
                        "file": doc["file"],
                        "page": doc["page"],
                        "bm25_score": score,
                    }
                )

            return results
        except Exception as e:
            logger.error(f"Failed BM25 retrieval: {e}", exc_info=True)
            return []

    def hybrid_retrieve(self, query: str, top_k: int = HYBRID_TOP_K) -> List[Dict[str, Any]]:
        """Hybrid retrieval: dense + BM25 with weighted score fusion."""
        dense_results = self.retrieve_dense(query, top_k=DENSE_TOP_K)
        bm25_results = self.retrieve_bm25(query, top_k=BM25_TOP_K)

        if not dense_results and not bm25_results:
            return []

        if HYBRID_MODE == "rrf":
            return self._hybrid_retrieve_rrf(dense_results, bm25_results, top_k=top_k)
        return self._hybrid_retrieve_score_fusion(dense_results, bm25_results, top_k=top_k)

    def _hybrid_retrieve_rrf(
        self,
        dense_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Rank-based fusion using Reciprocal Rank Fusion (RRF)."""
        merged: Dict[str, Dict[str, Any]] = {}
        dense_rank_map: Dict[str, int] = {}
        bm25_rank_map: Dict[str, int] = {}

        for rank, item in enumerate(dense_results, start=1):
            item_id = item["id"]
            dense_rank_map[item_id] = rank
            if item_id not in merged:
                merged[item_id] = {**item}

        for rank, item in enumerate(bm25_results, start=1):
            item_id = item["id"]
            bm25_rank_map[item_id] = rank
            if item_id not in merged:
                merged[item_id] = {**item}
            elif not merged[item_id].get("text") and item.get("text"):
                merged[item_id]["text"] = item["text"]

        k = max(1, RRF_K)
        for item_id, doc in merged.items():
            dense_rank = dense_rank_map.get(item_id)
            bm25_rank = bm25_rank_map.get(item_id)

            dense_rrf = 1.0 / (k + dense_rank) if dense_rank is not None else 0.0
            bm25_rrf = 1.0 / (k + bm25_rank) if bm25_rank is not None else 0.0
            hybrid_score = dense_rrf + bm25_rrf

            doc["dense_score"] = float(doc.get("dense_score", 0.0))
            doc["bm25_score"] = float(doc.get("bm25_score", 0.0))
            doc["dense_norm"] = None
            doc["bm25_norm"] = None
            doc["dense_rank"] = dense_rank
            doc["bm25_rank"] = bm25_rank
            doc["rrf_score"] = hybrid_score
            doc["hybrid_score"] = hybrid_score

        ranked = sorted(merged.values(), key=lambda x: x.get("hybrid_score", 0.0), reverse=True)
        return ranked[:top_k]

    @staticmethod
    def _resolve_missing_default(
        normalized_scores: Dict[str, float],
        strategy: str,
        epsilon: float,
    ) -> float:
        """Resolve default score for docs missing from one retrieval branch."""
        if strategy == "epsilon":
            return max(0.0, epsilon)
        if strategy == "min" and normalized_scores:
            return min(normalized_scores.values())
        return 0.0

    def _hybrid_retrieve_score_fusion(
        self,
        dense_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Score-based hybrid retrieval: normalized dense + BM25 weighted fusion."""

        merged: Dict[str, Dict[str, Any]] = {}
        dense_scores: Dict[str, float] = {}
        bm25_scores: Dict[str, float] = {}

        for item in dense_results:
            item_id = item["id"]
            merged[item_id] = {**item}
            dense_scores[item_id] = float(item.get("dense_score", 0.0))

        for item in bm25_results:
            item_id = item["id"]
            if item_id not in merged:
                merged[item_id] = {**item}
            else:
                # Keep richer metadata/text from whichever branch has it.
                if not merged[item_id].get("text") and item.get("text"):
                    merged[item_id]["text"] = item["text"]
            bm25_scores[item_id] = float(item.get("bm25_score", 0.0))

        dense_norm = self._minmax_normalize(dense_scores)
        bm25_norm = self._minmax_normalize(bm25_scores)

        missing_strategy = HYBRID_MISSING_STRATEGY
        if missing_strategy not in {"zero", "min", "epsilon"}:
            missing_strategy = "zero"

        dense_default = self._resolve_missing_default(
            dense_norm,
            strategy=missing_strategy,
            epsilon=HYBRID_MISSING_EPSILON,
        )
        bm25_default = self._resolve_missing_default(
            bm25_norm,
            strategy=missing_strategy,
            epsilon=HYBRID_MISSING_EPSILON,
        )

        alpha = max(0.0, HYBRID_ALPHA)
        beta = max(0.0, HYBRID_BETA)
        total = alpha + beta
        if total <= 0:
            alpha, beta = 0.5, 0.5
        else:
            alpha, beta = alpha / total, beta / total

        for item_id, doc in merged.items():
            d_score = dense_norm.get(item_id, dense_default)
            b_score = bm25_norm.get(item_id, bm25_default)
            hybrid_score = alpha * d_score + beta * b_score
            doc["dense_score"] = dense_scores.get(item_id, 0.0)
            doc["bm25_score"] = bm25_scores.get(item_id, 0.0)
            doc["dense_norm"] = d_score
            doc["bm25_norm"] = b_score
            doc["hybrid_score"] = hybrid_score

        ranked = sorted(merged.values(), key=lambda x: x.get("hybrid_score", 0.0), reverse=True)
        return ranked[:top_k]

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_n: int = RERANK_TOP_N_DEFAULT) -> List[Dict[str, Any]]:
        """Cross-encoder reranking for final context selection."""
        if not candidates:
            return []

        try:
            reranker = self._load_reranker()
            pairs = [(query, item.get("text", "")) for item in candidates]

            scores = reranker.predict(
                pairs,
                batch_size=RERANK_BATCH_SIZE,
                show_progress_bar=False,
            )

            # CrossEncoder may return np.ndarray with shape (n,) or (n,1)
            scores = np.asarray(scores).reshape(-1)
            for item, score in zip(candidates, scores):
                item["rerank_score"] = float(score)

            reranked = sorted(candidates, key=lambda x: x.get("rerank_score", -1e9), reverse=True)
            return reranked[:top_n]
        except Exception as e:
            logger.error(f"Failed reranking: {e}", exc_info=True)
            # Safe fallback: use hybrid ranking only.
            return candidates[:top_n]

    # ================================================================
    # PUBLIC APIs
    # ================================================================
    def reload_data(self, clean: bool = False):
        """Reload data for current subject or all subjects"""
        logger.info(f"🔄 Reloading data (clean={clean})")
        start_time = time.time()
        
        try:
            if clean:
                logger.info("♻️ Resetting collection...")
                self.qdrant_client.delete_collection(self.collection_name)
                self._ensure_collection()
                # Luôn index lại toàn bộ để tránh mất dữ liệu môn khác
                for subject in self.get_available_subjects():
                    new_chunks = self._process_subject_folder(subject)
                    self._add_chunks_to_vectorstore(new_chunks)

            if self.subject_filter:
                new_chunks = self._process_subject_folder(self.subject_filter)
                self._add_chunks_to_vectorstore(new_chunks)
            else:
                for subject in self.get_available_subjects():
                    new_chunks = self._process_subject_folder(subject)
                    self._add_chunks_to_vectorstore(new_chunks)

            # Keep lexical index in sync with vector store.
            self._build_bm25_index_from_qdrant()
            
            elapsed = time.time() - start_time
            logger.info(f"✅ Data reload completed in {elapsed:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to reload data: {e}")
            raise

    def _retrieve_relevant_chunks(self, query: str, k: int = None) -> List[Dict]:
        """Retrieve relevant chunks with hybrid retrieval and reranking."""
        final_top_n = k if k is not None else RERANK_TOP_N_DEFAULT

        try:
            # Stage 1: hybrid dense + BM25 retrieval
            hybrid_candidates = self.hybrid_retrieve(query, top_k=HYBRID_TOP_K)
            if not hybrid_candidates:
                logger.info("🔍 Hybrid retrieval returned 0 candidates")
                return []

            # Stage 2: rerank top-k candidates, return top-n
            relevant_chunks = self.rerank(query, hybrid_candidates, top_n=final_top_n)

            logger.info(
                "🔍 Retrieved %s candidates (hybrid=%s) and kept %s after rerank",
                len(hybrid_candidates),
                HYBRID_TOP_K,
                len(relevant_chunks),
            )
            return relevant_chunks
        except Exception as e:
            logger.error(f"Failed to retrieve chunks: {e}", exc_info=True)
            return []

    @staticmethod
    def _build_debug_score_rows(
        reranked_candidates: List[Dict[str, Any]],
        selected_chunk_ids: set,
        hybrid_rank_map: Dict[str, int],
    ) -> List[Dict[str, Any]]:
        """Build score table rows for retrieval debugging."""
        rows: List[Dict[str, Any]] = []

        for rerank_rank, item in enumerate(reranked_candidates, start=1):
            text = str(item.get("text", ""))
            text_preview = " ".join(text.split())[:240]
            item_id = item.get("id")

            rows.append(
                {
                    "rank": rerank_rank,
                    "rerank_rank": rerank_rank,
                    "hybrid_rank": hybrid_rank_map.get(item_id),
                    "selected_for_context": item_id in selected_chunk_ids,
                    "id": item_id,
                    "subject": item.get("subject", ""),
                    "file": item.get("file", ""),
                    "page": item.get("page", 0),
                    "dense_score": item.get("dense_score"),
                    "bm25_score": item.get("bm25_score"),
                    "dense_norm": item.get("dense_norm"),
                    "bm25_norm": item.get("bm25_norm"),
                    "hybrid_score": item.get("hybrid_score"),
                    "rerank_score": item.get("rerank_score"),
                    "dense_rank": item.get("dense_rank"),
                    "bm25_rank": item.get("bm25_rank"),
                    "rrf_score": item.get("rrf_score"),
                    "text_preview": text_preview,
                }
            )

        return rows

    def _retrieve_relevant_chunks_with_debug(
        self,
        query: str,
        k: int = None,
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Retrieve chunks and return a debug table of per-chunk scores."""
        final_top_n = k if k is not None else RERANK_TOP_N_DEFAULT

        try:
            hybrid_candidates = self.hybrid_retrieve(query, top_k=HYBRID_TOP_K)
            if not hybrid_candidates:
                logger.info("🔍 Hybrid retrieval returned 0 candidates (debug mode)")
                return [], []

            # Score all hybrid candidates with reranker, then slice top-n for context.
            reranked_candidates = self.rerank(
                query,
                hybrid_candidates,
                top_n=len(hybrid_candidates),
            )
            relevant_chunks = reranked_candidates[:final_top_n]

            selected_chunk_ids = {item.get("id") for item in relevant_chunks}
            hybrid_rank_map: Dict[str, int] = {
                item.get("id"): rank
                for rank, item in enumerate(hybrid_candidates, start=1)
                if item.get("id") is not None
            }
            debug_rows = self._build_debug_score_rows(
                reranked_candidates,
                selected_chunk_ids,
                hybrid_rank_map,
            )

            logger.info(
                "🔍 Debug retrieval: %s hybrid candidates, %s selected",
                len(debug_rows),
                len(relevant_chunks),
            )
            return relevant_chunks, debug_rows
        except Exception as e:
            logger.error(f"Failed retrieval debug mode: {e}", exc_info=True)
            return [], []

    @staticmethod
    def get_available_llm_models() -> Dict[str, Dict]:
        """Get list of available LLM models"""
        return LLMManager.list_available_models()

    def _add_to_history(self, role: str, content: str):
        """Add message to conversation history"""
        self.conversation_history.append({
            "role": role,
            "content": content
        })
        
        # Keep only last N messages (pairs of user+assistant)
        if len(self.conversation_history) > self.max_history_messages * 2:
            self.conversation_history = self.conversation_history[-(self.max_history_messages * 2):]
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        logger.info("🧹 Conversation history cleared")
    
    def set_history(self, messages: List[Dict[str, str]]):
        """Set conversation history from external source (e.g., database)"""
        self.conversation_history = messages[-self.max_history_messages * 2:] if messages else []
        logger.info(f"📝 Set conversation history: {len(self.conversation_history)} messages")
    
    def _build_prompt(self, context: str, question: str, use_history: bool = True) -> str:
        """
        Build prompt for LLM with optional conversation history
        
        Args:
            context: Retrieved context from documents
            question: User's question
            use_history: Whether to include conversation history
        """
        
        # Build history section if needed
        history_section = ""
        if use_history and self.conversation_history:
            history_parts = []
            for msg in self.conversation_history:
                role = "Người dùng" if msg["role"] == "user" else "Trợ lý AI"
                history_parts.append(f"{role}: {msg['content']}")
            
            history_text = "\n".join(history_parts)
            history_section = f"""Lịch sử hội thoại trước đó:
        {history_text}

    """
        
        prompt_template = """Bạn là một trợ lý AI được thiết kế để cung cấp các câu trả lời phù hợp và sâu sắc dựa trên ngữ cảnh từ nhiều tài liệu khác nhau.

    {history_section}Hãy sử dụng ngữ cảnh sau để trả lời câu hỏi của người dùng:

    Ngữ cảnh: {context}

    Câu hỏi{question_label}: {question}

    Câu trả lời của bạn cần:
    1. Rõ ràng, ngắn gọn và dựa trực tiếp vào ngữ cảnh đã cung cấp.
    {history_instruction}2. Bao gồm các chi tiết cụ thể từ tài liệu khi phù hợp.
    3. Nếu bạn không biết câu trả lời, hãy nói rõ.
    4. Nếu có nhiều tài liệu liên quan, hãy tổng hợp thông tin một cách mạch lạc.
    5. KHÔNG nêu nguồn trong câu trả lời - nguồn sẽ được thêm tự động.

    Câu trả lời của bạn:
    """
        
        # Dynamic text based on history usage
        question_label = " hiện tại của người dùng" if use_history else " của người dùng"
        history_instruction = "2. Tham khảo lịch sử hội thoại nếu câu hỏi hiện tại liên quan đến câu hỏi trước.\n" if use_history else ""
        
        return prompt_template.format(
            history_section=history_section,
            context=context,
            question=question,
            question_label=question_label,
            history_instruction=history_instruction
        )

    # Cập nhật các phương thức gọi:
    def get_response(self, query: str, use_history: bool = True) -> str:
        """Get response for query (non-streaming)"""
        logger.info(f"💬 Processing query with {self.llm_model_key}: {query[:100]}...")
        start_time = time.time()
        
        try:
            relevant_chunks = self._retrieve_relevant_chunks(query)
            if not relevant_chunks:
                logger.warning("No relevant chunks found")
                response = "❌ Không tìm thấy thông tin phù hợp trong tài liệu."
                if use_history:
                    self._add_to_history("user", query)
                    self._add_to_history("assistant", response)
                return response

            context = self._build_context(relevant_chunks)
            prompt = self._build_prompt(context, query, use_history=use_history)
            
            response = self.llm.invoke(prompt)
            
            # Add to history
            if use_history:
                self._add_to_history("user", query)
                self._add_to_history("assistant", response)
            
            # Add sources
            sources_text = self._format_sources(relevant_chunks)
            full_response = f"{response}\n\n{sources_text}"
            
            elapsed = time.time() - start_time
            logger.info(f"✅ Response generated in {elapsed:.2f}s")
            
            return full_response
            
        except Exception as e:
            logger.error(f"Failed to generate response: {e}", exc_info=True)
            error_response = f"❌ Lỗi khi xử lý câu hỏi: {str(e)}"
            if use_history:
                self._add_to_history("user", query)
                self._add_to_history("assistant", error_response)
            return error_response

    def get_response_with_debug(self, query: str, use_history: bool = True) -> Dict[str, Any]:
        """Get response with retrieval score table for debugging."""
        logger.info(f"💬 Processing debug query with {self.llm_model_key}: {query[:100]}...")
        start_time = time.time()

        try:
            relevant_chunks, debug_rows = self._retrieve_relevant_chunks_with_debug(query)
            if not relevant_chunks:
                response = "❌ Không tìm thấy thông tin phù hợp trong tài liệu."
                if use_history:
                    self._add_to_history("user", query)
                    self._add_to_history("assistant", response)
                return {
                    "answer": response,
                    "debug_scores": debug_rows,
                    "retrieval_meta": {
                        "hybrid_mode": HYBRID_MODE,
                        "hybrid_top_k": HYBRID_TOP_K,
                        "rerank_top_n": RERANK_TOP_N_DEFAULT,
                        "total_candidates": len(debug_rows),
                    },
                }

            context = self._build_context(relevant_chunks)
            prompt = self._build_prompt(context, query, use_history=use_history)
            response = self.llm.invoke(prompt)

            if use_history:
                self._add_to_history("user", query)
                self._add_to_history("assistant", response)

            sources_text = self._format_sources(relevant_chunks)
            full_response = f"{response}\n\n{sources_text}"

            elapsed = time.time() - start_time
            logger.info(f"✅ Debug response generated in {elapsed:.2f}s")

            return {
                "answer": full_response,
                "debug_scores": debug_rows,
                "retrieval_meta": {
                    "hybrid_mode": HYBRID_MODE,
                    "hybrid_top_k": HYBRID_TOP_K,
                    "rerank_top_n": RERANK_TOP_N_DEFAULT,
                    "total_candidates": len(debug_rows),
                    "selected_candidates": len(relevant_chunks),
                },
            }

        except Exception as e:
            logger.error(f"Failed to generate debug response: {e}", exc_info=True)
            error_response = f"❌ Lỗi khi xử lý câu hỏi: {str(e)}"
            if use_history:
                self._add_to_history("user", query)
                self._add_to_history("assistant", error_response)
            return {
                "answer": error_response,
                "debug_scores": [],
                "retrieval_meta": {
                    "hybrid_mode": HYBRID_MODE,
                    "hybrid_top_k": HYBRID_TOP_K,
                    "rerank_top_n": RERANK_TOP_N_DEFAULT,
                    "total_candidates": 0,
                },
            }


    def get_response_stream(self, query: str, use_history: bool = True) -> Generator[str, None, None]:
        """Get streaming response for query"""
        logger.info(f"💬 Processing streaming query with {self.llm_model_key}: {query[:100]}...")
        
        try:
            relevant_chunks = self._retrieve_relevant_chunks(query)
            if not relevant_chunks:
                error_msg = "❌ Không tìm thấy thông tin phù hợp trong tài liệu."
                if use_history:
                    self._add_to_history("user", query)
                    self._add_to_history("assistant", error_msg)
                yield error_msg
                return

            context = self._build_context(relevant_chunks)
            prompt = self._build_prompt(context, query, use_history=use_history)
            
            # Stream response and collect full response
            full_response = ""
            for chunk in self.llm.stream(prompt):
                full_response += chunk
                yield chunk
            
            # Add to history
            if use_history:
                self._add_to_history("user", query)
                self._add_to_history("assistant", full_response)
            
            # Add sources at the end
            yield "\n\n"
            yield self._format_sources(relevant_chunks)
            
        except Exception as e:
            logger.error(f"Failed to generate streaming response: {e}", exc_info=True)
            error_msg = f"\n\n❌ Lỗi: {str(e)}"
            if use_history:
                self._add_to_history("user", query)
                self._add_to_history("assistant", error_msg)
            yield error_msg
    

    def _build_context(self, chunks: List[Dict]) -> str:
        """Build context from relevant chunks"""
        context_parts = []
        total_length = 0
        
        for chunk in chunks:
            source_info = f"[{chunk['subject']} - {chunk['file']} - Trang {chunk['page']}]"
            chunk_text = f"{source_info}\n{chunk['text']}"
            
            # Check context length limit
            if total_length + len(chunk_text) > config.retrieval.max_context_length:
                break
            
            context_parts.append(chunk_text)
            total_length += len(chunk_text)
        
        return "\n\n".join(context_parts)

    def _format_sources(self, chunks: List[Dict]) -> str:
        """Format sources information"""
        seen = set()
        sources = []
        
        for chunk in chunks:
            key = (chunk['subject'], chunk['file'], chunk['page'])
            if key not in seen:
                seen.add(key)
                sources.append(
                    f"- {chunk['subject']} / {chunk['file']} / Trang {chunk['page']}"
                )
        
        if sources:
            return "📚 **Nguồn tham khảo:**\n" + "\n".join(sources)
        return ""

    @staticmethod
    def get_available_subjects() -> List[str]:
        """Get list of available subjects from source_docs and processed_chunks."""
        if not os.path.exists(config.app.data_folder):
            return []

        subjects = set()
        for root_name in ("source_docs", "processed_chunks"):
            root_path = os.path.join(config.app.data_folder, root_name)
            if not os.path.isdir(root_path):
                continue
            for folder in os.listdir(root_path):
                if os.path.isdir(os.path.join(root_path, folder)):
                    subjects.add(folder)

        return sorted(subjects)

    @staticmethod
    def get_subject_files(subject: str) -> List[str]:
        """Get list of source files (PDF/JSON) for a subject."""
        source_docs_path = os.path.join(config.app.data_folder, "source_docs", subject)
        processed_chunks_path = os.path.join(config.app.data_folder, "processed_chunks", subject)

        files: List[str] = []
        if os.path.isdir(source_docs_path):
            files.extend([f for f in os.listdir(source_docs_path) if f.lower().endswith(".pdf")])
        if os.path.isdir(processed_chunks_path):
            files.extend([f for f in os.listdir(processed_chunks_path) if f.lower().endswith(".json")])

        return sorted(files)

    def __del__(self):
        """Cleanup resources"""
        try:
            if hasattr(self, "qdrant_client"):
                self.qdrant_client.close()
                logger.info("🔒 Qdrant client closed")
        except Exception as e:
            logger.error(f"Error closing Qdrant client: {e}")