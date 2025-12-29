"""
Optimized RAG Processor with logging, better error handling, and performance improvements
"""
import hashlib
import logging
import os
import time
from typing import Dict, List, Optional, Generator
from datetime import datetime

import fitz  # PyMuPDF
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
from sentence_transformers import SentenceTransformer

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
            local_path = config.embedding.local_path
            
            if not os.path.exists(local_path):
                logger.info(f"📥 Downloading model {config.embedding.model_name}...")
                os.makedirs(config.embedding.cache_folder, exist_ok=True)
                model = SentenceTransformer(config.embedding.model_name)
                model.save(local_path)
                logger.info(f"✅ Model saved to {local_path}")
            else:
                logger.info(f"📂 Loading model from {local_path}")

            _embedding_model = SentenceTransformer(local_path, local_files_only=True)
            logger.info("✅ Embedding model loaded successfully")
            
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
        if self._collection_exists():
            logger.info(f"📂 Using existing collection: {self.collection_name}")
            return

        try:
            embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
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
    # PDF PROCESSING
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

    def _process_subject_folder(
        self, subject_name: str, subject_path: str
    ) -> List[Dict]:
        """Process all PDFs in a subject folder"""
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
                point = PointStruct(
                    id=chunk["id"],
                    vector=embedding.tolist(),
                    payload={
                        "text": chunk["text"],
                        "subject": chunk["metadata"]["subject"],
                        "file": chunk["metadata"]["file"],
                        "page": chunk["metadata"]["page"],
                        "chunk_id": chunk["metadata"]["chunk_id"],
                    },
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
                    subject_path = os.path.join(self.data_folder, subject)
                    new_chunks = self._process_subject_folder(subject, subject_path)
                    self._add_chunks_to_vectorstore(new_chunks)

            if self.subject_filter:
                subject_path = os.path.join(self.data_folder, self.subject_filter)
                new_chunks = self._process_subject_folder(self.subject_filter, subject_path)
                self._add_chunks_to_vectorstore(new_chunks)
            else:
                for subject in self.get_available_subjects():
                    subject_path = os.path.join(self.data_folder, subject)
                    new_chunks = self._process_subject_folder(subject, subject_path)
                    self._add_chunks_to_vectorstore(new_chunks)
            
            elapsed = time.time() - start_time
            logger.info(f"✅ Data reload completed in {elapsed:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to reload data: {e}")
            raise

    def _retrieve_relevant_chunks(self, query: str, k: int = None) -> List[Dict]:
        """Retrieve relevant chunks with score filtering"""
        if k is None:
            k = config.retrieval.top_k
        
        try:
            query_embedding = self.embedding_model.encode([query])[0]

            query_filter = None
            if self.subject_filter:
                query_filter = Filter(
                    must=[
                        FieldCondition(
                            key="subject", 
                            match=MatchValue(value=self.subject_filter)
                        )
                    ]
                )

            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                limit=k,
                query_filter=query_filter,
                score_threshold=config.retrieval.score_threshold,
            )

            relevant_chunks = []
            for result in search_results:
                relevant_chunks.append({
                    "text": result.payload["text"],
                    "subject": result.payload["subject"],
                    "file": result.payload["file"],
                    "page": result.payload["page"],
                    "score": result.score,
                })
            
            logger.info(f"🔍 Retrieved {len(relevant_chunks)} chunks (threshold: {config.retrieval.score_threshold})")
            return relevant_chunks
            
        except Exception as e:
            logger.error(f"Failed to retrieve chunks: {e}")
            return []

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
        """Get list of available subjects"""
        if not os.path.exists(config.app.data_folder):
            return []
        return [
            folder
            for folder in os.listdir(config.app.data_folder)
            if os.path.isdir(os.path.join(config.app.data_folder, folder))
        ]

    @staticmethod
    def get_subject_files(subject: str) -> List[str]:
        """Get list of PDF files for a subject"""
        subject_path = os.path.join(config.app.data_folder, subject)
        if not os.path.exists(subject_path):
            return []
        return [f for f in os.listdir(subject_path) if f.endswith(".pdf")]

    def __del__(self):
        """Cleanup resources"""
        try:
            if hasattr(self, "qdrant_client"):
                self.qdrant_client.close()
                logger.info("🔒 Qdrant client closed")
        except Exception as e:
            logger.error(f"Error closing Qdrant client: {e}")