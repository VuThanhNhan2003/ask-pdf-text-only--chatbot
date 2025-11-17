"""
Configuration management for RAG Chatbot
Centralized settings with absolute paths + GGUF support
"""
import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

# Get the project root directory (parent of src/)
PROJECT_ROOT = Path(__file__).parent.parent.absolute()


@dataclass
class EmbeddingConfig:
    """Embedding model configuration"""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    local_path: str = str(PROJECT_ROOT / "models" / "all-MiniLM-L6-v2")
    batch_size: int = 32
    cache_folder: str = str(PROJECT_ROOT / "models")


@dataclass
class QdrantConfig:
    """Qdrant vector database configuration"""
    host: str = "qdrant"
    port: int = 6333
    collection_name: str = "documents"
    timeout: int = 30
    
    def __post_init__(self):
        # Allow override from environment
        self.collection_name = os.getenv("QDRANT_COLLECTION", self.collection_name)
        
        # Support both Docker (qdrant) and local dev (localhost)
        qdrant_url = os.getenv("QDRANT_URL", "")
        if qdrant_url:
            # Parse QDRANT_URL if provided (e.g., http://qdrant:6333)
            if "://" in qdrant_url:
                parts = qdrant_url.split("://")[1].split(":")
                self.host = parts[0]
                if len(parts) > 1:
                    self.port = int(parts[1].split("/")[0])


@dataclass
class ChunkingConfig:
    """Text chunking configuration"""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    separators: list = None
    
    def __post_init__(self):
        if self.separators is None:
            self.separators = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]


@dataclass
class LLMConfig:
    """LLM configuration with GGUF support"""
    # Available models
    AVAILABLE_MODELS = {
        "gemini": {
            "name": "gemini-2.0-flash",
            "type": "api",
            "provider": "google",
        },
        # GGUF Models - Optimized for CPU
        "qwen2.5-3b-q4": {
            "name": "Qwen2.5-3B-Instruct-Q4_K_M.gguf",
            "type": "gguf",
            "provider": "llama.cpp",
            "repo": "Qwen/Qwen2.5-3B-Instruct-GGUF",
            "file": "qwen2.5-3b-instruct-q4_k_m.gguf",
            "description": "4-bit quantization, cân bằng tốc độ/chất lượng"
        },
        # "qwen2.5-3b-q5": {
        #     "name": "Qwen2.5-3B-Instruct-Q5_K_M.gguf",
        #     "type": "gguf",
        #     "provider": "llama.cpp",
        #     "repo": "Qwen/Qwen2.5-3B-Instruct-GGUF",
        #     "file": "qwen2.5-3b-instruct-q5_k_m.gguf",
        #     "description": "5-bit quantization, chất lượng cao hơn"
        # },
        # Legacy HuggingFace (giữ lại nếu cần)
        # "Qwen2.5-3B-Instruct": {
        #     "name": "Qwen/Qwen2.5-3B-Instruct",
        #     "type": "local",
        #     "provider": "huggingface",
        # }
    }
    
    # Current model selection
    current_model: str = os.getenv("LLM_MODEL", "qwen2.5-3b-q4")
    
    # Model-specific configs
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    max_output_tokens: int = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "1024"))  # Giảm xuống cho nhanh hơn
    api_key: str = os.getenv("GOOGLE_API_KEY", "")
    
    # GGUF models cache
    gguf_models_folder: str = str(PROJECT_ROOT / "models" / "gguf")
    
    # llama.cpp settings
    n_ctx: int = 4096  # Context window
    n_threads: int = int(os.getenv("LLAMA_THREADS", "4"))  # CPU threads
    n_gpu_layers: int = 0  # CPU only
    
    @classmethod
    def get_model_config(cls, model_key: str) -> dict:
        """Get configuration for specific model"""
        return cls.AVAILABLE_MODELS.get(model_key, cls.AVAILABLE_MODELS["gemini"])


@dataclass
class RetrievalConfig:
    """Retrieval configuration"""
    top_k: int = 3  # Giảm từ 5 xuống 3 cho nhanh hơn
    score_threshold: float = 0.3
    max_context_length: int = 3000  # Giảm từ 4000 xuống 3000


@dataclass
class AppConfig:
    """Application configuration"""
    data_folder: str = str(PROJECT_ROOT / "data")
    log_folder: str = str(PROJECT_ROOT / "logs")
    page_title: str = "RAG Chatbot"
    page_icon: str = "🤖"
    
    def __post_init__(self):
        # Ensure folders exist
        os.makedirs(self.data_folder, exist_ok=True)
        os.makedirs(self.log_folder, exist_ok=True)


class Config:
    """Main configuration class"""
    def __init__(self):
        self.embedding = EmbeddingConfig()
        self.qdrant = QdrantConfig()
        self.chunking = ChunkingConfig()
        self.llm = LLMConfig()
        self.retrieval = RetrievalConfig()
        self.app = AppConfig()
    
    def validate(self) -> bool:
        """Validate configuration"""
        try:
            # Check API key (chỉ cần nếu dùng API model)
            current_model = self.llm.get_model_config(self.llm.current_model)
            if current_model["type"] == "api" and not self.llm.api_key:
                print("⚠️ API key required for API models")
                return False
            
            # Check data folder
            if not os.path.exists(self.app.data_folder):
                print(f"⚠️ Data folder not found: {self.app.data_folder}")
                return False
            
            return True
        except Exception as e:
            print(f"❌ Configuration validation failed: {e}")
            return False
    
    def print_paths(self):
        """Debug: print all paths"""
        print(f"📂 PROJECT_ROOT: {PROJECT_ROOT}")
        print(f"📂 Data folder: {self.app.data_folder}")
        print(f"📂 Log folder: {self.app.log_folder}")
        print(f"📂 Model cache: {self.embedding.cache_folder}")
        print(f"📂 GGUF models: {self.llm.gguf_models_folder}")


# Global config instance
config = Config()