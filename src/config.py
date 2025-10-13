"""
Configuration management for RAG Chatbot
Centralized settings with absolute paths
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
    """LLM configuration"""
    # Available models
    AVAILABLE_MODELS = {
        "gemini": {
            "name": "gemini-1.5-pro",
            "type": "api",
            "provider": "google",
        },
        # "llama-3.1-8b": {
        #     "name": "meta-llama/Llama-3.1-8B-Instruct",
        #     "type": "local",
        #     "provider": "huggingface",
        # },
        "Qwen2.5-3B-Instruct": {
            "name": "Qwen/Qwen2.5-3B-Instruct",
            "type": "local",
            "provider": "huggingface",
        }
    }
    
    # Current model selection
    current_model: str = os.getenv("LLM_MODEL", "gemini")
    
    # Model-specific configs
    model_name: str = AVAILABLE_MODELS[current_model]["name"]
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    max_output_tokens: int = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "2048"))
    api_key: str = os.getenv("GOOGLE_API_KEY", "")
    
    # Local model cache
    local_models_folder: str = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "models", "llm"
    )
    
    @classmethod
    def get_model_config(cls, model_key: str) -> dict:
        """Get configuration for specific model"""
        return cls.AVAILABLE_MODELS.get(model_key, cls.AVAILABLE_MODELS["gemini"])


@dataclass
class RetrievalConfig:
    """Retrieval configuration"""
    top_k: int = 5
    score_threshold: float = 0.3
    max_context_length: int = 4000


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
            # Check API key
            if not self.llm.api_key:
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
        print(f"📂 Model path: {self.embedding.local_path}")


# Global config instance
config = Config()