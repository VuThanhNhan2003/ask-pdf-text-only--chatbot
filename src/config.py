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
    
    # ✅ FIX: Đảm bảo AVAILABLE_MODELS là class variable, KHÔNG phải instance variable
    AVAILABLE_MODELS = {
        "gemini": {
            "name": "gemini-2.5-flash",
            "type": "api",
            "provider": "google",
        },
        "qwen-7b": {
            "name": "Qwen/Qwen2.5-7B-Instruct",
            "type": "remote",
            "provider": "remote-gpu",
        },
    }
    
    def __post_init__(self):
        """Initialize instance variables after dataclass creation"""
        # Current model selection from env
        self.current_model = os.getenv("LLM_MODEL", "gemini")
        
        # Validate model exists
        if self.current_model not in self.AVAILABLE_MODELS:
            print(f"⚠️ Warning: LLM_MODEL '{self.current_model}' not found in AVAILABLE_MODELS")
            print(f"Available models: {list(self.AVAILABLE_MODELS.keys())}")
            print("Falling back to 'gemini'")
            self.current_model = "gemini"
        
        # Model-specific configs
        self.model_name = self.AVAILABLE_MODELS[self.current_model]["name"]
        self.temperature = float(os.getenv("LLM_TEMPERATURE", "0.7"))
        self.max_output_tokens = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "2048"))
        self.api_key = os.getenv("GOOGLE_API_KEY", "")
        
        # Remote LLM service URL
        self.remote_llm_url = os.getenv("REMOTE_LLM_URL", "")
        self.remote_llm_api_key = os.getenv("REMOTE_LLM_API_KEY", "")
        
        # Validate remote config if using remote model
        if self.AVAILABLE_MODELS[self.current_model]["type"] == "remote":
            if not self.remote_llm_url:
                raise ValueError("REMOTE_LLM_URL is required for remote models")
            if not self.remote_llm_api_key:
                raise ValueError("REMOTE_LLM_API_KEY is required for remote models")
    
    def get_model_config(self, model_key: str) -> dict:
        """Get configuration for specific model"""
        return self.AVAILABLE_MODELS.get(model_key, self.AVAILABLE_MODELS["gemini"])


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
            # Check API key for API models
            if self.llm.current_model == "gemini" and not self.llm.api_key:
                print("❌ GOOGLE_API_KEY not set for Gemini model")
                return False
            
            # Check remote URL for remote models
            model_config = self.llm.get_model_config(self.llm.current_model)
            if model_config["type"] == "remote":
                if not self.llm.remote_llm_url:
                    print("❌ REMOTE_LLM_URL not set for remote model")
                    return False
                if not self.llm.remote_llm_api_key:
                    print("❌ REMOTE_LLM_API_KEY not set for remote model")
                    return False
            
            # Check data folder
            if not os.path.exists(self.app.data_folder):
                print(f"⚠️ Data folder not found: {self.app.data_folder}")
                return False
            
            return True
        except Exception as e:
            print(f"❌ Configuration validation failed: {e}")
            return False
    
    def print_config(self):
        """Print current configuration for debugging"""
        print("=" * 60)
        print("Current Configuration:")
        print("=" * 60)
        print(f"📂 Data folder: {self.app.data_folder}")
        print(f"📂 Log folder: {self.app.log_folder}")
        print(f"📂 Model cache: {self.embedding.cache_folder}")
        print(f"🤖 LLM Model: {self.llm.current_model}")
        print(f"🌐 Model Type: {self.llm.get_model_config(self.llm.current_model)['type']}")
        if self.llm.get_model_config(self.llm.current_model)["type"] == "remote":
            print(f"🔗 Remote URL: {self.llm.remote_llm_url}")
            print(f"🔑 API Key: {'Set' if self.llm.remote_llm_api_key else 'Not set'}")
        print("=" * 60)


# Global config instance
config = Config()