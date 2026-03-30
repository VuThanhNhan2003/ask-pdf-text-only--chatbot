"""
Configuration management for RAG Chatbot
Updated with Proxy LLM support
"""
import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

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
    """
    LLM configuration
    
    Supports:
    - API models (Gemini) - direct API calls
    - Proxy models (vLLM) - via lightweight proxy with retry & fallback
    """
    
    # Available models
    AVAILABLE_MODELS = {
        "gemini": {
            "name": "gemini-2.5-flash",
            "type": "api",
            "provider": "google",
            "description": "Google Gemini (API)"
        },
        "qwen3-8b": {
            "name": "Qwen/Qwen3-8B-AWQ",
            "type": "proxy",
            "provider": "vllm",
            "description": "Qwen 8B via Proxy (vLLM + Gemini fallback)"
        },
    }
    
    def __post_init__(self):
        """Initialize instance variables"""
        # Current model
        self.current_model = os.getenv("LLM_MODEL", "gemini")
        
        # Validate model exists
        if self.current_model not in self.AVAILABLE_MODELS:
            print(f"⚠️ Warning: LLM_MODEL '{self.current_model}' not found")
            print(f"Available: {list(self.AVAILABLE_MODELS.keys())}")
            print("Falling back to 'gemini'")
            self.current_model = "gemini"
        
        # Model-specific configs
        self.model_name = self.AVAILABLE_MODELS[self.current_model]["name"]
        self.temperature = float(os.getenv("LLM_TEMPERATURE", "0.7"))
        self.max_output_tokens = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "2048"))
        
        # API keys
        self.api_key = os.getenv("GOOGLE_API_KEY", "")
        
        # Proxy URL (for proxy models)
        self.proxy_url = os.getenv("LLM_PROXY_URL", "http://localhost:5000")
        
        # Log config
        model_type = self.AVAILABLE_MODELS[self.current_model]["type"]
        print(f"✅ LLM: {self.current_model} ({model_type})")
        
        if model_type == "proxy":
            print(f"📡 Proxy: {self.proxy_url}")
    
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
            model_config = self.llm.get_model_config(self.llm.current_model)
            
            # Check API key for API models
            if model_config["type"] == "api":
                if model_config["provider"] == "google" and not self.llm.api_key:
                    print("❌ GOOGLE_API_KEY not set for Gemini model")
                    return False
            
            # Check proxy URL for proxy models
            if model_config["type"] == "proxy":
                if not self.llm.proxy_url:
                    print("❌ LLM_PROXY_URL not set for proxy model")
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
        """Print current configuration"""
        print("=" * 60)
        print("Configuration:")
        print("=" * 60)
        print(f"📂 Data:   {self.app.data_folder}")
        print(f"📂 Logs:   {self.app.log_folder}")
        print(f"📂 Models: {self.embedding.cache_folder}")
        print(f"🤖 LLM:    {self.llm.current_model}")
        
        model_config = self.llm.get_model_config(self.llm.current_model)
        print(f"🔧 Type:   {model_config['type']}")
        
        if model_config["type"] == "proxy":
            print(f"📡 Proxy:  {self.llm.proxy_url}")
        
        print("=" * 60)


# Global config instance
config = Config()