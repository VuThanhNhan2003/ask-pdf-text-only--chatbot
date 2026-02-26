"""
LLM Model Manager - Handles API and Remote GPU models
"""
import logging
import os
import requests
from typing import Optional, Dict, Any, Generator
from abc import ABC, abstractmethod

from langchain_google_genai import ChatGoogleGenerativeAI

from config import config

logger = logging.getLogger("RAGProcessor.LLM")


class BaseLLM(ABC):
    """Base class for LLM implementations"""
    
    @abstractmethod
    def invoke(self, prompt: str) -> str:
        """Generate response"""
        pass
    
    @abstractmethod
    def stream(self, prompt: str) -> Generator[str, None, None]:
        """Stream response"""
        pass


class GeminiLLM(BaseLLM):
    """Google Gemini API LLM"""
    
    def __init__(self, model_name: str, temperature: float, max_tokens: int, api_key: str):
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            max_output_tokens=max_tokens,
            google_api_key=api_key
        )
    
    def invoke(self, prompt: str) -> str:
        response = self.llm.invoke(prompt)
        return response.content
    
    def stream(self, prompt: str) -> Generator[str, None, None]:
        for chunk in self.llm.stream(prompt):
            yield chunk.content


class RemoteGPULLM(BaseLLM):
    """Remote GPU LLM via FastAPI"""
    
    def __init__(self, api_url: str, model_key: str, temperature: float, max_tokens: int, api_key: str):
        self.api_url = api_url.rstrip('/')
        self.model_key = model_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key
        
        # Test connection
        try:
            response = requests.get(
                f"{self.api_url}/",
                headers={"X-API-Key": self.api_key},
                timeout=10
            )
            response.raise_for_status()
            logger.info(f"✅ Connected to remote LLM service: {self.api_url}")
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Failed to connect to remote LLM service: {e}")
            raise ConnectionError(f"Cannot connect to LLM service at {self.api_url}. Error: {e}")
    
    def invoke(self, prompt: str) -> str:
        """Generate text via API call"""
        try:
            logger.info(f"🌐 Calling remote LLM: {self.api_url}/generate")
            
            response = requests.post(
                f"{self.api_url}/generate",
                headers={
                    "Content-Type": "application/json",
                    "X-API-Key": self.api_key
                },
                json={
                    "prompt": prompt,
                    "model_key": self.model_key,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "stream": False
                },
                timeout=180  # 3 minutes timeout
            )
            
            response.raise_for_status()
            result = response.json()
            
            logger.info(f"✅ Remote LLM response received ({len(result['text'])} chars)")
            return result["text"]
            
        except requests.exceptions.Timeout:
            logger.error("⏱️ Request timeout")
            raise TimeoutError("LLM request timeout after 180 seconds")
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                logger.error("🔒 Authentication failed - check API key")
                raise PermissionError("Invalid API key for remote LLM service")
            else:
                logger.error(f"❌ HTTP error: {e}")
                raise
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ API request failed: {e}")
            raise
    
    def stream(self, prompt: str) -> Generator[str, None, None]:
        """Simulate streaming by chunking the response"""
        # For simplicity, get full response and yield in chunks
        response = self.invoke(prompt)
        
        # Yield in chunks of 50 chars
        chunk_size = 50
        for i in range(0, len(response), chunk_size):
            yield response[i:i + chunk_size]


class LLMManager:
    """Manages multiple LLM models"""
    
    _instances: Dict[str, BaseLLM] = {}
    
    @classmethod
    def get_llm(cls, model_key: Optional[str] = None) -> BaseLLM:
        """Get or create LLM instance"""
        if model_key is None:
            model_key = config.llm.current_model
        
        # Return cached instance
        if model_key in cls._instances:
            logger.info(f"📦 Using cached LLM: {model_key}")
            return cls._instances[model_key]
        
        # Get model config - FIX: Gọi từ instance, không phải class
        model_config = config.llm.get_model_config(model_key)
        
        logger.info(f"🔧 Creating LLM instance: {model_key} (type: {model_config['type']})")
        
        if model_config["type"] == "api":
            if model_config["provider"] == "google":
                llm = GeminiLLM(
                    model_name=model_config["name"],
                    temperature=config.llm.temperature,
                    max_tokens=config.llm.max_output_tokens,
                    api_key=config.llm.api_key
                )
            else:
                raise ValueError(f"Unsupported API provider: {model_config['provider']}")
        
        elif model_config["type"] == "remote":
            # Remote GPU LLM
            llm = RemoteGPULLM(
                api_url=config.llm.remote_llm_url,
                model_key="qwen-7b",  # Model key on remote server
                temperature=config.llm.temperature,
                max_tokens=config.llm.max_output_tokens,
                api_key=config.llm.remote_llm_api_key
            )
        
        else:
            raise ValueError(f"Unsupported model type: {model_config['type']}")
        
        cls._instances[model_key] = llm
        logger.info(f"✅ Created LLM instance: {model_key}")
        
        return llm
    
    @classmethod
    def list_available_models(cls) -> Dict[str, Dict[str, Any]]:
        """List all available models"""
        return config.llm.AVAILABLE_MODELS
    
    @classmethod
    def clear_cache(cls):
        """Clear cached model instances"""
        cls._instances.clear()
        logger.info("🧹 Cleared LLM cache")