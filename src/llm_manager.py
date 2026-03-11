"""
LLM Model Manager - Handles API, Remote GPU, and Proxy models
"""
import logging
import os
import json
from typing import Optional, Dict, Any, Generator
from abc import ABC, abstractmethod

import httpx
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


class ProxyLLM(BaseLLM):
    """
    LLM via Lightweight Proxy (vLLM + Gemini fallback)
    
    This connects to the proxy server which handles:
    - vLLM server health checks
    - Retry logic (3x)
    - Automatic fallback to Gemini
    """
    
    def __init__(
        self,
        proxy_url: str,
        model_name: str,
        temperature: float,
        max_tokens: int
    ):
        self.proxy_url = proxy_url.rstrip('/')
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = 120  # 2 minutes
        
        logger.info(f"📡 Connecting to LLM Proxy: {self.proxy_url}")
        
        # Verify proxy is reachable
        self._check_proxy()
    
    def _check_proxy(self):
        """Check if proxy is reachable"""
        try:
            response = httpx.get(
                f"{self.proxy_url}/health",
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                healthy = data.get('healthy_servers', 0)
                total = data.get('total_servers', 0)
                logger.info(f"✅ Proxy healthy: {healthy}/{total} vLLM servers available")
            else:
                logger.warning(f"⚠️ Proxy returned {response.status_code}")
                
        except Exception as e:
            logger.error(f"❌ Cannot reach proxy at {self.proxy_url}: {e}")
            logger.warning("Proxy may not be running. Start with: python src/llm_proxy.py")
    
    def invoke(self, prompt: str) -> str:
        """Non-streaming generation via proxy"""
        try:
            # Prepare request
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "stream": False
            }
            
            logger.debug(f"→ Calling proxy: {self.proxy_url}/v1/chat/completions")
            
            # Call proxy
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{self.proxy_url}/v1/chat/completions",
                    json=payload
                )
                
                response.raise_for_status()
                result = response.json()
                
                # Extract content
                content = result["choices"][0]["message"]["content"]
                
                # Log which backend was used
                model_used = result.get("model", "unknown")
                if "gemini" in model_used.lower():
                    logger.info(f"✅ Response from Gemini (fallback)")
                else:
                    logger.info(f"✅ Response from vLLM")
                
                return content
                
        except httpx.TimeoutException:
            logger.error("⏱️ Request timeout")
            return "[ERROR] Request timeout - server may be overloaded"
        except httpx.HTTPStatusError as e:
            logger.error(f"❌ HTTP error: {e.response.status_code}")
            return f"[ERROR] Server error: {e.response.status_code}"
        except Exception as e:
            logger.error(f"❌ Request failed: {e}")
            return f"[ERROR] {str(e)}"
    
    def stream(self, prompt: str) -> Generator[str, None, None]:
        """Streaming generation via proxy"""
        try:
            # Prepare request
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "stream": True
            }
            
            logger.debug(f"→ Streaming from proxy: {self.proxy_url}")
            
            # Stream from proxy
            with httpx.Client(timeout=self.timeout) as client:
                with client.stream(
                    "POST",
                    f"{self.proxy_url}/v1/chat/completions",
                    json=payload
                ) as response:
                    
                    response.raise_for_status()
                    
                    # Parse SSE stream
                    for line in response.iter_lines():
                        if not line:
                            continue
                        
                        # Remove "data: " prefix
                        if line.startswith("data: "):
                            line = line[6:]
                        
                        # Check for end
                        if line == "[DONE]":
                            break
                        
                        try:
                            # Parse JSON chunk
                            chunk = json.loads(line)
                            
                            # Extract content
                            if "choices" in chunk:
                                delta = chunk["choices"][0].get("delta", {})
                                content = delta.get("content", "")
                                
                                if content:
                                    yield content
                        
                        except json.JSONDecodeError:
                            # Skip invalid JSON
                            continue
                    
        except httpx.TimeoutException:
            logger.error("⏱️ Streaming timeout")
            yield "\n\n[ERROR] Request timeout"
        except Exception as e:
            logger.error(f"❌ Streaming error: {e}")
            yield f"\n\n[ERROR] {str(e)}"


class RemoteGPULLM(BaseLLM):
    """
    DEPRECATED: Use ProxyLLM instead
    
    This class is kept for backward compatibility but should not be used.
    ProxyLLM provides better reliability with retry and fallback.
    """
    
    def __init__(self, api_url: str, model_key: str, temperature: float, max_tokens: int, api_key: str):
        logger.warning("⚠️ RemoteGPULLM is deprecated. Use ProxyLLM instead for better reliability.")
        self.api_url = api_url.rstrip('/')
        self.model_key = model_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key
    
    def invoke(self, prompt: str) -> str:
        raise NotImplementedError("RemoteGPULLM is deprecated. Use ProxyLLM instead.")
    
    def stream(self, prompt: str) -> Generator[str, None, None]:
        raise NotImplementedError("RemoteGPULLM is deprecated. Use ProxyLLM instead.")


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
            logger.debug(f"📦 Using cached LLM: {model_key}")
            return cls._instances[model_key]
        
        # Get model config
        model_config = config.llm.get_model_config(model_key)
        
        logger.info(f"🔧 Creating LLM instance: {model_key} (type: {model_config['type']})")
        
        # Create instance based on type
        if model_config["type"] == "api":
            # API models (Gemini)
            if model_config["provider"] == "google":
                llm = GeminiLLM(
                    model_name=model_config["name"],
                    temperature=config.llm.temperature,
                    max_tokens=config.llm.max_output_tokens,
                    api_key=config.llm.api_key
                )
            else:
                raise ValueError(f"Unsupported API provider: {model_config['provider']}")
        
        elif model_config["type"] == "proxy":
            # Proxy models (vLLM via proxy)
            proxy_url = os.getenv("LLM_PROXY_URL", "http://localhost:5000")
            
            llm = ProxyLLM(
                proxy_url=proxy_url,
                model_name=model_config["name"],
                temperature=config.llm.temperature,
                max_tokens=config.llm.max_output_tokens
            )
        
        elif model_config["type"] == "remote":
            # Legacy remote type - redirect to proxy
            logger.warning("⚠️ 'remote' type is deprecated. Use 'proxy' type instead.")
            logger.warning("   Falling back to proxy mode...")
            
            proxy_url = os.getenv("LLM_PROXY_URL", "http://localhost:5000")
            
            llm = ProxyLLM(
                proxy_url=proxy_url,
                model_name=model_config["name"],
                temperature=config.llm.temperature,
                max_tokens=config.llm.max_output_tokens
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