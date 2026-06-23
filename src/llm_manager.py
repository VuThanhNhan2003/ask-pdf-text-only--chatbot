"""
LLM Model Manager - Handles API, Groq, Remote GPU, and Proxy models
"""
import logging
import os
import json
from typing import Optional, Dict, Any, Generator
from abc import ABC, abstractmethod

import httpx
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI          # reused for Groq (OpenAI-compatible)

from config import config

logger = logging.getLogger("RAGProcessor.LLM")


# ============================================================
# Base class
# ============================================================

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


# ============================================================
# Gemini (Google API)
# ============================================================

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


# ============================================================
# Groq (OpenAI-compatible API)
# ============================================================

class GroqLLM(BaseLLM):
    """
    Groq API LLM using ChatOpenAI with Groq's OpenAI-compatible endpoint.

    Groq base URL : https://api.groq.com/openai/v1
    Default model : qwen/qwen3-32b
    Fixed params  : temperature=0.2, top_p=0.9, max_tokens=2048
                    (reasoning/thinking mode disabled via extra_body)

    Uses langchain-openai (already in requirements.txt) — no new dependency.
    """

    GROQ_BASE_URL = "https://api.groq.com/openai/v1"

    def __init__(
        self,
        model_name: str,
        api_key: str,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        top_p: float = 0.9,
    ):
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. "
                "Add it to your .env file: GROQ_API_KEY=gsk_..."
            )

        # langchain-openai 0.1.x uses `openai_api_base` and `openai_api_key`.
        # `model_kwargs` is forwarded as extra body params to the Groq endpoint,
        # which lets us pass top_p.
        # `extra_body` with thinking disabled is passed via model_kwargs too.
        self.llm = ChatOpenAI(
            model=model_name,
            openai_api_key=api_key,
            openai_api_base=self.GROQ_BASE_URL,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            model_kwargs={
                "reasoning_effort": "none"
            },
            streaming=True,   # enable streaming support
        )

        logger.info(
            f"✅ GroqLLM ready | model={model_name} | "
            f"temp={temperature} | top_p={top_p} | max_tokens={max_tokens}"
        )

    def invoke(self, prompt: str) -> str:
        """Non-streaming generation via Groq."""
        response = self.llm.invoke(prompt)
        return response.content

    def stream(self, prompt: str) -> Generator[str, None, None]:
        """Token-by-token streaming via Groq."""
        for chunk in self.llm.stream(prompt):
            if chunk.content:
                yield chunk.content


# ============================================================
# Proxy (vLLM + Gemini fallback)
# ============================================================

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

    def _fallback_to_gemini_text(self, prompt: str) -> str:
        """Direct Gemini fallback used when proxy is unavailable."""
        if not config.llm.api_key:
            raise RuntimeError("GOOGLE_API_KEY is not configured for Gemini fallback")

        gemini = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
            google_api_key=config.llm.api_key,
        )
        response = gemini.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)

    def _fallback_to_gemini_stream(self, prompt: str) -> Generator[str, None, None]:
        """Direct Gemini fallback stream used when proxy is unavailable."""
        if not config.llm.api_key:
            raise RuntimeError("GOOGLE_API_KEY is not configured for Gemini fallback")

        gemini = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
            google_api_key=config.llm.api_key,
        )

        for chunk in gemini.stream(prompt):
            if getattr(chunk, "content", None):
                yield chunk.content
    
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
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "stream": False,
                "presence_penalty": float(os.getenv("LLM_PRESENCE_PENALTY", "1.5")),
                "chat_template_kwargs": {"enable_thinking": False},
            }
            
            logger.debug(f"→ Calling proxy: {self.proxy_url}/v1/chat/completions")
            
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{self.proxy_url}/v1/chat/completions",
                    json=payload
                )
                
                response.raise_for_status()
                result = response.json()
                
                content = result["choices"][0]["message"]["content"]
                
                model_used = result.get("model", "unknown")
                if "gemini" in model_used.lower():
                    logger.info(f"✅ Response from Gemini (fallback)")
                else:
                    logger.info(f"✅ Response from vLLM")
                
                return content
                
        except httpx.TimeoutException:
            logger.error("⏱️ Request timeout")
            try:
                logger.warning("🔄 Falling back to Gemini after proxy timeout")
                return self._fallback_to_gemini_text(prompt)
            except Exception as fallback_error:
                logger.error(f"❌ Gemini fallback failed: {fallback_error}")
                return "[ERROR] Request timeout - server may be overloaded"
        except httpx.HTTPStatusError as e:
            logger.error(f"❌ HTTP error: {e.response.status_code}")
            if e.response.status_code == 503:
                try:
                    logger.warning("🔄 Falling back to Gemini after proxy 503")
                    return self._fallback_to_gemini_text(prompt)
                except Exception as fallback_error:
                    logger.error(f"❌ Gemini fallback failed: {fallback_error}")
            return f"[ERROR] Server error: {e.response.status_code}"
        except Exception as e:
            logger.error(f"❌ Request failed: {e}")
            try:
                logger.warning("🔄 Falling back to Gemini after request failure")
                return self._fallback_to_gemini_text(prompt)
            except Exception as fallback_error:
                logger.error(f"❌ Gemini fallback failed: {fallback_error}")
                return f"[ERROR] {str(e)}"
    
    def stream(self, prompt: str) -> Generator[str, None, None]:
        """Streaming generation via proxy"""
        try:
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "stream": True,
                "presence_penalty": float(os.getenv("LLM_PRESENCE_PENALTY", "1.5")),
                "chat_template_kwargs": {"enable_thinking": False},
            }
            
            logger.debug(f"→ Streaming from proxy: {self.proxy_url}")
            
            with httpx.Client(timeout=self.timeout) as client:
                with client.stream(
                    "POST",
                    f"{self.proxy_url}/v1/chat/completions",
                    json=payload
                ) as response:
                    
                    response.raise_for_status()
                    
                    for line in response.iter_lines():
                        if not line:
                            continue
                        
                        if line.startswith("data: "):
                            line = line[6:]
                        
                        if line == "[DONE]":
                            break
                        
                        try:
                            chunk = json.loads(line)

                            if "error" in chunk:
                                yield f"\n\n[ERROR] {chunk['error']}"
                                continue

                            if "choices" in chunk and isinstance(chunk["choices"], list) and chunk["choices"]:
                                first_choice = chunk["choices"][0]
                                message = first_choice.get("message", {}) if isinstance(first_choice, dict) else {}
                                if message and message.get("content"):
                                    yield message["content"]
                                    continue
                            
                            if "choices" in chunk:
                                delta = chunk["choices"][0].get("delta", {})
                                content = delta.get("content", "")
                                
                                if content:
                                    yield content
                        
                        except json.JSONDecodeError:
                            continue
                    
        except httpx.TimeoutException:
            logger.error("⏱️ Streaming timeout")
            try:
                logger.warning("🔄 Falling back to Gemini stream after proxy timeout")
                for chunk in self._fallback_to_gemini_stream(prompt):
                    yield chunk
            except Exception as fallback_error:
                logger.error(f"❌ Gemini fallback failed: {fallback_error}")
                yield "\n\n[ERROR] Request timeout"
        except httpx.HTTPStatusError as e:
            logger.error(f"❌ HTTP error: {e.response.status_code}")
            if e.response.status_code == 503:
                try:
                    logger.warning("🔄 Falling back to Gemini stream after proxy 503")
                    for chunk in self._fallback_to_gemini_stream(prompt):
                        yield chunk
                    return
                except Exception as fallback_error:
                    logger.error(f"❌ Gemini fallback failed: {fallback_error}")
            yield f"\n\n[ERROR] Server error: {e.response.status_code}"
        except Exception as e:
            logger.error(f"❌ Streaming error: {e}")
            try:
                logger.warning("🔄 Falling back to Gemini stream after request failure")
                for chunk in self._fallback_to_gemini_stream(prompt):
                    yield chunk
            except Exception as fallback_error:
                logger.error(f"❌ Gemini fallback failed: {fallback_error}")
                yield f"\n\n[ERROR] {str(e)}"


# ============================================================
# Deprecated
# ============================================================

class RemoteGPULLM(BaseLLM):
    """
    DEPRECATED: Use ProxyLLM instead.
    Kept for backward compatibility only.
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


# ============================================================
# LLM Manager (factory + cache)
# ============================================================

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
        
        # ----------------------------------------------------------
        # Instantiate the correct backend
        # ----------------------------------------------------------
        if model_config["type"] == "api":
            # Google Gemini
            if model_config["provider"] == "google":
                llm = GeminiLLM(
                    model_name=model_config["name"],
                    temperature=config.llm.temperature,
                    max_tokens=config.llm.max_output_tokens,
                    api_key=config.llm.api_key,
                )
            else:
                raise ValueError(f"Unsupported API provider: {model_config['provider']}")

        elif model_config["type"] == "groq":
            # ── NEW: Groq (OpenAI-compatible) ──────────────────────────
            llm = GroqLLM(
                model_name=model_config["name"],
                api_key=config.llm.groq_api_key,
                temperature=config.llm.groq_temperature,
                max_tokens=config.llm.groq_max_tokens,
                top_p=config.llm.groq_top_p,
            )

        elif model_config["type"] == "proxy":
            # vLLM via proxy
            proxy_url = os.getenv("LLM_PROXY_URL", "http://localhost:5000")
            llm = ProxyLLM(
                proxy_url=proxy_url,
                model_name=model_config["name"],
                temperature=config.llm.temperature,
                max_tokens=config.llm.max_output_tokens,
            )

        elif model_config["type"] == "remote":
            # Legacy – redirect to proxy
            logger.warning("⚠️ 'remote' type is deprecated. Redirecting to proxy mode...")
            proxy_url = os.getenv("LLM_PROXY_URL", "http://localhost:5000")
            llm = ProxyLLM(
                proxy_url=proxy_url,
                model_name=model_config["name"],
                temperature=config.llm.temperature,
                max_tokens=config.llm.max_output_tokens,
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