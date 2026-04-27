"""
Lightweight LLM Proxy - Direct passthrough with retry & fallback
Features:
- Health check vLLM servers
- Retry 3x on failure
- Fallback to Gemini API
- Streaming passthrough
- Request logging
"""
import os
import time
import logging
from typing import List, Optional
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LLMProxy")

# ==================== CONFIG ====================
VLLM_SERVERS = [
    "http://10.11.9.51:8001",  # Primary vLLM via VPN
    # "http://10.11.9.51:8002",  # Backup (uncomment if needed)
]

GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY", "")
MAX_RETRIES = 3
HEALTH_CHECK_TIMEOUT = 5
REQUEST_TIMEOUT = 120

VLLM_MODEL_NAME = os.getenv("VLLM_MODEL_NAME", "Qwen/Qwen3-8B-AWQ")

# ==================== MODELS ====================
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 2048
    stream: bool = False

# ==================== VLLM MANAGER ====================
class VLLMManager:
    """Manage vLLM servers with health checks"""
    
    def __init__(self, servers: List[str]):
        self.servers = servers
        self.health_status = {server: True for server in servers}
    
    def check_health(self, server: str) -> bool:
        """Check if vLLM server is healthy"""
        try:
            with httpx.Client(timeout=HEALTH_CHECK_TIMEOUT) as client:
                response = client.get(f"{server}/health")
                is_healthy = response.status_code == 200
                
                self.health_status[server] = is_healthy
                
                if is_healthy:
                    logger.debug(f"✅ {server} is healthy")
                else:
                    logger.warning(f"⚠️ {server} unhealthy: {response.status_code}")
                
                return is_healthy
                
        except Exception as e:
            logger.warning(f"⚠️ {server} health check failed: {e}")
            self.health_status[server] = False
            return False
    
    def get_healthy_server(self) -> Optional[str]:
        """Get first healthy server"""
        for server in self.servers:
            if self.check_health(server):
                return server
        return None
    
    async def forward_to_vllm(
        self,
        server: str,
        request: ChatCompletionRequest,
        stream: bool = False
    ):
        """Forward request to vLLM server"""
        
        url = f"{server}/v1/chat/completions"
        
        payload = {
            "model": request.model,
            "messages": [msg.model_dump() for msg in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": stream,
            # Qwen3: tắt thinking mode để tránh <think>...</think> leak vào response
           "chat_template_kwargs": {"enable_thinking": False},
        }
        
        if stream:
            # Streaming response
            return await self._stream_from_vllm(url, payload)
        else:
            # Non-streaming
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                return response.json()
    
    async def _stream_from_vllm(self, url: str, payload: dict):
        """Stream response from vLLM"""
        
        async def generate():
            try:
                async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                    async with client.stream("POST", url, json=payload) as response:
                        response.raise_for_status()
                        
                        async for chunk in response.aiter_bytes():
                            if chunk:
                                yield chunk
                                
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                error_msg = f"data: {{\"error\": \"{str(e)}\"}}\n\n"
                yield error_msg.encode()
        
        return StreamingResponse(generate(), media_type="text/event-stream")


# ==================== GEMINI FALLBACK ====================
async def fallback_to_gemini(request: ChatCompletionRequest, stream: bool = False):
    """Fallback to Gemini API when vLLM unavailable."""
    
    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="vLLM unavailable and no Gemini API key configured"
        )
    
    try:
        logger.info("🔄 Falling back to Gemini API")
        
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=request.temperature,
            max_output_tokens=request.max_tokens,
            google_api_key=GEMINI_API_KEY
        )
        
        # Convert messages to prompt
        prompt = "\n".join([
            f"{msg.role}: {msg.content}" for msg in request.messages
        ])
        
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)

        result = {
            "id": f"gemini-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "gemini-2.5-flash",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }

        if not stream:
            return result

        async def gemini_sse():
            # Return a minimal OpenAI-compatible SSE stream for stream clients.
            chunk = {
                "id": result["id"],
                "object": "chat.completion.chunk",
                "created": result["created"],
                "model": result["model"],
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant", "content": content},
                    "finish_reason": "stop"
                }]
            }
            import json
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode("utf-8")
            yield b"data: [DONE]\n\n"

        return StreamingResponse(gemini_sse(), media_type="text/event-stream")
        
    except Exception as e:
        logger.error(f"Gemini fallback failed: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Both vLLM and Gemini failed: {str(e)}"
        )


# ==================== FASTAPI APP ====================
vllm_manager = VLLMManager(VLLM_SERVERS)

app = FastAPI(
    title="LLM Proxy",
    description="Lightweight proxy with retry & fallback",
    version="1.0.0"
)


@app.get("/")
def root():
    """Status endpoint"""
    return {
        "status": "online",
        "servers": VLLM_SERVERS,
        "health": vllm_manager.health_status
    }


@app.get("/health")
def health():
    """Health check"""
    healthy_servers = [
        s for s, h in vllm_manager.health_status.items() if h
    ]
    
    return {
        "status": "healthy" if healthy_servers else "degraded",
        "healthy_servers": len(healthy_servers),
        "total_servers": len(VLLM_SERVERS),
        "servers": vllm_manager.health_status
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    import time
    t0 = time.time()
    logger.info(f"📨 Request received: model={request.model}, "
                f"messages={len(request.messages)}, "
                f"prompt_len={sum(len(m.content) for m in request.messages)}")

    for attempt in range(MAX_RETRIES):
        try:
            server = vllm_manager.get_healthy_server()
            if server is None:
                break

            t1 = time.time()
            result = await vllm_manager.forward_to_vllm(server, request, stream=request.stream)
            t2 = time.time()

            # Stream mode returns a StreamingResponse, not JSON dict.
            if request.stream:
                logger.info(f"✅ vLLM stream started in {t2-t1:.2f}s | total_wall={t2-t0:.2f}s")
                return result

            usage = result.get("usage", {})
            logger.info(f"✅ vLLM done in {t2-t1:.2f}s | "
                        f"prompt={usage.get('prompt_tokens')} | "
                        f"completion={usage.get('completion_tokens')} | "
                        f"total_wall={t2-t0:.2f}s")
            return result

        except Exception as e:
            logger.warning(f"⚠️ Attempt {attempt+1} failed: {e}")
            if server:
                vllm_manager.health_status[server] = False
            await asyncio.sleep(1)

    logger.warning(f"🔄 Falling back to Gemini after {time.time()-t0:.2f}s")
    return await fallback_to_gemini(request, stream=request.stream)


@app.get("/servers")
def list_servers():
    """List configured servers and their status"""
    return {
        "servers": [
            {
                "url": server,
                "healthy": vllm_manager.health_status[server]
            }
            for server in VLLM_SERVERS
        ]
    }


# ==================== RUN ====================
if __name__ == "__main__":
    import uvicorn
    import asyncio
    
    logger.info("🚀 Starting LLM Proxy (Lightweight)")
    logger.info(f"vLLM Servers: {VLLM_SERVERS}")
    logger.info(f"Gemini fallback: {'enabled' if GEMINI_API_KEY else 'disabled'}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=5000,
        log_level="info"
    )