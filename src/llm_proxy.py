"""
LLM Proxy - Pull Model Architecture (Safe for Slurm)

CHANGES FROM OLD VERSION:
- ❌ NO reverse tunnels
- ❌ NO auto-requeue
- ✅ Workers PULL tasks from proxy
- ✅ Task queue system
- ✅ Fallback to Gemini when no workers
- ✅ Safe for Slurm policies

Architecture:
1. Proxy receives user requests → Adds to queue
2. Workers poll proxy every 30s: "Any tasks?"
3. Proxy assigns task to worker
4. Worker processes and returns result
5. No workers? Use Gemini API fallback
"""
import asyncio
import time
import logging
from typing import List, Dict, Optional
from datetime import datetime
from collections import deque
import uuid
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager

# ==================== LOGGING ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("LLMProxy")

# ==================== MODELS ====================
class WorkerRegister(BaseModel):
    """Worker registration (via polling)"""
    worker_id: str
    model_name: Optional[str] = None
    gpu_type: Optional[str] = None
    slurm_job_id: Optional[str] = None
    metadata: Optional[Dict] = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 2048
    stream: bool = False


class TaskRequest(BaseModel):
    """Task that worker can pull"""
    task_id: str
    request: ChatCompletionRequest
    created_at: float


class TaskResult(BaseModel):
    """Result from worker"""
    task_id: str
    result: Dict
    error: Optional[str] = None


class WorkerInfo:
    """Worker information"""
    def __init__(
        self,
        worker_id: str,
        model_name: str,
        gpu_type: str,
        slurm_job_id: str,
        metadata: Dict
    ):
        self.worker_id = worker_id
        self.model_name = model_name or "unknown"
        self.gpu_type = gpu_type or "unknown"
        self.slurm_job_id = slurm_job_id or "unknown"
        self.metadata = metadata or {}
        
        self.last_heartbeat = time.time()
        self.is_healthy = True
        self.request_count = 0
        self.error_count = 0
        self.registered_at = datetime.now()
        self.current_task_id = None  # Task currently assigned
    
    def mark_healthy(self):
        self.is_healthy = True
        self.last_heartbeat = time.time()
    
    def mark_unhealthy(self):
        self.is_healthy = False
        self.error_count += 1
    
    def assign_task(self, task_id: str):
        self.current_task_id = task_id
        self.request_count += 1
    
    def complete_task(self):
        self.current_task_id = None
    
    def to_dict(self) -> Dict:
        return {
            "worker_id": self.worker_id,
            "model_name": self.model_name,
            "gpu_type": self.gpu_type,
            "slurm_job_id": self.slurm_job_id,
            "is_healthy": self.is_healthy,
            "last_heartbeat": self.last_heartbeat,
            "request_count": self.request_count,
            "error_count": self.error_count,
            "current_task": self.current_task_id,
            "registered_at": self.registered_at.isoformat(),
            "metadata": self.metadata
        }


# ==================== TASK QUEUE ====================
class TaskQueue:
    """Queue for pending tasks"""
    
    def __init__(self):
        self.pending_tasks: Dict[str, TaskRequest] = {}
        self.completed_tasks: Dict[str, TaskResult] = {}
        self.task_timeout = 120  # 2 minutes
    
    def add_task(self, request: ChatCompletionRequest) -> str:
        """Add new task to queue"""
        task_id = str(uuid.uuid4())
        task = TaskRequest(
            task_id=task_id,
            request=request,
            created_at=time.time()
        )
        self.pending_tasks[task_id] = task
        logger.info(f"📥 New task queued: {task_id}")
        return task_id
    
    def get_next_task(self) -> Optional[TaskRequest]:
        """Get next pending task (FIFO)"""
        if not self.pending_tasks:
            return None
        
        # Get oldest task
        task_id = next(iter(self.pending_tasks))
        task = self.pending_tasks.pop(task_id)
        
        # Check if timed out
        if time.time() - task.created_at > self.task_timeout:
            logger.warning(f"⏱️ Task {task_id} timed out")
            return None
        
        return task
    
    def complete_task(self, result: TaskResult):
        """Mark task as completed"""
        self.completed_tasks[result.task_id] = result
        logger.info(f"✅ Task completed: {result.task_id}")
    
    async def wait_for_result(self, task_id: str, timeout: int = 60) -> Optional[Dict]:
        """Wait for task result"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if task_id in self.completed_tasks:
                result = self.completed_tasks.pop(task_id)
                
                if result.error:
                    raise Exception(result.error)
                
                return result.result
            
            await asyncio.sleep(0.5)
        
        # Timeout
        if task_id in self.pending_tasks:
            self.pending_tasks.pop(task_id)
        
        return None


# ==================== WORKER MANAGER ====================
class WorkerManager:
    """Manages GPU workers with pull model"""
    
    def __init__(self, worker_timeout: int = 120):
        self.workers: Dict[str, WorkerInfo] = {}
        self.worker_timeout = worker_timeout
        self._cleanup_task = None
    
    async def start(self):
        """Start background cleanup"""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("✅ Worker manager started (pull model)")
    
    async def stop(self):
        """Stop background tasks"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        logger.info("🛑 Worker manager stopped")
    
    def register_worker(self, worker: WorkerRegister) -> Dict:
        """Register or update worker heartbeat"""
        
        if worker.worker_id in self.workers:
            # Update heartbeat
            self.workers[worker.worker_id].mark_healthy()
            logger.debug(f"💓 Heartbeat: {worker.worker_id}")
        else:
            # New worker
            worker_info = WorkerInfo(
                worker_id=worker.worker_id,
                model_name=worker.model_name,
                gpu_type=worker.gpu_type,
                slurm_job_id=worker.slurm_job_id,
                metadata=worker.metadata
            )
            self.workers[worker.worker_id] = worker_info
            logger.info(f"➕ New worker registered: {worker.worker_id} (Job: {worker.slurm_job_id})")
        
        return {
            "status": "registered",
            "worker_id": worker.worker_id,
            "total_workers": len(self.workers)
        }
    
    def get_healthy_workers(self) -> List[WorkerInfo]:
        """Get list of healthy workers"""
        return [w for w in self.workers.values() if w.is_healthy]
    
    def get_idle_worker(self) -> Optional[WorkerInfo]:
        """Get idle worker (no current task)"""
        healthy = self.get_healthy_workers()
        
        for worker in healthy:
            if worker.current_task_id is None:
                return worker
        
        return None
    
    async def _cleanup_loop(self):
        """Remove dead workers"""
        while True:
            try:
                await asyncio.sleep(30)
                
                current_time = time.time()
                workers_to_remove = []
                
                for worker_id, worker in self.workers.items():
                    # Remove workers with no heartbeat in timeout period
                    if current_time - worker.last_heartbeat > self.worker_timeout:
                        logger.warning(f"⏱️ Worker {worker_id} timeout (no heartbeat)")
                        workers_to_remove.append(worker_id)
                
                for worker_id in workers_to_remove:
                    del self.workers[worker_id]
                
                if workers_to_remove:
                    logger.info(f"🧹 Cleaned up {len(workers_to_remove)} dead workers")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Cleanup error: {e}")


# ==================== GEMINI FALLBACK ====================
async def fallback_to_gemini(request: ChatCompletionRequest) -> Dict:
    """Fallback to Gemini API when no workers available"""
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        import os
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise Exception("No Gemini API key configured")
        
        logger.info("🔄 Falling back to Gemini API")
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=request.temperature,
            max_output_tokens=request.max_tokens,
            google_api_key=api_key
        )
        
        # Convert messages to prompt
        prompt = "\n".join([f"{msg.role}: {msg.content}" for msg in request.messages])
        
        response = llm.invoke(prompt)
        
        return {
            "id": str(uuid.uuid4()),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "gemini-2.0-flash",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response.content
                },
                "finish_reason": "stop"
            }]
        }
        
    except Exception as e:
        logger.error(f"❌ Gemini fallback failed: {e}")
        raise HTTPException(status_code=503, detail=f"No workers and fallback failed: {str(e)}")


# ==================== GLOBAL STATE ====================
worker_manager = WorkerManager()
task_queue = TaskQueue()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown"""
    logger.info("🚀 Starting LLM Proxy (Pull Model)...")
    await worker_manager.start()
    yield
    logger.info("🛑 Stopping LLM Proxy...")
    await worker_manager.stop()


app = FastAPI(
    title="LLM Proxy (Pull Model)",
    description="Safe proxy for Slurm - workers pull tasks",
    version="2.0.0",
    lifespan=lifespan
)


# ==================== ENDPOINTS ====================
@app.get("/")
async def root():
    """Root endpoint"""
    healthy_workers = worker_manager.get_healthy_workers()
    return {
        "status": "online",
        "architecture": "pull_model",
        "total_workers": len(worker_manager.workers),
        "healthy_workers": len(healthy_workers),
        "pending_tasks": len(task_queue.pending_tasks),
        "workers": [w.to_dict() for w in worker_manager.workers.values()]
    }


@app.get("/health")
async def health():
    """Health check"""
    healthy_workers = worker_manager.get_healthy_workers()
    return {
        "status": "healthy",
        "healthy_workers": len(healthy_workers),
        "total_workers": len(worker_manager.workers),
        "pending_tasks": len(task_queue.pending_tasks)
    }


@app.post("/register")
async def register_worker(worker: WorkerRegister):
    """
    Worker heartbeat endpoint
    Workers call this every 30s to stay registered
    """
    return worker_manager.register_worker(worker)


@app.get("/workers")
async def list_workers():
    """List all workers"""
    return {
        "workers": [w.to_dict() for w in worker_manager.workers.values()],
        "total": len(worker_manager.workers),
        "healthy": len(worker_manager.get_healthy_workers())
    }


@app.get("/tasks/next")
async def get_next_task():
    """
    Worker polls this to get next task
    Called by worker every 30s
    """
    task = task_queue.get_next_task()
    
    if task is None:
        return {"task": None}
    
    # Assign to an idle worker (track who got it)
    return {
        "task": {
            "task_id": task.task_id,
            "request": task.request.dict()
        }
    }


@app.post("/tasks/complete")
async def complete_task(result: TaskResult):
    """Worker submits completed task"""
    task_queue.complete_task(result)
    return {"status": "completed"}


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint
    
    Flow:
    1. Add request to queue
    2. Wait for worker to pick it up
    3. If timeout → fallback to Gemini
    """
    
    # Check if we have workers
    healthy_workers = worker_manager.get_healthy_workers()
    
    if len(healthy_workers) == 0:
        logger.warning("⚠️ No workers available, using Gemini fallback")
        return await fallback_to_gemini(request)
    
    # Add to task queue
    task_id = task_queue.add_task(request)
    
    logger.info(f"📋 Task queued: {task_id} (Workers: {len(healthy_workers)})")
    
    try:
        # Wait for result (workers poll /tasks/next and submit result)
        result = await task_queue.wait_for_result(task_id, timeout=60)
        
        if result is None:
            logger.warning(f"⏱️ Task {task_id} timeout, falling back to Gemini")
            return await fallback_to_gemini(request)
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Task {task_id} failed: {e}")
        
        # Try fallback
        logger.info("🔄 Attempting Gemini fallback...")
        return await fallback_to_gemini(request)


# ==================== RUN ====================
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=5000,
        log_level="info"
    )