"""
LLM Model Manager - Handles multiple LLM models (API, Local, GGUF)
"""
import logging
import os
from typing import Optional, Dict, Any, Generator
from abc import ABC, abstractmethod

from langchain_google_genai import ChatGoogleGenerativeAI
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# NEW: llama.cpp for GGUF
from llama_cpp import Llama

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


class HuggingFaceLLM(BaseLLM):
    """Local HuggingFace model LLM (Legacy - chậm)"""
    
    def __init__(self, model_name: str, temperature: float, max_tokens: int):
        logger.info(f"Loading local model: {model_name}")
        
        # Check if model exists locally
        local_path = os.path.join(config.llm.local_models_folder, model_name.replace("/", "--"))
        
        if not os.path.exists(local_path):
            logger.info(f"Downloading model {model_name} from HuggingFace...")
            os.makedirs(config.llm.local_models_folder, exist_ok=True)
            
            # Download model and tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                low_cpu_mem_usage=True
            )
            
            # Save locally
            self.tokenizer.save_pretrained(local_path)
            self.model.save_pretrained(local_path)
            logger.info(f"Model saved to {local_path}")
        else:
            logger.info(f"Loading model from {local_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(local_path, local_files_only=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                local_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                local_files_only=True,
                low_cpu_mem_usage=True
            )
        
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Create pipeline
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True if temperature > 0 else False,
        )
        
        logger.info("Local model loaded successfully")
    
    def invoke(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        
        # Apply chat template if available
        if hasattr(self.tokenizer, "apply_chat_template"):
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        else:
            formatted_prompt = prompt
        
        result = self.pipe(formatted_prompt)
        return result[0]["generated_text"].replace(formatted_prompt, "").strip()
    
    def stream(self, prompt: str) -> Generator[str, None, None]:
        # For simplicity, yield full response (streaming with HF models is more complex)
        response = self.invoke(prompt)
        # Simulate streaming by yielding in chunks
        chunk_size = 50
        for i in range(0, len(response), chunk_size):
            yield response[i:i + chunk_size]


class GGUFLLM(BaseLLM):
    """
    llama.cpp GGUF model LLM - Optimized for CPU
    Nhanh hơn 3-10x so với HuggingFace transformers
    """
    
    def __init__(self, model_config: dict, temperature: float, max_tokens: int):
        logger.info(f"Loading GGUF model: {model_config['name']}")
        
        # Create models folder
        os.makedirs(config.llm.gguf_models_folder, exist_ok=True)
        
        # Model path
        model_filename = model_config.get("file", model_config["name"])
        model_path = os.path.join(config.llm.gguf_models_folder, model_filename)
        
        # Download model if not exists
        if not os.path.exists(model_path):
            logger.info(f"📥 Downloading GGUF model from HuggingFace...")
            logger.info(f"Repo: {model_config['repo']}, File: {model_filename}")
            
            try:
                from huggingface_hub import hf_hub_download
                
                model_path = hf_hub_download(
                    repo_id=model_config["repo"],
                    filename=model_filename,
                    local_dir=config.llm.gguf_models_folder,
                    local_dir_use_symlinks=False
                )
                logger.info(f"✅ Model downloaded to {model_path}")
                
            except Exception as e:
                logger.error(f"❌ Failed to download model: {e}")
                raise RuntimeError(
                    f"Failed to download GGUF model. Please download manually:\n"
                    f"Repo: {model_config['repo']}\n"
                    f"File: {model_filename}\n"
                    f"Save to: {config.llm.gguf_models_folder}"
                )
        
        # Initialize llama.cpp
        logger.info(f"Loading model from {model_path}")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=config.llm.n_ctx,
            n_threads=config.llm.n_threads,
            n_gpu_layers=config.llm.n_gpu_layers,
            verbose=False
        )
        
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Detect model type for proper formatting
        self.model_name = model_config.get("name", "").lower()
        self.is_qwen = "qwen" in self.model_name
        
        logger.info(f"✅ GGUF model loaded successfully")
        logger.info(f"Config: n_ctx={config.llm.n_ctx}, n_threads={config.llm.n_threads}")
        logger.info(f"Model type: {'Qwen' if self.is_qwen else 'Generic'}")
    
    def _format_prompt(self, prompt: str) -> str:
        """Format prompt according to model's chat template"""
        if self.is_qwen:
            # Qwen chat template: <|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n
            formatted = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            return formatted
        else:
            # Generic format
            return prompt
    
    def invoke(self, prompt: str) -> str:
        """Generate complete response"""
        # Format prompt for model
        formatted_prompt = self._format_prompt(prompt)
        
        response = self.llm(
            formatted_prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            # CRITICAL: Stop sequences cho Qwen models
            stop=[
                "\n\nCâu hỏi:",
                "\nCâu hỏi:",
                "\n\nNgười dùng:",
                "\nNgười dùng:",
                "User:",
                "\nUser:",
                "###",
                "<|endoftext|>",
                "<|im_end|>",
                "<|im_start|>",
            ],
            echo=False,
            repeat_penalty=1.1,  # Penalize repetition
            top_p=0.9,  # Nucleus sampling
            top_k=40,  # Top-k sampling
        )
        
        return response["choices"][0]["text"].strip()
    
    def stream(self, prompt: str) -> Generator[str, None, None]:
        """Stream response token by token"""
        # Format prompt for model
        formatted_prompt = self._format_prompt(prompt)
        
        stream = self.llm(
            formatted_prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stop=[
                "\n\nCâu hỏi:",
                "\nCâu hỏi:",
                "\n\nNgười dùng:",
                "\nNgười dùng:",
                "User:",
                "\nUser:",
                "###",
                "<|endoftext|>",
                "<|im_end|>",
                "<|im_start|>",
            ],
            echo=False,
            repeat_penalty=1.1,
            top_p=0.9,
            top_k=40,
            stream=True
        )
        
        for output in stream:
            text = output["choices"][0]["text"]
            yield text


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
            return cls._instances[model_key]
        
        # Create new instance
        model_config = config.llm.get_model_config(model_key)
        
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
        
        elif model_config["type"] == "gguf":
            # NEW: GGUF model via llama.cpp
            llm = GGUFLLM(
                model_config=model_config,
                temperature=config.llm.temperature,
                max_tokens=config.llm.max_output_tokens
            )
        
        elif model_config["type"] == "local":
            # Legacy HuggingFace
            if model_config["provider"] == "huggingface":
                llm = HuggingFaceLLM(
                    model_name=model_config["name"],
                    temperature=config.llm.temperature,
                    max_tokens=config.llm.max_output_tokens
                )
            else:
                raise ValueError(f"Unsupported local provider: {model_config['provider']}")
        
        else:
            raise ValueError(f"Unsupported model type: {model_config['type']}")
        
        cls._instances[model_key] = llm
        logger.info(f"Created LLM instance: {model_key}")
        
        return llm
    
    @classmethod
    def list_available_models(cls) -> Dict[str, Dict[str, Any]]:
        """List all available models"""
        return config.llm.AVAILABLE_MODELS
    
    @classmethod
    def clear_cache(cls):
        """Clear cached model instances"""
        cls._instances.clear()
        logger.info("Cleared LLM cache")