from enum import Enum
from typing import Dict, Any


class LLMProviderConfig:
    """Configuration container for LLM providers."""
    
    def __init__(self):
        self.providers = {}
    
    def add_provider(self, provider_config):
        """Add a provider configuration."""
        self.providers[provider_config.name] = provider_config


class ProviderConfig:
    """Configuration for a specific LLM provider."""
    
    def __init__(self, name: str, base_url: str, default_model: str, extra_params: Dict[str, Any]):
        self.name = name
        self.base_url = base_url
        self.default_model = default_model
        self.extra_params = extra_params


class ModelType(Enum):
    """Supported model types with their characteristics."""
    # Qwen3 models (32K native, 131K with YaRN)
    QWEN3_0_6B = "Qwen/Qwen3-0.6B"
    QWEN3_4B = "Qwen/Qwen3-4B-AWQ"
    QWEN3_8B = "Qwen/Qwen3-8B-AWQ"
    QWEN3_14B = "Qwen/Qwen3-14B-AWQ"
    QWEN3_32B = "Qwen/Qwen3-32B-AWQ"
    QWEN3_30B_A3B = "unsloth/Qwen3-30B-A3B-128K-GGUF"
    QWEN3_30B_A3B_GPTQ = "Qwen/Qwen3-30B-A3B-GPTQ-Int4"

    # Devstral
    DEVSTRAL_SMALL = "mistralai/devstral-small-2505"

    # GEMMA3 models (latest generation)
    GEMMA3_3B = "google/gemma-3-3b-it"
    GEMMA3_9B = "google/gemma-3-9b-it" 
    GEMMA3_27B = "google/gemma-3-27b-it"

    @property
    def supports_tools(self) -> bool:
        """Check if model supports tool/function calling."""
        # Qwen3, Devstral, and GEMMA3 support function calling
        return self in {
            self.QWEN3_4B, self.QWEN3_8B, self.QWEN3_14B, self.QWEN3_32B,
            self.QWEN3_30B_A3B, self.QWEN3_30B_A3B_GPTQ,
            self.DEVSTRAL_SMALL,
            self.GEMMA3_3B, self.GEMMA3_9B, self.GEMMA3_27B
        }

    @property
    def supports_reasoning(self) -> bool:
        """Check if model supports reasoning/thinking output."""
        # Qwen3 models support thinking mode with enable_thinking parameter
        return self in {
            self.QWEN3_0_6B, self.QWEN3_4B, self.QWEN3_8B, self.QWEN3_14B,
            self.QWEN3_32B, self.QWEN3_30B_A3B, self.QWEN3_30B_A3B_GPTQ,
        }

    @property
    def context_length(self) -> int:
        """Get default context length for the model (native support)."""
        context_map = {
            # Qwen3 models (32K native, 131K with YaRN rope scaling)
            self.QWEN3_0_6B: 32768,
            self.QWEN3_4B: 32768,
            self.QWEN3_8B: 32768,
            self.QWEN3_14B: 32768,
            self.QWEN3_32B: 32768,
            self.QWEN3_30B_A3B: 128000,  # Unsloth has extended context via template - check unsloth docs 
            self.QWEN3_30B_A3B_GPTQ: 32768,

            # Devstral
            self.DEVSTRAL_SMALL: 32768,

            # GEMMA3 models
            self.GEMMA3_3B: 8192,
            self.GEMMA3_9B: 8192,
            self.GEMMA3_27B: 8192,
        }
        return context_map.get(self, 4096)

    @property
    def extended_context_length(self) -> int:
        """Get extended context length with rope scaling (YaRN for Qwen3)."""
        if self in {
            self.QWEN3_0_6B, self.QWEN3_4B, self.QWEN3_8B, self.QWEN3_14B,
            self.QWEN3_32B, self.QWEN3_30B_A3B, self.QWEN3_30B_A3B_GPTQ,
        }:
            return 131072  # Qwen3 with YaRN scaling
        return self.context_length  # No extended context for other models

    @property
    def is_quantized(self) -> bool:
        """Check if model is quantized for memory efficiency."""
        model_name = self.value.upper()
        return any(quant_type in model_name for quant_type in ['AWQ', 'GPTQ', 'GGUF'])

    @property
    def quantization_type(self) -> str:
        """Get quantization type for the model."""
        model_name = self.value.upper()
        if 'AWQ' in model_name:
            return "awq"
        elif 'GGUF' in model_name:
            return "gguf"
        elif 'GPTQ' in model_name:
            return "gptq"
        return "none"


def get_default_config() -> LLMProviderConfig:
    """Get default configuration optimised for Qwen3 models."""
    config = LLMProviderConfig()

    # vLLM provider for local model serving
    vllm_config = ProviderConfig(
        name="vllm",
        base_url="http://localhost:8000/v1",
        default_model="Qwen/Qwen3-8B-AWQ",
        extra_params={
            # Qwen3 thinking mode settings (default recommended)
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 1.5,  # Reduces repetition in quantized models
            "max_tokens": 32768,
            # Qwen3-specific parameters
            "chat_template_kwargs": {
                "enable_thinking": True  # Enable thinking mode by default
            }
        }
    )
    config.add_provider(vllm_config)

    return config


def get_model_config(model: ModelType) -> Dict[str, Any]:
    """Get recommended configuration for a specific model."""
    base_config = {
        "max_tokens": 32768,
    }

    if model in {
        ModelType.QWEN3_0_6B, ModelType.QWEN3_4B, ModelType.QWEN3_8B,
        ModelType.QWEN3_14B, ModelType.QWEN3_32B, ModelType.QWEN3_30B_A3B,
        ModelType.QWEN3_30B_A3B_GPTQ
    }:
        # Qwen3 thinking mode settings (recommended by Qwen team)
        base_config.update({
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 1.5,  # Reduces repetition, especially for quantized models
            "chat_template_kwargs": {"enable_thinking": True}
        })
    elif model == ModelType.DEVSTRAL_SMALL:
        # Devstral specific settings for coding tasks
        base_config.update({
            "temperature": 0.6,  # Lower temp for more focused code generation
            "top_p": 0.95,
            "top_k": 40,
            "repetition_penalty": 1.05,
            "max_tokens": 4096,  # Higher token limit for code generation
        })
    elif model.name.startswith("GEMMA3"):
        # GEMMA3 specific settings
        base_config.update({
            "temperature": 0.9,  # Gemma tends to be more deterministic
            "top_p": 0.8,
            "top_k": 40,
            "repetition_penalty": 1.15,
            "max_tokens": 4096,  # Reasonable limit for Gemma models
        })

    return base_config


def get_qwen3_non_thinking_config(model: ModelType) -> Dict[str, Any]:
    """Get Qwen3 configuration for non-thinking mode (more efficient)."""
    if not model.supports_reasoning:
        raise ValueError(f"Model {model} does not support reasoning/thinking modes")
    
    return {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
        "max_tokens": 32768,
        "chat_template_kwargs": {"enable_thinking": False}
    } 