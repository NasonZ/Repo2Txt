"""AI module for LLM integration and intelligent code analysis."""

from .adapter import (
    AdapterFactory,
    create_adapter,
    create_vllm_adapter,
    create_openai_adapter,
    BaseLLMAdapter,
    OpenAIAdapter,
    VLLMAdapter,
)
from .models.common import (
    LLMRequest,
    LLMResponse,
    StreamEvent,
    StreamEventType,
    Tool,
    ToolCall,
    Message,
    MessageRole,
    TokenUsage,
)
from .models.capabilities import ProviderCapabilities
from .config import ModelType, get_model_config, get_default_config

__all__ = [
    # Factory and creation functions
    "AdapterFactory",
    "create_adapter",
    "create_vllm_adapter",
    "create_openai_adapter",
    
    # Adapter classes
    "BaseLLMAdapter",
    "OpenAIAdapter", 
    "VLLMAdapter",
    
    # Data models
    "LLMRequest",
    "LLMResponse",
    "StreamEvent",
    "StreamEventType",
    "Tool",
    "ToolCall",
    "Message",
    "MessageRole",
    "TokenUsage",
    "ProviderCapabilities",
    
    # Configuration
    "ModelType",
    "get_model_config",
    "get_default_config",
]