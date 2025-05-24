"""Factory for creating LLM adapters with unified configuration."""

from typing import Dict, Optional, Type, Union

from .base import BaseLLMAdapter
from .vllm_adapter import VLLMAdapter
from .openai_adapter import OpenAIAdapter
from ..config import ModelType, get_model_config
from ..models.capabilities import ProviderCapabilities


class AdapterFactory:
    """Factory for creating and managing LLM adapter instances."""
    
    # Registry of available adapter classes
    _adapters: Dict[str, Type[BaseLLMAdapter]] = {
        "vllm": VLLMAdapter,
        "openai": OpenAIAdapter,
    }
    
    @classmethod
    def register_adapter(cls, provider: str, adapter_class: Type[BaseLLMAdapter]):
        """Register a new adapter type."""
        cls._adapters[provider] = adapter_class
    
    @classmethod
    def get_provider_for_model(cls, model: Union[str, ModelType]) -> str:
        """Determine the appropriate provider for a given model."""
        if isinstance(model, ModelType):
            model_name = model.value
        else:
            model_name = model.lower()
        
        # Qwen3 models through vLLM (self-hosted)
        if any(name in model_name for name in ["qwen3", "unsloth/qwen3"]):
            return "vllm"
        
        # Mistral models through vLLM (self-hosted)  
        if any(name in model_name for name in ["mistral", "devstral"]):
            return "vllm"
        
        # Gemma models through vLLM (self-hosted)
        if "gemma" in model_name:
            return "vllm"
        
        # DeepSeek models through vLLM
        if "deepseek" in model_name:
            return "vllm"
        
        # OpenAI models through OpenAI API
        if any(name in model_name for name in ["gpt-3.5", "gpt-4", "gpt-4o"]):
            return "openai"
        
        # Claude models through OpenAI-compatible API (if configured)
        if "claude" in model_name:
            return "openai"  # Anthropic uses OpenAI-compatible API format
        
        # Default to vLLM for self-hosted models
        return "vllm"
    
    @classmethod
    def create_adapter(
        cls,
        model: Union[str, ModelType],
        provider: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs
    ) -> BaseLLMAdapter:
        """Create an adapter instance for the given model and provider.
        
        Args:
            model: Model name or ModelType enum
            provider: Provider name (auto-detected if not specified)
            base_url: Base URL for API endpoint
            api_key: API key for authentication
            **kwargs: Additional adapter-specific parameters
            
        Returns:
            Configured adapter instance
            
        Raises:
            ValueError: If provider is unsupported or model is invalid
        """
        # Convert ModelType to string if needed
        if isinstance(model, ModelType):
            model_name = model.value
        else:
            model_name = model
        
        # Auto-detect provider if not specified
        if provider is None:
            provider = cls.get_provider_for_model(model_name)
        
        # Check if provider is supported
        if provider not in cls._adapters:
            available = ", ".join(cls._adapters.keys())
            raise ValueError(f"Unsupported provider '{provider}'. Available: {available}")
        
        # Get adapter class
        adapter_class = cls._adapters[provider]
        
        # Get model configuration
        try:
            if isinstance(model, ModelType):
                config = get_model_config(model)
            else:
                # Try to find matching ModelType
                model_type = None
                for mt in ModelType:
                    if mt.value == model_name or model_name in mt.value:
                        model_type = mt
                        break
                
                if model_type:
                    config = get_model_config(model_type)
                else:
                    # Use default configuration for unknown models
                    from ..config import get_default_config
                    config = get_default_config()
        except Exception:
            # Fallback to default config
            from ..config import get_default_config
            config = get_default_config()
        
        # Prepare adapter parameters
        adapter_params = {
            "model": model_name,
            **kwargs
        }
        
        # Provider-specific parameter handling
        if provider == "vllm":
            # vLLM defaults
            adapter_params["base_url"] = base_url or "http://localhost:8000"
            
            # Add vLLM-specific config if available
            if hasattr(config, 'vllm') and config.vllm:
                adapter_params.update(config.vllm)
                
        elif provider == "openai":
            # OpenAI API parameters
            adapter_params["api_key"] = api_key
            if base_url:
                adapter_params["base_url"] = base_url
            
            # Add OpenAI-specific config if available  
            if hasattr(config, 'openai') and config.openai:
                adapter_params.update(config.openai)
        
        # Create and return adapter instance
        return adapter_class(**adapter_params)
    
    @classmethod
    def create_vllm_adapter(
        cls,
        model: Union[str, ModelType],
        base_url: str = "http://localhost:8000",
        **kwargs
    ) -> VLLMAdapter:
        """Convenience method to create a vLLM adapter."""
        return cls.create_adapter(
            model=model,
            provider="vllm", 
            base_url=base_url,
            **kwargs
        )
    
    @classmethod
    def create_openai_adapter(
        cls,
        model: Union[str, ModelType],
        api_key: str,
        base_url: Optional[str] = None,
        **kwargs
    ) -> OpenAIAdapter:
        """Convenience method to create an OpenAI adapter."""
        return cls.create_adapter(
            model=model,
            provider="openai",
            api_key=api_key,
            base_url=base_url,
            **kwargs
        )
    
    @classmethod
    def get_available_providers(cls) -> list[str]:
        """Get list of available provider names."""
        return list(cls._adapters.keys())
    
    @classmethod
    def get_adapter_capabilities(cls, provider: str, model: str) -> ProviderCapabilities:
        """Get capabilities for a specific provider and model."""
        if provider not in cls._adapters:
            raise ValueError(f"Unknown provider: {provider}")
        
        # Create temporary adapter to get capabilities
        try:
            adapter = cls.create_adapter(model=model, provider=provider)
            return adapter.capabilities
        except Exception:
            # Return basic capabilities as fallback
            from ..models.capabilities import ProviderCapabilities
            return ProviderCapabilities(
                supports_streaming=True,
                supports_tools=False,
                supports_reasoning=False,
                max_context_length=4096,
                cost_per_input_token=0.0,
                cost_per_output_token=0.0
            )
    
    @classmethod
    def is_model_supported(cls, model: Union[str, ModelType]) -> bool:
        """Check if a model is supported by any provider."""
        try:
            provider = cls.get_provider_for_model(model)
            return provider in cls._adapters
        except Exception:
            return False
    
    @classmethod
    def list_supported_models(cls) -> Dict[str, list[str]]:
        """List supported models grouped by provider."""
        supported_models = {
            "vllm": [
                # Qwen3 models
                "unsloth/Qwen3-0.6B-128K-GGUF",
                "unsloth/Qwen3-4B-128K-GGUF", 
                "unsloth/Qwen3-8B-128K-GGUF",
                "unsloth/Qwen3-14B-128K-GGUF",
                "unsloth/Qwen3-32B-128K-GGUF",
                "unsloth/Qwen3-30B-A3B-128K-GGUF",
                "unsloth/Qwen3-235B-A22B-128K-GGUF",
                # Mistral models
                "mistralai/Mistral-7B-v0.3",
                "mistralai/Mistral-Nemo-12B-2407",
                "mistralai/Devstral-7B-v0.1",
                # Gemma models
                "google/gemma-2-2b",
                "google/gemma-2-9b",
                "google/gemma-2-27b",
                # Any other vLLM-compatible model
            ],
            "openai": [
                "gpt-3.5-turbo",
                "gpt-4",
                "gpt-4-turbo",
                "gpt-4o",
                "gpt-4o-mini",
            ]
        }
        return supported_models


# Convenience functions for backward compatibility
def create_adapter(model: Union[str, ModelType], **kwargs) -> BaseLLMAdapter:
    """Create an adapter instance (convenience function)."""
    return AdapterFactory.create_adapter(model, **kwargs)


def create_vllm_adapter(model: Union[str, ModelType], **kwargs) -> VLLMAdapter:
    """Create a vLLM adapter (convenience function)."""
    return AdapterFactory.create_vllm_adapter(model, **kwargs)


def create_openai_adapter(model: Union[str, ModelType], **kwargs) -> OpenAIAdapter:
    """Create an OpenAI adapter (convenience function)."""
    return AdapterFactory.create_openai_adapter(model, **kwargs)