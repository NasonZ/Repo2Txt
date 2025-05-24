"""Unified adapter manager for LLM providers."""

import asyncio
from typing import Dict, List, Optional, Union, AsyncGenerator, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

from .models.common import (
    LLMRequest, LLMResponse, Tool, Message, MessageRole,
    StreamEvent, TokenUsage, RequestMetadata, ConversationContext
)
from .models.capabilities import ProviderCapabilities, get_vllm_capabilities
from .adapter.base import BaseLLMAdapter
from .adapter.vllm_adapter import VLLMAdapter
from .adapter.openai_adapter import OpenAIAdapter
from .config import ModelType, get_model_config


logger = logging.getLogger(__name__)


class ProviderType(str, Enum):
    """Supported LLM providers."""
    VLLM = "vllm"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    MISTRAL = "mistral"


@dataclass
class AdapterConfig:
    """Configuration for an adapter instance."""
    provider: ProviderType
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_retries: int = 3
    timeout: float = 300.0
    custom_params: Dict[str, Any] = field(default_factory=dict)


class AdapterManager:
    """Manages multiple LLM adapter instances and provides unified access."""
    
    def __init__(self):
        self._adapters: Dict[str, BaseLLMAdapter] = {}
        self._default_adapter: Optional[str] = None
        self._conversation_contexts: Dict[str, ConversationContext] = {}
        
    # Adapter lifecycle management
    
    def register_adapter(self, name: str, config: AdapterConfig) -> BaseLLMAdapter:
        """Register a new adapter instance."""
        if name in self._adapters:
            logger.warning(f"Adapter '{name}' already exists, replacing...")
            
        # Create adapter based on provider type
        adapter = self._create_adapter(config)
        self._adapters[name] = adapter
        
        # Set as default if it's the first one
        if self._default_adapter is None:
            self._default_adapter = name
            
        logger.info(f"Registered adapter '{name}' for {config.provider} with model {config.model}")
        return adapter
    
    def register_vllm_adapter(self, name: str, model: str, base_url: str = "http://localhost:8000") -> BaseLLMAdapter:
        """Convenience method to register a vLLM adapter."""
        config = AdapterConfig(
            provider=ProviderType.VLLM,
            model=model,
            base_url=base_url
        )
        return self.register_adapter(name, config)
    
    def register_openai_adapter(self, name: str, model: str, api_key: str) -> BaseLLMAdapter:
        """Convenience method to register an OpenAI adapter."""
        config = AdapterConfig(
            provider=ProviderType.OPENAI,
            model=model,
            api_key=api_key
        )
        return self.register_adapter(name, config)
    
    def remove_adapter(self, name: str) -> bool:
        """Remove an adapter instance."""
        if name not in self._adapters:
            return False
            
        del self._adapters[name]
        
        # Update default if we removed it
        if self._default_adapter == name:
            self._default_adapter = next(iter(self._adapters.keys())) if self._adapters else None
            
        logger.info(f"Removed adapter '{name}'")
        return True
    
    def set_default_adapter(self, name: str) -> bool:
        """Set the default adapter for requests."""
        if name not in self._adapters:
            return False
            
        self._default_adapter = name
        logger.info(f"Set default adapter to '{name}'")
        return True
    
    def get_adapter(self, name: Optional[str] = None) -> Optional[BaseLLMAdapter]:
        """Get an adapter instance by name or the default one."""
        if name is None:
            name = self._default_adapter
            
        return self._adapters.get(name) if name else None
    
    def list_adapters(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all registered adapters."""
        return {
            name: {
                "provider": adapter.provider_name,
                "model": adapter.model,
                "capabilities": adapter.capabilities,
                "is_default": name == self._default_adapter
            }
            for name, adapter in self._adapters.items()
        }
    
    # Unified completion interface
    
    async def complete(
        self,
        messages: Union[List[Message], List[Dict[str, str]], str],
        adapter_name: Optional[str] = None,
        model: Optional[str] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[LLMResponse, AsyncGenerator[StreamEvent, None]]:
        """
        Generate completion using specified or best available adapter.
        
        Args:
            messages: Messages to complete (various formats supported)
            adapter_name: Specific adapter to use
            model: Specific model to use (will try to find compatible adapter)
            stream: Whether to stream the response
            **kwargs: Additional parameters for the request
            
        Returns:
            LLMResponse or async generator of StreamEvents
        """
        # Convert messages to standard format
        if isinstance(messages, str):
            messages = [Message(role=MessageRole.USER, content=messages)]
        elif isinstance(messages, list) and messages and isinstance(messages[0], dict):
            messages = [
                Message(
                    role=MessageRole(msg.get("role", "user")),
                    content=msg.get("content", "")
                )
                for msg in messages
            ]
        
        # Find appropriate adapter
        adapter = self._select_adapter(adapter_name, model, kwargs)
        if not adapter:
            raise ValueError("No suitable adapter found")
        
        # Create request
        request = LLMRequest(
            messages=messages,
            model=model or adapter.model,
            stream=stream,
            **kwargs
        )
        
        # Validate request can be handled
        can_handle, issues = adapter.can_handle_request(request)
        if not can_handle:
            raise ValueError(f"Adapter cannot handle request: {', '.join(issues)}")
        
        # Execute request
        return await adapter.complete(request, stream=stream)
    
    async def complete_with_tools(
        self,
        messages: Union[List[Message], List[Dict[str, str]], str],
        tools: List[Tool],
        adapter_name: Optional[str] = None,
        model: Optional[str] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[LLMResponse, AsyncGenerator[StreamEvent, None]]:
        """Generate completion with tool use capability."""
        # Convert messages format
        if isinstance(messages, str):
            messages = [Message(role=MessageRole.USER, content=messages)]
        elif isinstance(messages, list) and messages and isinstance(messages[0], dict):
            messages = [
                Message(
                    role=MessageRole(msg.get("role", "user")),
                    content=msg.get("content", "")
                )
                for msg in messages
            ]
        
        # Find adapter that supports tools
        adapter = self._select_adapter(adapter_name, model, {"requires_tools": True})
        if not adapter:
            raise ValueError("No adapter found that supports tools")
        
        if not adapter.capabilities.supports_tools:
            raise ValueError(f"Adapter {adapter} does not support tools")
        
        # Create request
        request = LLMRequest(
            messages=messages,
            model=model or adapter.model,
            tools=tools,
            stream=stream,
            **kwargs
        )
        
        return await adapter.complete_with_tools(request, tools, stream=stream)
    
    # Conversation management
    
    def create_conversation(self, conversation_id: str, system_message: Optional[str] = None) -> ConversationContext:
        """Create a new conversation context."""
        context = ConversationContext(
            conversation_id=conversation_id,
            system_message=system_message
        )
        self._conversation_contexts[conversation_id] = context
        return context
    
    def get_conversation(self, conversation_id: str) -> Optional[ConversationContext]:
        """Get an existing conversation context."""
        return self._conversation_contexts.get(conversation_id)
    
    async def continue_conversation(
        self,
        conversation_id: str,
        message: str,
        adapter_name: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Continue an existing conversation."""
        context = self.get_conversation(conversation_id)
        if not context:
            raise ValueError(f"Conversation '{conversation_id}' not found")
        
        # Add user message
        context.add_message(MessageRole.USER, message)
        
        # Generate response
        request = context.to_llm_request(**kwargs)
        adapter = self._select_adapter(adapter_name, None, kwargs)
        if not adapter:
            raise ValueError("No suitable adapter found")
        
        response = await adapter.complete(request)
        
        # Add assistant response to context
        if isinstance(response, LLMResponse):
            context.add_message(MessageRole.ASSISTANT, response.content)
            
        return response
    
    # Health and diagnostics
    
    async def health_check_all(self) -> Dict[str, bool]:
        """Check health of all registered adapters."""
        results = {}
        
        tasks = [
            (name, adapter.health_check())
            for name, adapter in self._adapters.items()
        ]
        
        for name, task in tasks:
            try:
                results[name] = await asyncio.wait_for(task, timeout=10.0)
            except asyncio.TimeoutError:
                results[name] = False
            except Exception:
                results[name] = False
                
        return results
    
    def get_usage_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get usage statistics for all adapters."""
        return {
            name: {
                "total_requests": len(adapter.get_request_history()),
                "total_tokens": adapter.get_total_tokens_used(),
                "total_cost": adapter.get_total_cost()
            }
            for name, adapter in self._adapters.items()
        }
    
    # Internal methods
    
    def _create_adapter(self, config: AdapterConfig) -> BaseLLMAdapter:
        """Create adapter instance based on configuration."""
        if config.provider == ProviderType.VLLM:
            return VLLMAdapter(
                model=config.model,
                base_url=config.base_url or "http://localhost:8000",
                **config.custom_params
            )
        elif config.provider == ProviderType.OPENAI:
            return OpenAIAdapter(
                model=config.model,
                api_key=config.api_key,
                **config.custom_params
            )
        else:
            raise ValueError(f"Unsupported provider: {config.provider}")
    
    def _select_adapter(
        self, 
        adapter_name: Optional[str], 
        model: Optional[str], 
        requirements: Dict[str, Any]
    ) -> Optional[BaseLLMAdapter]:
        """Select the best adapter for the request."""
        # Use specific adapter if requested
        if adapter_name:
            return self.get_adapter(adapter_name)
        
        # Try to find adapter by model
        if model:
            for adapter in self._adapters.values():
                if adapter.model == model:
                    return adapter
        
        # Find best adapter based on requirements
        best_adapter = None
        best_score = 0.0
        
        for adapter in self._adapters.values():
            # Check if adapter can handle requirements
            if requirements.get("requires_tools") and not adapter.capabilities.supports_tools:
                continue
            if requirements.get("requires_streaming") and not adapter.capabilities.supports_streaming:
                continue
            if requirements.get("requires_reasoning") and not adapter.capabilities.supports_reasoning:
                continue
                
            # Calculate compatibility score
            score = self._calculate_adapter_score(adapter, requirements)
            if score > best_score:
                best_score = score
                best_adapter = adapter
        
        return best_adapter or self.get_adapter()  # Fall back to default
    
    def _calculate_adapter_score(self, adapter: BaseLLMAdapter, requirements: Dict[str, Any]) -> float:
        """Calculate how well an adapter matches requirements."""
        score = 1.0  # Base score
        
        # Bonus for capabilities
        caps = adapter.capabilities
        if caps.supports_tools:
            score += 0.2
        if caps.supports_streaming:
            score += 0.1
        if caps.supports_reasoning:
            score += 0.3
        if caps.supports_structured_output:
            score += 0.1
            
        # Penalty for cost (if available)
        if caps.cost_per_input_token and caps.cost_per_input_token > 0:
            score -= min(caps.cost_per_input_token * 1000, 0.5)
            
        return score


# Global adapter manager instance
_global_manager: Optional[AdapterManager] = None


def get_adapter_manager() -> AdapterManager:
    """Get the global adapter manager instance."""
    global _global_manager
    if _global_manager is None:
        _global_manager = AdapterManager()
    return _global_manager


def initialize_default_adapters(vllm_base_url: str = "http://localhost:8000") -> AdapterManager:
    """Initialize adapter manager with common local setups."""
    manager = get_adapter_manager()
    
    # Register Qwen3 models
    for model_type in [ModelType.QWEN3_8B, ModelType.QWEN3_32B, ModelType.QWEN3_30B_A3B]:
        model_name = model_type.value.split("/")[-1]  # Extract model name
        adapter_name = f"qwen3_{model_name.split('-')[1]}"  # e.g., "qwen3_8b"
        
        try:
            manager.register_vllm_adapter(
                name=adapter_name,
                model=model_type.value,
                base_url=vllm_base_url
            )
        except Exception as e:
            logger.warning(f"Failed to register {adapter_name}: {e}")
    
    # Set Qwen3-8B as default if available
    if "qwen3_8b" in manager._adapters:
        manager.set_default_adapter("qwen3_8b")
    
    return manager 