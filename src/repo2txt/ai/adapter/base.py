"""Abstract base adapter for LLM providers."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, AsyncGenerator, Any
import uuid
from datetime import datetime

from ..models.common import (
    LLMRequest, LLMResponse, Tool, Message, MessageRole,
    TokenUsage, StreamEvent, RequestMetadata, ErrorInfo
)
from ..models.capabilities import ProviderCapabilities


class BaseLLMAdapter(ABC):
    """Abstract base class for all LLM provider adapters."""
    
    def __init__(self, model: str, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self._capabilities = None
        
        # Request tracking
        self._request_history: List[RequestMetadata] = []
    
    @property
    @abstractmethod
    def capabilities(self) -> ProviderCapabilities:
        """Return the capabilities of this provider."""
        pass
    
    @property
    def provider_name(self) -> str:
        """Return the name of this provider."""
        return self.__class__.__name__.replace("Adapter", "").lower()
    
    # Core completion methods
    
    @abstractmethod
    async def complete(
        self, 
        request: LLMRequest,
        stream: bool = False
    ) -> Union[LLMResponse, AsyncGenerator[StreamEvent, None]]:
        """
        Generate a completion for the given request.
        
        Args:
            request: The LLM request
            stream: Whether to stream the response
        
        Returns:
            Either a complete LLMResponse or an async generator of StreamEvents
        """
        pass
    
    @abstractmethod
    async def complete_with_tools(
        self,
        request: LLMRequest,
        tools: List[Tool],
        stream: bool = False
    ) -> Union[LLMResponse, AsyncGenerator[StreamEvent, None]]:
        """
        Generate completion with tool use capability.
        
        Args:
            request: The LLM request
            tools: Available tools for the LLM
            stream: Whether to stream the response
            
        Returns:
            Either a complete LLMResponse or an async generator of StreamEvents
        """
        pass
    
    # Token and cost management
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in the given text for this model."""
        pass
    
    def estimate_request_tokens(self, request: LLMRequest) -> int:
        """Estimate total input tokens for a request."""
        total_tokens = 0
        
        # Count message tokens
        for message in request.messages:
            total_tokens += self.count_tokens(message.content)
            
        # Add system message if present
        system_msg = request.get_system_message()
        if system_msg:
            total_tokens += self.count_tokens(system_msg)
            
        # Add tool schema tokens if tools are provided
        if request.tools:
            for tool in request.tools:
                tool_schema = f"{tool.name}: {tool.description}\n{tool.parameters}"
                total_tokens += self.count_tokens(tool_schema)
                
        return total_tokens
    
    def get_max_tokens(self) -> int:
        """Get maximum tokens supported by this model."""
        return self.capabilities.max_tokens
    
    def get_max_context_length(self) -> int:
        """Get maximum context length supported by this model."""
        return self.capabilities.max_context_length
    
    @abstractmethod
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate the cost for the given token usage."""
        pass
    
    # Thinking mode support (primarily for Qwen3)
    
    def supports_thinking_mode(self) -> bool:
        """Check if this provider supports thinking/reasoning mode."""
        return self.capabilities.supports_thinking_mode
    
    def should_enable_thinking(self, request: LLMRequest) -> bool:
        """Determine if thinking mode should be enabled for this request."""
        # Explicit request setting takes precedence
        if request.enable_reasoning:
            return self.supports_thinking_mode()
            
        # Check for thinking mode commands in messages
        for message in request.messages:
            content = message.content.lower().strip()
            if content.startswith(("/think ", "/reasoning ")):
                return self.supports_thinking_mode()
            elif content.startswith(("/no_think ", "/direct ")):
                return False
                
        return False
    
    def preprocess_thinking_commands(self, request: LLMRequest) -> LLMRequest:
        """
        Preprocess messages to handle thinking mode commands.
        
        Removes /think, /no_think commands from message content and
        updates the enable_reasoning flag accordingly.
        """
        processed_messages = []
        thinking_mode = request.enable_reasoning
        
        for message in request.messages:
            content = message.content.strip()
            
            # Check for thinking mode commands
            if content.lower().startswith("/think "):
                thinking_mode = True
                content = content[7:].strip()  # Remove "/think "
            elif content.lower().startswith("/no_think "):
                thinking_mode = False
                content = content[10:].strip()  # Remove "/no_think "
            elif content.lower().startswith("/reasoning "):
                thinking_mode = True
                content = content[11:].strip()  # Remove "/reasoning "
            elif content.lower().startswith("/direct "):
                thinking_mode = False
                content = content[8:].strip()  # Remove "/direct "
                
            # Create new message with processed content
            processed_message = Message(
                role=message.role,
                content=content,
                timestamp=message.timestamp,
                name=message.name,
                tool_call_id=message.tool_call_id
            )
            processed_messages.append(processed_message)
        
        # Create new request with processed messages and updated thinking mode
        return LLMRequest(
            messages=processed_messages,
            model=request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            tools=request.tools,
            structured_output_schema=request.structured_output_schema,
            enable_reasoning=thinking_mode,
            system_message=request.system_message,
            stream=request.stream,
            extra_params=request.extra_params
        )
    
    # Request tracking and metadata
    
    def _create_request_metadata(self, request: LLMRequest) -> RequestMetadata:
        """Create metadata for tracking a request."""
        return RequestMetadata(
            request_id=str(uuid.uuid4()),
            provider=self.provider_name,
            model=self.model,
            start_time=datetime.now(),
            token_count_estimate=self.estimate_request_tokens(request)
        )
    
    def _update_request_metadata(self, metadata: RequestMetadata, 
                               response: LLMResponse) -> None:
        """Update request metadata with response information."""
        metadata.mark_completed(response.usage)
        self._request_history.append(metadata)
    
    def _handle_request_error(self, metadata: RequestMetadata, 
                            error: Exception) -> None:
        """Handle and record request errors."""
        error_info = ErrorInfo(
            error_type=type(error).__name__,
            message=str(error),
            is_recoverable=self._is_recoverable_error(error)
        )
        metadata.mark_failed(error_info)
        self._request_history.append(metadata)
    
    def _is_recoverable_error(self, error: Exception) -> bool:
        """Determine if an error is recoverable (for retry logic)."""
        error_str = str(error).lower()
        
        # Non-recoverable errors
        if any(term in error_str for term in ["unauthorized", "api key", "permission"]):
            return False
        if "context length" in error_str or "token limit" in error_str:
            return False
            
        # Recoverable errors
        if any(term in error_str for term in ["rate limit", "timeout", "connection", "overloaded"]):
            return True
            
        return True  # Default to recoverable
    
    # Health check and diagnostics
    
    async def health_check(self) -> bool:
        """Check if the provider is accessible and working."""
        try:
            test_request = LLMRequest(
                messages=[Message(role=MessageRole.USER, content="Hello")],
                model=self.model,
                max_tokens=1,
                temperature=0.1
            )
            
            response = await self.complete(test_request)
            return isinstance(response, LLMResponse)
            
        except Exception:
            return False
    
    async def get_available_models(self) -> List[str]:
        """Get list of available models from this provider."""
        # Default implementation returns current model
        # Providers can override to query their API
        return [self.model]
    
    def get_request_history(self) -> List[RequestMetadata]:
        """Get history of requests made to this adapter."""
        return self._request_history.copy()
    
    def get_total_tokens_used(self) -> int:
        """Get total tokens used across all requests."""
        total = 0
        for metadata in self._request_history:
            if metadata.actual_token_usage:
                total += metadata.actual_token_usage.total_tokens
        return total
    
    def get_total_cost(self) -> float:
        """Get total estimated cost across all requests."""
        total_cost = 0.0
        for metadata in self._request_history:
            if metadata.actual_token_usage:
                cost = self.estimate_cost(
                    metadata.actual_token_usage.input_tokens,
                    metadata.actual_token_usage.output_tokens
                )
                total_cost += cost
        return total_cost
    
    # Utility methods
    
    def can_handle_request(self, request: LLMRequest) -> tuple[bool, List[str]]:
        """
        Check if this adapter can handle the given request.
        
        Returns:
            Tuple of (can_handle, list_of_issues)
        """
        issues = []
        
        # Check token limits
        estimated_tokens = self.estimate_request_tokens(request)
        if estimated_tokens > self.get_max_context_length():
            issues.append(f"Request tokens ({estimated_tokens}) exceed context length ({self.get_max_context_length()})")
        
        # Check tool support
        if request.tools and not self.capabilities.supports_tools:
            issues.append("Provider does not support tools")
            
        # Check streaming support
        if request.stream and not self.capabilities.supports_streaming:
            issues.append("Provider does not support streaming")
            
        # Check structured output support
        if request.structured_output_schema and not self.capabilities.supports_structured_output:
            issues.append("Provider does not support structured output")
            
        # Check reasoning support
        if request.enable_reasoning and not self.capabilities.supports_reasoning:
            issues.append("Provider does not support reasoning/thinking mode")
        
        return len(issues) == 0, issues
    
    def __repr__(self) -> str:
        """String representation of the adapter."""
        return f"{self.__class__.__name__}(model='{self.model}', provider='{self.provider_name}')"