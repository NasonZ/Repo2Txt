"""OpenAI adapter implementation with streaming and tool calling support."""

import json
import os
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)

from .base import BaseLLMAdapter
from ..models.common import (
    LLMRequest,
    LLMResponse,
    StreamEvent,
    StreamEventType,
    Tool,
    ToolCall,
    TokenUsage,
    Message,
    MessageRole,
)
from ..models.capabilities import ProviderCapabilities


class OpenAIAdapter(BaseLLMAdapter):
    """OpenAI adapter implementation with full feature support."""
    
    def __init__(self, model: str, api_key: Optional[str] = None, **kwargs):
        """Initialize OpenAI adapter."""
        super().__init__(model=model, api_key=api_key, **kwargs)
        
        # Get API key from parameter or environment
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        # Initialize client
        self.client = AsyncOpenAI(api_key=self.api_key)
        
        # Use provided model or fallback to default
        self.default_model = model or "gpt-4o-mini"
        
        # Token counting
        self._init_tokenizer()
    
    def _init_tokenizer(self):
        """Initialize tokenizer for accurate token counting."""
        try:
            import tiktoken
            # Get encoding for the model
            try:
                self.encoding = tiktoken.encoding_for_model(self.default_model)
            except KeyError:
                # Fallback to cl100k_base for newer models
                self.encoding = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            self.encoding = None
    
    @property
    def capabilities(self) -> ProviderCapabilities:
        """Return OpenAI capabilities."""
        from ..models.capabilities import get_openai_capabilities
        return get_openai_capabilities(self.model)
    
    async def complete(
        self, 
        request: LLMRequest,
        stream: bool = False
    ) -> Union[LLMResponse, AsyncGenerator[StreamEvent, None]]:
        """Generate completion with optional streaming."""
        # Override stream parameter if specified
        if stream is not None:
            request.stream = stream
        
        # Prepare messages
        messages = self._prepare_messages(request.messages)
        
        # Build request parameters
        params = self._build_params(request)
        params["messages"] = messages
        
        # Add structured output if specified
        if request.response_format:
            params["response_format"] = request.response_format
        
        # Handle tools if provided
        if request.tools:
            return await self.complete_with_tools(request, request.tools)
        
        # Execute request
        if request.stream:
            return self._stream_completion(params)
        else:
            return await self._complete(params)
    
    async def complete_with_tools(
        self,
        request: LLMRequest,
        tools: List[Tool],
        stream: bool = False
    ) -> Union[LLMResponse, AsyncGenerator[StreamEvent, None]]:
        """Generate completion with tool calling support."""
        # Override stream parameter if specified
        if stream is not None:
            request.stream = stream
        
        # Prepare tool definitions
        tool_params = self._prepare_tools(tools)
        
        # Build parameters
        params = self._build_params(request)
        params["messages"] = self._prepare_messages(request.messages)
        params["tools"] = tool_params
        
        # Add tool choice if specified
        if request.tool_choice:
            params["tool_choice"] = request.tool_choice
        
        # Handle parallel tool calls setting
        if hasattr(request, "parallel_tool_calls"):
            params["parallel_tool_calls"] = request.parallel_tool_calls
        
        # Execute request
        if request.stream:
            return self._stream_completion_with_tools(params)
        else:
            return await self._complete_with_tools(params)
    
    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        """Count tokens using tiktoken."""
        if not self.encoding:
            # Fallback estimation: ~4 characters per token
            return len(text) // 4
        
        try:
            return len(self.encoding.encode(text))
        except Exception:
            # Fallback on any encoding error
            return len(text) // 4
    
    def estimate_cost(
        self, 
        input_tokens: int, 
        output_tokens: int,
        model: Optional[str] = None
    ) -> float:
        """Estimate cost based on OpenAI pricing."""
        model = model or self.default_model
        
        # Pricing per 1M tokens (as of 2025)
        pricing = {
            "gpt-4.1": {"input": 10.0, "output": 30.0},
            "gpt-4.1-nano": {"input": 0.3, "output": 1.2},
            "gpt-4-turbo-preview": {"input": 10.0, "output": 30.0},
            "gpt-4": {"input": 30.0, "output": 60.0},
            "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
            "o4-mini": {"input": 3.0, "output": 12.0},
            "o3-mini": {"input": 15.0, "output": 60.0},
        }
        
        # Get rates or use default
        rates = pricing.get(model, {"input": 10.0, "output": 30.0})
        
        # Calculate costs
        input_cost = (input_tokens / 1_000_000) * rates["input"]
        output_cost = (output_tokens / 1_000_000) * rates["output"]
        
        return input_cost + output_cost
    
    def _build_params(self, request: LLMRequest) -> Dict[str, Any]:
        """Build API parameters from request."""
        params = {
            "model": request.model or self.default_model,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "stream": request.stream,
        }
        
        # Add optional parameters
        if request.max_tokens:
            params["max_tokens"] = request.max_tokens
        
        # Handle reasoning models
        if request.model and request.model.startswith(("o3", "o4")):
            # Add reasoning effort for o-series models
            if hasattr(request, "reasoning_effort"):
                params["reasoning_effort"] = request.reasoning_effort
            # Enable storage for reasoning models
            params["store"] = True
        
        return params
    
    def _prepare_messages(
        self, 
        messages: List[Message]
    ) -> List[ChatCompletionMessageParam]:
        """Convert messages to OpenAI format."""
        prepared = []
        for msg in messages:
            # Simple text content for now
            prepared.append({
                "role": msg.role.value,
                "content": msg.content,
            })
        return prepared
    
    def _prepare_tools(self, tools: List[Tool]) -> List[ChatCompletionToolParam]:
        """Convert tools to OpenAI format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                    "strict": True,  # Enable strict mode for reliable outputs
                }
            }
            for tool in tools
        ]
    
    async def _complete(self, params: Dict[str, Any]) -> LLMResponse:
        """Non-streaming completion."""
        response = await self.client.chat.completions.create(**params)
        
        # Extract message
        message = response.choices[0].message
        content = message.content or ""
        
        # Handle refusal
        if hasattr(message, "refusal") and message.refusal:
            content = f"[REFUSAL] {message.refusal}"
        
        # Extract usage
        usage = None
        if response.usage:
            usage = TokenUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )
        
        return LLMResponse(
            content=content,
            usage=usage,
            model=response.model,
            finish_reason=response.choices[0].finish_reason,
        )
    
    async def _complete_with_tools(self, params: Dict[str, Any]) -> LLMResponse:
        """Non-streaming completion with tools."""
        response = await self.client.chat.completions.create(**params)
        
        # Extract message
        message = response.choices[0].message
        content = message.content or ""
        
        # Extract tool calls
        tool_calls = None
        if message.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc.id,
                    tool_name=tc.function.name,
                    input=json.loads(tc.function.arguments),
                )
                for tc in message.tool_calls
            ]
        
        # Extract usage
        usage = None
        if response.usage:
            usage = TokenUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )
        
        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            usage=usage,
            model=response.model,
            finish_reason=response.choices[0].finish_reason,
        )
    
    async def _stream_completion(
        self, 
        params: Dict[str, Any]
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream completion events."""
        # Send start event
        yield StreamEvent(
            event_type=StreamEventType.MESSAGE_START,
            metadata={"model": params["model"]},
        )
        
        # Stream response
        accumulated_content = ""
        stream = await self.client.chat.completions.create(**params)
        
        async for chunk in stream:
            chunk: ChatCompletionChunk
            
            # Extract delta content
            if chunk.choices and chunk.choices[0].delta.content:
                delta = chunk.choices[0].delta.content
                accumulated_content += delta
                
                yield StreamEvent(
                    event_type=StreamEventType.CONTENT_DELTA,
                    content=accumulated_content,
                    delta=delta,
                )
            
            # Check for completion
            if chunk.choices and chunk.choices[0].finish_reason:
                # Extract usage if available
                usage = None
                if hasattr(chunk, "usage") and chunk.usage:
                    usage = TokenUsage(
                        input_tokens=chunk.usage.prompt_tokens,
                        output_tokens=chunk.usage.completion_tokens,
                        total_tokens=chunk.usage.total_tokens,
                    )
                
                yield StreamEvent(
                    event_type=StreamEventType.MESSAGE_END,
                    content=accumulated_content,
                    usage=usage,
                    metadata={
                        "finish_reason": chunk.choices[0].finish_reason,
                        "model": chunk.model,
                    },
                )
    
    async def _stream_completion_with_tools(
        self, 
        params: Dict[str, Any]
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream completion with tool calling."""
        # Send start event
        yield StreamEvent(
            event_type=StreamEventType.MESSAGE_START,
            metadata={"model": params["model"]},
        )
        
        # Track state
        accumulated_content = ""
        tool_calls: Dict[int, Dict[str, Any]] = {}
        
        stream = await self.client.chat.completions.create(**params)
        
        async for chunk in stream:
            chunk: ChatCompletionChunk
            
            # Extract content delta
            if chunk.choices and chunk.choices[0].delta.content:
                delta = chunk.choices[0].delta.content
                accumulated_content += delta
                
                yield StreamEvent(
                    event_type=StreamEventType.CONTENT_DELTA,
                    content=accumulated_content,
                    delta=delta,
                )
            
            # Extract tool call deltas
            if chunk.choices and chunk.choices[0].delta.tool_calls:
                for tc_delta in chunk.choices[0].delta.tool_calls:
                    index = tc_delta.index
                    
                    # Initialize tool call tracking
                    if index not in tool_calls:
                        tool_calls[index] = {
                            "id": tc_delta.id,
                            "type": "function",
                            "function": {
                                "name": "",
                                "arguments": "",
                            }
                        }
                    
                    # Update tool call data
                    if tc_delta.id:
                        tool_calls[index]["id"] = tc_delta.id
                    
                    if tc_delta.function:
                        if tc_delta.function.name:
                            tool_calls[index]["function"]["name"] = tc_delta.function.name
                        if tc_delta.function.arguments:
                            tool_calls[index]["function"]["arguments"] += tc_delta.function.arguments
                    
                    # Try to parse and emit complete tool calls
                    tc = tool_calls[index]
                    if tc["function"]["name"] and tc["function"]["arguments"]:
                        try:
                            parsed_args = json.loads(tc["function"]["arguments"])
                            yield StreamEvent(
                                event_type=StreamEventType.TOOL_USE,
                                tool_call=ToolCall(
                                    id=tc["id"],
                                    tool_name=tc["function"]["name"],
                                    input=parsed_args,
                                ),
                            )
                        except json.JSONDecodeError:
                            # Arguments not complete yet
                            pass
            
            # Check for completion
            if chunk.choices and chunk.choices[0].finish_reason:
                # Extract usage if available
                usage = None
                if hasattr(chunk, "usage") and chunk.usage:
                    usage = TokenUsage(
                        input_tokens=chunk.usage.prompt_tokens,
                        output_tokens=chunk.usage.completion_tokens,
                        total_tokens=chunk.usage.total_tokens,
                    )
                
                yield StreamEvent(
                    event_type=StreamEventType.MESSAGE_END,
                    content=accumulated_content,
                    usage=usage,
                    metadata={
                        "finish_reason": chunk.choices[0].finish_reason,
                        "model": chunk.model,
                    },
                )