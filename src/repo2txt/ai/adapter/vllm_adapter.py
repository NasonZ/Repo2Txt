"""vLLM adapter implementation with support for multiple LLM families."""

import json
import asyncio
from typing import Any, AsyncGenerator, Dict, List, Optional, Union
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionChunk

from .base import BaseLLMAdapter
from ...utils.logging_config import get_logger
from ..models.common import (
    LLMRequest, LLMResponse, Tool, Message, MessageRole,
    StreamEvent, StreamEventType, ToolCall, TokenUsage
)
from ..models.capabilities import ProviderCapabilities, get_vllm_capabilities


class VLLMAdapter(BaseLLMAdapter):
    """vLLM adapter using OpenAI-compatible API with multi-model support."""
    
    def __init__(self, model: str, base_url: str = "http://localhost:8000", **kwargs):
        """Initialize vLLM adapter with OpenAI-compatible client."""
        super().__init__(model=model, base_url=base_url, **kwargs)
        self.logger = get_logger(__name__)
        
        # Initialize OpenAI-compatible client for vLLM
        self.client = AsyncOpenAI(
            api_key="EMPTY",  # vLLM doesn't require real API key
            base_url=f"{base_url}/v1" if not base_url.endswith("/v1") else base_url,
        )
        
        # Initialize tokenizer for accurate token counting
        self._init_tokenizer()
        
        self.logger.info(f"🤖 vLLM adapter ready", extra={
            "adapter_model": model,
            "adapter_server": base_url,
            "has_tokenizer": self.tokenizer is not None,
            "capabilities": {
                "tools": self.capabilities.supports_tools,
                "streaming": self.capabilities.supports_streaming,
                "reasoning": self.capabilities.supports_reasoning
            }
        })
        
    def _init_tokenizer(self):
        """Initialize tokenizer for token counting."""
        try:
            from transformers import AutoTokenizer
            # Try to load tokenizer for the model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model, trust_remote_code=True)
        except Exception:
            # Fallback: no tokenizer available
            self.tokenizer = None
    
    @property
    def capabilities(self) -> ProviderCapabilities:
        """Return vLLM capabilities based on the model."""
        return get_vllm_capabilities(self.model)
    
    def count_tokens(self, text: str) -> int:
        """Count tokens using model tokenizer."""
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except Exception:
                pass
        
        # Fallback estimation: ~4 characters per token
        return len(text) // 4
    
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """vLLM is self-hosted, so cost is zero."""
        return 0.0
    
    def _is_qwen3_model(self) -> bool:
        """Check if this is a Qwen3 model that supports thinking mode."""
        return "qwen3" in self.model.lower()
    
    def _is_reasoning_model(self) -> bool:
        """Check if this model supports reasoning/thinking."""
        model_lower = self.model.lower()
        return any(name in model_lower for name in ["qwen3", "deepseek-r1", "qwq", "devstral"])
    
    def _prepare_messages(self, messages: List[Message]) -> List[Dict[str, str]]:
        """Convert internal messages to OpenAI-compatible format."""
        openai_messages = []
        
        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                openai_messages.append({"role": "system", "content": msg.content})
            elif msg.role == MessageRole.USER:
                openai_messages.append({"role": "user", "content": msg.content})
            elif msg.role == MessageRole.ASSISTANT:
                message_dict = {"role": "assistant"}
                
                if msg.content:
                    message_dict["content"] = msg.content
                    
                # Handle tool calls
                if msg.tool_calls:
                    message_dict["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.tool_name,
                                "arguments": json.dumps(tc.input)
                            }
                        }
                        for tc in msg.tool_calls
                    ]
                
                openai_messages.append(message_dict)
            elif msg.role == MessageRole.TOOL:
                openai_messages.append({
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "content": msg.content
                })
        
        return openai_messages
    
    def _prepare_tools(self, tools: List[Tool]) -> List[Dict[str, Any]]:
        """Convert tools to OpenAI-compatible format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters
                }
            }
            for tool in tools
        ]
    
    def _build_request_params(self, request: LLMRequest) -> Dict[str, Any]:
        """Build vLLM request parameters."""
        params = {
            "model": self.model,
            "messages": self._prepare_messages(request.messages),
            "temperature": request.temperature,
            "top_p": request.top_p,
            "stream": request.stream,
        }
        
        # Add optional parameters
        if request.max_tokens:
            params["max_tokens"] = request.max_tokens
            
        if request.top_k is not None:
            params["extra_body"] = params.get("extra_body", {})
            params["extra_body"]["top_k"] = request.top_k
        
        # Handle system message
        if request.system_message:
            # Prepend system message
            params["messages"].insert(0, {"role": "system", "content": request.system_message})
        
        # Handle thinking mode for Qwen3
        if self._is_qwen3_model():
            extra_body = params.get("extra_body", {})
            chat_template_kwargs = extra_body.get("chat_template_kwargs", {})
            
            # Determine thinking mode setting
            enable_thinking = request.enable_reasoning if request.enable_reasoning is not None else True
            chat_template_kwargs["enable_thinking"] = enable_thinking
            
            extra_body["chat_template_kwargs"] = chat_template_kwargs
            params["extra_body"] = extra_body
        
        # Add any extra parameters from request
        if request.extra_params:
            extra_body = params.get("extra_body", {})
            extra_body.update(request.extra_params)
            params["extra_body"] = extra_body
        
        return params
    
    async def complete(
        self, 
        request: LLMRequest,
        stream: bool = False
    ) -> Union[LLMResponse, AsyncGenerator[StreamEvent, None]]:
        """Generate completion for the given request."""
        # Preprocess thinking commands
        if self._is_reasoning_model():
            request = self.preprocess_thinking_commands(request)
        
        # Create request metadata for tracking
        metadata = self._create_request_metadata(request)
        
        # Override stream parameter if specified
        if stream is not None:
            request.stream = stream
        
        try:
            # Build request parameters
            params = self._build_request_params(request)
            
            if request.stream:
                return self._stream_completion(params, metadata)
            else:
                response = await self._complete(params)
                self._update_request_metadata(metadata, response)
                return response
                
        except Exception as e:
            self._handle_request_error(metadata, e)
            raise
    
    async def complete_with_tools(
        self,
        request: LLMRequest,
        tools: List[Tool],
        stream: bool = False
    ) -> Union[LLMResponse, AsyncGenerator[StreamEvent, None]]:
        """Generate completion with tool calling support."""
        # Check if model supports tools
        if not self.capabilities.supports_tools:
            raise ValueError(f"Model {self.model} does not support tool calling")
        
        # Preprocess thinking commands
        if self._is_reasoning_model():
            request = self.preprocess_thinking_commands(request)
        
        # Create request metadata
        metadata = self._create_request_metadata(request)
        
        # Override stream parameter
        if stream is not None:
            request.stream = stream
        
        try:
            # Build parameters with tools
            params = self._build_request_params(request)
            params["tools"] = self._prepare_tools(tools)
            
            # Add tool choice if specified
            if request.tool_choice:
                params["tool_choice"] = request.tool_choice
                
            if request.stream:
                return self._stream_completion_with_tools(params, metadata)
            else:
                response = await self._complete_with_tools(params)
                self._update_request_metadata(metadata, response)
                return response
                
        except Exception as e:
            self._handle_request_error(metadata, e)
            raise
    
    async def _complete(self, params: Dict[str, Any]) -> LLMResponse:
        """Execute non-streaming completion."""
        response = await self.client.chat.completions.create(**params)
        
        # Extract message content
        message = response.choices[0].message
        content = message.content or ""
        
        # Extract reasoning content if available
        reasoning = None
        if hasattr(message, 'reasoning_content') and message.reasoning_content:
            reasoning = message.reasoning_content
        elif self._is_qwen3_model() and content and "<think>" in content:
            # Parse Qwen3 thinking tags manually
            reasoning = self._extract_qwen3_reasoning(content)
            # Extract just the final answer without thinking tags
            content = self._extract_qwen3_answer(content)
        
        # Extract usage information
        usage = None
        if response.usage:
            usage = TokenUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens
            )
        
        return LLMResponse(
            content=content,
            reasoning=reasoning,
            usage=usage,
            model=response.model,
            finish_reason=response.choices[0].finish_reason,
            metadata={"provider": "vllm"}
        )
    
    async def _complete_with_tools(self, params: Dict[str, Any]) -> LLMResponse:
        """Execute non-streaming completion with tools."""
        response = await self.client.chat.completions.create(**params)
        
        # Extract message
        message = response.choices[0].message
        content = message.content or ""
        
        # Extract tool calls
        tool_calls = None
        if hasattr(message, 'tool_calls') and message.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc.id,
                    tool_name=tc.function.name,
                    input=json.loads(tc.function.arguments),
                    status="pending"
                )
                for tc in message.tool_calls
            ]
        
        # Extract reasoning if available
        reasoning = None
        if hasattr(message, 'reasoning_content') and message.reasoning_content:
            reasoning = message.reasoning_content
        
        # Extract usage
        usage = None
        if response.usage:
            usage = TokenUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens
            )
        
        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            reasoning=reasoning,
            usage=usage,
            model=response.model,
            finish_reason=response.choices[0].finish_reason,
            metadata={"provider": "vllm"}
        )
    
    async def _stream_completion(
        self, 
        params: Dict[str, Any], 
        metadata
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream completion events."""
        # Send start event
        yield StreamEvent(
            event_type=StreamEventType.MESSAGE_START,
            metadata={"model": params["model"], "provider": "vllm"}
        )
        
        accumulated_content = ""
        accumulated_reasoning = ""
        
        try:
            stream = await self.client.chat.completions.create(**params)
            
            async for chunk in stream:
                chunk: ChatCompletionChunk
                
                if not chunk.choices:
                    continue
                
                delta = chunk.choices[0].delta
                
                # Handle content delta
                if delta.content:
                    accumulated_content += delta.content
                    
                    yield StreamEvent(
                        event_type=StreamEventType.CONTENT_DELTA,
                        content=accumulated_content,
                        delta=delta.content
                    )
                
                # Handle reasoning content (vLLM-specific)
                if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                    accumulated_reasoning += delta.reasoning_content
                    
                    yield StreamEvent(
                        event_type=StreamEventType.THINKING,
                        reasoning=accumulated_reasoning,
                        delta=delta.reasoning_content
                    )
                
                # Check for completion
                if chunk.choices[0].finish_reason:
                    # Extract final usage
                    usage = None
                    if hasattr(chunk, 'usage') and chunk.usage:
                        usage = TokenUsage(
                            input_tokens=chunk.usage.prompt_tokens,
                            output_tokens=chunk.usage.completion_tokens,
                            total_tokens=chunk.usage.total_tokens
                        )
                    
                    # Create final response for metadata tracking
                    final_response = LLMResponse(
                        content=accumulated_content,
                        reasoning=accumulated_reasoning if accumulated_reasoning else None,
                        usage=usage,
                        model=chunk.model,
                        finish_reason=chunk.choices[0].finish_reason
                    )
                    self._update_request_metadata(metadata, final_response)
                    
                    yield StreamEvent(
                        event_type=StreamEventType.MESSAGE_END,
                        content=accumulated_content,
                        reasoning=accumulated_reasoning if accumulated_reasoning else None,
                        usage=usage,
                        metadata={
                            "finish_reason": chunk.choices[0].finish_reason,
                            "model": chunk.model,
                            "provider": "vllm"
                        }
                    )
                    
        except Exception as e:
            self._handle_request_error(metadata, e)
            yield StreamEvent(
                event_type=StreamEventType.ERROR,
                error=str(e)
            )
    
    async def _stream_completion_with_tools(
        self, 
        params: Dict[str, Any], 
        metadata
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream completion with tool calling."""
        yield StreamEvent(
            event_type=StreamEventType.MESSAGE_START,
            metadata={"model": params["model"], "provider": "vllm"}
        )
        
        accumulated_content = ""
        accumulated_reasoning = ""
        tool_calls_buffer = {}
        
        try:
            stream = await self.client.chat.completions.create(**params)
            
            async for chunk in stream:
                chunk: ChatCompletionChunk
                
                if not chunk.choices:
                    continue
                
                delta = chunk.choices[0].delta
                
                # Handle content
                if delta.content:
                    accumulated_content += delta.content
                    yield StreamEvent(
                        event_type=StreamEventType.CONTENT_DELTA,
                        content=accumulated_content,
                        delta=delta.content
                    )
                    
                # Handle reasoning
                if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                    accumulated_reasoning += delta.reasoning_content
                    yield StreamEvent(
                        event_type=StreamEventType.THINKING,
                        reasoning=accumulated_reasoning,
                        delta=delta.reasoning_content
                    )
                    
                # Handle tool calls
                if hasattr(delta, 'tool_calls') and delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        index = tc_delta.index
                        
                        # Initialize tool call buffer
                        if index not in tool_calls_buffer:
                            tool_calls_buffer[index] = {
                                "id": tc_delta.id or "",
                                "function": {
                                    "name": "",
                                    "arguments": ""
                                }
                            }
                        
                        # Update buffer
                        if tc_delta.id:
                            tool_calls_buffer[index]["id"] = tc_delta.id
                        
                        if tc_delta.function:
                            if tc_delta.function.name:
                                tool_calls_buffer[index]["function"]["name"] = tc_delta.function.name
                            if tc_delta.function.arguments:
                                tool_calls_buffer[index]["function"]["arguments"] += tc_delta.function.arguments
                
                        # Try to emit complete tool calls
                        tc_data = tool_calls_buffer[index]
                        if tc_data["function"]["name"] and tc_data["function"]["arguments"]:
                            try:
                                args = json.loads(tc_data["function"]["arguments"])
                                tool_call = ToolCall(
                                    id=tc_data["id"],
                                    tool_name=tc_data["function"]["name"],
                                    input=args,
                                    status="pending"
                                )
                                
                                yield StreamEvent(
                                    event_type=StreamEventType.TOOL_USE,
                                    tool_call=tool_call
                                )
                            except json.JSONDecodeError:
                                # Arguments still incomplete
                                pass
                
                # Check for completion
                if chunk.choices[0].finish_reason:
                    # Extract usage
                    usage = None
                    if hasattr(chunk, 'usage') and chunk.usage:
                        usage = TokenUsage(
                            input_tokens=chunk.usage.prompt_tokens,
                            output_tokens=chunk.usage.completion_tokens,
                            total_tokens=chunk.usage.total_tokens
                        )
                    
                    # Build final tool calls list
                    final_tool_calls = []
                    for tc_data in tool_calls_buffer.values():
                        if tc_data["function"]["name"] and tc_data["function"]["arguments"]:
                            try:
                                args = json.loads(tc_data["function"]["arguments"])
                                final_tool_calls.append(ToolCall(
                                    id=tc_data["id"],
                                    tool_name=tc_data["function"]["name"],
                                    input=args,
                                    status="pending"
                                ))
                            except json.JSONDecodeError:
                                pass
                    
                    # Create final response for tracking
                    final_response = LLMResponse(
                        content=accumulated_content,
                        tool_calls=final_tool_calls if final_tool_calls else None,
                        reasoning=accumulated_reasoning if accumulated_reasoning else None,
                        usage=usage,
                        model=chunk.model,
                        finish_reason=chunk.choices[0].finish_reason
                    )
                    self._update_request_metadata(metadata, final_response)
                    
                    yield StreamEvent(
                        event_type=StreamEventType.MESSAGE_END,
                        content=accumulated_content,
                        tool_calls=final_tool_calls if final_tool_calls else None,
                        reasoning=accumulated_reasoning if accumulated_reasoning else None,
                        usage=usage,
                        metadata={
                            "finish_reason": chunk.choices[0].finish_reason,
                            "model": chunk.model,
                            "provider": "vllm"
                        }
                    )
                
        except Exception as e:
            self._handle_request_error(metadata, e)
            yield StreamEvent(
                event_type=StreamEventType.ERROR,
                error=str(e)
            )
    
    async def get_available_models(self) -> List[str]:
        """Get available models from vLLM server."""
        try:
            models = await self.client.models.list()
            return [model.id for model in models.data]
        except Exception:
            # Fallback to current model
            return [self.model]
    
    async def classify(
        self,
        text: str,
        labels: List[str],
        temperature: float = 0.1,
        **kwargs
    ) -> Dict[str, float]:
        """Use vLLM's classification capabilities with guided generation."""
        # Create classification messages
        messages = [
            Message(
                role=MessageRole.SYSTEM, 
                content=f"Classify the following text into one of these categories: {', '.join(labels)}. Respond with only the category name."
            ),
            Message(role=MessageRole.USER, content=text)
        ]
        
        # Create request for classification
        request = LLMRequest(
            messages=messages,
            temperature=temperature,
            max_tokens=50,
            stream=False,
            **kwargs
        )
        
        # Use guided choice for accurate classification if available
        if hasattr(self, '_supports_guided_generation') and self._supports_guided_generation():
            request.extra_params = request.extra_params or {}
            request.extra_params["guided_choice"] = labels
        
        try:
            response = await self.complete(request)
            
            # Extract and validate the classification result
            if response.content:
                predicted_label = response.content.strip().lower()
                
                # Find matching label (case-insensitive)
                for label in labels:
                    if label.lower() == predicted_label:
                        return {l: 1.0 if l == label else 0.0 for l in labels}
                
                # Fuzzy matching if exact match fails
                for label in labels:
                    if predicted_label in label.lower() or label.lower() in predicted_label:
                        return {l: 1.0 if l == label else 0.0 for l in labels}
            
            # Fallback to equal probabilities if classification fails
            return {label: 1.0 / len(labels) for label in labels}
            
        except Exception:
            # Return equal probabilities on error
            return {label: 1.0 / len(labels) for label in labels}
    
    async def generate_structured(
        self,
        messages: List[Message],
        schema: Union[Dict, List[str]],
        method: str = "json_schema",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Union[Dict, str]:
        """Generate structured output using vLLM's guided generation.
        
        Methods:
        - json_schema: Use JSON schema for structured output
        - regex: Use regex-based guided generation
        - grammar: Use grammar-based guided generation  
        - choice: Use guided choice selection
        """
        # Create base request
        request = LLMRequest(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
            **kwargs
        )
        
        # Configure structured generation based on method
        extra_params = {}
        
        if method == "json_schema":
            if isinstance(schema, dict):
                # JSON schema provided
                request.extra_params = {"response_format": {"type": "json_object", "schema": schema}}
            else:
                raise ValueError("JSON schema method requires a dictionary schema")
                
        elif method == "regex":
            # Use regex-based guided generation
            extra_params["guided_regex"] = schema
            
        elif method == "grammar":
            # Use grammar-based guided generation
            extra_params["guided_grammar"] = schema
            
        elif method == "choice":
            # Use guided choice selection
            if isinstance(schema, list):
                extra_params["guided_choice"] = schema
            else:
                raise ValueError("Choice method requires a list of choices")
        else:
            raise ValueError(f"Unsupported structured generation method: {method}")
        
        # Add extra parameters
        if extra_params:
            request.extra_params = request.extra_params or {}
            request.extra_params.update(extra_params)
        
        try:
            response = await self.complete(request)
            
            # Parse and return the structured content
            if response.content:
                if method == "json_schema":
                    try:
                        return json.loads(response.content)
                    except json.JSONDecodeError:
                        return {"raw_content": response.content}
                else:
                    return response.content.strip()
            else:
                raise ValueError("No content in structured response")
                
        except Exception as e:
            raise RuntimeError(f"Structured generation failed: {str(e)}")
    
    def extract_reasoning(self, content: str) -> Dict[str, str]:
        """Extract thinking and response content from model output.
        
        This is useful for models that generate thinking content inline.
        Returns dict with 'thinking' and 'response' keys.
        """
        # Handle Qwen3-style thinking tags
        if "<thinking>" in content and "</thinking>" in content:
            thinking_start = content.index("<thinking>") + len("<thinking>")
            thinking_end = content.index("</thinking>")
            thinking = content[thinking_start:thinking_end].strip()
            response = content[thinking_end + len("</thinking>"):].strip()
            return {"thinking": thinking, "response": response}
        
        # Handle other reasoning markers
        if "**Thinking:**" in content:
            parts = content.split("**Thinking:**", 1)
            if len(parts) == 2:
                thinking_part = parts[1]
                if "**Response:**" in thinking_part:
                    thinking, response = thinking_part.split("**Response:**", 1)
                    return {"thinking": thinking.strip(), "response": response.strip()}
                else:
                    return {"thinking": thinking_part.strip(), "response": ""}
        
        # Handle reasoning delimiters
        reasoning_patterns = [
            ("<!-- thinking -->", "<!-- /thinking -->"),
            ("[THINKING]", "[/THINKING]"),
            ("REASONING:", "ANSWER:"),
        ]
        
        for start_marker, end_marker in reasoning_patterns:
            if start_marker in content and end_marker in content:
                thinking_start = content.index(start_marker) + len(start_marker)
                thinking_end = content.index(end_marker)
                thinking = content[thinking_start:thinking_end].strip()
                response = content[thinking_end + len(end_marker):].strip()
                return {"thinking": thinking, "response": response}
        
        # No thinking content found
        return {"thinking": "", "response": content}
    
    def _supports_guided_generation(self) -> bool:
        """Check if the current vLLM setup supports guided generation."""
        # This would typically check vLLM version or server capabilities
        # For now, assume it's supported for most models
        return True
    
    def _extract_qwen3_reasoning(self, content: str) -> Optional[str]:
        """Extract reasoning content from Qwen3 <think> tags."""
        if not content or "<think>" not in content:
            return None
        
        start_idx = content.find("<think>") + len("<think>")
        
        if "</think>" in content:
            end_idx = content.find("</think>")
            return content[start_idx:end_idx].strip()
        else:
            # No closing tag - take everything after <think>
            return content[start_idx:].strip()
    
    def _extract_qwen3_answer(self, content: str) -> str:
        """Extract the final answer from Qwen3 response, removing <think> tags."""
        if not content or "<think>" not in content:
            return content
        
        if "</think>" in content:
            end_idx = content.find("</think>") + len("</think>")
            answer = content[end_idx:].strip()
            return answer if answer else content
        else:
            # No closing tag - return original content
            return content