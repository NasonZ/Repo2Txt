"""Common data models for LLM adapters."""

from typing import List, Dict, Any, Optional, Union, Literal
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class MessageRole(str, Enum):
    """Valid message roles."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class ToolCall(BaseModel):
    """A tool call made by the LLM."""
    id: str
    tool_name: str  # Changed from 'tool' to match vllm_adapter expectations
    input: Dict[str, Any]
    status: Literal["pending", "success", "error"] = "pending"


class Message(BaseModel):
    """A message in the conversation."""
    role: MessageRole
    content: str
    tool_calls: Optional[List[ToolCall]] = None  # For assistant messages with tool calls
    timestamp: Optional[datetime] = Field(default_factory=datetime.now)
    name: Optional[str] = None  # For tool messages
    tool_call_id: Optional[str] = None  # For tool responses


class Tool(BaseModel):
    """Definition of a tool that can be called by the LLM."""
    name: str
    description: str
    parameters: Dict[str, Any]
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "select_files",
                "description": "Select files from repository based on criteria",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_paths": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of file paths to select"
                        },
                        "reasoning": {
                            "type": "string", 
                            "description": "Explanation for the selection"
                        }
                    },
                    "required": ["file_paths"]
                }
            }
        }


class ToolResult(BaseModel):
    """Result of a tool call execution."""
    tool_call_id: str
    output: Any
    error: Optional[str] = None
    execution_time: Optional[float] = None


class TokenUsage(BaseModel):
    """Token usage statistics."""
    input_tokens: int
    output_tokens: int
    cache_creation_tokens: Optional[int] = None
    cache_read_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None  # For Qwen3 thinking mode
    
    @property
    def total_tokens(self) -> int:
        """Calculate total tokens used."""
        return self.input_tokens + self.output_tokens


class StreamEventType(str, Enum):
    """Types of streaming events."""
    MESSAGE_START = "message_start"
    CONTENT_START = "content_start"
    CONTENT_DELTA = "content_delta"
    CONTENT_STOP = "content_stop"
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_DELTA = "tool_call_delta"
    TOOL_CALL_STOP = "tool_call_stop"
    THINKING_START = "thinking_start"
    THINKING_DELTA = "thinking_delta"
    THINKING_STOP = "thinking_stop"
    MESSAGE_STOP = "message_stop"
    ERROR = "error"
    PING = "ping"


class StreamEvent(BaseModel):
    """A streaming event from an LLM response."""
    event_type: StreamEventType
    content: Optional[str] = None
    delta: Optional[str] = None
    tool_call: Optional[Dict[str, Any]] = None
    thinking: Optional[str] = None
    usage: Optional[TokenUsage] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class LLMRequest(BaseModel):
    """Request to an LLM provider."""
    messages: List[Message]
    model: str
    max_tokens: Optional[int] = None
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: Optional[int] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    structured_output_schema: Optional[Dict[str, Any]] = None
    enable_reasoning: bool = False  # For Qwen3 thinking mode
    system_message: Optional[str] = None
    stream: bool = False
    
    # Provider-specific parameters
    extra_params: Dict[str, Any] = Field(default_factory=dict)
    
    def add_message(self, role: MessageRole, content: str, **kwargs) -> None:
        """Add a message to the request."""
        message = Message(role=role, content=content, **kwargs)
        self.messages.append(message)
    
    def get_system_message(self) -> Optional[str]:
        """Get the system message from messages or system_message field."""
        # Check for explicit system_message field first
        if self.system_message:
            return self.system_message
            
        # Look for system message in messages
        for message in self.messages:
            if message.role == MessageRole.SYSTEM:
                return message.content
                
        return None
    
    def get_conversation_messages(self) -> List[Message]:
        """Get all non-system messages."""
        return [msg for msg in self.messages if msg.role != MessageRole.SYSTEM]


class LLMResponse(BaseModel):
    """Response from an LLM provider."""
    content: str
    tool_calls: Optional[List[ToolCall]] = None
    reasoning: Optional[str] = None  # For Qwen3 thinking mode output
    usage: Optional[TokenUsage] = None
    model: str
    finish_reason: str
    response_id: Optional[str] = None
    created: datetime = Field(default_factory=datetime.now)
    
    # Provider-specific metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @property
    def has_tool_calls(self) -> bool:
        """Check if response contains tool calls."""
        return self.tool_calls is not None and len(self.tool_calls) > 0
    
    @property
    def has_reasoning(self) -> bool:
        """Check if response contains reasoning/thinking output."""
        return self.reasoning is not None and len(self.reasoning.strip()) > 0
    
    def get_pending_tool_calls(self) -> List[ToolCall]:
        """Get tool calls that haven't been executed yet."""
        if not self.tool_calls:
            return []
        return [tc for tc in self.tool_calls if tc.status == "pending"]


class ConversationContext(BaseModel):
    """Context for a conversation with an LLM."""
    conversation_id: str
    messages: List[Message] = Field(default_factory=list)
    tools: List[Tool] = Field(default_factory=list)
    total_tokens_used: int = 0
    session_started: datetime = Field(default_factory=datetime.now)
    
    def add_user_message(self, content: str) -> None:
        """Add a user message to the conversation."""
        self.messages.append(Message(role=MessageRole.USER, content=content))
    
    def add_assistant_response(self, response: LLMResponse) -> None:
        """Add an assistant response to the conversation."""
        self.messages.append(Message(
            role=MessageRole.ASSISTANT, 
            content=response.content
        ))
        self.total_tokens_used += response.usage.total_tokens
    
    def add_tool_result(self, tool_result: ToolResult) -> None:
        """Add a tool result to the conversation."""
        self.messages.append(Message(
            role=MessageRole.TOOL,
            content=str(tool_result.output),
            tool_call_id=tool_result.tool_call_id
        ))
    
    def to_llm_request(self, model: str, **kwargs) -> LLMRequest:
        """Convert conversation context to an LLM request."""
        return LLMRequest(
            messages=self.messages,
            model=model,
            tools=self.tools,
            **kwargs
        )


class ThinkingModeConfig(BaseModel):
    """Configuration for Qwen3 thinking mode."""
    enabled: bool = False
    budget_tokens: int = 16000  # Max tokens for thinking
    include_in_response: bool = True  # Whether to include thinking in response
    auto_enable_for_complex: bool = True  # Auto-enable for complex queries
    
    # Patterns that trigger thinking mode
    complexity_patterns: List[str] = Field(default_factory=lambda: [
        "analyze", "explain", "debug", "review", "architecture", 
        "design", "optimize", "refactor", "understand", "why"
    ])


class ErrorInfo(BaseModel):
    """Information about an error that occurred."""
    error_type: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    retry_count: int = 0
    is_recoverable: bool = True


class RequestMetadata(BaseModel):
    """Metadata about a request for debugging and monitoring."""
    request_id: str
    provider: str
    model: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_duration: Optional[float] = None
    token_count_estimate: Optional[int] = None
    actual_token_usage: Optional[TokenUsage] = None
    error: Optional[ErrorInfo] = None
    
    def mark_completed(self, usage: TokenUsage) -> None:
        """Mark request as completed with token usage."""
        self.end_time = datetime.now()
        self.total_duration = (self.end_time - self.start_time).total_seconds()
        self.actual_token_usage = usage
    
    def mark_failed(self, error: ErrorInfo) -> None:
        """Mark request as failed with error details."""
        self.end_time = datetime.now()
        self.total_duration = (self.end_time - self.start_time).total_seconds()
        self.error = error 