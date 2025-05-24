"""Provider capabilities model for LLM adapters."""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class ProviderCapabilities:
    """Defines the capabilities of an LLM provider."""
    
    # Core capabilities
    supports_streaming: bool
    supports_tools: bool
    supports_structured_output: bool
    supports_reasoning: bool
    supports_vision: bool
    max_tokens: int
    max_context_length: int
    supports_system_messages: bool
    supports_function_calling: bool
    
    # Tool-specific capabilities
    supports_parallel_tool_calls: bool = False
    supports_tool_streaming: bool = False
    
    # Streaming-specific capabilities
    supports_delta_streaming: bool = False
    supports_event_types: bool = False
    
    # Advanced features
    supports_json_mode: bool = False
    supports_response_format: bool = False
    supports_thinking_mode: bool = False
    supports_context_extension: bool = False
    
    # Cost information (for API providers)
    cost_per_input_token: Optional[float] = None
    cost_per_output_token: Optional[float] = None


class CapabilityChecker:
    """Helper class to check and validate provider capabilities."""
    
    @staticmethod
    def can_handle_request(capabilities: ProviderCapabilities, 
                          requires_tools: bool = False,
                          requires_streaming: bool = False,
                          requires_reasoning: bool = False,
                          token_count: int = 0) -> tuple[bool, List[str]]:
        """
        Check if provider can handle a specific request.
        
        Returns:
            Tuple of (can_handle, list_of_issues)
        """
        issues = []
        
        if requires_tools and not capabilities.supports_tools:
            issues.append("Provider does not support tools")
            
        if requires_streaming and not capabilities.supports_streaming:
            issues.append("Provider does not support streaming")
            
        if requires_reasoning and not capabilities.supports_reasoning:
            issues.append("Provider does not support reasoning/thinking mode")
            
        if token_count > capabilities.max_context_length:
            issues.append(f"Token count ({token_count}) exceeds max context length ({capabilities.max_context_length})")
            
        return len(issues) == 0, issues
    
    @staticmethod
    def get_capability_score(capabilities: ProviderCapabilities, 
                           task_requirements: Dict[str, bool]) -> float:
        """
        Calculate a capability score (0-1) based on how well provider matches requirements.
        """
        total_requirements = len(task_requirements)
        if total_requirements == 0:
            return 1.0
            
        matched = 0
        for requirement, needed in task_requirements.items():
            if not needed:
                matched += 1
                continue
                
            provider_has = getattr(capabilities, requirement, False)
            if provider_has:
                matched += 1
                
        return matched / total_requirements


def get_anthropic_capabilities() -> ProviderCapabilities:
    """Get capabilities for Anthropic Claude models."""
    return ProviderCapabilities(
        supports_streaming=True,
        supports_tools=True,
        supports_structured_output=True,
        supports_reasoning=True,
        supports_vision=True,
        max_tokens=4096,
        max_context_length=200000,
        supports_system_messages=True,
        supports_function_calling=True,
        supports_parallel_tool_calls=True,
        supports_tool_streaming=True,
        supports_delta_streaming=True,
        supports_event_types=True,
        supports_thinking_mode=True,
        cost_per_input_token=0.000015,  # $15 per million tokens
        cost_per_output_token=0.000075,  # $75 per million tokens
    )


def get_openai_capabilities(model: str = "gpt-4") -> ProviderCapabilities:
    """Get capabilities for OpenAI models."""
    base_capabilities = ProviderCapabilities(
        supports_streaming=True,
        supports_tools=True,
        supports_structured_output=True,
        supports_reasoning=True,
        supports_vision=True,
        max_tokens=4096,
        max_context_length=128000,
        supports_system_messages=True,
        supports_function_calling=True,
        supports_parallel_tool_calls=True,
        supports_delta_streaming=True,
        supports_json_mode=True,
        supports_response_format=True,
        cost_per_input_token=0.000030,  # GPT-4 pricing
        cost_per_output_token=0.000060,
    )
    
    # Adjust for specific models
    if "gpt-4o" in model.lower():
        base_capabilities.cost_per_input_token = 0.000005
        base_capabilities.cost_per_output_token = 0.000015
    elif "gpt-3.5" in model.lower():
        base_capabilities.cost_per_input_token = 0.000001
        base_capabilities.cost_per_output_token = 0.000002
        base_capabilities.max_context_length = 16385
        
    return base_capabilities


def get_vllm_capabilities(model: str) -> ProviderCapabilities:
    """Get capabilities for vLLM models based on model family."""
    model_lower = model.lower()
    
    # Base vLLM capabilities
    base_capabilities = ProviderCapabilities(
        supports_streaming=True,
        supports_tools=True,
        supports_structured_output=True,
        supports_reasoning=False,
        supports_vision=False,
        max_tokens=4096,
        max_context_length=32768,
        supports_system_messages=True,
        supports_function_calling=True,
        supports_parallel_tool_calls=True,
        supports_tool_streaming=True,
        supports_delta_streaming=True,
        supports_json_mode=True,
        supports_response_format=True,
        cost_per_input_token=0.0,  # Self-hosted, no cost
        cost_per_output_token=0.0,
    )
    
    # Qwen3 specific enhancements
    if "qwen3" in model_lower:
        base_capabilities.supports_reasoning = True
        base_capabilities.supports_thinking_mode = True
        base_capabilities.supports_context_extension = True
        
        # Context length based on model size
        if any(size in model_lower for size in ["30b", "235b", "14b", "32b", "8b"]):
            base_capabilities.max_context_length = 128000
        elif "128k" in model_lower:
            base_capabilities.max_context_length = 128000
        
        # Token limits based on model size
        if any(size in model_lower for size in ["30b", "235b", "32b"]):
            base_capabilities.max_tokens = 8192
        
    # Devstral specific enhancements
    elif "devstral" in model_lower:
        base_capabilities.supports_reasoning = True
        base_capabilities.max_tokens = 8192
        
    # Gemma family
    elif "gemma" in model_lower:
        base_capabilities.max_context_length = 8192
        if "27b" in model_lower or "9b" in model_lower:
            base_capabilities.supports_tools = True
        else:
            base_capabilities.supports_tools = False
        
    return base_capabilities 