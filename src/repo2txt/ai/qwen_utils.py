"""
Utilities for Qwen model-specific functionality.

This module contains helper functions for handling Qwen-specific
features like thinking tags.
"""

import re
from typing import Optional


def is_qwen_model(model_name: str) -> bool:
    """Check if the model is a Qwen model.
    
    Args:
        model_name: The model name to check
        
    Returns:
        True if it's a Qwen model, False otherwise
    """
    return "qwen" in model_name.lower()


def get_thinking_tag(enable_thinking: bool) -> str:
    """Get the appropriate thinking tag for Qwen models.
    
    Args:
        enable_thinking: Whether thinking is enabled
        
    Returns:
        The thinking tag to append (" /think" or " /no_think")
    """
    return " /think" if enable_thinking else " /no_think"


def add_thinking_tag_to_message(message: str, model_name: str, enable_thinking: bool) -> str:
    """Add thinking tag to a message if it's for a Qwen model.
    
    Args:
        message: The message to add tag to
        model_name: The model name
        enable_thinking: Whether thinking is enabled
        
    Returns:
        The message with thinking tag appended (if applicable)
    """
    if not is_qwen_model(model_name):
        return message
        
    # Check if tag already present
    if message.endswith("/think") or message.endswith("/no_think"):
        return message
        
    return message + get_thinking_tag(enable_thinking)


def clean_thinking_tags(content: str) -> str:
    """Remove <think>...</think> blocks from content.
    
    This is used to clean up Qwen model outputs that contain
    thinking blocks.
    
    Args:
        content: The content to clean
        
    Returns:
        Content with thinking blocks removed
    """
    if not content:
        return content
        
    # Remove think blocks and clean up extra whitespace
    cleaned = re.sub(r'<think>.*?</think>\s*', '', content, flags=re.DOTALL)
    return cleaned.strip()


def ensure_message_has_thinking_tag(messages: list, model_name: str, enable_thinking: bool) -> None:
    """Ensure the last user message has appropriate thinking tag.
    
    This modifies the messages list in place.
    
    Args:
        messages: List of message dictionaries
        model_name: The model name
        enable_thinking: Whether thinking is enabled
    """
    if not is_qwen_model(model_name):
        return
        
    thinking_tag = get_thinking_tag(enable_thinking)
    
    # Find last user message and add tag if not already present
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "user":
            content = messages[i].get("content", "")
            if content and not content.endswith("/think") and not content.endswith("/no_think"):
                messages[i]["content"] = content + thinking_tag
            break