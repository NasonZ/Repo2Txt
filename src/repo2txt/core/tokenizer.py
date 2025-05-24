"""
Token counting functionality for repo2txt.

This module provides token counting capabilities using OpenAI's tiktoken library.
It's designed to gracefully handle cases where tiktoken is not installed.
"""

import logging
from typing import Optional, Dict, Any

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    tiktoken = None
    TIKTOKEN_AVAILABLE = False


class TokenCounter:
    """
    Handles token counting for text content.
    
    This class provides a clean interface for token counting that gracefully
    handles the absence of the tiktoken library.
    """
    
    def __init__(self, encoding_name: str = "cl100k_base"):
        """
        Initialize the token counter.
        
        Args:
            encoding_name: The name of the tiktoken encoding to use.
                         Default is cl100k_base (used by GPT-4).
        """
        self.encoding_name = encoding_name
        self.encoder: Optional[Any] = None
        self._initialized = False
        
        if TIKTOKEN_AVAILABLE:
            try:
                self.encoder = tiktoken.get_encoding(encoding_name)
                self._initialized = True
            except Exception as e:
                logging.warning(f"Failed to initialize token encoder '{encoding_name}': {e}")
    
    @property
    def is_available(self) -> bool:
        """Check if token counting is available."""
        return self._initialized and self.encoder is not None
    
    def count(self, text: str) -> int:
        """
        Count tokens in the given text.
        
        Args:
            text: The text to count tokens for.
            
        Returns:
            Number of tokens, or 0 if counting is unavailable.
        """
        if not self.is_available or not text:
            return 0
        
        try:
            return len(self.encoder.encode(text))
        except Exception as e:
            logging.debug(f"Error counting tokens: {e}")
            return 0
    
    def count_batch(self, texts: Dict[str, str]) -> Dict[str, int]:
        """
        Count tokens for multiple texts efficiently.
        
        Args:
            texts: Dictionary mapping identifiers to text content.
            
        Returns:
            Dictionary mapping identifiers to token counts.
        """
        if not self.is_available:
            return {key: 0 for key in texts}
        
        results = {}
        for key, text in texts.items():
            results[key] = self.count(text)
        
        return results
    
    @staticmethod
    def estimate_tokens(text: str) -> int:
        """
        Estimate token count without tiktoken.
        
        This provides a rough estimate based on common patterns.
        Useful as a fallback when tiktoken is not available.
        
        Args:
            text: The text to estimate tokens for.
            
        Returns:
            Estimated number of tokens.
        """
        if not text:
            return 0
        
        # Rough estimation: ~4 characters per token on average
        # This is based on empirical observations with GPT models
        char_count = len(text)
        word_count = len(text.split())
        
        # Use a weighted average of character and word-based estimates
        char_estimate = char_count / 4
        word_estimate = word_count * 1.3
        
        return int((char_estimate + word_estimate) / 2)