"""
File analysis module for repo2txt.

This module handles individual file processing including:
- Binary file detection
- Encoding detection with fallbacks
- File content reading
- Token counting
"""

import os
import mimetypes
import logging
from pathlib import Path
from typing import Optional, Tuple

try:
    import tiktoken
except ImportError:
    tiktoken = None

from .models import Config


class FileAnalyzer:
    """Handles file analysis and content extraction."""
    
    def __init__(self, config: Config):
        self.config = config
        self.token_encoder = None
        if tiktoken and config.enable_token_counting:
            try:
                self.token_encoder = tiktoken.get_encoding(config.token_encoder)
            except Exception as e:
                logging.warning(f"Failed to initialize token encoder: {e}")
    
    def is_binary_file(self, file_path: str, content: bytes = None) -> bool:
        """
        Improved binary file detection.
        
        Uses multiple methods to determine if a file is binary:
        1. Check file extension against known binary extensions
        2. Check against skip patterns (minified files, lockfiles, etc.)
        3. Use mimetypes to guess content type
        4. Check for null bytes in content
        5. Attempt UTF-8 decode as final check
        """
        # Check extension first
        path = Path(file_path)
        if path.suffix.lower() in self.config.binary_extensions:
            return True
        
        # Check skip patterns
        for pattern in self.config.skip_patterns:
            if pattern.startswith('*'):
                if path.name.endswith(pattern[1:]):
                    return True
            elif path.name == pattern:
                return True
        
        # If we have content, check for binary data first (more reliable than mimetypes)
        if content:
            # Check for null bytes in first N bytes (configurable)
            sample = content[:self.config.binary_sample_size]
            if b'\x00' in sample:
                return True
            
            # Try to decode as text
            try:
                sample.decode('utf-8')
                # If we can decode it, use mimetypes as final check
                mime_type, _ = mimetypes.guess_type(file_path)
                if mime_type:
                    return not mime_type.startswith('text/')
                return False
            except UnicodeDecodeError:
                return True
        
        # If no content provided, use mimetypes as fallback
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type:
            return not mime_type.startswith('text/')
        
        return False
    
    def read_file_content(self, file_path: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Read file content with multiple encoding fallbacks.
        
        Returns:
            Tuple of (content, error_message)
            If successful, content is the file text and error_message is None
            If failed, content is None and error_message describes the issue
        """
        try:
            # Check file size first
            file_size = os.path.getsize(file_path)
            if file_size > self.config.max_file_size:
                return None, f"File too large ({file_size:,} bytes)"
            
            # Read raw content
            with open(file_path, 'rb') as f:
                raw_content = f.read()
            
            # Check if binary
            if self.is_binary_file(file_path, raw_content):
                return None, "Binary file"
            
            # Try different encodings
            for encoding in self.config.encoding_fallbacks:
                try:
                    return raw_content.decode(encoding), None
                except UnicodeDecodeError:
                    continue
            
            return None, "Unable to decode file with available encodings"
            
        except PermissionError:
            return None, "Permission denied"
        except Exception as e:
            return None, f"Error reading file: {str(e)}"
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using tiktoken.
        
        Returns 0 if token counting is disabled or unavailable.
        """
        if not self.token_encoder:
            return 0
        try:
            return len(self.token_encoder.encode(text))
        except Exception:
            return 0