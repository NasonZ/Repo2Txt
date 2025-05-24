"""
Encoding detection and handling utilities.

This module provides robust encoding detection and text decoding
capabilities with multiple fallback strategies.
"""

import logging
from typing import Optional, List, Tuple


# Common encodings to try, ordered by likelihood
DEFAULT_ENCODINGS = [
    'utf-8',
    'utf-8-sig',  # UTF-8 with BOM
    'latin-1',
    'cp1252',     # Windows-1252
    'iso-8859-1',
    'ascii',
]

# Set up module logger
logger = logging.getLogger(__name__)


class EncodingDetector:
    """Handles encoding detection and text decoding."""
    
    def __init__(self, fallback_encodings: Optional[List[str]] = None):
        """
        Initialize the encoding detector.
        
        Args:
            fallback_encodings: List of encodings to try. If None, uses defaults.
        """
        self.encodings = fallback_encodings or DEFAULT_ENCODINGS
    
    def decode_bytes(self, content: bytes, file_path: Optional[str] = None) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Attempt to decode bytes to string using multiple encodings.
        
        Args:
            content: Raw bytes to decode.
            file_path: Optional file path for better error messages.
            
        Returns:
            Tuple of (decoded_text, encoding_used, error_message).
            If successful: (text, encoding, None)
            If failed: (None, None, error_message)
        """
        # Check for BOM first
        has_bom, bom_encoding = self.has_bom(content)
        if has_bom:
            try:
                decoded = content.decode(bom_encoding)
                logger.debug(f"Decoded {file_path} using BOM-detected {bom_encoding}")
                return decoded, bom_encoding, None
            except Exception as e:
                logger.debug(f"BOM decode failed for {file_path}: {e}")
        
        # Try each encoding
        last_error = None
        for encoding in self.encodings:
            try:
                decoded = content.decode(encoding)
                logger.debug(f"Decoded {file_path} using {encoding}")
                return decoded, encoding, None
            except UnicodeDecodeError as e:
                last_error = e
                continue
            except Exception as e:
                logger.warning(f"Unexpected error with {encoding} for {file_path}: {type(e).__name__}")
                last_error = e
                continue
        
        # All encodings failed - return user-friendly error
        error_msg = f"Unable to decode file with available encodings ({', '.join(self.encodings[:3])}, ...)"
        if last_error and hasattr(last_error, 'start'):
            error_msg += f" - failed at byte {last_error.start}"
        
        logger.info(f"Encoding detection failed for {file_path}: tried {len(self.encodings)} encodings")
        return None, None, error_msg
    
    def detect_encoding(self, content: bytes, sample_size: int = 8192) -> Optional[str]:
        """
        Detect the most likely encoding for the given content.
        
        Args:
            content: Raw bytes to analyze.
            sample_size: Number of bytes to sample for detection.
            
        Returns:
            The name of the most likely encoding, or None if detection fails.
        """
        sample = content[:sample_size]
        
        # Check for BOM
        has_bom, bom_encoding = self.has_bom(content)
        if has_bom:
            return bom_encoding
        
        # Try each encoding
        for encoding in self.encodings:
            try:
                sample.decode(encoding)
                # Verify with more content
                content[:sample_size * 4].decode(encoding)
                return encoding
            except UnicodeDecodeError:
                continue
            except Exception:
                continue
        
        return None
    
    def has_bom(self, content: bytes) -> Tuple[bool, Optional[str]]:
        """
        Check if content starts with a Byte Order Mark (BOM).
        
        Args:
            content: Raw bytes to check.
            
        Returns:
            Tuple of (has_bom, encoding_name).
        """
        bom_checks = [
            (b'\xef\xbb\xbf', 'utf-8-sig'),
            (b'\xff\xfe', 'utf-16-le'),
            (b'\xfe\xff', 'utf-16-be'),
            (b'\xff\xfe\x00\x00', 'utf-32-le'),
            (b'\x00\x00\xfe\xff', 'utf-32-be'),
        ]
        
        for bom, encoding in bom_checks:
            if content.startswith(bom):
                return True, encoding
        
        return False, None
    
    @staticmethod
    def is_likely_binary(content: bytes, sample_size: int = 8192) -> bool:
        """
        Check if content is likely binary by looking for null bytes.
        
        Args:
            content: Raw bytes to check.
            sample_size: Number of bytes to check.
            
        Returns:
            True if content appears to be binary.
        """
        sample = content[:sample_size]
        
        # Check for null bytes
        if b'\x00' in sample:
            return True
        
        # Try UTF-8 decode
        try:
            sample.decode('utf-8')
            return False
        except UnicodeDecodeError:
            pass
        
        # Check for high ratio of non-printable characters
        non_printable = 0
        for byte in sample:
            if byte < 32 and byte not in (9, 10, 13):  # tab, newline, carriage return
                non_printable += 1
        
        return non_printable > len(sample) * 0.3