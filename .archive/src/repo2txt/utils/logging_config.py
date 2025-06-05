"""
Structured logging configuration for repo2txt AI integration.

Provides clean, production-ready logging for debugging and RCA:
- Structured JSON format for production
- Context tracking for request flows  
- Clean console output for development
- Separate integration test logging
"""

import logging
import json
import sys
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging in production."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON with context."""
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add extra fields from record
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'lineno', 'funcName', 'created', 
                          'msecs', 'relativeCreated', 'thread', 'threadName',
                          'processName', 'process', 'getMessage', 'exc_info',
                          'exc_text', 'stack_info']:
                log_data[key] = value
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data, default=str)


class ContextFilter(logging.Filter):
    """Add request context to log records."""
    
    def __init__(self):
        super().__init__()
        self.context: Dict[str, Any] = {}
    
    def set_context(self, **kwargs):
        """Set context for subsequent log records."""
        self.context.update(kwargs)
    
    def clear_context(self):
        """Clear all context."""
        self.context.clear()
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add context to log record."""
        for key, value in self.context.items():
            setattr(record, key, value)
        return True


# Global context filter instance
_context_filter = ContextFilter()


def get_logger(name: str) -> logging.Logger:
    """Get a configured logger instance."""
    return logging.getLogger(name)


def set_context(**kwargs):
    """Set logging context for current request/operation."""
    _context_filter.set_context(**kwargs)


def clear_context():
    """Clear logging context."""
    _context_filter.clear_context()


def configure_production_logging(log_file: str = "repo2txt.log", level: str = "INFO"):
    """Configure structured JSON logging for production use."""
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(JSONFormatter())
    file_handler.addFilter(_context_filter)
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level.upper()))
    logger.addHandler(file_handler)
    
    # Reduce noise from external libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    

def configure_development_logging(level: str = "DEBUG"):
    """Configure clean console logging for development."""
    
    # Console formatter with color support
    console_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.addFilter(_context_filter)
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level.upper()))
    logger.addHandler(console_handler)
    
    # Reduce noise from external libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)


def configure_integration_logging():
    """Configure logging specifically for integration tests."""
    
    # Integration test log file
    log_file = Path("integration_tests.log")
    
    # File handler with JSON format
    file_handler = logging.FileHandler(log_file, mode='w')  # Overwrite on each run
    file_handler.setFormatter(JSONFormatter())
    file_handler.addFilter(_context_filter)
    
    # Console handler for test output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(
        logging.Formatter('%(levelname)s | %(name)s | %(message)s')
    )
    console_handler.setLevel(logging.WARNING)  # Only warnings/errors to console
    
    # Configure logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Context for integration tests
    set_context(
        test_session=datetime.now().isoformat(),
        test_type="integration"
    )


def configure_logging(level: str = "DEBUG", console_output: bool = True):
    """Configure logging for general use with optional console output."""
    if console_output:
        return configure_development_logging(level)
    else:
        return configure_quiet_logging()


def configure_quiet_logging():
    """Configure minimal logging for quiet operation."""
    
    # Only log errors to stderr
    error_handler = logging.StreamHandler(sys.stderr)
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(
        logging.Formatter('ERROR: %(message)s')
    )
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.ERROR)
    logger.addHandler(error_handler)


class LogContext:
    """Context manager for temporary logging context."""
    
    def __init__(self, **context):
        self.context = context
        self.previous_context = {}
    
    def __enter__(self):
        # Save current context
        self.previous_context = _context_filter.context.copy()
        # Set new context
        set_context(**self.context)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore previous context
        _context_filter.context = self.previous_context


# Default configuration for library use
if not logging.getLogger().handlers:
    configure_development_logging()