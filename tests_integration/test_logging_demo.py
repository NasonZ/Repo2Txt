#!/usr/bin/env python3
"""
Demo script to show improved logging capabilities.

USAGE:
    python tests_integration/test_logging_demo.py --model Qwen/Qwen3-8B-AWQ --debug
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.repo2txt.utils.logging_config import configure_integration_logging, get_logger, LogContext


async def demo_logging():
    """Demo the logging capabilities."""
    logger = get_logger(__name__)
    
    logger.info("Starting logging demonstration")
    
    # Demo structured logging with context
    with LogContext(logger, demo_id="logging-demo-001", component="evaluation"):
        logger.info("Demonstrating structured logging")
        
        # Demo different log levels
        logger.debug("Debug message with detailed info", extra={
            "step": 1,
            "action": "debug_demo",
            "data": {"key": "value"}
        })
        
        logger.info("Info message about operation", extra={
            "step": 2,
            "operation": "model_evaluation",
            "progress": "50%"
        })
        
        logger.warning("Warning about potential issue", extra={
            "step": 3,
            "issue": "rate_limit_approaching",
            "current_rate": 95,
            "limit": 100
        })
        
        # Demo exception logging
        try:
            raise ValueError("Demo exception for logging")
        except ValueError as e:
            logger.error("Caught demo exception", extra={
                "step": 4,
                "exception_type": type(e).__name__,
                "error_code": "DEMO_ERROR"
            }, exc_info=True)
    
    logger.info("Logging demonstration completed")


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()
    
    # Configure logging
    configure_integration_logging()
    if args.debug:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    print("🔍 Logging Demo")
    print("📁 Check integration_tests.log for detailed structured logs")
    print("=" * 50)
    
    asyncio.run(demo_logging())
    
    print("\n✅ Demo completed! Check the log files:")
    print("   📄 integration_tests.log - Structured logs for RCA")
    print("   🔍 Console output - Human-readable logs")


if __name__ == "__main__":
    main()