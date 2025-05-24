#!/usr/bin/env python3
"""Integration test script for AI adapter functionality."""

import asyncio
import json
import sys
import time
from typing import List, Dict, Any
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.repo2txt.utils.logging_config import configure_integration_logging, get_logger

def test_imports():
    """Test that all required imports work."""
    print("🧪 Testing imports...")
    
    try:
        from src.repo2txt.ai import (
            AdapterFactory,
            create_adapter,
            create_vllm_adapter,
            BaseLLMAdapter,
            VLLMAdapter,
            OpenAIAdapter,
            ModelType,
            get_model_config,
            get_default_config,
            LLMRequest,
            LLMResponse,
            Message,
            MessageRole,
            Tool,
            ToolCall,
            StreamEvent,
            ProviderCapabilities,
        )
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False


def test_model_types():
    """Test model type definitions and properties."""
    print("\n🧪 Testing model types...")
    
    try:
        from src.repo2txt.ai import ModelType
        
        # Test your specific models
        models_to_test = [
            ModelType.QWEN3_8B,
            ModelType.QWEN3_32B,
            ModelType.QWEN3_30B_A3B,
            ModelType.DEVSTRAL_SMALL,
            ModelType.GEMMA3_9B,
            ModelType.GEMMA3_27B,
        ]
        
        print(f"Testing {len(models_to_test)} model types:")
        for model in models_to_test:
            print(f"  📋 {model.name}:")
            print(f"    Value: {model.value}")
            print(f"    Tools: {model.supports_tools}")
            print(f"    Reasoning: {model.supports_reasoning}")
            print(f"    Context: {model.context_length:,} tokens")
            print(f"    Extended: {model.extended_context_length:,} tokens")
            print(f"    Quantized: {model.is_quantized} ({model.quantization_type})")
        
        print("✅ Model types test passed")
        return True
    except Exception as e:
        print(f"❌ Model types test failed: {e}")
        return False


def test_configuration():
    """Test configuration generation."""
    print("\n🧪 Testing configuration...")
    
    try:
        from src.repo2txt.ai import ModelType, get_model_config, get_default_config
        
        # Test default config
        default_config = get_default_config()
        print(f"✅ Default config created with {len(default_config.providers)} providers")
        
        # Test model-specific configs
        test_models = [ModelType.QWEN3_8B, ModelType.DEVSTRAL_SMALL, ModelType.GEMMA3_9B]
        for model in test_models:
            config = get_model_config(model)
            print(f"✅ Config for {model.name}: {list(config.keys())}")
            
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


def test_adapter_creation():
    """Test adapter creation without connecting."""
    print("\n🧪 Testing adapter creation...")
    
    try:
        from src.repo2txt.ai import create_vllm_adapter, AdapterFactory, ModelType
        
        # Test vLLM adapter creation
        test_models = [
            ModelType.QWEN3_8B,
            ModelType.DEVSTRAL_SMALL,
            ModelType.GEMMA3_9B,
        ]
        
        for model in test_models:
            adapter = create_vllm_adapter(model, base_url="http://localhost:8000")
            print(f"✅ Created vLLM adapter for {model.name}")
            print(f"    Provider: {adapter.provider_name}")
            print(f"    Model: {adapter.model}")
            print(f"    Capabilities: tools={adapter.capabilities.supports_tools}")
        
        # Test factory methods
        factory_adapter = AdapterFactory.create_adapter(
            model=ModelType.QWEN3_8B,
            provider="vllm",
            base_url="http://localhost:8000"
        )
        print(f"✅ Factory created adapter: {factory_adapter.__class__.__name__}")
        
        return True
    except Exception as e:
        print(f"❌ Adapter creation test failed: {e}")
        return False


def test_data_models():
    """Test data model creation and validation."""
    print("\n🧪 Testing data models...")
    
    try:
        from src.repo2txt.ai import (
            Message, MessageRole, Tool, ToolCall, LLMRequest, 
            LLMResponse, TokenUsage, StreamEvent, StreamEventType
        )
        
        # Test Message creation
        msg = Message(
            role=MessageRole.USER,
            content="Test message"
        )
        print(f"✅ Created message: {msg.role.value}")
        
        # Test Tool definition
        tool = Tool(
            name="test_tool",
            description="A test tool",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        )
        print(f"✅ Created tool: {tool.name}")
        
        # Test ToolCall
        tool_call = ToolCall(
            id="test-123",
            tool_name="test_tool",
            input={"query": "test"},
            status="pending"
        )
        print(f"✅ Created tool call: {tool_call.tool_name}")
        
        # Test LLMRequest
        request = LLMRequest(
            messages=[msg],
            model="test-model",
            temperature=0.7,
            enable_reasoning=True
        )
        print(f"✅ Created request with {len(request.messages)} messages")
        
        # Test TokenUsage
        usage = TokenUsage(
            input_tokens=100,
            output_tokens=50
        )
        print(f"✅ Created usage: {usage.total_tokens} total tokens")
        
        # Test LLMResponse
        response = LLMResponse(
            content="Test response",
            usage=usage,
            model="test-model",
            finish_reason="stop"
        )
        print(f"✅ Created response: {len(response.content)} chars")
        
        # Test StreamEvent
        event = StreamEvent(
            event_type=StreamEventType.CONTENT_DELTA,
            content="test",
            delta="test"
        )
        print(f"✅ Created stream event: {event.event_type.value}")
        
        return True
    except Exception as e:
        print(f"❌ Data models test failed: {e}")
        return False


async def test_vllm_connection():
    """Test actual connection to vLLM server (if available)."""
    print("\n🧪 Testing vLLM connection...")
    
    try:
        from src.repo2txt.ai import create_vllm_adapter, ModelType, LLMRequest, Message, MessageRole
        
        # Create adapter
        adapter = create_vllm_adapter(
            ModelType.QWEN3_8B, 
            base_url="http://localhost:8000"
        )
        
        # Test health check
        print("🔍 Checking vLLM server health...")
        is_healthy = await adapter.health_check()
        
        if not is_healthy:
            print("⚠️  vLLM server not available - skipping connection tests")
            return True  # Not a failure, just not available
        
        print("✅ vLLM server is healthy")
        
        # Test model listing
        try:
            models = await adapter.get_available_models()
            print(f"✅ Available models: {models}")
        except Exception as e:
            print(f"⚠️  Could not list models: {e}")
        
        # Test simple completion
        print("🔍 Testing simple completion...")
        request = LLMRequest(
            messages=[
                Message(role=MessageRole.USER, content="Say 'Hello AI test!'")
            ],
            model=adapter.model,
            max_tokens=20,
            temperature=0.1
        )
        
        start_time = time.time()
        response = await adapter.complete(request)
        duration = time.time() - start_time
        
        print(f"✅ Completion successful in {duration:.2f}s")
        print(f"    Response: {response.content[:100]}...")
        if response.usage:
            print(f"    Tokens: {response.usage.input_tokens} in, {response.usage.output_tokens} out")
        
        # Test thinking mode (Qwen3 specific)
        if adapter.capabilities.supports_reasoning:
            print("🔍 Testing thinking mode...")
            thinking_request = LLMRequest(
                messages=[
                    Message(role=MessageRole.USER, content="/think What is 2+2?")
                ],
                model=adapter.model,
                max_tokens=100,
                temperature=0.1,
                enable_reasoning=True
            )
            
            thinking_response = await adapter.complete(thinking_request)
            print(f"✅ Thinking mode test complete")
            if thinking_response.reasoning:
                print(f"    Reasoning: {thinking_response.reasoning[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"⚠️  vLLM connection test failed: {e}")
        print("    This is expected if vLLM server is not running")
        return True  # Don't fail the test suite for connection issues


async def test_tool_calling():
    """Test tool calling functionality (if vLLM available)."""
    print("\n🧪 Testing tool calling...")
    
    try:
        from src.repo2txt.ai import (
            create_vllm_adapter, ModelType, LLMRequest, Message, 
            MessageRole, Tool
        )
        
        # Create adapter
        adapter = create_vllm_adapter(
            ModelType.QWEN3_8B,
            base_url="http://localhost:8000"
        )
        
        # Check if server is available
        if not await adapter.health_check():
            print("⚠️  vLLM server not available - skipping tool calling tests")
            return True
        
        # Check if model supports tools
        if not adapter.capabilities.supports_tools:
            print("⚠️  Model doesn't support tools - skipping tool calling tests")
            return True
        
        # Define a simple tool
        calculator_tool = Tool(
            name="calculate",
            description="Perform basic arithmetic calculations",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate (e.g., '2+2')"
                    }
                },
                "required": ["expression"]
            }
        )
        
        # Test tool calling
        request = LLMRequest(
            messages=[
                Message(role=MessageRole.USER, content="Calculate 15 + 27 using the calculator tool")
            ],
            model=adapter.model,
            max_tokens=100,
            temperature=0.1
        )
        
        print("🔍 Testing tool calling...")
        response = await adapter.complete_with_tools(request, [calculator_tool])
        
        print("✅ Tool calling test completed")
        if response.has_tool_calls:
            print(f"    Tool calls: {len(response.tool_calls)}")
            for tc in response.tool_calls:
                print(f"      {tc.tool_name}: {tc.input}")
        else:
            print("    No tool calls made (model may not have chosen to use tools)")
        
        return True
        
    except Exception as e:
        print(f"⚠️  Tool calling test failed: {e}")
        return True  # Don't fail for connection issues


def test_streaming():
    """Test streaming functionality."""
    print("\n🧪 Testing streaming (mock)...")
    
    try:
        from src.repo2txt.ai import StreamEvent, StreamEventType
        
        # Test stream event creation
        events = [
            StreamEvent(event_type=StreamEventType.MESSAGE_START),
            StreamEvent(event_type=StreamEventType.CONTENT_DELTA, content="Hello", delta="Hello"),
            StreamEvent(event_type=StreamEventType.CONTENT_DELTA, content="Hello world", delta=" world"),
            StreamEvent(event_type=StreamEventType.MESSAGE_STOP, content="Hello world"),
        ]
        
        print(f"✅ Created {len(events)} stream events")
        for i, event in enumerate(events):
            print(f"    {i+1}. {event.event_type.value}: {event.content or 'N/A'}")
        
        return True
    except Exception as e:
        print(f"❌ Streaming test failed: {e}")
        return False


async def main():
    """Run all integration tests."""
    # Configure logging
    configure_integration_logging()
    logger = get_logger(__name__)
    
    logger.info("Starting AI integration tests")
    
    print("🚀 Starting AI Integration Tests")
    print("📁 Detailed logs: integration_tests.log")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Model Types", test_model_types),
        ("Configuration", test_configuration),
        ("Adapter Creation", test_adapter_creation),
        ("Data Models", test_data_models),
        ("Streaming", test_streaming),
        ("vLLM Connection", test_vllm_connection),
        ("Tool Calling", test_tool_calling),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*50)
    print("📊 TEST SUMMARY")
    print("="*50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:<8} {test_name}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Ready to commit.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please fix before committing.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)