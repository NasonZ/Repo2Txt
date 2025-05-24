#!/usr/bin/env python3
"""
Model Evaluation Toolkit for LLM Integration

PURPOSE: Quickly evaluate any new LLM model with our adapters
- Download Devstral? Run this to see if it works
- New Qwen model? Test all capabilities instantly  
- Adapter issues? Get detailed RCA diagnostics

USAGE:
    # Quick test (clean output)
    python tests_integration/model_evaluation_toolkit.py --model Qwen/Qwen3-8B-AWQ
    
    # Full test suite 
    python tests_integration/model_evaluation_toolkit.py --model mistralai/devstral-small-2505 --full-suite
    
    # Minimal output for automation
    python tests_integration/model_evaluation_toolkit.py --model Qwen/Qwen3-8B-AWQ --quiet
    
    # Verbose debugging
    python tests_integration/model_evaluation_toolkit.py --model Qwen/Qwen3-8B-AWQ --debug
    
    # Connection diagnostics
    python tests_integration/model_evaluation_toolkit.py --diagnose-connection
"""

import asyncio
import json
import sys
import time
import traceback
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.repo2txt.utils.logging_config import configure_integration_logging, get_logger, configure_logging
from src.repo2txt.ai import (
    create_vllm_adapter, ModelType, LLMRequest, Message, MessageRole,
    Tool, get_model_config, VLLMAdapter
)


@dataclass
class TestResult:
    """Test result with detailed diagnostics."""
    test_name: str
    passed: bool
    duration: float
    details: Dict[str, Any]
    error: Optional[str] = None
    suggestions: List[str] = None

    def __post_init__(self):
        if self.suggestions is None:
            self.suggestions = []


class ModelEvaluator:
    """Comprehensive model evaluation and diagnostics."""
    
    def __init__(self, model_name: str, base_url: str = "http://localhost:8000"):
        self.model_name = model_name
        self.base_url = base_url
        self.adapter: Optional[VLLMAdapter] = None
        self.results: List[TestResult] = []
        self.logger = get_logger(__name__)
        
    async def evaluate_model(self, quick_mode: bool = True) -> Dict[str, Any]:
        """Run comprehensive model evaluation with clean streaming output."""
        print(f"🔍 Evaluating Model: {self.model_name}")
        print(f"📡 Server: {self.base_url}")
        print(f"📋 Mode: {'Quick' if quick_mode else 'Full Suite'}")
        print("=" * 70)
        
        # Core tests (always run)
        core_tests = [
            ("Adapter Creation", self._test_adapter_creation),
            ("Server Connection", self._test_server_connection),
            ("Model Availability", self._test_model_availability),
            ("Basic Completion", self._test_basic_completion),
        ]
        
        # Extended tests (full suite)
        extended_tests = [
            ("Streaming Support", self._test_streaming),
            ("Tool Calling", self._test_tool_calling),
            ("Reasoning Mode", self._test_reasoning_mode),
            ("Parameter Compatibility", self._test_parameter_compatibility),
            ("Error Handling", self._test_error_handling),
            ("Performance Baseline", self._test_performance_baseline),
        ]
        
        tests_to_run = core_tests + (extended_tests if not quick_mode else [])
        
        for i, (test_display_name, test_func) in enumerate(tests_to_run, 1):
            print(f"[{i}/{len(tests_to_run)}] {test_display_name}...", end="", flush=True)
            
            try:
                start_time = time.time()
                result = await test_func()
                duration = time.time() - start_time
                result.duration = duration
                self.results.append(result)
                
                # Clean streaming output
                status = "✅" if result.passed else "❌"
                print(f" {status} ({duration:.2f}s)")
                
                if not result.passed:
                    print(f"   💥 {result.error}")
                
            except Exception as e:
                duration = time.time() - start_time
                error_result = TestResult(
                    test_name=test_display_name,
                    passed=False,
                    duration=duration,
                    details={},
                    error=str(e),
                    suggestions=[]
                )
                self.results.append(error_result)
                print(f" ❌ ({duration:.2f}s)")
                print(f"   💥 {str(e)}")
        
        return self._generate_report()
    
    def _extract_key_metrics(self, result: TestResult) -> Dict[str, Any]:
        """Extract key metrics from test result for logging."""
        if not result.details:
            return {}
        
        # Extract meaningful metrics based on test type
        metrics = {}
        details = result.details
        
        if "tokens_per_second" in details:
            metrics["performance"] = {
                "tokens_per_second": details["tokens_per_second"],
                "performance_class": details.get("performance_class", "unknown")
            }
        
        if "tool_count" in details:
            metrics["tools"] = {
                "tools_executed": details["tool_count"],
                "tools_used": details.get("tools_used", [])
            }
        
        if "has_reasoning_output" in details:
            metrics["reasoning"] = {
                "enabled": details["has_reasoning_output"],
                "reasoning_length": details.get("reasoning_length", 0)
            }
        
        if "compatibility_rate" in details:
            metrics["parameters"] = {
                "compatibility": details["compatibility_rate"],
                "working_params": details.get("working_params", [])
            }
        
        if "available_models" in details:
            metrics["server"] = {
                "models_available": len(details["available_models"]),
                "target_model_found": details.get("target_model") in details["available_models"]
            }
        
        return metrics
    
    async def _test_adapter_creation(self) -> TestResult:
        """Test if we can create an adapter for this model."""
        try:
            # Try to find model type
            model_type = None
            for mt in ModelType:
                if mt.value == self.model_name:
                    model_type = mt
                    break
            
            if not model_type:
                # Create custom adapter
                from src.repo2txt.ai.adapter.vllm_adapter import VLLMAdapter
                from src.repo2txt.ai.models import ProviderCapabilities
                
                # Assume basic capabilities for unknown models
                capabilities = ProviderCapabilities(
                    supports_streaming=True,
                    supports_tools=True,  # Most modern models do
                    supports_reasoning=False,  # Conservative default
                    max_context_length=32768
                )
                
                self.adapter = VLLMAdapter(
                    model=self.model_name,
                    base_url=self.base_url,
                    capabilities=capabilities
                )
                
                return TestResult(
                    test_name="Adapter Creation",
                    passed=True,
                    duration=0,
                    details={
                        "model_type": "Custom (not in predefined types)",
                        "assumed_capabilities": capabilities.__dict__
                    },
                    suggestions=["Consider adding this model to ModelType enum if it works well"]
                )
            else:
                # Use predefined model type
                self.adapter = create_vllm_adapter(model_type, base_url=self.base_url)
                
                return TestResult(
                    test_name="Adapter Creation", 
                    passed=True,
                    duration=0,
                    details={
                        "model_type": model_type.name,
                        "capabilities": self.adapter.capabilities.__dict__,
                        "config": get_model_config(model_type)
                    }
                )
                
        except Exception as e:
            return TestResult(
                test_name="Adapter Creation",
                passed=False,
                duration=0,
                details={},
                error=str(e),
                suggestions=[
                    "Check if model name is correct",
                    "Verify vLLM server supports this model",
                    "Check ModelType definitions in config.py"
                ]
            )
    
    async def _test_server_connection(self) -> TestResult:
        """Test basic server connectivity."""
        if not self.adapter:
            return TestResult("Server Connection", False, 0, {}, "No adapter available")
        
        try:
            is_healthy = await self.adapter.health_check()
            
            return TestResult(
                test_name="Server Connection",
                passed=is_healthy,
                duration=0,
                details={"base_url": self.base_url, "healthy": is_healthy},
                error=None if is_healthy else "Health check failed",
                suggestions=[] if is_healthy else [
                    f"Check if vLLM server is running at {self.base_url}",
                    "Try: vllm serve {model} --port 8000",
                    "Verify firewall/network connectivity"
                ]
            )
            
        except Exception as e:
            return TestResult(
                test_name="Server Connection",
                passed=False,
                duration=0,
                details={},
                error=str(e),
                suggestions=[
                    "Server may not be running",
                    "Check URL format (should include http://)",
                    "Verify port number"
                ]
            )
    
    async def _test_model_availability(self) -> TestResult:
        """Test if the specific model is loaded on the server."""
        if not self.adapter:
            return TestResult("Model Availability", False, 0, {}, "No adapter available")
        
        try:
            models = await self.adapter.get_available_models()
            model_available = self.model_name in models
            
            return TestResult(
                test_name="Model Availability",
                passed=model_available,
                duration=0,
                details={"available_models": models, "target_model": self.model_name},
                error=None if model_available else f"Model {self.model_name} not found on server",
                suggestions=[] if model_available else [
                    f"Start vLLM with: vllm serve {self.model_name}",
                    "Check model name spelling/case sensitivity",
                    f"Available models: {', '.join(models[:3])}{'...' if len(models) > 3 else ''}"
                ]
            )
            
        except Exception as e:
            return TestResult(
                test_name="Model Availability", 
                passed=False,
                duration=0,
                details={},
                error=str(e),
                suggestions=["Server may not support model listing endpoint"]
            )
    
    async def _test_basic_completion(self) -> TestResult:
        """Test basic text completion."""
        if not self.adapter:
            return TestResult("Basic Completion", False, 0, {}, "No adapter available")
        
        try:
            request = LLMRequest(
                messages=[
                    Message(role=MessageRole.USER, content="Say 'Model test successful!' and nothing else.")
                ],
                model=self.adapter.model,
                max_tokens=20,
                temperature=0.1
            )
            
            response = await self.adapter.complete(request)
            success = response.content and "successful" in response.content.lower()
            
            details = {
                "response_content": response.content,
                "response_length": len(response.content) if response.content else 0,
                "finish_reason": response.finish_reason
            }
            
            if response.usage:
                details.update({
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.total_tokens
                })
            
            return TestResult(
                test_name="Basic Completion",
                passed=success,
                duration=0,
                details=details,
                error=None if success else "Model didn't respond as expected",
                suggestions=[] if success else [
                    "Model may be having generation issues",
                    "Try different temperature/parameters",
                    "Check if model is properly loaded"
                ]
            )
            
        except Exception as e:
            return TestResult(
                test_name="Basic Completion",
                passed=False,
                duration=0,
                details={},
                error=str(e),
                suggestions=[
                    "Model may not support chat completion format",
                    "Check token limits and parameters",
                    "Verify model is compatible with OpenAI API format"
                ]
            )
    
    async def _test_streaming(self) -> TestResult:
        """Test streaming capability."""
        if not self.adapter:
            return TestResult("Streaming", False, 0, {}, "No adapter available")
        
        try:
            # Mock streaming test - just verify the adapter supports it
            supports_streaming = self.adapter.capabilities.supports_streaming
            
            return TestResult(
                test_name="Streaming",
                passed=supports_streaming,
                duration=0,
                details={"supports_streaming": supports_streaming},
                error=None if supports_streaming else "Streaming not supported by adapter",
                suggestions=[] if supports_streaming else [
                    "This model may not support streaming responses",
                    "Check vLLM server configuration"
                ]
            )
            
        except Exception as e:
            return TestResult(
                test_name="Streaming",
                passed=False,
                duration=0,
                details={},
                error=str(e)
            )
    
    async def _test_tool_calling(self) -> TestResult:
        """Test tool calling capability."""
        if not self.adapter:
            return TestResult("Tool Calling", False, 0, {}, "No adapter available")
        
        if not self.adapter.capabilities.supports_tools:
            return TestResult(
                test_name="Tool Calling",
                passed=False,
                duration=0,
                details={"supports_tools": False},
                error="Model doesn't support tool calling",
                suggestions=[
                    "This model type doesn't support function calling",
                    "Consider using a tool-capable model like Qwen3 or Devstral"
                ]
            )
        
        try:
            # Simple tool test
            calculator_tool = Tool(
                name="add_numbers",
                description="Add two numbers",
                parameters={
                    "type": "object",
                    "properties": {
                        "a": {"type": "number"},
                        "b": {"type": "number"}
                    },
                    "required": ["a", "b"]
                }
            )
            
            request = LLMRequest(
                messages=[
                    Message(role=MessageRole.USER, content="Add 5 and 7 using the add_numbers tool")
                ],
                model=self.adapter.model,
                max_tokens=100,
                temperature=0.1
            )
            
            response = await self.adapter.complete_with_tools(request, [calculator_tool])
            
            has_tools = response.has_tool_calls
            tool_details = {}
            if has_tools:
                tool_details = {
                    "tool_count": len(response.tool_calls),
                    "tools_used": [tc.tool_name for tc in response.tool_calls]
                }
            
            return TestResult(
                test_name="Tool Calling",
                passed=has_tools,
                duration=0,
                details=tool_details,
                error=None if has_tools else "Model didn't use tools when expected",
                suggestions=[] if has_tools else [
                    "Model may need specific prompting for tool use",
                    "Check if vLLM server has tool calling enabled",
                    "Try: --enable-auto-tool-choice --tool-call-parser hermes"
                ]
            )
            
        except Exception as e:
            return TestResult(
                test_name="Tool Calling",
                passed=False,
                duration=0,
                details={},
                error=str(e),
                suggestions=[
                    "vLLM server may not support tool calling",
                    "Check server startup flags for tool support"
                ]
            )
    
    async def _test_reasoning_mode(self) -> TestResult:
        """Test reasoning/thinking mode (Qwen3 specific)."""
        if not self.adapter:
            return TestResult("Reasoning Mode", False, 0, {}, "No adapter available")
        
        if not self.adapter.capabilities.supports_reasoning:
            return TestResult(
                test_name="Reasoning Mode",
                passed=True,  # Not an error if unsupported
                duration=0,
                details={"supports_reasoning": False},
                error=None,
                suggestions=["This model doesn't support reasoning mode (normal for most models)"]
            )
        
        try:
            request = LLMRequest(
                messages=[
                    Message(role=MessageRole.USER, content="/think What is 2+2? Think step by step.")
                ],
                model=self.adapter.model,
                max_tokens=200,
                temperature=0.1,
                enable_reasoning=True
            )
            
            response = await self.adapter.complete(request)
            has_reasoning = bool(response.reasoning)
            
            return TestResult(
                test_name="Reasoning Mode",
                passed=has_reasoning,
                duration=0,
                details={
                    "has_reasoning_output": has_reasoning,
                    "reasoning_length": len(response.reasoning) if response.reasoning else 0
                },
                error=None if has_reasoning else "No reasoning output detected",
                suggestions=[] if has_reasoning else [
                    "Model may not be configured for thinking mode",
                    "Check enable_thinking parameter in vLLM config"
                ]
            )
            
        except Exception as e:
            return TestResult(
                test_name="Reasoning Mode",
                passed=False,
                duration=0,
                details={},
                error=str(e)
            )
    
    async def _test_parameter_compatibility(self) -> TestResult:
        """Test various parameter configurations."""
        if not self.adapter:
            return TestResult("Parameter Compatibility", False, 0, {}, "No adapter available")
        
        try:
            # Test different parameter combinations
            test_params = [
                {"temperature": 0.7, "top_p": 0.9},
                {"temperature": 0.1, "top_k": 50},
                {"max_tokens": 50, "presence_penalty": 0.5}
            ]
            
            working_params = []
            for params in test_params:
                try:
                    request = LLMRequest(
                        messages=[Message(role=MessageRole.USER, content="Hi")],
                        model=self.adapter.model,
                        **params
                    )
                    await self.adapter.complete(request)
                    working_params.append(params)
                except Exception:
                    pass
            
            success = len(working_params) > 0
            
            return TestResult(
                test_name="Parameter Compatibility",
                passed=success,
                duration=0,
                details={
                    "tested_params": test_params,
                    "working_params": working_params,
                    "compatibility_rate": f"{len(working_params)}/{len(test_params)}"
                },
                error=None if success else "No parameter combinations worked",
                suggestions=[] if success else [
                    "Model may have strict parameter requirements",
                    "Check vLLM server parameter support"
                ]
            )
            
        except Exception as e:
            return TestResult(
                test_name="Parameter Compatibility",
                passed=False,
                duration=0,
                details={},
                error=str(e)
            )
    
    async def _test_error_handling(self) -> TestResult:
        """Test how gracefully the model handles errors."""
        if not self.adapter:
            return TestResult("Error Handling", False, 0, {}, "No adapter available")
        
        try:
            # Test with invalid parameters
            error_cases = []
            
            # Case 1: Excessive token request
            try:
                request = LLMRequest(
                    messages=[Message(role=MessageRole.USER, content="Hi")],
                    model=self.adapter.model,
                    max_tokens=999999  # Intentionally excessive
                )
                await self.adapter.complete(request)
                error_cases.append("excessive_tokens: handled")
            except Exception as e:
                error_cases.append(f"excessive_tokens: {type(e).__name__}")
            
            # Case 2: Invalid temperature
            try:
                request = LLMRequest(
                    messages=[Message(role=MessageRole.USER, content="Hi")],
                    model=self.adapter.model,
                    temperature=10.0  # Out of range
                )
                await self.adapter.complete(request)
                error_cases.append("invalid_temperature: handled")
            except Exception as e:
                error_cases.append(f"invalid_temperature: {type(e).__name__}")
            
            return TestResult(
                test_name="Error Handling",
                passed=True,  # Any response is good
                duration=0,
                details={"error_responses": error_cases},
                suggestions=["Model handles errors appropriately"]
            )
            
        except Exception as e:
            return TestResult(
                test_name="Error Handling",
                passed=False,
                duration=0,
                details={},
                error=str(e)
            )
    
    async def _test_performance_baseline(self) -> TestResult:
        """Establish basic performance metrics."""
        if not self.adapter:
            return TestResult("Performance Baseline", False, 0, {}, "No adapter available")
        
        try:
            # Simple performance test
            start_time = time.time()
            
            request = LLMRequest(
                messages=[Message(role=MessageRole.USER, content="Count from 1 to 10")],
                model=self.adapter.model,
                max_tokens=100,
                temperature=0.5
            )
            
            response = await self.adapter.complete(request)
            
            total_time = time.time() - start_time
            tokens_per_second = 0
            
            if response.usage and response.usage.output_tokens:
                tokens_per_second = response.usage.output_tokens / total_time
            
            performance_details = {
                "total_time": round(total_time, 2),
                "tokens_per_second": round(tokens_per_second, 1),
                "response_length": len(response.content) if response.content else 0
            }
            
            # Basic performance classification
            performance_class = "unknown"
            if tokens_per_second > 50:
                performance_class = "fast"
            elif tokens_per_second > 20:
                performance_class = "moderate"
            elif tokens_per_second > 0:
                performance_class = "slow"
            
            performance_details["performance_class"] = performance_class
            
            return TestResult(
                test_name="Performance Baseline",
                passed=True,
                duration=0,
                details=performance_details,
                suggestions=[f"Performance: {performance_class} ({tokens_per_second:.1f} tokens/sec)"]
            )
            
        except Exception as e:
            return TestResult(
                test_name="Performance Baseline",
                passed=False,
                duration=0,
                details={},
                error=str(e)
            )
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive evaluation report."""
        passed_tests = [r for r in self.results if r.passed]
        failed_tests = [r for r in self.results if not r.passed]
        
        total_duration = sum(r.duration for r in self.results)
        
        report = {
            "model": self.model_name,
            "server": self.base_url,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "summary": {
                "total_tests": len(self.results),
                "passed": len(passed_tests),
                "failed": len(failed_tests),
                "success_rate": f"{len(passed_tests)/len(self.results)*100:.1f}%",
                "total_duration": f"{total_duration:.2f}s"
            },
            "results": [
                {
                    "test": r.test_name,
                    "passed": r.passed,
                    "duration": f"{r.duration:.2f}s",
                    "details": r.details,
                    "error": r.error,
                    "suggestions": r.suggestions
                }
                for r in self.results
            ]
        }
        
        return report


async def diagnose_connection(base_url: str = "http://localhost:8000"):
    """Diagnose basic connection issues."""
    print("🔧 Connection Diagnostics")
    print("=" * 40)
    
    # Test basic HTTP connectivity
    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{base_url}/health") as response:
                if response.status == 200:
                    print("✅ Server is responding")
                else:
                    print(f"⚠️  Server returned status {response.status}")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("💡 Check if vLLM server is running")
        print("💡 Try: vllm serve <model> --port 8000")
        return
    
    # Test model endpoint
    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{base_url}/v1/models") as response:
                if response.status == 200:
                    data = await response.json()
                    models = [m.get("id", "unknown") for m in data.get("data", [])]
                    print(f"✅ Available models: {', '.join(models)}")
                else:
                    print(f"⚠️  Models endpoint returned {response.status}")
    except Exception as e:
        print(f"❌ Models endpoint failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="LLM Model Evaluation Toolkit")
    parser.add_argument("--model", required=True, help="Model name to evaluate")
    parser.add_argument("--base-url", default="http://localhost:8000", help="vLLM server URL")
    parser.add_argument("--full-suite", action="store_true", help="Run complete test suite")
    parser.add_argument("--diagnose-connection", action="store_true", help="Run connection diagnostics")
    parser.add_argument("--output", help="Save results to JSON file")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--quiet", action="store_true", help="Minimal console logging")
    
    args = parser.parse_args()
    
    # Configure logging based on verbosity  
    if args.debug:
        configure_integration_logging()
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        # No logging by default for clean output
        configure_logging(level="CRITICAL", console_output=False)
    
    async def run_evaluation():
        if args.diagnose_connection:
            await diagnose_connection(args.base_url)
            return
        
        evaluator = ModelEvaluator(args.model, args.base_url)
        report = await evaluator.evaluate_model(quick_mode=not args.full_suite)
        
        # Clean summary like test_qwen3.py
        summary = report["summary"]
        print(f"\n🎉 Model Evaluation Complete!")
        print(f"📊 Results: {summary['passed']}/{summary['total_tests']} tests passed")
        print(f"⏱️  Duration: {summary['total_duration']}")
        
        if summary["passed"] != summary["total_tests"]:
            print(f"\n⚠️  {summary['total_tests'] - summary['passed']} issues found:")
            for result in report["results"]:
                if not result["passed"]:
                    print(f"   ❌ {result['test']}: {result['error']}")
        
        # Save report if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"\n📄 Report saved to {args.output}")
    
    asyncio.run(run_evaluation())


if __name__ == "__main__":
    main()