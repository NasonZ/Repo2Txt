#!/usr/bin/env python3
"""
Devstral Model Evaluation Script

Quick evaluation specifically for Devstral models to validate compatibility
with our adapters and identify any model-specific issues.

USAGE:
    # Quick test when you download Devstral
    python tests_integration/devstral_evaluation.py
    
    # Full evaluation with performance metrics
    python tests_integration/devstral_evaluation.py --full
    
    # Test specific Devstral variant
    python tests_integration/devstral_evaluation.py --model mistralai/devstral-small-2505
"""

import asyncio
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.repo2txt.ai import (
    create_vllm_adapter, ModelType, LLMRequest, Message, MessageRole, Tool
)


class DevstralEvaluator:
    """Devstral-specific model evaluation."""
    
    def __init__(self, model_name: str = None, base_url: str = "http://localhost:8000"):
        self.model_name = model_name or ModelType.DEVSTRAL_SMALL.value
        self.base_url = base_url
        
    async def run_devstral_tests(self, full_suite: bool = False):
        """Run Devstral-specific evaluation tests."""
        print("🤖 Devstral Model Evaluation")
        print("=" * 50)
        print(f"Model: {self.model_name}")
        print(f"Server: {self.base_url}")
        print()
        
        try:
            # Create adapter
            adapter = create_vllm_adapter(ModelType.DEVSTRAL_SMALL, base_url=self.base_url)
            print("✅ Devstral adapter created successfully")
            
            # Test 1: Health check
            print("\n🔍 Testing server connection...")
            if await adapter.health_check():
                print("✅ vLLM server is healthy")
            else:
                print("❌ Server health check failed")
                print("💡 Start Devstral with: vllm serve mistralai/devstral-small-2505")
                return
            
            # Test 2: Model availability
            print("\n🔍 Checking model availability...")
            try:
                models = await adapter.get_available_models()
                if self.model_name in models:
                    print(f"✅ Devstral model loaded: {self.model_name}")
                else:
                    print(f"❌ Model not found. Available: {models}")
                    return
            except Exception as e:
                print(f"⚠️  Could not check models: {e}")
            
            # Test 3: Basic code completion (Devstral's specialty)
            print("\n🔍 Testing code completion capabilities...")
            await self._test_code_completion(adapter)
            
            # Test 4: Code explanation
            print("\n🔍 Testing code explanation...")
            await self._test_code_explanation(adapter)
            
            # Test 5: Tool calling for development tasks
            print("\n🔍 Testing development tool calling...")
            await self._test_development_tools(adapter)
            
            if full_suite:
                # Test 6: Multiple programming languages
                print("\n🔍 Testing multi-language support...")
                await self._test_multi_language_support(adapter)
                
                # Test 7: Performance with code tasks
                print("\n🔍 Performance testing with code generation...")
                await self._test_code_performance(adapter)
            
            print("\n🎉 Devstral evaluation completed!")
            
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            print("💡 Check that vLLM server is running with Devstral")
    
    async def _test_code_completion(self, adapter):
        """Test Devstral's code completion abilities."""
        code_prompt = """Complete this Python function:

def fibonacci(n):
    '''Return the nth Fibonacci number'''
    if n <= 1:
        return n
    # Complete this function"""
        
        try:
            request = LLMRequest(
                messages=[Message(role=MessageRole.USER, content=code_prompt)],
                model=adapter.model,
                max_tokens=200,
                temperature=0.2
            )
            
            start_time = time.time()
            response = await adapter.complete(request)
            duration = time.time() - start_time
            
            if response.content and "fibonacci" in response.content.lower():
                print(f"✅ Code completion successful ({duration:.2f}s)")
                print(f"   Generated {len(response.content)} characters")
                if response.usage:
                    print(f"   Tokens: {response.usage.output_tokens} output")
            else:
                print("❌ Code completion failed - no relevant code generated")
                print(f"   Response: {response.content[:100]}...")
                
        except Exception as e:
            print(f"❌ Code completion error: {e}")
    
    async def _test_code_explanation(self, adapter):
        """Test Devstral's code explanation abilities."""
        code_to_explain = """
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)
"""
        
        try:
            request = LLMRequest(
                messages=[Message(
                    role=MessageRole.USER, 
                    content=f"Explain this code in detail:\n{code_to_explain}"
                )],
                model=adapter.model,
                max_tokens=300,
                temperature=0.3
            )
            
            response = await adapter.complete(request)
            
            if response.content and any(word in response.content.lower() 
                                     for word in ["quicksort", "pivot", "recursive", "algorithm"]):
                print("✅ Code explanation successful")
                print(f"   Explanation length: {len(response.content)} characters")
            else:
                print("❌ Code explanation unclear or incomplete")
                print(f"   Response: {response.content[:150]}...")
                
        except Exception as e:
            print(f"❌ Code explanation error: {e}")
    
    async def _test_development_tools(self, adapter):
        """Test tool calling for development-specific tasks."""
        if not adapter.capabilities.supports_tools:
            print("⚠️  Devstral adapter doesn't support tools")
            return
        
        # Define development-focused tools
        dev_tools = [
            Tool(
                name="run_code",
                description="Execute code and return results",
                parameters={
                    "type": "object",
                    "properties": {
                        "language": {"type": "string", "enum": ["python", "javascript", "bash"]},
                        "code": {"type": "string", "description": "Code to execute"}
                    },
                    "required": ["language", "code"]
                }
            ),
            Tool(
                name="format_code",
                description="Format code according to style guidelines",
                parameters={
                    "type": "object",
                    "properties": {
                        "language": {"type": "string"},
                        "code": {"type": "string"}
                    },
                    "required": ["language", "code"]
                }
            )
        ]
        
        try:
            request = LLMRequest(
                messages=[Message(
                    role=MessageRole.USER,
                    content="I need to run this Python code: print('Hello from Devstral!'). Use the appropriate tool."
                )],
                model=adapter.model,
                max_tokens=150,
                temperature=0.2
            )
            
            response = await adapter.complete_with_tools(request, dev_tools)
            
            if response.has_tool_calls:
                print(f"✅ Development tool calling successful")
                for tc in response.tool_calls:
                    print(f"   Tool used: {tc.tool_name}")
                    print(f"   Arguments: {tc.input}")
            else:
                print("❌ No development tools were called")
                print(f"   Response: {response.content[:100]}...")
                
        except Exception as e:
            print(f"❌ Development tool calling error: {e}")
    
    async def _test_multi_language_support(self, adapter):
        """Test Devstral's support for multiple programming languages."""
        language_tests = [
            ("Python", "def hello(): print('Hello World')"),
            ("JavaScript", "function hello() { console.log('Hello World'); }"),
            ("Rust", "fn main() { println!(\"Hello World\"); }"),
            ("Go", "func main() { fmt.Println(\"Hello World\") }")
        ]
        
        successful_languages = []
        
        for lang, code in language_tests:
            try:
                request = LLMRequest(
                    messages=[Message(
                        role=MessageRole.USER,
                        content=f"Explain this {lang} code and suggest improvements:\n{code}"
                    )],
                    model=adapter.model,
                    max_tokens=150,
                    temperature=0.2
                )
                
                response = await adapter.complete(request)
                
                if response.content and lang.lower() in response.content.lower():
                    successful_languages.append(lang)
                    
            except Exception:
                pass
        
        print(f"✅ Multi-language support: {len(successful_languages)}/4 languages")
        print(f"   Supported: {', '.join(successful_languages)}")
        
        if len(successful_languages) >= 3:
            print("   🎉 Excellent multi-language support!")
        elif len(successful_languages) >= 2:
            print("   👍 Good multi-language support")
        else:
            print("   ⚠️  Limited multi-language support")
    
    async def _test_code_performance(self, adapter):
        """Test performance with code generation tasks."""
        complex_prompt = """Create a Python class that implements a binary search tree with the following methods:
- insert(value)
- search(value) 
- delete(value)
- in_order_traversal()

Include proper error handling and docstrings."""
        
        try:
            start_time = time.time()
            
            request = LLMRequest(
                messages=[Message(role=MessageRole.USER, content=complex_prompt)],
                model=adapter.model,
                max_tokens=500,
                temperature=0.3
            )
            
            response = await adapter.complete(request)
            generation_time = time.time() - start_time
            
            # Analyze the response
            code_indicators = ["class", "def", "insert", "search", "delete", "traversal"]
            code_quality = sum(1 for indicator in code_indicators 
                             if indicator in response.content.lower())
            
            tokens_per_second = 0
            if response.usage and response.usage.output_tokens:
                tokens_per_second = response.usage.output_tokens / generation_time
            
            print(f"✅ Complex code generation completed")
            print(f"   Generation time: {generation_time:.2f}s")
            print(f"   Tokens/second: {tokens_per_second:.1f}")
            print(f"   Code quality indicators: {code_quality}/{len(code_indicators)}")
            print(f"   Response length: {len(response.content)} characters")
            
            if code_quality >= 4:
                print("   🎉 High-quality code generation!")
            elif code_quality >= 2:
                print("   👍 Decent code generation")
            else:
                print("   ⚠️  Code generation needs improvement")
                
        except Exception as e:
            print(f"❌ Performance test error: {e}")


async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Devstral Model Evaluation")
    parser.add_argument("--model", help="Specific Devstral model to test")
    parser.add_argument("--base-url", default="http://localhost:8000", help="vLLM server URL")
    parser.add_argument("--full", action="store_true", help="Run full evaluation suite")
    
    args = parser.parse_args()
    
    evaluator = DevstralEvaluator(args.model, args.base_url)
    await evaluator.run_devstral_tests(args.full)


if __name__ == "__main__":
    asyncio.run(main())