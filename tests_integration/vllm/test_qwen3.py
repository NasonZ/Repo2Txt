#!/usr/bin/env python3
"""
Simple vLLM Qwen3-8B-AWQ Test Script
Parses <think> tags manually without reasoning parser

USAGE:
    # Run direct vLLM testing
    python tests_integration/vllm/test_qwen3.py
    
    # Or run full integration suite
    python tests_integration/test_ai_adapters.py
    
    # Or use comprehensive evaluation toolkit
    python tests_integration/model_evaluation_toolkit.py --model Qwen/Qwen3-8B-AWQ --full-suite

REQUIRES:
    vLLM server running with: vllm serve Qwen/Qwen3-8B-AWQ --enable-auto-tool-choice --tool-call-parser hermes
"""

import time
from openai import OpenAI

# Configuration
API_KEY = "EMPTY"
BASE_URL = "http://localhost:8000/v1"
MODEL = "Qwen/Qwen3-8B-AWQ"

def parse_thinking_response(response_text):
    """Extract thinking and final answer from response"""
    if not response_text:
        return None, "No response"
    
    if "<think>" in response_text:
        start = response_text.find("<think>") + 7
        
        # Look for closing tag
        if "</think>" in response_text:
            end = response_text.find("</think>")
            thinking = response_text[start:end].strip()
            answer = response_text[end + 8:].strip()
        else:
            # No closing tag - take everything after <think> as thinking
            thinking = response_text[start:].strip()
            answer = "No final answer (thinking may be truncated)"
        
        return thinking, answer
    else:
        return None, response_text

def test_thinking_mode():
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    
    print("Test 1: Thinking Mode Enabled")
    print("-" * 50)
    
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "What is 15 + 28? Think step by step."}],
        temperature=0.6,
        top_p=0.95,
        max_tokens=1000,
        presence_penalty=1.5,
        extra_body={
            "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": True}
        }
    )
    
    full_response = response.choices[0].message.content
    thinking, answer = parse_thinking_response(full_response)
    
    print(f"Full response: {full_response[:100]}...")
    print(f"\nThinking extracted: {thinking[:100] if thinking else 'None'}...")
    print(f"Answer extracted: {answer}")
    print()

def test_non_thinking_mode():
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    
    print("\nTest 2: Non-Thinking Mode")
    print("-" * 50)
    
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "What is the capital of Japan?"}],
        temperature=0.7,
        top_p=0.8,
        max_tokens=100,
        presence_penalty=1.5,
        extra_body={
            "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": False}
        }
    )
    
    full_response = response.choices[0].message.content
    print(f"Response: {full_response}")
    print()

def test_streaming():
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    
    print("\nTest 3: Streaming Response")
    print("-" * 50)
    
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Count from 1 to 5 slowly."}],
        stream=True,
        temperature=0.7,
        top_p=0.8,
        max_tokens=100,
        presence_penalty=1.5,
        extra_body={
            "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": False}
        }
    )
    
    print("Streaming: ", end="")
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end='', flush=True)
    print("\n")

def test_tool_calling():
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    
    print("\nTest 4: Tool Calling")
    print("-" * 50)
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "The unit of temperature"
                        }
                    },
                    "required": ["location"]
                }
            }
        },
        {
            "type": "function", 
            "function": {
                "name": "calculate",
                "description": "Perform basic arithmetic calculations",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "The mathematical expression to evaluate"
                        }
                    },
                    "required": ["expression"]
                }
            }
        }
    ]
    
    try:
        # Test 1: Simple tool call
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
            tools=tools,
            tool_choice="auto",
            temperature=0.7,
            presence_penalty=1.5,
            extra_body={
                "top_k": 20,
                "chat_template_kwargs": {"enable_thinking": False}
            }
        )
        
        message = response.choices[0].message
        if message.tool_calls:
            print("Tool calls detected:")
            for tool_call in message.tool_calls:
                print(f"  - Function: {tool_call.function.name}")
                print(f"    Arguments: {tool_call.function.arguments}")
        else:
            print(f"No tool calls. Response: {message.content}")
        
        # Test 2: Multiple tool calls
        print("\nTesting multiple tool calls...")
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "What's the weather in both Tokyo and New York? Also calculate 42 * 17."}],
            tools=tools,
            tool_choice="auto",
            temperature=0.7,
            presence_penalty=1.5,
            extra_body={
                "top_k": 20,
                "chat_template_kwargs": {"enable_thinking": False}
            }
        )
        
        message = response.choices[0].message
        if message.tool_calls:
            print(f"Found {len(message.tool_calls)} tool calls:")
            for tool_call in message.tool_calls:
                print(f"  - {tool_call.function.name}: {tool_call.function.arguments}")
        else:
            print(f"No tool calls. Response: {message.content}")
            
    except Exception as e:
        print(f"Tool calling test error: {e}")
        print("Make sure vLLM was started with --enable-auto-tool-choice --tool-call-parser hermes")

def test_tool_calling_with_streaming():
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    
    print("\nTest 5: Tool Calling with Streaming")
    print("-" * 50)
    
    tools = [{
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get the current time in a timezone",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {"type": "string", "description": "The timezone, e.g. 'UTC', 'EST'"}
                },
                "required": ["timezone"]
            }
        }
    }]
    
    try:
        stream = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "What time is it in UTC?"}],
            tools=tools,
            tool_choice="auto",
            stream=True,
            temperature=0.7,
            presence_penalty=1.5,
            extra_body={
                "top_k": 20,
                "chat_template_kwargs": {"enable_thinking": False}
            }
        )
        
        print("Streaming tool call response:")
        tool_calls = []
        for chunk in stream:
            if chunk.choices[0].delta.tool_calls:
                for tool_call in chunk.choices[0].delta.tool_calls:
                    if tool_call.index >= len(tool_calls):
                        tool_calls.append({"name": "", "arguments": ""})
                    if tool_call.function.name:
                        tool_calls[tool_call.index]["name"] = tool_call.function.name
                    if tool_call.function.arguments:
                        tool_calls[tool_call.index]["arguments"] += tool_call.function.arguments
            elif chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end='', flush=True)
        
        if tool_calls:
            print("\nStreamed tool calls:")
            for tc in tool_calls:
                print(f"  - {tc['name']}: {tc['arguments']}")
        print()
        
    except Exception as e:
        print(f"Streaming tool call error: {e}")

def main():
    print("vLLM Qwen3-8B-AWQ Simple Test")
    print("="*50)
    print(f"Endpoint: {BASE_URL}")
    print(f"Model: {MODEL}")
    print("="*50)
    print()
    
    try:
        test_thinking_mode()
        time.sleep(1)
        
        test_non_thinking_mode()
        time.sleep(1)
        
        test_streaming()
        time.sleep(1)
        
        test_tool_calling()
        time.sleep(1)
        
        test_tool_calling_with_streaming()
        
        print("\nAll tests completed!")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure vLLM is running with appropriate flags")

if __name__ == "__main__":
    main()