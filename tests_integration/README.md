# Integration Tests

Real LLM testing with live model servers - evaluate any model's compatibility with our adapters.

## Quick Commands

```bash
# Test any model you download:
python tests_integration/model_evaluation_toolkit.py --model mistralai/devstral-small-2505

# Devstral code-specific testing:
python tests_integration/devstral_evaluation.py

# Full adapter validation:
python tests_integration/test_ai_adapters.py

# Connection problems?
python tests_integration/model_evaluation_toolkit.py --diagnose-connection
```

## Test Files

### `model_evaluation_toolkit.py` - **Universal Model Testing**
Test ANY LLM model with comprehensive diagnostics and RCA:
- ✅ Adapter compatibility validation
- ✅ Server connection diagnostics
- ✅ Capability detection (tools, streaming, reasoning)
- ✅ Performance benchmarking
- ✅ Error handling validation

### `devstral_evaluation.py` - **Code Model Testing**
Devstral-specific evaluation for development tasks:
- 🤖 Code completion capabilities
- 🤖 Code explanation quality  
- 🤖 Development tool integration
- 🤖 Multi-language support
- 🤖 Complex code generation

### `test_ai_adapters.py` - **Full Integration Suite**
Comprehensive adapter testing (the original working tests):
- ✅ Model configurations and adapter creation
- ✅ Data model validation
- ✅ Live vLLM connection testing
- ✅ Tool calling and streaming
- ✅ Qwen3 thinking mode

## Test Requirements

### Core Tests (Always Run)
- Python 3.8+
- Project dependencies installed
- Proper import paths

### Live Integration Tests (Optional)
- vLLM server running at `http://localhost:8000`
- Qwen3-8B-AWQ model loaded in vLLM
- Tool calling enabled (`--enable-auto-tool-choice --tool-call-parser hermes`)

## Expected Output

All tests should pass:
```
Results: 8/8 tests passed
🎉 All tests passed! Ready to commit.
```

If vLLM server is not available, connection tests will be skipped but still marked as passing.

## Supported Models

The tests verify these model types:
- **Qwen3**: 0.6B, 4B, 8B, 14B, 32B, 30B-A3B variants
- **Devstral**: Small-2505 coding model  
- **Gemma3**: 3B, 9B, 27B instruction-tuned models

## Features Tested

- **Thinking Mode**: Qwen3 reasoning with `/think` commands
- **Tool Calling**: Function calling with JSON schema validation
- **Streaming**: Delta-based response streaming
- **Quantization**: AWQ, GPTQ, GGUF model support
- **Context Extension**: YaRN scaling for long contexts