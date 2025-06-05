import pytest
from unittest.mock import patch, MagicMock
from repo2txt.core.tokenizer import TokenCounter


class TestTokenCounter:
    def test_initialization_with_tiktoken(self):
        with patch('repo2txt.core.tokenizer.tiktoken') as mock_tiktoken:
            with patch('repo2txt.core.tokenizer.TIKTOKEN_AVAILABLE', True):
                mock_encoder = MagicMock()
                mock_tiktoken.get_encoding.return_value = mock_encoder
                
                counter = TokenCounter()
                
                assert counter.is_available is True
                assert counter.encoder == mock_encoder
                mock_tiktoken.get_encoding.assert_called_once_with("cl100k_base")

    def test_initialization_without_tiktoken(self):
        with patch('repo2txt.core.tokenizer.TIKTOKEN_AVAILABLE', False):
            counter = TokenCounter()
            
            assert counter.is_available is False
            assert counter.encoder is None

    def test_initialization_with_custom_encoding(self):
        with patch('repo2txt.core.tokenizer.tiktoken') as mock_tiktoken:
            with patch('repo2txt.core.tokenizer.TIKTOKEN_AVAILABLE', True):
                mock_encoder = MagicMock()
                mock_tiktoken.get_encoding.return_value = mock_encoder
                
                counter = TokenCounter("gpt2")
                
                assert counter.encoding_name == "gpt2"
                mock_tiktoken.get_encoding.assert_called_once_with("gpt2")

    def test_count_with_encoder(self):
        with patch('repo2txt.core.tokenizer.tiktoken') as mock_tiktoken:
            with patch('repo2txt.core.tokenizer.TIKTOKEN_AVAILABLE', True):
                mock_encoder = MagicMock()
                mock_encoder.encode.return_value = [1, 2, 3, 4, 5]
                mock_tiktoken.get_encoding.return_value = mock_encoder
                
                counter = TokenCounter()
                result = counter.count("test text")
                
                assert result == 5
                mock_encoder.encode.assert_called_once_with("test text")

    def test_count_without_encoder(self):
        with patch('repo2txt.core.tokenizer.TIKTOKEN_AVAILABLE', False):
            counter = TokenCounter()
            
            result = counter.count("hello world")
            
            assert result == 0  # No encoder means 0 tokens

    def test_count_empty_string(self):
        counter = TokenCounter()
        
        assert counter.count("") == 0
        # Whitespace-only strings still have tokens
        # assert counter.count("   ") == 0  # This would fail

    def test_estimate_tokens(self):
        # Test the static estimation method
        assert TokenCounter.estimate_tokens("") == 0
        assert TokenCounter.estimate_tokens("hello") > 0
        assert TokenCounter.estimate_tokens("hello world") > TokenCounter.estimate_tokens("hello")

    def test_count_batch(self):
        with patch('repo2txt.core.tokenizer.tiktoken') as mock_tiktoken:
            with patch('repo2txt.core.tokenizer.TIKTOKEN_AVAILABLE', True):
                mock_encoder = MagicMock()
                mock_encoder.encode.side_effect = [[1, 2], [1, 2, 3], [1]]
                mock_tiktoken.get_encoding.return_value = mock_encoder
                
                counter = TokenCounter()
                texts = {"file1": "hello", "file2": "hello world", "file3": "!"}
                results = counter.count_batch(texts)
                
                assert results == {"file1": 2, "file2": 3, "file3": 1}

    def test_encoder_error_handling(self):
        with patch('repo2txt.core.tokenizer.tiktoken') as mock_tiktoken:
            with patch('repo2txt.core.tokenizer.TIKTOKEN_AVAILABLE', True):
                mock_encoder = MagicMock()
                mock_encoder.encode.side_effect = Exception("Encoding error")
                mock_tiktoken.get_encoding.return_value = mock_encoder
                
                counter = TokenCounter()
                result = counter.count("test text")
                
                assert result == 0  # Error returns 0