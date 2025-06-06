import pytest
from unittest.mock import patch, mock_open, MagicMock
from repo2txt.core.file_analyzer import FileAnalyzer
from repo2txt.core.models import Config


class TestFileAnalyzer:
    @pytest.fixture
    def analyzer(self):
        config = Config()
        return FileAnalyzer(config)

    def test_is_binary_file_by_extension(self, analyzer):
        assert analyzer.is_binary_file("/path/to/file.exe") is True
        assert analyzer.is_binary_file("/path/to/file.pdf") is True
        assert analyzer.is_binary_file("/path/to/file.py") is False
        assert analyzer.is_binary_file("/path/to/file.txt") is False

    def test_is_binary_file_by_mimetype(self, analyzer):
        with patch('mimetypes.guess_type') as mock_guess:
            mock_guess.return_value = ('application/octet-stream', None)
            assert analyzer.is_binary_file("/path/to/unknown") is True
            
            mock_guess.return_value = ('text/plain', None)
            assert analyzer.is_binary_file("/path/to/unknown") is False

    def test_is_binary_file_by_content(self, analyzer):
        content = b'hello world'
        assert analyzer.is_binary_file("/path/to/file.txt", content) is False

    def test_is_binary_file_with_null_bytes(self, analyzer):
        content = b'hello\x00world'
        # .txt files are not binary by extension, so it goes to content check
        assert analyzer.is_binary_file("/path/to/file", content) is True

    @patch('os.path.getsize', return_value=100)
    @patch('builtins.open', new_callable=mock_open, read_data=b'hello world')
    def test_read_file_content_success(self, mock_file, mock_size, analyzer):
        content, error = analyzer.read_file_content("/path/to/file.txt")
        assert content == "hello world"
        assert error is None

    @patch('os.path.getsize', return_value=2 * 1024 * 1024)
    def test_read_file_content_too_large(self, mock_size, analyzer):
        content, error = analyzer.read_file_content("/path/to/large.txt")
        assert content is None
        assert "File too large" in error

    @patch('os.path.getsize', return_value=100)
    @patch('builtins.open', new_callable=mock_open, read_data=b'\x89PNG\r\n\x1a\n')
    def test_read_file_content_binary(self, mock_file, mock_size, analyzer):
        content, error = analyzer.read_file_content("/path/to/image.png")
        assert content is None
        assert error == "Binary file"

    @patch('os.path.getsize', return_value=100)
    def test_read_file_content_permission_error(self, mock_size, analyzer):
        with patch('builtins.open', side_effect=PermissionError("Access denied")):
            content, error = analyzer.read_file_content("/path/to/file.txt")
            assert content is None
            assert error == "Permission denied"

    def test_count_tokens_with_encoder(self, analyzer):
        # Mock the encoder
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = [1, 2, 3, 4, 5]
        analyzer.token_encoder = mock_encoder
        
        result = analyzer.count_tokens("test content")
        assert result == 5

    def test_count_tokens_no_encoder(self, analyzer):
        analyzer.token_encoder = None
        
        result = analyzer.count_tokens("test content")
        assert result == 0


class TestFileAnalyzerSecurity:
    """Essential security tests for FileAnalyzer - real threats only."""
    
    @pytest.fixture
    def analyzer(self):
        config = Config()
        return FileAnalyzer(config)
    
    def test_large_file_protection(self, analyzer):
        """CRITICAL: Test protection against DoS via huge files."""
        with patch('os.path.getsize', return_value=10 * 1024 * 1024 * 1024):  # 10GB
            content, error = analyzer.read_file_content("/path/to/huge.txt")
            assert content is None, "Huge file was not blocked"
            assert "File too large" in error
    
    def test_null_byte_detection(self, analyzer):
        """IMPORTANT: Test null byte detection prevents content injection."""
        import tempfile
        import os
        
        # Create real temporary file with null bytes
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(b'content\x00injection')
            tmp_file.flush()
            
            try:
                content, error = analyzer.read_file_content(tmp_file.name)
                
                # Should detect as binary due to null bytes
                assert content is None, "File with null bytes was not detected as binary"
                assert error == "Binary file"
            finally:
                os.unlink(tmp_file.name)
    
    def test_binary_content_detection(self, analyzer):
        """IMPORTANT: Test binary detection prevents malicious content parsing."""
        # File that looks like text but has binary markers
        malicious_content = b'#!/bin/bash\necho "hello"\x00\xff\xfe'
        
        is_binary = analyzer.is_binary_file("/path/to/script.sh", malicious_content)
        assert is_binary is True, "Malicious binary content was not detected"