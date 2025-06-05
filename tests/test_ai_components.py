"""
Tests for refactored AI components.

This module tests the extracted components from the AI file selector
refactoring to ensure they work correctly in isolation.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import tempfile
import time

from repo2txt.ai.state import FileSelectionState, StateManager, TokenCache
from repo2txt.ai.qwen_utils import (
    is_qwen_model, 
    get_thinking_tag, 
    add_thinking_tag_to_message,
    clean_thinking_tags,
    ensure_message_has_thinking_tag
)
from repo2txt.ai.command_handler import CommandHandler
from repo2txt.ai.agent_session import AgentSession, SessionConfig
from repo2txt.core.models import AnalysisResult


class TestFileSelectionState:
    """Test the FileSelectionState dataclass."""
    
    def test_initial_state(self):
        """Test initial state values."""
        state = FileSelectionState()
        assert state.selected_files == []
        assert state.token_budget == 50000
        assert state.total_tokens_selected == 0
        assert state.previous_files == []
        assert state.previous_tokens == 0
        
    def test_budget_calculations(self):
        """Test budget calculation methods."""
        state = FileSelectionState(token_budget=1000, total_tokens_selected=250)
        
        assert state.get_budget_usage_percent() == 25.0
        assert state.get_budget_remaining() == 750
        assert not state.is_over_budget()
        
        # Test over budget
        state.total_tokens_selected = 1500
        assert state.get_budget_usage_percent() == 150.0
        assert state.get_budget_remaining() == -500
        assert state.is_over_budget()


class TestQwenUtils:
    """Test Qwen-specific utility functions."""
    
    def test_is_qwen_model(self):
        """Test model detection."""
        assert is_qwen_model("qwen3-7b")
        assert is_qwen_model("Qwen-14B")
        assert is_qwen_model("QWEN-TURBO")
        assert not is_qwen_model("gpt-4")
        assert not is_qwen_model("claude-3")
        
    def test_get_thinking_tag(self):
        """Test thinking tag generation."""
        assert get_thinking_tag(True) == " /think"
        assert get_thinking_tag(False) == " /no_think"
        
    def test_add_thinking_tag_to_message(self):
        """Test adding thinking tags to messages."""
        # Non-Qwen model - no change
        msg = add_thinking_tag_to_message("Hello", "gpt-4", True)
        assert msg == "Hello"
        
        # Qwen model - add tag
        msg = add_thinking_tag_to_message("Hello", "qwen3", True)
        assert msg == "Hello /think"
        
        msg = add_thinking_tag_to_message("Hello", "qwen3", False)
        assert msg == "Hello /no_think"
        
        # Already has tag - no change
        msg = add_thinking_tag_to_message("Hello /think", "qwen3", False)
        assert msg == "Hello /think"
        
    def test_clean_thinking_tags(self):
        """Test cleaning thinking tags from content."""
        # No tags
        assert clean_thinking_tags("Hello world") == "Hello world"
        
        # Single tag
        content = "Let me <think>consider this</think> help you"
        assert clean_thinking_tags(content) == "Let me help you"
        
        # Multiple tags
        content = "<think>First thought</think>Answer<think>Second thought</think>More"
        assert clean_thinking_tags(content) == "AnswerMore"
        
        # Multiline
        content = """<think>
        This is a long
        thinking block
        </think>
        The answer is 42"""
        assert clean_thinking_tags(content).strip() == "The answer is 42"
        
    def test_ensure_message_has_thinking_tag(self):
        """Test ensuring messages have thinking tags."""
        # Non-Qwen model - no change
        messages = [
            {"role": "user", "content": "Hello"}
        ]
        ensure_message_has_thinking_tag(messages, "gpt-4", True)
        assert messages[0]["content"] == "Hello"
        
        # Qwen model - add tag
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Hello"}
        ]
        ensure_message_has_thinking_tag(messages, "qwen3", True)
        assert messages[1]["content"] == "Hello /think"
        
        # Multiple user messages - only last one
        messages = [
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Response"},
            {"role": "user", "content": "Second"}
        ]
        ensure_message_has_thinking_tag(messages, "qwen3", False)
        assert messages[0]["content"] == "First"  # Unchanged
        assert messages[2]["content"] == "Second /no_think"  # Changed


class TestTokenCache:
    """Test the TokenCache class."""
    
    def test_cache_initialization(self):
        """Test cache initialization."""
        cache = TokenCache()
        assert cache.get_stats()['cache_size'] == 0
        assert cache.get_stats()['hits'] == 0
        assert cache.get_stats()['misses'] == 0
        
    def test_cache_with_file(self):
        """Test caching with real files."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("Hello world")
            temp_path = f.name
            
        try:
            cache = TokenCache()
            
            # Mock the token counter
            with patch.object(cache, '_counter') as mock_counter:
                mock_counter.count.return_value = 10
                cache._counter = mock_counter
                
                # First call - cache miss
                count1 = cache.get_token_count(temp_path)
                assert count1 == 10
                assert cache.get_stats()['misses'] == 1
                assert cache.get_stats()['hits'] == 0
                
                # Second call - cache hit
                count2 = cache.get_token_count(temp_path)
                assert count2 == 10
                assert cache.get_stats()['hits'] == 1
                assert cache.get_stats()['misses'] == 1
                
                # Modify file - cache miss
                time.sleep(0.01)  # Ensure mtime changes
                Path(temp_path).touch()
                count3 = cache.get_token_count(temp_path)
                assert count3 == 10
                assert cache.get_stats()['hits'] == 1
                assert cache.get_stats()['misses'] == 2
                
        finally:
            Path(temp_path).unlink()
            
    def test_cache_invalidation(self):
        """Test cache invalidation."""
        cache = TokenCache()
        
        # Add some fake entries
        cache._cache['file1.txt'] = (123.45, 100)
        cache._cache['file2.txt'] = (678.90, 200)
        
        assert cache.get_stats()['cache_size'] == 2
        
        # Invalidate one
        cache.invalidate('file1.txt')
        assert cache.get_stats()['cache_size'] == 1
        assert 'file1.txt' not in cache._cache
        
        # Clear all
        cache.clear()
        assert cache.get_stats()['cache_size'] == 0


class TestCommandHandler:
    """Test the CommandHandler class."""
    
    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent for testing."""
        agent = Mock()
        agent.session = Mock()
        agent.ui = Mock()
        agent.state_manager = Mock()
        agent.message_manager = Mock()
        agent.prompt_generator = Mock()
        return agent
        
    def test_command_routing(self, mock_agent):
        """Test command routing."""
        handler = CommandHandler(mock_agent)
        
        # Test help command
        result = handler.handle_command('/help')
        assert result is True
        mock_agent.ui.print_section.assert_called_once()
        
        # Test quit command
        result = handler.handle_command('/quit')
        assert result is False
        
        # Test unknown command
        result = handler.handle_command('/unknown')
        assert result is True
        mock_agent.ui.print_warning.assert_called()
        
    def test_clear_command(self, mock_agent):
        """Test /clear command."""
        handler = CommandHandler(mock_agent)
        mock_agent._create_system_prompt.return_value = "System prompt"
        
        result = handler._handle_clear()
        assert result is True
        
        mock_agent.session.clear_conversation.assert_called_once()
        mock_agent.message_manager.add_system_message.assert_called_with("System prompt")
        mock_agent.ui.print_success.assert_called()
        
    def test_toggle_commands(self, mock_agent):
        """Test /toggle commands."""
        handler = CommandHandler(mock_agent)
        
        # Test streaming toggle
        mock_agent.use_streaming = True
        result = handler._handle_toggle(['toggle', 'streaming'])
        assert result is True
        assert mock_agent.use_streaming is False
        
        # Test thinking toggle for non-Qwen
        mock_agent.model = "gpt-4"
        result = handler._handle_toggle(['toggle', 'thinking'])
        assert result is True
        mock_agent.ui.print_warning.assert_called_with(
            "Thinking mode is only available for Qwen models"
        )


class TestAgentSession:
    """Test the AgentSession class."""
    
    @pytest.fixture
    def mock_analysis_result(self):
        """Create a mock analysis result."""
        result = Mock(spec=AnalysisResult)
        result.repo_name = "test_repo"
        result.total_files = 100
        result.total_tokens = 10000
        result.file_list = [
            {'path': 'file1.txt', 'tokens': 100},
            {'path': 'file2.py', 'tokens': 200}
        ]
        return result
        
    def test_session_initialization(self, mock_analysis_result):
        """Test session initialization."""
        config = SessionConfig(
            repo_path=Path("/test/repo"),
            model="gpt-4",
            token_budget=50000
        )
        
        with patch('repo2txt.ai.agent_session.LLMClient'), \
             patch('repo2txt.ai.agent_session.ChatConsole'):
            session = AgentSession(config, mock_analysis_result, "test_key")
            
            assert session.config == config
            assert session.analysis_result == mock_analysis_result
            assert session.use_streaming is True
            assert session.enable_thinking is True
            assert len(session.state_snapshots) == 0
            
    def test_session_toggles(self, mock_analysis_result):
        """Test session toggle methods."""
        config = SessionConfig(repo_path=Path("/test"), model="gpt-4")
        
        with patch('repo2txt.ai.agent_session.LLMClient'), \
             patch('repo2txt.ai.agent_session.ChatConsole'):
            session = AgentSession(config, mock_analysis_result, "test_key")
            
            # Test streaming toggle
            assert session.use_streaming is True
            assert session.toggle_streaming() is False
            assert session.use_streaming is False
            
            # Test thinking toggle
            assert session.enable_thinking is True
            assert session.toggle_thinking() is False
            assert session.enable_thinking is False
            
            # Test prompt style cycling
            assert session.prompt_style == "standard"
            assert session.cycle_prompt_style() == "meta-reasoning"
            assert session.prompt_style == "meta-reasoning"
            assert session.cycle_prompt_style() == "xml"
            assert session.cycle_prompt_style() == "standard"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])