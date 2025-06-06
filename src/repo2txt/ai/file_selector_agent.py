#!/usr/bin/env python3
"""
File Selector Agent

Provides AI-assisted file selection capabilities for repo2txt,
allowing users to interactively select files from repositories
using natural language queries and intelligent recommendations.
"""

import asyncio
import json
from pathlib import Path
from typing import Dict, List, Optional, Any

from .state import StateManager
from .tools import ToolExecutor, ToolCall, ToolResult, create_file_selection_tools, parse_openai_tool_calls
from .prompts import PromptGenerator
from .llm import LLMClient, MessageManager
from .console_chat import ChatConsole
from .command_handler import CommandHandler
from .agent_session import AgentSession, SessionConfig
from .chat_orchestrator import ChatOrchestrator
from ..core.models import Config, AnalysisResult
from ..adapters import create_adapter


class FileSelectorAgent:
    def __init__(self, repo_path: str, openai_api_key: str, model: str, base_url: str = None, 
                 theme: str = "green", token_budget: int = 50000, debug_mode: bool = False, 
                 prompt_style: str = "standard", analysis_result: Optional[AnalysisResult] = None):
        """Initialize the file selector agent.
        
        Args:
            repo_path: Path to repository (can be empty if analysis_result is provided)
            openai_api_key: OpenAI API key
            model: Model name to use
            base_url: Optional base URL for API
            theme: UI theme
            token_budget: Token budget for selection
            debug_mode: Enable debug output
            prompt_style: System prompt style
            analysis_result: Pre-analyzed repository data (if provided, skips analysis)
        """
        # Handle repository path and analysis
        if analysis_result:
            # Use pre-analyzed data
            repo_path_obj = Path(repo_path) if repo_path else Path(".")
            self.analysis_result = analysis_result
        else:
            # Analyze repository from path
            repo_path_obj = Path(repo_path).resolve()
            if not repo_path_obj.is_dir():
                raise ValueError(f"Repository path '{repo_path}' not found or not a directory.")
            # We'll analyze after session is created
            self.analysis_result = None
            
        # Create session configuration
        session_config = SessionConfig(
            repo_path=repo_path_obj,
            model=model,
            base_url=base_url,
            theme=theme,
            token_budget=token_budget,
            debug_mode=debug_mode,
            prompt_style=prompt_style
        )
        
        # Store for backward compatibility
        self.repo_path = repo_path_obj
        self.openai_api_key = openai_api_key
        
        # Analyze repository if needed
        if not self.analysis_result:
            # Create temporary UI for analysis
            self.ui = ChatConsole(theme=theme, debug_mode=debug_mode)
            self._analyze_repository()
            
        # Initialize session with analysis result
        self.session = AgentSession(session_config, self.analysis_result, openai_api_key)
        
        # Set up proxies for backward compatibility
        self._setup_compatibility_proxies()
        
        # Register tools
        self._register_tools()
        
        # Initialize command handler and chat orchestrator
        self.command_handler = CommandHandler(self)
        self.chat_orchestrator = ChatOrchestrator(self.session)
    
    def _setup_compatibility_proxies(self):
        """Set up property proxies for backward compatibility.
        
        This allows existing code to continue accessing properties directly
        on the agent while the actual state lives in the session.
        """
        # Component proxies (these don't change)
        self.ui = self.session.ui
        self.llm_client = self.session.llm_client
        self.message_manager = self.session.message_manager
        self.state_manager = self.session.state_manager
        self.prompt_generator = self.session.prompt_generator
        self.tool_executor = self.session.tool_executor
        self.state_snapshots = self.session.state_snapshots
        self.config = self.session.repo_config
        
    # Properties for mutable state (with getters/setters)
    @property
    def model(self):
        return self.session.config.model
        
    @property
    def base_url(self):
        return self.session.config.base_url
        
    @property
    def debug_mode(self):
        return self.session.debug_mode
        
    @debug_mode.setter
    def debug_mode(self, value):
        self.session.debug_mode = value
        self.session.ui.debug_mode = value
        
    @property
    def prompt_style(self):
        return self.session.prompt_style
        
    @prompt_style.setter  
    def prompt_style(self, value):
        self.session.set_prompt_style(value)
        
    @property
    def use_streaming(self):
        return self.session.use_streaming
        
    @use_streaming.setter
    def use_streaming(self, value):
        self.session.use_streaming = value
        
    @property
    def enable_thinking(self):
        return self.session.enable_thinking
        
    @enable_thinking.setter
    def enable_thinking(self, value):
        self.session.enable_thinking = value

    def _analyze_repository(self):
        """Analyze repository using repo2txt components."""
        self.ui.print_info(f"Analyzing repository: {self.repo_path}")
        
        try:
            # Create adapter for the repository
            adapter = create_adapter(str(self.repo_path), self.config, validate_size=False)
            repo_name = adapter.get_name()
            
            # Build file tree and get file list
            file_tree = adapter.build_file_tree()
            file_list = adapter.get_file_list()
            
            # Get README content
            readme_content = adapter.get_readme_content()
            
            # Create analysis result
            self.analysis_result = AnalysisResult(
                repo_name=repo_name,
                branch=None,
                readme_content=readme_content,
                structure="",
                file_contents="",
                token_data={},
                total_tokens=sum(f.get('tokens', 0) for f in file_list),
                total_files=len(file_list),
                errors=adapter.errors,
                file_tree=file_tree,
                file_list=file_list
            )
            
            self.ui.print_info(f"Found {self.analysis_result.total_files} files with {self.analysis_result.total_tokens:,} tokens")
            
        except Exception as e:
            self.ui.print_error(f"Failed to analyze repository: {e}")
            raise





    
    def _create_system_prompt(self) -> str:
        """Create system prompt using selected style."""
        system_prompt = self.prompt_generator.generate(self.prompt_style)
        
        # Show system prompt in debug mode
        self.ui.print_debug_system_prompt(system_prompt)
        return system_prompt

    # Tool implementations (now delegate to state manager)
    def select_files_impl(self, paths: List[str], reasoning: str) -> Dict[str, Any]:
        """Select initial set of files via state manager."""
        return self.state_manager.replace_selection(paths, reasoning)

    def adjust_selection_impl(self, add_files: Optional[List[str]] = None, remove_files: Optional[List[str]] = None, reasoning: str = "") -> Dict[str, Any]:
        """Adjust current selection via state manager."""
        return self.state_manager.modify_selection(add_files, remove_files, reasoning)

    def _register_tools(self):
        """Register tools with state manager integration."""
        tool_defs = create_file_selection_tools()
        
        for tool_obj, method_name in tool_defs:
            # Map method names to actual methods
            if method_name == "select_files_impl":
                func = self.select_files_impl
            elif method_name == "adjust_selection_impl":
                func = self.adjust_selection_impl
            else:
                raise ValueError(f"Unknown tool method: {method_name}")
                
            self.tool_executor.register_tool(tool_obj, func)


    
    def _save_state_snapshot(self):
        """Save current state for undo functionality."""
        self.session.save_snapshot()
    
    
    def _handle_command(self, command: str) -> bool:
        """Delegate to CommandHandler."""
        return self.command_handler.handle_command(command)
    
    
    async def _execute_tool_calls_and_update_history(self, tool_calls: List[ToolCall]):
        """Execute tool calls and update conversation history."""
        tool_results_for_history = []
        for tool_call in tool_calls:
            self.ui.print_tool_call(tool_call)
            result = await self.tool_executor.execute_tool(tool_call)
            self.ui.print_tool_result(result)
            
            # Show state diff after tool execution in BOTH debug and clean modes
            self.ui.print_state_diff(self.state_manager)
            
            # Only show error feedback in clean mode (success is shown in state diff)
            if not self.debug_mode and result.error:
                self.ui.print_error(f"Tool {tool_call.tool_name} failed: {result.error}")
            
            output_content = json.dumps(result.output) if result.is_json_output else str(result.output)
            if result.error:
                output_content = json.dumps({"error": result.error, "details": str(result.output)})

            tool_results_for_history.append({
                "role": "tool", 
                "tool_call_id": result.tool_call_id, 
                "name": tool_call.tool_name, 
                "content": output_content
            })
        
        self.message_manager.add_tool_results(tool_results_for_history)


    async def chat_loop(self):
        """Main chat loop."""
        await self.chat_orchestrator.run_chat_loop(
            prompt_generator_func=self._create_system_prompt,
            save_snapshot_func=self._save_state_snapshot,
            handle_command_func=self._handle_command,
            execute_tools_func=self._execute_tool_calls_and_update_history
        )

    def run(self):
        """Run the agent interactively and return selected files."""
        self.chat_orchestrator.show_banner()
        
        try:
            asyncio.run(self.chat_loop())
        except KeyboardInterrupt:
            self.ui.print_warning("\nInterrupted by user.")
            self.ui.print_success("File Selector Agent shutting down. Goodbye! 👋")
            raise KeyboardInterrupt("AI selection interrupted")
        finally:
            pass
        
        # Return the selected files
        return self.get_selected_files()
    
    def run_with_query(self, query: str) -> List[str]:
        """Run the agent with a specific query and return selected files.
        
        Args:
            query: The query to use for file selection
            
        Returns:
            List of selected file paths
        """
        # Initialize system prompt
        system_prompt = self._create_system_prompt()
        self.message_manager.clear()
        self.message_manager.add_system_message(system_prompt)
        
        # Add the query as a user message
        self.message_manager.add_user_message(query)
        
        # Run one iteration of the chat to get AI response
        try:
            # Get AI response with tools
            openai_tools_fmt = self.tool_executor.get_tools_for_openai()
            messages = self.message_manager.get_messages()
            
            completion = self.llm_client.create_completion(
                messages,
                openai_tools_fmt,
                stream=False
            )
            
            msg = completion.choices[0].message
            
            # Process any tool calls
            if msg.tool_calls:
                invoked_tool_calls = parse_openai_tool_calls(msg.tool_calls)
                
                # Show tool calls even in non-interactive mode
                for idx, tool_call in enumerate(invoked_tool_calls):
                    self.ui.print(f"\n[{self.ui.colors['warning']}]🔧 Calling tool {idx + 1}: {tool_call.tool_name}[/{self.ui.colors['warning']}]")
                
                # Execute tools with full display
                for tool_call in invoked_tool_calls:
                    result = asyncio.run(self.tool_executor.execute_tool(tool_call))
                    
                    # Show state diff after tool execution
                    self.ui.print_state_diff(self.state_manager)
                    
                    # Only show error feedback in clean mode (success is shown in state diff)
                    if not self.debug_mode and result.error:
                        self.ui.print_error(f"Tool {tool_call.tool_name} failed: {result.error}")
            
            # Return selected files
            return self.get_selected_files()
            
        except Exception as e:
            self.ui.print_error(f"Error during query execution: {e}")
            return []
    
    def get_selected_files(self) -> List[str]:
        """Get the currently selected files.
        
        Returns:
            List of selected file paths
        """
        return self.session.get_selected_files()
