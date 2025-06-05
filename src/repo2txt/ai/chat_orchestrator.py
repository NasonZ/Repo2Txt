"""
Chat Orchestration for AI File Selection

Handles the main chat loop, streaming responses, and coordination
between user input, AI responses, and tool execution.
"""

import asyncio
import logging
from typing import List, Tuple, Optional, TYPE_CHECKING

from openai import APIError
from rich.prompt import Prompt

from .tools import ToolCall, parse_openai_tool_calls
from .qwen_utils import add_thinking_tag_to_message, ensure_message_has_thinking_tag

if TYPE_CHECKING:
    from .agent_session import AgentSession


class ChatOrchestrator:
    """Orchestrates the chat interaction flow."""
    
    def __init__(self, session: 'AgentSession'):
        """Initialize with agent session.
        
        Args:
            session: The agent session containing all state and components
        """
        self.session = session
        self.ui = session.ui
        self.llm_client = session.llm_client
        self.message_manager = session.message_manager
        self.tool_executor = session.tool_executor
        
    async def run_chat_loop(self, prompt_generator_func, save_snapshot_func, 
                           handle_command_func, execute_tools_func) -> None:
        """Run the main chat loop.
        
        Args:
            prompt_generator_func: Function to generate system prompts
            save_snapshot_func: Function to save state snapshots
            handle_command_func: Function to handle commands
            execute_tools_func: Function to execute tool calls
        """
        self.ui.print_section("CHAT MODE", "Type '/help' for commands, /generate for reports and /quit to exit.")
        
        # Initialize with system prompt
        system_prompt = prompt_generator_func()
        self.message_manager.add_system_message(system_prompt)
        
        while True:
            try:
                # Get user input
                user_input = await self._get_user_input()
                
                if user_input is None:  # User wants to quit
                    break
                    
                # Handle commands
                if user_input.startswith('/'):
                    command_result = handle_command_func(user_input.strip())
                    if command_result is False:  # /quit or /exit
                        break
                    elif command_result == "REDO":  # Special case for redo
                        # Don't continue - let it fall through to regenerate
                        pass
                    else:
                        continue
                        
                if not user_input:
                    continue
                
                # Process user message
                self._process_user_message(user_input, save_snapshot_func)
                
                # Update system prompt
                updated_system_prompt = prompt_generator_func()
                self.message_manager.update_system_message(updated_system_prompt)
                
                # Get AI response
                response_content, tool_calls = await self._get_ai_response()
                
                # Store assistant message
                self._store_assistant_message(response_content, tool_calls)
                
                # Execute tools if any
                if tool_calls:
                    await self._handle_tool_execution(
                        tool_calls, 
                        prompt_generator_func,
                        execute_tools_func
                    )
                    
            except APIError as e:
                self.ui.print_error(f"OpenAI API Error: {e.code} - {e.message}")
            except KeyboardInterrupt:
                self.ui.print_warning("\nInterrupted by user.")
                break
            except Exception as e:
                self.ui.print_error(f"Unexpected error: {e}")
                logging.error("Chat loop error", exc_info=True)
                break
    
    async def _get_user_input(self) -> Optional[str]:
        """Get user input from prompt.
        
        Returns:
            User input string or None if user wants to quit
        """
        user_input = Prompt.ask(f"\n[{self.ui.colors['primary']}][>][/{self.ui.colors['primary']}]").strip()
        
        if user_input.lower() in ['quit', 'exit']:
            return None
            
        return user_input
    
    def _process_user_message(self, user_input: str, save_snapshot_func) -> None:
        """Process user message before sending to AI.
        
        Args:
            user_input: The user's input
            save_snapshot_func: Function to save state snapshot
        """
        # Save state snapshot before processing (unless it's a redo)
        if not (user_input.startswith('/') and user_input.strip().split()[0].lower() == '/redo'):
            save_snapshot_func()
            
            # Add thinking tag for Qwen models
            user_input = add_thinking_tag_to_message(
                user_input, 
                self.session.config.model, 
                self.session.enable_thinking
            )
                
            self.message_manager.add_user_message(user_input)
    
    async def _get_ai_response(self) -> Tuple[str, List[ToolCall]]:
        """Get response from AI with optional streaming.
        
        Returns:
            Tuple of (response_content, tool_calls)
        """
        openai_tools_fmt = self.tool_executor.get_tools_for_openai()
        messages = self.message_manager.get_messages()
        
        if self.session.use_streaming:
            stream = self.llm_client.create_completion(messages, openai_tools_fmt, stream=True)
            response_content, tool_calls = await self.llm_client.handle_streaming_response(stream, self.ui)
        else:
            completion = self.llm_client.create_completion(messages, openai_tools_fmt, stream=False)
            msg = completion.choices[0].message
            response_content = msg.content or ""
            
            # Display non-streaming response
            self.ui.print(f"\n[{self.ui.colors['accent']}][<] [/{self.ui.colors['accent']}] {self.ui._sanitize_content(response_content)}")
            
            # Parse tool calls
            tool_calls = parse_openai_tool_calls(msg.tool_calls)
            
            # Show tool calls in non-streaming mode
            for idx, tool_call in enumerate(tool_calls):
                self.ui.print(f"\n[{self.ui.colors['warning']}]🔧 Calling tool {idx + 1}: {tool_call.tool_name}[/{self.ui.colors['warning']}]")
                
        return response_content, tool_calls
    
    def _store_assistant_message(self, response_content: str, tool_calls: List[ToolCall]) -> None:
        """Store assistant message in history.
        
        Args:
            response_content: The assistant's response
            tool_calls: Any tool calls made by the assistant
        """
        if response_content:
            # Clean thinking tags before storing
            cleaned_content = self.llm_client.clean_thinking_tags(response_content)
            self.message_manager.add_assistant_message(cleaned_content, tool_calls)
        elif tool_calls:
            self.message_manager.add_assistant_message(None, tool_calls)
    
    async def _handle_tool_execution(self, tool_calls: List[ToolCall], 
                                   prompt_generator_func, execute_tools_func) -> None:
        """Handle tool execution and follow-up response.
        
        Args:
            tool_calls: Tool calls to execute
            prompt_generator_func: Function to generate system prompts
            execute_tools_func: Function to execute tool calls
        """
        if self.session.debug_mode:
            self.ui.print_debug_info(f"Executing {len(tool_calls)} tool call(s)")
        
        # Execute tools
        await execute_tools_func(tool_calls)
        
        # Update system prompt after tool execution
        final_system_prompt = prompt_generator_func()
        self.message_manager.update_system_message(final_system_prompt)
        
        # Ensure Qwen thinking tag
        ensure_message_has_thinking_tag(
            self.message_manager.messages,
            self.session.config.model, 
            self.session.enable_thinking
        )
        
        if self.session.debug_mode:
            self.ui.print(f"\n[{self.ui.colors['accent']}]🔄 Getting final response after tools...[/{self.ui.colors['accent']}]")
        
        # Get final response after tool execution
        final_response = await self._get_final_response()
        
        if final_response:
            # Clean and store final response
            cleaned_final = self.llm_client.clean_thinking_tags(final_response)
            self.message_manager.add_assistant_message(cleaned_final)
    
    async def _get_final_response(self) -> str:
        """Get final response after tool execution.
        
        Returns:
            The final response content
        """
        final_messages = self.message_manager.get_messages()
        
        if self.session.use_streaming:
            final_stream = self.llm_client.create_completion(final_messages, stream=True)
            final_response_content, _ = await self.llm_client.handle_streaming_response(
                final_stream, self.ui
            )
        else:
            final_completion = self.llm_client.create_completion(final_messages, stream=False)
            final_response_content = final_completion.choices[0].message.content or ""
            self.ui.print(f"\n[{self.ui.colors['accent']}][<] [/{self.ui.colors['accent']}] {self.ui._sanitize_content(final_response_content)}")
            
        return final_response_content
    
    def show_banner(self) -> None:
        """Show the startup banner with session info."""
        self.ui.print_banner()
        prompt_style_display = f" [{self.session.prompt_style.upper()}]" if self.session.prompt_style != "standard" else ""
        
        # Get debug info for display
        debug_info = self.session.get_debug_info()
        
        self.ui.print_info_with_heading("Model:", debug_info['model'])
        self.ui.print_info_with_heading("Repository:", self.session.analysis_result.repo_name)
        self.ui.print_info_with_heading("Files:", str(self.session.analysis_result.total_files))
        self.ui.print_info_with_heading("Total tokens:", f"{self.session.analysis_result.total_tokens:,}")
        self.ui.print_info_with_heading("Token budget:", f"{debug_info['token_budget']:,}")
        self.ui.print_info_with_heading("Tools:", ', '.join(self.tool_executor._registered_tools.keys()))
        self.ui.print_info_with_heading("Prompt style:", f"{debug_info['prompt_style']}{prompt_style_display}")
        
        if debug_info['debug_mode']:
            self.ui.print_debug_info("Debug mode enabled - system internals will be shown")