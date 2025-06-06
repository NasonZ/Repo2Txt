"""
Command Handler for File Selector Agent

Handles all slash commands for the chat interface, keeping command logic
separate from the main agent orchestration.
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .file_selector_agent import FileSelectorAgent


class CommandHandler:
    """Handles all slash commands for the file selector agent."""
    
    def __init__(self, agent: 'FileSelectorAgent'):
        """Initialize with reference to parent agent."""
        self.agent = agent
        self.ui = agent.ui
        self.state_manager = agent.state_manager
        self.message_manager = agent.message_manager
        self.prompt_generator = agent.prompt_generator
        
    def handle_command(self, command: str) -> bool:
        """Handle slash commands. Returns True if command was processed, False to exit, or 'REDO' for regeneration."""
        parts = command.split()
        cmd = parts[0].lower()
        
        handlers = {
            '/clear': self._handle_clear,
            '/toggle': lambda: self._handle_toggle(parts),
            '/debug': lambda: self._handle_debug(parts),
            '/undo': self._handle_undo,
            '/redo': self._handle_redo,
            '/generate': lambda: self._handle_generate(parts[1:] if len(parts) > 1 else []),
            '/save': lambda: self._handle_save(parts[1:] if len(parts) > 1 else []),
            '/quit': lambda: False,
            '/exit': lambda: False,
            '/help': self._handle_help,
        }
        
        handler = handlers.get(cmd)
        if handler:
            return handler()
        else:
            self.ui.print_warning(f"Unknown command: {cmd}")
            self._handle_help()
            return True
    
    def _handle_clear(self) -> bool:
        """Clear conversation and selection."""
        self.agent.session.clear_conversation()
        new_system_prompt = self.agent._create_system_prompt()
        self.message_manager.add_system_message(new_system_prompt)
        self.ui.print_success("Conversation history and selection cleared.")
        return True
    
    def _handle_toggle(self, parts: List[str]) -> bool:
        """Handle toggle commands."""
        if len(parts) < 2:
            self.ui.print_warning("Usage: /toggle [streaming|thinking|reasoning]")
            return True
            
        toggle_type = parts[1].lower()
        
        if toggle_type == 'streaming':
            self.agent.use_streaming = not self.agent.use_streaming
            status = "enabled" if self.agent.use_streaming else "disabled"
            self.ui.print_success(f"Streaming {status}")
            return True
            
        elif toggle_type == 'thinking':
            if "qwen" not in self.agent.model.lower():
                self.ui.print_warning("Thinking mode is only available for Qwen models")
                return True
            self.agent.enable_thinking = not self.agent.enable_thinking
            status = "enabled" if self.agent.enable_thinking else "disabled"
            self.ui.print_success(f"Thinking mode {status}")
            return True
            
        elif toggle_type == 'prompt':
            # Cycle through available styles
            if self.agent.prompt_style == "standard":
                new_style = "meta-reasoning"
            elif self.agent.prompt_style == "meta-reasoning":
                new_style = "xml"
            else:  # xml
                new_style = "standard"
            
            self.agent.prompt_style = new_style
            # Update system prompt immediately
            updated_system_prompt = self.agent._create_system_prompt()
            self.message_manager.update_system_message(updated_system_prompt)
            self.ui.print_success(f"Prompt style changed to: {new_style}")
            return True
            
        elif toggle_type == 'budget':
            if len(parts) < 3:
                self.ui.print_warning("Usage: /toggle budget <number>")
                return True
            try:
                new_budget = int(parts[2])
                if new_budget <= 0:
                    self.ui.print_warning("Budget must be a positive number")
                    return True
                old_budget = self.agent.session.set_token_budget(new_budget)
                # Update system prompt with new budget
                updated_system_prompt = self.agent._create_system_prompt()
                self.message_manager.update_system_message(updated_system_prompt)
                self.ui.print_success(f"Token budget changed from {old_budget:,} to {new_budget:,}")
                return True
            except ValueError:
                self.ui.print_warning("Budget must be a valid number")
                return True
                
        else:
            self.ui.print_warning("Available toggles: streaming, thinking, prompt, budget")
            return True
    
    def _handle_debug(self, parts: List[str]) -> bool:
        """Handle debug commands."""
        if len(parts) < 2:
            self.agent.debug_mode = not self.agent.debug_mode
            status = "enabled" if self.agent.debug_mode else "disabled"
            self.ui.print_success(f"Debug mode {status}")
            return True
        else:
            debug_type = parts[1].lower()
            if debug_type == 'state':
                self._show_debug_state()
                return True
            else:
                self.ui.print_warning("Available debug options: /debug or /debug state")
                return True
    
    def _handle_undo(self) -> bool:
        """Undo last action."""
        if self.agent.session.restore_snapshot():
            # Update system prompt to reflect restored state
            updated_system_prompt = self.agent._create_system_prompt()
            self.message_manager.update_system_message(updated_system_prompt)
            self.ui.print_success("Undone last action - restored previous state")
        else:
            self.ui.print_warning("No actions to undo")
        return True
    
    def _handle_redo(self) -> str:
        """Regenerate last AI response."""
        messages = self.message_manager.get_messages()
        if len(messages) <= 1:  # Only system message
            self.ui.print_warning("No message to regenerate")
            return True
        
        # Find the last user message
        last_user_idx = -1
        for i in range(len(messages) - 1, 0, -1):
            if messages[i]["role"] == "user":
                last_user_idx = i
                break
        
        if last_user_idx == -1:
            self.ui.print_warning("No user message found to regenerate response for")
            return True
        
        # Remove all messages after the last user message
        self.message_manager.messages = messages[:last_user_idx + 1]
        
        # Update system prompt
        updated_system_prompt = self.agent._create_system_prompt()
        self.message_manager.update_system_message(updated_system_prompt)
        
        self.ui.print_info("Regenerating AI response...")
        # Return special marker to trigger regeneration in chat loop
        return "REDO"
    
    def _handle_generate(self, args: List[str]) -> bool:
        """Handle the /generate command to create output files."""
        # Check if any files are selected
        if not self.state_manager.state.selected_files:
            self.ui.print_warning("No files selected. Select some files first before generating output.")
            return True
        
        # Parse arguments
        format_type = "markdown"  # default
        include_tokens = True     # default
        generate_json = False
        generate_all = False
        output_dir = "output"     # default
        
        # Parse command arguments
        i = 0
        while i < len(args):
            arg = args[i].lower()
            if arg in ["xml", "markdown"]:
                format_type = arg
            elif arg == "json":
                format_type = None  # JSON only
                generate_json = True
                include_tokens = False
            elif arg == "all":
                generate_all = True
            elif arg == "no-tokens":
                include_tokens = False
            elif arg.startswith("output-dir="):
                output_dir = arg.split("=", 1)[1]
            i += 1
        
        try:
            # Import required modules
            from ..core.analyzer import RepositoryAnalyzer
            from ..core.models import AnalysisResult, FileNode
            from ..adapters import create_adapter
            
            # Check output directory permissions
            try:
                test_dir = os.path.join(output_dir, f".test_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                os.makedirs(test_dir, exist_ok=True)
                os.rmdir(test_dir)
            except PermissionError:
                self.ui.print_error(f"No write permission for output directory: {output_dir}")
                return True
            except Exception as e:
                self.ui.print_error(f"Cannot create output directory: {str(e)}")
                return True
            
            # Get the adapter to fetch file contents (skip size validation for output generation)
            adapter = create_adapter(str(self.agent.repo_path), self.agent.config, validate_size=False)
            
            # Collect file contents for selected files
            file_contents_list = []
            file_token_data = {}
            failed_files = []
            
            self.ui.print_info("Generating output files...")
            self.ui.print(f"Selected files: {len(self.state_manager.state.selected_files)}")
            
            for file_path in self.state_manager.state.selected_files:
                content, error = adapter.get_file_content(file_path)
                if error:
                    failed_files.append((file_path, error))
                    file_contents_list.append(f"\n{'='*60}\nFile: {file_path}\n{'='*60}\n\nError: {error}\n")
                elif content:
                    # Format content based on output format
                    if format_type == "xml":
                        file_contents_list.append(f'<file path="{file_path}">\n{content}\n</file>\n')
                    else:  # markdown
                        file_contents_list.append(f"\n{'='*60}\nFile: {file_path}\n{'='*60}\n\n{content}\n")
                    
                    # Use cached token count from state manager
                    if include_tokens and file_path in self.state_manager.file_tokens:
                        file_token_data[file_path] = self.state_manager.file_tokens[file_path]
            
            # Warn about failed files
            if failed_files:
                self.ui.print_warning(f"\nFailed to read {len(failed_files)} file(s):")
                for path, error in failed_files[:5]:  # Show first 5
                    self.ui.print(f"  [dim]>[/dim] {path}: {error}")
                if len(failed_files) > 5:
                    self.ui.print(f"  [dim]... and {len(failed_files) - 5} more[/dim]")
            
            file_contents = "".join(file_contents_list)
            
            # Create AnalysisResult
            # First, create a simple file tree from selected files
            root_node = FileNode(
                path=self.agent.analysis_result.repo_name,
                name=self.agent.analysis_result.repo_name,
                type="dir"
            )
            
            # Add selected files to the tree
            for file_path in self.state_manager.state.selected_files:
                file_node = FileNode(
                    path=file_path,
                    name=os.path.basename(file_path),
                    type="file",
                    token_count=file_token_data.get(file_path, 0) if include_tokens else 0
                )
                root_node.children.append(file_node)
            
            result = AnalysisResult(
                repo_path=str(self.agent.repo_path),
                repo_name=self.agent.analysis_result.repo_name,
                file_tree=root_node,
                file_paths=self.state_manager.state.selected_files,
                total_files=len(self.state_manager.state.selected_files),
                branch=getattr(self.agent.analysis_result, 'branch', None),
                readme_content=self.agent.analysis_result.readme_content,
                file_contents=file_contents,
                token_data=file_token_data if include_tokens else {},
                total_tokens=self.state_manager.state.total_tokens_selected if include_tokens else 0,
                errors=[]
            )
            
            # Generate the tree structure for selected files only - 
            # The AnalysisResult.structure property will automatically generate from file_tree
            
            # Save results based on requested format(s)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            repo_output_dir = os.path.join(output_dir, f"{result.repo_name}_{timestamp}")
            os.makedirs(repo_output_dir, exist_ok=True)
            
            output_files = []
            
            # Create a temporary analyzer for saving
            temp_config = self.agent.config.__class__()
            temp_config.enable_token_counting = include_tokens
            temp_config.github_token = self.agent.config.github_token
            
            analyzer = RepositoryAnalyzer(temp_config, self.ui.theme)
            
            if generate_all:
                # Generate all formats
                formats_to_generate = [("markdown", True), ("xml", True), ("json", True)]
            elif generate_json and not format_type:
                # JSON only
                formats_to_generate = [("json", True)]
            else:
                # Specific format
                formats_to_generate = [(format_type, include_tokens)]
            
            for fmt, tokens in formats_to_generate:
                if fmt == "json":
                    # Generate JSON export
                    if file_token_data or not include_tokens:
                        json_data = {
                            'repo_name': result.repo_name,
                            'branch': result.branch,
                            'total_tokens': result.total_tokens,
                            'total_files': result.total_files,
                            'files': file_token_data,
                            'timestamp': timestamp,
                            'selected_files': self.state_manager.state.selected_files
                        }
                        json_path = os.path.join(repo_output_dir, f"{result.repo_name}_tokens.json")
                        try:
                            with open(json_path, 'w', encoding='utf-8') as f:
                                json.dump(json_data, f, indent=2)
                            output_files.append(('json', json_path))
                        except Exception as e:
                            self.ui.print_error(f"Failed to write JSON file: {str(e)}")
                else:
                    # Set format for this iteration
                    temp_config.output_format = fmt
                    temp_config.enable_token_counting = tokens
                    
                    # Generate the output
                    saved_files = analyzer.save_results(result, output_dir)
                    for file_type, file_path in saved_files.items():
                        output_files.append((file_type, file_path))
            
            # Display results
            self.ui.print_success("\nOutput files generated successfully!")
            self.ui.print(f"\n[info]OUTPUT LOCATION:[/info] [path]{repo_output_dir}[/path]")
            self.ui.print("\n[info]FILES CREATED:[/info]")
            for file_type, file_path in output_files:
                rel_path = os.path.relpath(file_path)
                self.ui.print(f"  [dim]>[/dim] {file_type.upper()}: [path]{rel_path}[/path]")
            
        except Exception as e:
            self.ui.print_error(f"Failed to generate output: {str(e)}")
            if self.agent.debug_mode:
                import traceback
                self.ui.print_debug_info(traceback.format_exc())
        
        return True
    
    def _handle_save(self, args: List[str]) -> bool:
        """Save chat history to JSON file."""
        try:
            # Default filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chat_history_{timestamp}.json"
            
            # Check if custom filename provided
            if args:
                filename = args[0]
                if not filename.endswith('.json'):
                    filename += '.json'
            
            # Create output directory if needed
            output_dir = "output/chat_history"
            os.makedirs(output_dir, exist_ok=True)
            filepath = os.path.join(output_dir, filename)
            
            # Prepare chat data
            chat_data = {
                'timestamp': timestamp,
                'model': self.agent.model,
                'prompt_style': self.agent.prompt_style,
                'token_budget': self.state_manager.state.token_budget,
                'total_tokens_selected': self.state_manager.state.total_tokens_selected,
                'selected_files': self.state_manager.state.selected_files,
                'conversation': []
            }
            
            # Get messages, excluding system prompt for readability
            messages = self.message_manager.get_messages()
            for msg in messages[1:]:  # Skip system prompt
                if msg['role'] == 'user':
                    chat_data['conversation'].append({
                        'role': 'user',
                        'content': msg.get('content', '')
                    })
                elif msg['role'] == 'assistant':
                    # Handle assistant messages with optional content
                    content = msg.get('content', '')
                    
                    # Check for tool calls
                    if 'tool_calls' in msg and msg['tool_calls']:
                        # Assistant message with tool calls
                        saved_msg = {
                            'role': 'assistant',
                            'content': content,
                            'tool_calls': [
                                {
                                    'id': tc.get('id', ''),
                                    'function': {
                                        'name': tc.get('function', {}).get('name', ''),
                                        'arguments': tc.get('function', {}).get('arguments', '')
                                    }
                                }
                                for tc in msg['tool_calls']
                            ]
                        }
                        chat_data['conversation'].append(saved_msg)
                    elif content:
                        # Assistant message with content - parse thinking tags if present
                        thinking_match = content.find('<thinking>')
                        if thinking_match >= 0:
                            thinking_end = content.find('</thinking>')
                            if thinking_end > thinking_match:
                                thinking = content[thinking_match+10:thinking_end]
                                visible = content[thinking_end+11:].strip()
                                chat_data['conversation'].append({
                                    'role': 'assistant',
                                    'thinking': thinking,
                                    'content': visible
                                })
                            else:
                                chat_data['conversation'].append({
                                    'role': 'assistant',
                                    'content': content
                                })
                        else:
                            chat_data['conversation'].append({
                                'role': 'assistant',
                                'content': content
                            })
                    # Skip empty assistant messages (no content and no tool calls)
                elif msg['role'] == 'tool':
                    # Include tool results
                    chat_data['conversation'].append({
                        'role': 'tool',
                        'tool_name': msg.get('name', 'unknown'),
                        'tool_call_id': msg.get('tool_call_id', ''),
                        'content': msg.get('content', '')
                    })
            
            # Save to file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(chat_data, f, indent=2, ensure_ascii=False)
            
            self.ui.print_success(f"Chat history saved to: {filepath}")
            
        except Exception as e:
            self.ui.print_error(f"Failed to save chat history: {str(e)}")
            if self.agent.debug_mode:
                import traceback
                self.ui.print_debug_info(traceback.format_exc())
        
        return True
    
    def _handle_help(self) -> bool:
        """Show available commands."""
        # TODO: /help <query> should call the LLM which is prompted with the usage instructions ebabling it to teach the user how to use the application.
        self.ui.print_section("COMMANDS", """
/clear                  - Clear conversation and selection
/undo                   - Undo last action (restore previous state)
/redo                   - Regenerate last AI response
/save                   - Save chat history to JSON file
/generate               - Generate output files (markdown + token report)
/generate xml           - Generate XML format output + token report
/generate markdown      - Generate markdown format output + token report
/generate json          - Generate JSON token data export only
/generate all           - Generate all formats (markdown, XML, token report, JSON)
/generate <fmt> no-tokens - Generate without token report
/toggle streaming       - Toggle streaming on/off
/toggle thinking        - Toggle Qwen thinking mode (Qwen models only)
/toggle prompt          - Cycle through prompt styles (standard/meta-reasoning/xml)
/toggle budget <n>      - Set token budget to <n> tokens
/debug                  - Toggle debug mode
/debug state            - Show current configuration
/help                   - Show this help
/quit or /exit         - Exit the application
        """.strip())
        return True
    
    def _show_debug_state(self):
        """Show current debug state information."""
        debug_info = self.agent.session.get_debug_info()
        self.ui.print_section("DEBUG STATE", f"""
Streaming: {debug_info['streaming']}
Thinking: {debug_info['thinking']} {'(Qwen only)' if not debug_info['thinking_available'] else ''}
Prompt style: {debug_info['prompt_style']}
Debug mode: {debug_info['debug_mode']}
Model: {debug_info['model']}
Base URL: {debug_info['base_url']}
Selected files: {debug_info['selected_files']}
Token budget: {debug_info['token_budget']:,}
Tokens used: {debug_info['tokens_used']:,}
        """.strip())