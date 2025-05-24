"""Main repository analyzer orchestrator."""
import os
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Optional

from ..core.models import Config, AnalysisResult
from ..adapters import create_adapter


class RepositoryAnalyzer:
    """Main analyzer for repository analysis and report generation."""
    
    def __init__(self, config: Config):
        """Initialize analyzer with configuration."""
        self.config = config
    
    def analyze(self, repo_url_or_path: str) -> AnalysisResult:
        """
        Analyze a repository and generate results.
        
        Args:
            repo_url_or_path: GitHub URL or local directory path
            
        Returns:
            AnalysisResult with all analysis data
        """
        # Create appropriate adapter
        adapter = create_adapter(repo_url_or_path, self.config)
        
        # Get repository name
        repo_name = adapter.get_name()
        
        # Select branch if GitHub
        branch = None
        if hasattr(adapter, 'select_branch'):
            branch = adapter.select_branch()
        
        # Get README
        print("\n[>] Fetching README...")
        readme_content = adapter.get_readme_content()
        
        # Interactive traversal
        print("\n[>] Starting interactive file selection...")
        structure, selected_paths, token_data = adapter.traverse_interactive()
        
        # Get file contents
        if selected_paths:
            file_contents, file_token_data = adapter.get_file_contents(list(selected_paths))
            token_data.update(file_token_data)
        else:
            file_contents = "No files selected.\n"
        
        # Calculate totals
        total_tokens = sum(token_data.values())
        total_files = len(selected_paths)
        
        return AnalysisResult(
            repo_name=repo_name,
            branch=branch,
            readme_content=readme_content,
            structure=structure,
            file_contents=file_contents,
            token_data=token_data,
            total_tokens=total_tokens,
            total_files=total_files,
            errors=adapter.errors
        )
    
    def generate_instructions(self, result: AnalysisResult) -> str:
        """Generate analysis instructions for the output."""
        branch_info = f" (branch: {result.branch})" if result.branch else ""
        
        if self.config.output_format == 'xml':
            instructions = f"""<repository>
<repo_name>{self._escape_xml(result.repo_name)}</repo_name>
<branch>{self._escape_xml(result.branch or 'main')}</branch>
<instructions>
Please analyze this repository to understand its structure, purpose, and functionality. Follow these steps:

1. README Review: Start by reading the README to understand the project's purpose, setup, and usage.
2. Structure Analysis: Examine the repository structure to understand the organization and architecture.
3. Entry Points: Identify the main entry point(s) of the application and trace the execution flow.
4. Dependencies: Note the key dependencies and libraries used in the project.
5. Core Components: Identify and understand the main components, modules, or classes.
6. Configuration: Look for configuration files and understand the project's settings.
7. Data Flow: Trace how data flows through the application.
8. Testing: Review the test structure and coverage if available.

Key Questions to Answer:
- What is the primary purpose of this repository?
- What are the main technologies and frameworks used?
- How is the code organized and structured?
- What are the entry points and main execution flows?
- Are there any notable design patterns or architectural decisions?
- What external dependencies does it have?
- How is testing implemented?
</instructions>
<structure>
"""
        else:  # markdown
            instructions = f"""# Repository Analysis: {result.repo_name}{branch_info}

## Analysis Instructions

Please analyze this repository to understand its structure, purpose, and functionality. Follow these steps:

1. **README Review**: Start by reading the README to understand the project's purpose, setup, and usage.

2. **Structure Analysis**: Examine the repository structure to understand the organization and architecture.

3. **Entry Points**: Identify the main entry point(s) of the application and trace the execution flow.

4. **Dependencies**: Note the key dependencies and libraries used in the project.

5. **Core Components**: Identify and understand the main components, modules, or classes.

6. **Configuration**: Look for configuration files and understand the project's settings.

7. **Data Flow**: Trace how data flows through the application.

8. **Testing**: Review the test structure and coverage if available.

## Key Questions to Answer

- What is the primary purpose of this repository?
- What are the main technologies and frameworks used?
- How is the code organized and structured?
- What are the entry points and main execution flows?
- Are there any notable design patterns or architectural decisions?
- What external dependencies does it have?
- How is testing implemented?

## Repository Structure

"""
        return instructions
    
    def _escape_xml(self, text: str) -> str:
        """Escape special XML characters."""
        if not text:
            return ""
        return (text
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;")
                .replace("'", "&apos;"))
    
    def generate_token_report(self, token_data: Dict[str, int]) -> str:
        """
        Generate a focused token report for iterative file selection.
        
        Two main views:
        1. Tree representation - visual structure understanding
        2. Full table - precise token counts for budgeting
        """
        if not token_data:
            return "No token data available.\n"
        
        report = "TOKEN ANALYSIS REPORT\n"
        report += "=" * 80 + "\n\n"
        
        # Calculate totals
        total_tokens = sum(token_data.values())
        total_files = len(token_data)
        
        # Build directory tree with aggregated token counts
        tree = defaultdict(lambda: {"tokens": 0, "files": 0, "subdirs": {}})
        dir_totals = {}
        
        for file_path, tokens in token_data.items():
            parts = file_path.split(os.sep)
            
            # Track all directory paths and their totals
            for i in range(len(parts)):
                dir_path = os.sep.join(parts[:i+1])
                if i < len(parts) - 1:  # It's a directory
                    if dir_path not in dir_totals:
                        dir_totals[dir_path] = {"tokens": 0, "files": 0}
                    dir_totals[dir_path]["tokens"] += tokens
                    dir_totals[dir_path]["files"] += 1
            
            # Build tree structure
            current = tree
            for i, part in enumerate(parts[:-1]):
                if part not in current:
                    current[part] = {"tokens": 0, "files": 0, "subdirs": {}}
                current[part]["tokens"] += tokens
                current = current[part]["subdirs"]
            
            # Add the file
            file_name = parts[-1]
            current[file_name] = {"tokens": tokens, "files": 1, "subdirs": {}}
            
            # Update parent directory counts
            current = tree
            for part in parts[:-1]:
                current[part]["files"] += 1
                current = current[part]["subdirs"]
        
        # 1. TREE VIEW
        report += "Directory Tree:\n"
        report += "-" * 80 + "\n"
        
        def print_tree(node: dict, name: str = "root", prefix: str = "", is_last: bool = True) -> str:
            output = ""
            if name != "root":
                output += prefix + ("└── " if is_last else "├── ")
                if node.get("subdirs"):  # Directory
                    output += f"{name}/ (Tokens: {node['tokens']:,}, Files: {node['files']})\n"
                else:  # File
                    output += f"{name} (Tokens: {node['tokens']:,})\n"
            
            if node.get("subdirs"):
                items = sorted(node["subdirs"].items())
                for i, (child_name, child_node) in enumerate(items):
                    is_last_child = (i == len(items) - 1)
                    child_prefix = prefix + ("    " if is_last else "│   ") if name != "root" else ""
                    output += print_tree(child_node, child_name, child_prefix, is_last_child)
            
            return output
        
        # Print tree
        for name, node in sorted(tree.items()):
            is_last = (name == sorted(tree.keys())[-1]) if tree else True
            report += print_tree(node, name, "", is_last)
        
        report += f"\nTOTAL: {total_tokens:,} tokens, {total_files} files\n"
        
        # 2. STATISTICS
        report += "\n" + "=" * 80 + "\n"
        report += "Token Statistics:\n"
        report += "-" * 80 + "\n"
        
        # Calculate stats
        token_counts = list(token_data.values())
        if token_counts:
            min_tokens = min(token_counts)
            max_tokens = max(token_counts)
            avg_tokens = total_tokens // total_files
            
            # Median
            sorted_counts = sorted(token_counts)
            median_tokens = sorted_counts[len(sorted_counts) // 2]
            
            report += f"Total tokens:  {total_tokens:,}\n"
            report += f"Total files:   {total_files}\n"
            report += f"Average:       {avg_tokens:,} tokens/file\n"
            report += f"Median:        {median_tokens:,} tokens/file\n"
            report += f"Range:         {min_tokens:,} - {max_tokens:,} tokens\n"
        
        # Top 10 largest directories
        report += "\nTop 10 largest directories:\n"
        sorted_dirs = sorted(dir_totals.items(), key=lambda x: x[1]['tokens'], reverse=True)[:10]
        for dir_path, info in sorted_dirs:
            report += f"  {info['tokens']:>8,} tokens: {dir_path}/ ({info['files']} files)\n"
        
        # Top 10 largest files
        report += "\nTop 10 largest files:\n"
        sorted_files = sorted(token_data.items(), key=lambda x: x[1], reverse=True)[:10]
        for file_path, tokens in sorted_files:
            report += f"  {tokens:>8,} tokens: {file_path}\n"
        
        report += "\n" + "=" * 80 + "\n"
        
        return report
    
    def save_results(self, result: AnalysisResult, output_dir: str = "output") -> Dict[str, str]:
        """
        Save analysis results to files.
        
        Args:
            result: AnalysisResult object
            output_dir: Directory to save output files
            
        Returns:
            Dictionary of output file paths
        """
        # Create output directory structure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        repo_output_dir = os.path.join(output_dir, f"{result.repo_name}_{timestamp}")
        os.makedirs(repo_output_dir, exist_ok=True)
        
        output_files = {}
        
        # Generate main analysis file
        main_content = self.generate_instructions(result)
        
        if self.config.output_format == 'xml':
            main_content += self._escape_xml(result.structure) + "\n"
            main_content += f"</structure>\n"
            main_content += f"<total_files>{result.total_files}</total_files>\n"
            if self.config.enable_token_counting:
                main_content += f"<total_tokens>{result.total_tokens}</total_tokens>\n"
            main_content += f"<readme>\n{self._escape_xml(result.readme_content)}\n</readme>\n"
            main_content += "<files>\n"
            main_content += result.file_contents
            main_content += "</files>\n"
            
            # Add error summary if any
            if result.has_errors():
                main_content += "<errors>\n"
                for error in result.errors:
                    main_content += f"<error>{self._escape_xml(error)}</error>\n"
                main_content += "</errors>\n"
            
            main_content += "</repository>\n"
        else:  # markdown
            main_content += result.structure + "\n"
            main_content += f"\n## Total Files Selected: {result.total_files}\n"
            if self.config.enable_token_counting:
                main_content += f"## Total Tokens: {result.total_tokens:,}\n"
            main_content += "\n## README\n\n"
            main_content += result.readme_content + "\n"
            main_content += "\n## File Contents\n\n"
            main_content += result.file_contents
            
            # Add error summary if any
            if result.has_errors():
                main_content += "\n## Errors Encountered\n\n"
                main_content += result.get_error_summary()
        
        # Save main analysis with appropriate extension
        file_extension = '.md' if self.config.output_format == 'markdown' else '.txt'
        main_path = os.path.join(repo_output_dir, f"{result.repo_name}_analysis{file_extension}")
        with open(main_path, 'w', encoding='utf-8') as f:
            f.write(main_content)
        output_files['main'] = main_path
        
        # Save token report if enabled
        if self.config.enable_token_counting and result.token_data:
            token_report = self.generate_token_report(result.token_data)
            token_path = os.path.join(repo_output_dir, f"{result.repo_name}_token_report.txt")
            with open(token_path, 'w', encoding='utf-8') as f:
                f.write(token_report)
            output_files['tokens'] = token_path
        
        # Save token data as JSON if requested
        if hasattr(self.config, 'export_json') and self.config.export_json and result.token_data:
            import json
            json_data = {
                'repo_name': result.repo_name,
                'branch': result.branch,
                'total_tokens': result.total_tokens,
                'total_files': result.total_files,
                'files': result.token_data,
                'timestamp': timestamp
            }
            json_path = os.path.join(repo_output_dir, f"{result.repo_name}_tokens.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2)
            output_files['json'] = json_path
        
        return output_files