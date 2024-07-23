import os
import sys
import argparse
from github import Github
from tqdm import tqdm
import tiktoken

# Set your GitHub token here
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN', 'xxx')

def get_readme_content(repo, branch=None):
    """
    Retrieve the content of the README file.

    Args:
        repo: The repository object or local path.
        branch: The branch name (optional).

    Returns:
        The content of the README file or a message if not found.
    """
    if isinstance(repo, str):  # Local path
        readme_path = os.path.join(repo, "README.md")
        if os.path.exists(readme_path):
            with open(readme_path, 'r', encoding='utf-8') as f:
                return f.read()
        return "README not found."
    else:  # GitHub repo
        try:
            readme = repo.get_contents("README.md", ref=branch)
            return readme.decoded_content.decode('utf-8')
        except Exception:
            return "README not found."

def parse_range(range_str):
    """
    Parse a string of ranges into a list of integers.

    Args:
        range_str: A string representing ranges (e.g., "1-3,5,7-9").

    Returns:
        A list of integers.
    """
    ranges = []
    try:
        for part in range_str.split(','):
            if '-' in part:
                start, end = map(int, part.split('-'))
                ranges.extend(range(start, end + 1))
            else:
                ranges.append(int(part))
        return ranges
    except ValueError:
        return []

def count_tokens(text):
    """
    Count the number of tokens in the given text using the cl100k_base encoding.

    Args:
        text: The input text to tokenize.

    Returns:
        The number of tokens in the text.
    """
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    return len(tokens)

def traverse_local_repo_interactively(repo_path, current_path="", selected_paths=None, excluded_dirs=None, count_tokens_flag=False):
    """
    Traverse the local repository interactively, allowing user to select folders and files.

    Args:
        repo_path: The root path of the local repository.
        current_path: The current path relative to repo_path being traversed.
        selected_paths: A set to store selected paths (optional).
        excluded_dirs: A set of directories to exclude (optional).
        count_tokens_flag: Whether to count tokens for files (optional).

    Returns:
        The structure of the repository, the selected paths, and token data (if counting tokens).
    """
    if selected_paths is None:
        selected_paths = set()
    if excluded_dirs is None:
        excluded_dirs = {'__pycache__', '.git', '.hg', '.svn', '.idea', '.vscode', 'node_modules'}

    structure = ""
    token_data = {}
    full_path = os.path.join(repo_path, current_path)

    try:
        items = []
        for item in os.listdir(full_path):
            item_path = os.path.join(full_path, item)
            if os.path.isdir(item_path) and item not in excluded_dirs:
                items.append((item, 'dir'))
            elif os.path.isfile(item_path):
                items.append((item, 'file'))

        print(f"\nContents of {current_path or '.'}:")
        for i, (item, item_type) in enumerate(items, start=1):
            print(f"{i}. {item} ({item_type})")

        while True:
            selected_indices = input("Enter the indices of the folders/files you want to extract (e.g., 1-5,7,9-12) or 'a' for all: ")
            if selected_indices.lower() == 'a':
                selected_indices = list(range(1, len(items) + 1))
                break
            else:
                selected_indices = parse_range(selected_indices)
                if selected_indices:
                    break
                print("Invalid input. Please enter the indices in the correct format (e.g., 1-3,5).")

        for i, (item, item_type) in enumerate(items, start=1):
            if i in selected_indices:
                item_path = os.path.join(full_path, item)
                rel_item_path = os.path.relpath(item_path, repo_path)
                if item_type == 'dir':
                    structure += f"{rel_item_path}/\n"
                    while True:
                        sub_folders_choice = input(f"Do you want to select sub-folders in {rel_item_path}? (y/n/a): ").lower()
                        if sub_folders_choice == 'y':
                            sub_structure, sub_selected_paths, sub_token_data = traverse_local_repo_interactively(
                                repo_path, 
                                os.path.join(current_path, item), 
                                selected_paths, 
                                excluded_dirs,
                                count_tokens_flag
                            )
                            structure += sub_structure
                            selected_paths.update(sub_selected_paths)
                            token_data.update(sub_token_data)
                            break
                        elif sub_folders_choice == 'a':
                            for root, dirs, files in os.walk(item_path):
                                dirs[:] = [d for d in dirs if d not in excluded_dirs]
                                rel_root = os.path.relpath(root, repo_path)
                                for sub_dir in dirs:
                                    sub_dir_path = os.path.join(rel_root, sub_dir)
                                    structure += f"{sub_dir_path}/\n"
                                for sub_file in files:
                                    sub_file_path = os.path.join(rel_root, sub_file)
                                    structure += f"{sub_file_path}\n"
                                    selected_paths.add(sub_file_path)
                                    if count_tokens_flag:
                                        try:
                                            with open(os.path.join(root, sub_file), 'r', encoding='utf-8') as f:
                                                content = f.read()
                                            token_count = count_tokens(content)
                                            token_data[sub_file_path] = token_count
                                        except Exception as e:
                                            print(f"Error counting tokens in {sub_file_path}: {e}")
                            break
                        elif sub_folders_choice == 'n':
                            break
                        else:
                            print("Invalid choice. Please enter 'y', 'n', or 'a'.")
                else:
                    structure += f"{rel_item_path}\n"
                    selected_paths.add(rel_item_path)
                    if count_tokens_flag:
                        try:
                            with open(item_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                            token_count = count_tokens(content)
                            token_data[rel_item_path] = token_count
                        except Exception as e:
                            print(f"Error counting tokens in {rel_item_path}: {e}")
            else:
                item_path = os.path.join(full_path, item)
                rel_item_path = os.path.relpath(item_path, repo_path)
                if item_type == 'dir':
                    structure += f"{rel_item_path}/ (Omitted for brevity)\n"
                else:
                    structure += f"{rel_item_path}\n"

    except PermissionError:
        print(f"Permission denied to access {full_path}. Skipping.")
    except Exception as e:
        print(f"An error occurred while processing {full_path}: {str(e)}. Skipping.")

    return structure, selected_paths, token_data

def traverse_repo_interactively(repo, path="", selected_paths=None, excluded_dirs=None, count_tokens_flag=False):
    """
    Traverse the GitHub repository interactively, allowing user to select folders and files.

    Args:
        repo: The GitHub repository object.
        path: The current path in the repository.
        selected_paths: A set to store selected paths (optional).
        excluded_dirs: A set of directories to exclude (optional).
        count_tokens_flag: Whether to count tokens for files (optional).

    Returns:
        The structure of the repository, the selected paths, and token data (if counting tokens).
    """
    if selected_paths is None:
        selected_paths = set()
    if excluded_dirs is None:
        excluded_dirs = {'__pycache__', '.git', '.hg', '.svn', '.idea', '.vscode', 'node_modules'}

    structure = ""
    token_data = {}
    contents = repo.get_contents(path)

    print(f"\nContents of {path}:")
    for i, content in enumerate(contents, start=1):
        print(f"{i}. {content.name} ({'dir' if content.type == 'dir' else 'file'})")

    while True:
        selected_indices = input("Enter the indices of the folders/files you want to extract (e.g., 1-5,7,9-12) or 'a' for all: ")
        if selected_indices.lower() == 'a':
            selected_indices = list(range(1, len(contents) + 1))
            break
        else:
            selected_indices = parse_range(selected_indices)
            if selected_indices:
                break
            print("Invalid input. Please enter the indices in the correct format (e.g., 1-3,5).")

    for i, content in enumerate(contents, start=1):
        if i in selected_indices:
            if content.type == "dir":
                if content.name not in excluded_dirs:
                    structure += f"{path}/{content.name}/\n"
                    while True:
                        sub_folders_choice = input(f"Do you want to select sub-folders in {content.path}? (y/n/a): ").lower()
                        if sub_folders_choice == 'y':
                            sub_structure, sub_selected_paths, sub_token_data = traverse_repo_interactively(
                                repo, 
                                content.path, 
                                selected_paths, 
                                excluded_dirs,
                                count_tokens_flag
                            )
                            structure += sub_structure
                            selected_paths.update(sub_selected_paths)
                            token_data.update(sub_token_data)
                            break
                        elif sub_folders_choice == 'a':
                            sub_contents = repo.get_contents(content.path)
                            for sub_content in sub_contents:
                                if sub_content.type == "dir":
                                    if sub_content.name not in excluded_dirs:
                                        sub_dir_path = f"{content.path}/{sub_content.name}/"
                                        structure += f"{sub_dir_path}\n"
                                        sub_structure, sub_selected_paths, sub_token_data = traverse_repo_interactively(
                                            repo, 
                                            sub_content.path, 
                                            selected_paths, 
                                            excluded_dirs,
                                            count_tokens_flag
                                        )
                                        structure += sub_structure
                                        selected_paths.update(sub_selected_paths)
                                        token_data.update(sub_token_data)
                                else:
                                    sub_file_path = f"{content.path}/{sub_content.name}"
                                    structure += f"{sub_file_path}\n"
                                    selected_paths.add(sub_file_path)
                                    if count_tokens_flag:
                                        try:
                                            decoded_content = sub_content.decoded_content.decode('utf-8')
                                            token_count = count_tokens(decoded_content)
                                            token_data[sub_file_path] = token_count
                                        except Exception as e:
                                            print(f"Error counting tokens in {sub_file_path}: {e}")
                            break
                        elif sub_folders_choice == 'n':
                            break
                        else:
                            print("Invalid choice. Please enter 'y', 'n', or 'a'.")
            else:
                structure += f"{path}/{content.name}\n"
                selected_paths.add(f"{path}/{content.name}")
                if count_tokens_flag:
                    try:
                        decoded_content = content.decoded_content.decode('utf-8')
                        token_count = count_tokens(decoded_content)
                        token_data[f"{path}/{content.name}"] = token_count
                    except Exception as e:
                        print(f"Error counting tokens in {path}/{content.name}: {e}")
        else:
            if content.type == "dir":
                structure += f"{path}/{content.name}/ (Omitted for brevity)\n"
            else:
                structure += f"{path}/{content.name}\nContent: Omitted for brevity\n\n"

    return structure, selected_paths, token_data

def get_selected_file_contents(repo, selected_files, is_local=False, count_tokens_flag=False):
    """
    Get the contents of the selected files.

    Args:
        repo: The repository object or local path.
        selected_files: A list of selected files.
        is_local: A flag indicating if the repository is local.
        count_tokens_flag: Whether to count tokens for files.

    Returns:
        The contents of the selected files and token data (if counting tokens).
    """
    file_contents = ""
    token_data = {}
    binary_extensions = [
        '.exe', '.dll', '.so', '.a', '.lib', '.dylib', '.o', '.obj', '.zip', '.tar', '.tar.gz', '.tgz', 
        '.rar', '.7z', '.bz2', '.gz', '.xz', '.z', '.lz', '.lzma', '.lzo', '.rz', '.sz', '.dz', 
        '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', '.odt', '.ods', '.odp', 
        '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp', '.svg', '.mp3', '.mp4', 
        '.wav', '.flac', '.ogg', '.avi', '.mkv', '.mov', '.webm', '.wmv', '.m4a', '.aac', 
        '.eps', '.ai', '.iso', '.vmdk', '.qcow2', '.vdi', '.vhd', '.vhdx', '.ova', '.ovf', 
        '.db', '.sqlite', '.mdb', '.accdb', '.frm', '.ibd', '.dbf', '.sql', '.jar', '.class', 
        '.war', '.ear', '.jpi', '.pyc', '.pyo', '.pyd', '.egg', '.whl', '.o', '.ko', '.obj', 
        '.elf', '.lib', '.a', '.la', '.lo', '.dll', '.so', '.dylib', '.exe', '.out', '.app', 
        '.sl', '.framework', '.eot', '.otf', '.ttf', '.woff', '.woff2', '.ico', '.icns', '.cur', 
        '.cab', '.dmp', '.sys', '.msi', '.msix', '.msp', '.msm', '.msu', '.dmg', '.pkg', 
        '.deb', '.rpm', '.snap', '.flatpak', '.appimage', '.apk', '.aab', '.ipa', '.pem', 
        '.crt', '.ca-bundle', '.p7b', '.p7c', '.p12', '.pfx', '.cer', '.der', '.key', '.keystore', 
        '.jks', '.p8', '.sig', '.svn', '.git', '.gitignore', '.gitattributes', '.gitmodules', 
        '.iml', '.ipr', '.iws', '.project', '.cproject', '.buildpath', '.classpath', 
        '.metadata', '.settings', '.idea', '.vscode', 'bin/', 'obj/', 'build/', 'dist/', 
        'target/', '/node_modules/', 'vendor/', 'packages/', '.log', '.tlog', '.tmp', '.temp', 
        '.swp', '.bak', '.cache', '.ini', '.cfg', '.config', '.conf', '.properties', '.prefs', 
        '.htaccess', '.htpasswd', '.env', '.dockerignore', '.chm', '.epub', '.mobi', '.img', 
        '.iso', '.vcd', '.bak', '.gho', '.ori', '.orig', '.dat', '.data', '.dump', '.bin', 
        '.raw', '.crx', '.xpi', '.lockb', 'package-lock.json', '.rlib', '.pdb', '.idb', 
        '.ilk', '.exp', '.map', '.sdf', '.suo', '.VC.db', '.aps', '.res', '.rc', '.nupkg', 
        '.snupkg', '.vsix', '.bpl', '.dcu', '.dcp', '.dcpil', '.drc', '.DS_Store', 
        '.localized', '.manifest', '.lance', '.txt', '.one', '.notebook', '.nmbak', '.enex', 
        '.nd', '.gdoc', '.gsheet', '.gslides', '.ipynb', '.qvnotebook', '.bear', '.csv', 
        '.tsv', '.json', '.xml', '.yaml', '.yml'
    ]

    for file_path in tqdm(selected_files, desc="Reviewing files"):
        try:
            if is_local:
                full_path = os.path.join(repo, file_path)
                if any(file_path.endswith(ext) for ext in binary_extensions):
                    file_contents += f"File: {file_path}\nContent: Skipped binary file\n\n"
                else:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    file_contents += f"File: {file_path}\nContent:\n{content}\n\n"
                    if count_tokens_flag:
                        token_count = count_tokens(content)
                        token_data[file_path] = token_count
            else:
                content = repo.get_contents(file_path)
                if content.type == "dir":
                    file_contents += f"File: {file_path}\nContent: Skipped (directory)\n\n"
                    continue
                if any(content.name.endswith(ext) for ext in binary_extensions):
                    file_contents += f"File: {file_path}\nContent: Skipped binary file\n\n"
                else:
                    file_contents += f"File: {file_path}\n"
                    try:
                        if content.encoding is None or content.encoding == 'none':
                            file_contents += "Content: Skipped due to missing encoding\n\n"
                        else:
                            try:
                                decoded_content = content.decoded_content.decode('utf-8')
                                file_contents += f"Content:\n{decoded_content}\n\n"
                                if count_tokens_flag:
                                    token_count = count_tokens(decoded_content)
                                    token_data[file_path] = token_count
                            except UnicodeDecodeError:
                                try:
                                    decoded_content = content.decoded_content.decode('latin-1')
                                    file_contents += f"Content (Latin-1 Decoded):\n{decoded_content}\n\n"
                                    if count_tokens_flag:
                                        token_count = count_tokens(decoded_content)
                                        token_data[file_path] = token_count
                                except UnicodeDecodeError:
                                    file_contents += "Content: Skipped due to unsupported encoding\n\n"
                    except (AttributeError, UnicodeDecodeError):
                        file_contents += "Content: Skipped due to decoding error or missing decoded_content\n\n"
        except Exception as e:
            print(f"Error reviewing file {file_path}: {e}")
            file_contents += f"File: {file_path}\nContent: Skipped due to error: {e}\n\n"
    return file_contents, token_data

def generate_tree_representation(token_data):
    """
    Generate a tree representation of the repository structure with token counts.

    Args:
        token_data: A dictionary mapping file paths to token counts.

    Returns:
        A string representing the tree structure with token counts.
    """
    tree = {}
    total_tokens = 0
    total_files = 0
    for path, count in token_data.items():
        total_tokens += count
        total_files += 1
        parts = path.split(os.path.sep)
        current = tree
        for part in parts[:-1]:
            if part not in current:
                current[part] = {"__token_count": 0, "__files": 0}
            current = current[part]
        current[parts[-1]] = {"__token_count": count, "__files": 1}
        
        current = tree
        for part in parts[:-1]:
            current[part]["__token_count"] += count
            current[part]["__files"] += 1
            current = current[part]
    
    def print_tree(node, indent=""):
        output = ""
        items = sorted(node.items())
        for i, (key, value) in enumerate(items):
            if key == "__token_count" or key == "__files":
                continue
            is_last = (i == len(items) - 1)
            if isinstance(value, dict) and "__token_count" in value:
                token_count = value["__token_count"]
                file_count = value["__files"]
                branch = "└── " if is_last else "├── "
                if file_count > 1:  # It's a directory
                    output += f"{indent}{branch}{key}/ (Tokens: {token_count}, Files: {file_count})\n"
                else:  # It's a file
                    output += f"{indent}{branch}{key} (Tokens: {token_count})\n"
                if file_count > 1:  # Only recurse if it's a directory
                    sub_indent = indent + ("    " if is_last else "│   ")
                    output += print_tree(value, sub_indent)
            else:
                branch = "└── " if is_last else "├── "
                output += f"{indent}{branch}{key} (Tokens: {value['__token_count']})\n"
        return output

    tree_output = print_tree(tree)
    tree_output += f"\nTOTAL: {total_tokens} tokens, {total_files} files"
    return tree_output

def generate_summary_table(token_data):
    """
    Generate a summary table of token counts for files and directories.

    Args:
        token_data: A dictionary mapping file paths to token counts.

    Returns:
        A string representing the summary table.
    """
    summary = "Token Count Summary:\n"
    summary += "-------------------\n"
    summary += f"{'File/Directory':<70} | {'Token Count':>12} | {'File Count':>10}\n"
    summary += "-" * 98 + "\n"

    tree = {}
    total_tokens = 0
    total_files = 0
    for path, count in token_data.items():
        total_tokens += count
        total_files += 1
        parts = path.split(os.path.sep)
        current = tree
        for part in parts[:-1]:
            if part not in current:
                current[part] = {"__token_count": 0, "__files": 0}
            current = current[part]
        current[parts[-1]] = {"__token_count": count, "__files": 1}
        
        current = tree
        for part in parts[:-1]:
            current[part]["__token_count"] += count
            current[part]["__files"] += 1
            current = current[part]

    def process_tree(node, path="", level=0):
        nonlocal summary
        items = sorted(node.items())
        for i, (key, value) in enumerate(items):
            if key == "__token_count" or key == "__files":
                continue
            full_path = os.path.join(path, key)
            indent = "  " * level
            display_path = indent + full_path
            if isinstance(value, dict):
                token_count = value["__token_count"]
                file_count = value["__files"]
                if level == 0:
                    summary += "-" * 98 + "\n"
                summary += f"{display_path:<70} | {token_count:>12} | {file_count:>10}\n"
                process_tree(value, full_path, level + 1)
            else:
                summary += f"{display_path:<70} | {value['__token_count']:>12} | {1:>10}\n"

    process_tree(tree)
    
    # Add the overall total at the end
    summary += "-" * 98 + "\n"
    summary += f"{'TOTAL':<70} | {total_tokens:>12} | {total_files:>10}\n"
    
    return summary

def get_repo_contents(repo_path_or_url, branch=None, count_tokens_flag=False):
    """
    Main function to get repository contents.

    Args:
        repo_path_or_url: The GitHub repository URL or local path.
        branch: The branch name (optional).
        count_tokens_flag: Whether to count tokens for files.

    Returns:
        The repository name, instructions, README content, repository structure, and file contents.
    """
    is_local = os.path.isdir(repo_path_or_url)
    
    if is_local:
        repo_name = os.path.basename(os.path.abspath(repo_path_or_url))
        print(f"repo_name: {repo_name}")
        repo = repo_path_or_url
        branch_info = ""
    else:
        if not GITHUB_TOKEN:
            raise ValueError("Please set the 'GITHUB_TOKEN' environment variable or the 'GITHUB_TOKEN' in the script.")
        g = Github(GITHUB_TOKEN)
        repo = g.get_repo(repo_path_or_url.replace('https://github.com/', ''))
        repo_name = repo.name
        branch_info = f" (branch: {branch})" if branch else ""

    print(f"Fetching README for: {repo_name}{branch_info}")
    readme_content = get_readme_content(repo, branch)

    print(f"\nFetching repository structure for: {repo_name}{branch_info}")
    if is_local:
        repo_structure, selected_paths, token_data = traverse_local_repo_interactively(repo, count_tokens_flag=count_tokens_flag)
    else:
        repo_structure, selected_paths, token_data = traverse_repo_interactively(repo, count_tokens_flag=count_tokens_flag)

    print(f"\nFetching contents of selected files for: {repo_name}{branch_info}")
    file_contents, file_token_data = get_selected_file_contents(repo, selected_paths, is_local, count_tokens_flag)
    
    if count_tokens_flag:
        token_data.update(file_token_data)
        tree_representation = generate_tree_representation(token_data)
        summary_table = generate_summary_table(token_data)
        
        token_output_filename = f'{repo_name}_token_data.txt'
        with open(token_output_filename, 'w', encoding='utf-8') as f:
            f.write("Token Data Tree Representation:\n")
            f.write("--------------------------------\n")
            f.write(tree_representation)
            f.write("\n\n")
            f.write(summary_table)
        print(f"Token data saved to '{token_output_filename}'.")

    instructions = f"Prompt: Analyse the {repo_name}{branch_info} repository to understand its structure, purpose, and functionality. Follow these steps to study the codebase:\n\n"
    instructions += "1. Read the README file to gain an overview of the project, its goals, and any setup instructions.\n\n"
    instructions += "2. Examine the repository structure to understand how the files and directories are organised.\n\n"
    instructions += "3. Identify the main entry point of the application (e.g., main.py, app.py, index.js) and start analysing the code flow from there.\n\n"
    instructions += "4. Study the dependencies and libraries used in the project to understand the external tools and frameworks being utilised.\n\n"
    instructions += "5. Analyse the core functionality of the project by examining the key modules, classes, and functions.\n\n"
    instructions += "6. Look for any configuration files (e.g., config.py, .env) to understand how the project is configured and what settings are available.\n\n"
    instructions += "7. Investigate any tests or test directories to see how the project ensures code quality and handles different scenarios.\n\n"
    instructions += "8. Review any documentation or inline comments to gather insights into the codebase and its intended behaviour.\n\n"
    instructions += "9. Identify any potential areas for improvement, optimisation, or further exploration based on your analysis.\n\n"
    instructions += "10. Provide a summary of your findings, including the project's purpose, key features, and any notable observations or recommendations.\n\n"
    instructions += "Use the files and contents provided below to complete this analysis:\n\n"

    return repo_name, instructions, readme_content, repo_structure, file_contents

def main():
    """
    Main function to parse command-line arguments and execute the script.
    """
    parser = argparse.ArgumentParser(description='Interactively traverse and analyse the contents of a GitHub repository or local folder.')
    parser.add_argument('repo_path_or_url', type=str, help='The GitHub repository URL or the path to a local folder')
    parser.add_argument('--count-tokens', action='store_true', help='Count tokens in files and generate token statistics')
    args = parser.parse_args()

    repo_path_or_url = args.repo_path_or_url
    count_tokens_flag = args.count_tokens

    if os.path.isdir(repo_path_or_url):
        try:
            repo_name, instructions, readme_content, repo_structure, file_contents = get_repo_contents(repo_path_or_url, count_tokens_flag=count_tokens_flag)
            output_filename = f'{repo_name}_contents.txt'
            with open(output_filename, 'w', encoding='utf-8') as f:
                f.write(instructions)
                f.write(f"README:\n{readme_content}\n\n")
                f.write(f"Repo structure:\n{repo_structure}\n\n")
                f.write('\n\n')
                f.write(file_contents)
            print(f"Repository contents saved to '{output_filename}'.")
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Please check the local folder path and try again.")
    else:
        if not GITHUB_TOKEN:
            print("Error: GitHub token is not set. Please set the GITHUB_TOKEN environment variable or update the script.")
            sys.exit(1)

        try:
            g = Github(GITHUB_TOKEN)
            repo = g.get_repo(repo_path_or_url.replace('https://github.com/', ''))
            branches = [branch.name for branch in repo.get_branches()]
            
            print("Available branches:")
            for i, branch in enumerate(branches, 1):
                print(f"{i}. {branch}")
            
            branch_choice = int(input("Enter the number of the branch you want to analyse (or 0 for default branch): "))
            branch = branches[branch_choice - 1] if branch_choice > 0 else None
            
            branch_info = f"_{branch} " if branch else ""
            repo_name, instructions, readme_content, repo_structure, file_contents = get_repo_contents(repo_path_or_url, branch, count_tokens_flag)
            output_filename = f'{repo_name}{branch_info.replace(" ", "_")}_contents.txt'
            with open(output_filename, 'w', encoding='utf-8') as f:
                f.write(instructions)
                f.write(f"README:\n{readme_content}\n\n")
                f.write(repo_structure)
                f.write('\n\n')
                f.write(file_contents)
            print(f"Repository contents saved to '{output_filename}'.")
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Please check the repository URL and try again.")

if __name__ == '__main__':
    main()