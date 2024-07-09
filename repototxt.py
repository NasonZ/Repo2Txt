import os
import sys
import argparse
from github import Github
from tqdm import tqdm

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

def traverse_local_repo_interactively(repo_path, selected_paths=None, excluded_dirs=None):
    """
    Traverse the local repository interactively, allowing user to select folders and files.

    Args:
        repo_path: The local repository path.
        selected_paths: A set to store selected paths (optional).
        excluded_dirs: A set of directories to exclude (optional).

    Returns:
        The structure of the repository and the selected paths.
    """
    if selected_paths is None:
        selected_paths = set()
    if excluded_dirs is None:
        excluded_dirs = {'__pycache__', '.git', '.hg', '.svn', '.idea', '.vscode', 'node_modules', '.lancedb'}

    structure = ""
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in excluded_dirs]  # Exclude specific directories
        rel_root = os.path.relpath(root, repo_path)
        if rel_root == '.':
            rel_root = ''
        print(f"\nContents of {rel_root}:")
        items = dirs + files
        for i, item in enumerate(items, start=1):
            print(f"{i}. {item} ({'dir' if item in dirs else 'file'})")

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

        for i, item in enumerate(items, start=1):
            if i in selected_indices:
                item_path = os.path.join(root, item)
                rel_item_path = os.path.relpath(item_path, repo_path)
                if os.path.isdir(item_path):
                    structure += f"{rel_item_path}/\n"
                    while True:
                        sub_folders_choice = input(f"Do you want to select sub-folders in {rel_item_path}? (y/n/a): ").lower()
                        if sub_folders_choice == 'y':
                            sub_structure, sub_selected_paths = traverse_local_repo_interactively(item_path, selected_paths, excluded_dirs)
                            structure += sub_structure
                            selected_paths.update(sub_selected_paths)
                            break
                        elif sub_folders_choice == 'a':
                            for sub_root, sub_dirs, sub_files in os.walk(item_path):
                                sub_dirs[:] = [d for d in sub_dirs if d not in excluded_dirs]  # Exclude specific directories
                                rel_sub_root = os.path.relpath(sub_root, repo_path)
                                for sub_dir in sub_dirs:
                                    sub_dir_path = os.path.join(rel_sub_root, sub_dir)
                                    structure += f"{sub_dir_path}/\n"
                                for sub_file in sub_files:
                                    sub_file_path = os.path.join(rel_sub_root, sub_file)
                                    structure += f"{sub_file_path}\n"
                                    selected_paths.add(sub_file_path)
                            break
                        elif sub_folders_choice == 'n':
                            break
                        else:
                            print("Invalid choice. Please enter 'y', 'n', or 'a'.")
                else:
                    structure += f"{rel_item_path}\n"
                    selected_paths.add(rel_item_path)
            else:
                item_path = os.path.join(root, item)
                rel_item_path = os.path.relpath(item_path, repo_path)
                if os.path.isdir(item_path):
                    structure += f"{rel_item_path}/ (Omitted for brevity)\n"
                else:
                    structure += f"{rel_item_path}"

        break  # Exit after processing the current directory

    return structure, selected_paths

def traverse_repo_interactively(repo, path="", selected_paths=None, excluded_dirs=None):
    """
    Traverse the GitHub repository interactively, allowing user to select folders and files.

    Args:
        repo: The GitHub repository object.
        path: The current path in the repository.
        selected_paths: A set to store selected paths (optional).
        excluded_dirs: A set of directories to exclude (optional).

    Returns:
        The structure of the repository and the selected paths.
    """
    if selected_paths is None:
        selected_paths = set()
    if excluded_dirs is None:
        excluded_dirs = {'__pycache__', '.git', '.hg', '.svn', '.idea', '.vscode', 'node_modules'}

    structure = ""
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
                            sub_structure, sub_selected_paths = traverse_repo_interactively(repo, content.path, selected_paths, excluded_dirs)
                            structure += sub_structure
                            selected_paths.update(sub_selected_paths)
                            break
                        elif sub_folders_choice == 'a':
                            sub_contents = repo.get_contents(content.path)
                            for sub_content in sub_contents:
                                if sub_content.type == "dir":
                                    if sub_content.name not in excluded_dirs:
                                        sub_dir_path = f"{content.path}/{sub_content.name}/"
                                        structure += f"{sub_dir_path}\n"
                                        sub_structure, sub_selected_paths = traverse_repo_interactively(repo, sub_content.path, selected_paths, excluded_dirs)
                                        structure += sub_structure
                                        selected_paths.update(sub_selected_paths)
                                else:
                                    sub_file_path = f"{content.path}/{sub_content.name}"
                                    structure += f"{sub_file_path}\n"
                                    selected_paths.add(sub_file_path)
                            break
                        elif sub_folders_choice == 'n':
                            break
                        else:
                            print("Invalid choice. Please enter 'y', 'n', or 'a'.")
            else:
                structure += f"{path}/{content.name}\n"
                selected_paths.add(f"{path}/{content.name}")
        else:
            if content.type == "dir":
                structure += f"{path}/{content.name}/ (Omitted for brevity)\n"
            else:
                structure += f"{path}/{content.name}\nContent: Omitted for brevity\n\n"

    return structure, selected_paths

def get_selected_file_contents(repo, selected_files, is_local=False):
    """
    Get the contents of the selected files.

    Args:
        repo: The repository object or local path.
        selected_files: A list of selected files.
        is_local: A flag indicating if the repository is local.

    Returns:
        The contents of the selected files.
    """
    file_contents = ""
    binary_extensions = [
        # Compiled executables and libraries
        '.exe', '.dll', '.so', '.a', '.lib', '.dylib', '.o', '.obj',
        # Compressed archives
        '.zip', '.tar', '.tar.gz', '.tgz', '.rar', '.7z', '.bz2', '.gz', '.xz', '.z', '.lz', '.lzma', '.lzo', '.rz', '.sz', '.dz',
        # Application-specific files
        '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', '.odt', '.ods', '.odp',
        # Media files
        '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp', '.svg',
        '.mp3', '.mp4', '.wav', '.flac', '.ogg', '.avi', '.mkv', '.mov', '.webm', '.wmv', '.m4a', '.aac',
        '.eps', '.ai',
        # Virtual machine and container images
        '.iso', '.vmdk', '.qcow2', '.vdi', '.vhd', '.vhdx', '.ova', '.ovf',
        # Database files
        '.db', '.sqlite', '.mdb', '.accdb', '.frm', '.ibd', '.dbf', '.sql',
        # Java-related files
        '.jar', '.class', '.war', '.ear', '.jpi',
        # Python bytecode and packages
        '.pyc', '.pyo', '.pyd', '.egg', '.whl',
        # Other programming language-specific files
        '.o', '.ko', '.obj', '.elf', '.lib', '.a', '.la', '.lo', '.dll', '.so', '.dylib',
        '.exe', '.out', '.app', '.sl', '.framework',
        # Font files
        '.eot', '.otf', '.ttf', '.woff', '.woff2',
        # Icon files
        '.ico', '.icns', '.cur',
        # Windows-specific files
        '.cab', '.dmp', '.sys', '.msi', '.msix', '.msp', '.msm', '.msu',
        # macOS-specific files
        '.dmg', '.pkg',
        # Linux package files
        '.deb', '.rpm', '.snap', '.flatpak', '.appimage',
        # Android package files
        '.apk', '.aab',
        # iOS package files
        '.ipa',
        # Certificate and key files
        '.pem', '.crt', '.ca-bundle', '.p7b', '.p7c', '.p12', '.pfx',
        '.cer', '.der', '.key', '.keystore', '.jks', '.p8', '.sig',
        # Version control system files
        '.svn', '.git', '.gitignore', '.gitattributes', '.gitmodules',
        # IDE and editor files
        '.iml', '.ipr', '.iws', '.project', '.cproject', '.buildpath',
        '.classpath', '.metadata', '.settings', '.idea', '.vscode',
        # Build output directories
        'bin/', 'obj/', 'build/', 'dist/', 'target/',
        # Dependency directories
        '/node_modules/', 'vendor/', 'packages/',
        # Log files
        '.log', '.tlog',
        # Temporary files
        '.tmp', '.temp', '.swp', '.bak', '.cache',
        # Configuration files
        '.ini', '.cfg', '.config', '.conf', '.properties', '.prefs',
        '.htaccess', '.htpasswd', '.env', '.dockerignore',
        # Documentation files
        '.chm', '.epub', '.mobi',
        # Disk image files
        '.img', '.iso', '.vcd',
        # Backup files
        '.bak', '.gho', '.ori', '.orig',
        # Miscellaneous
        '.dat', '.data', '.dump', '.bin', '.raw',
        '.crx', '.xpi', '.lockb', 'package-lock.json',
        '.rlib', '.pdb', '.idb', '.ilk', '.exp', '.map',
        '.sdf', '.suo', '.VC.db', '.aps', '.res', '.rc',
        '.nupkg', '.snupkg', '.vsix',
        '.bpl', '.dcu', '.dcp', '.dcpil', '.drc',
        '.DS_Store', '.localized',
        '.manifest', '.lance', '.txt',
        # Note-taking application files
        '.one', '.notebook', '.nmbak',  # Microsoft OneNote
        '.enex',  # Evernote
        '.nd',  # Notability
        '.gdoc', '.gsheet', '.gslides',  # Google Docs/Sheets/Slides
        '.ipynb',  # Jupyter Notebook
        '.qvnotebook',  # Quiver
        '.bear',  # Bear
        # Large text files that might cause issues
        '.csv', '.tsv', '.json', '.xml', '.yaml', '.yml',
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
                            except UnicodeDecodeError:
                                try:
                                    decoded_content = content.decoded_content.decode('latin-1')
                                    file_contents += f"Content (Latin-1 Decoded):\n{decoded_content}\n\n"
                                except UnicodeDecodeError:
                                    file_contents += "Content: Skipped due to unsupported encoding\n\n"
                    except (AttributeError, UnicodeDecodeError):
                        file_contents += "Content: Skipped due to decoding error or missing decoded_content\n\n"
        except Exception as e:
            print(f"Error reviewing file {file_path}: {e}")
            file_contents += f"File: {file_path}\nContent: Skipped due to error: {e}\n\n"
    return file_contents

def get_repo_contents(repo_path_or_url, branch=None):
    """
    Main function to get repository contents.

    Args:
        repo_path_or_url: The GitHub repository URL or local path.
        branch: The branch name (optional).

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
        repo_structure, selected_paths = traverse_local_repo_interactively(repo)
    else:
        repo_structure, selected_paths = traverse_repo_interactively(repo)

    print(f"\nFetching contents of selected files for: {repo_name}{branch_info}")
    file_contents = get_selected_file_contents(repo, selected_paths, is_local)

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
    parser = argparse.ArgumentParser(description='Interactively traverse and analyse the contents of a GitHub repository or local folder.')
    parser.add_argument('repo_path_or_url', type=str, help='The GitHub repository URL or the path to a local folder')
    args = parser.parse_args()

    repo_path_or_url = args.repo_path_or_url

    if os.path.isdir(repo_path_or_url):
        print(f"repo_path_or_url: {repo_path_or_url}")
        # Local folder
        branch = None
        try:
            repo_name, instructions, readme_content, repo_structure, file_contents = get_repo_contents(repo_path_or_url, branch)
            #print(f"file_contents: {file_contents}")
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
        # GitHub repository
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
            repo_name, instructions, readme_content, repo_structure, file_contents = get_repo_contents(repo_path_or_url, branch)
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
