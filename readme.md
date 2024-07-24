# Repo2Txt

`Repo2Txt` is a Python script that allows you to interactively traverse and analyse the contents of a GitHub repository or a local folder. It extracts the structure and contents of selected files and folders and saves the information to a text file.

## Features

- Traverse and analyse both local directories and GitHub repositories.
- Saves the analysis, including repository structure and file contents, to a text file.
- Skips binary files, handles different encodings for text files, and excludes junk directories (e.g., `__pycache__`, `.git`, `.hg`, `.svn`, `.idea`, `.vscode`, `node_modules`).

**Additional Features/Improvements in This Repo (not present in /Doriandarko/RepoToTextForLLMs):**

- Interactively select specific branches, folders, and files for analysis, with an option to include or exclude sub-folders.
- Count tokens for selected files and include token statistics in the analysis for easier prompt pruning.
## Prerequisites

- Python 3.6 or later
- `PyGithub` library: Install it using `pip install PyGithub`
- `tqdm` library: Install it using `pip install tqdm`
- `tiktoken` library: Install it using `pip install tiktoken`
- GitHub Personal Access Token (PAT) for accessing private repositories

## Installation

1. Clone the repository or download the script.

    ```sh
    git clone https://github.com/your-username/repo2txt.git
    ```

2. Navigate to the directory containing the script.

    ```sh
    cd repo2txt
    ```

3. Install the required Python packages.

    ```sh
    pip install PyGithub tqdm
    ```

## Usage

1. Ensure you have a GitHub Personal Access Token (PAT). Set it as an environment variable named `GITHUB_TOKEN`.

    ```sh
    export GITHUB_TOKEN='your_github_token'
    ```

2. Run the script.

    ```sh
    python repo2txt.py
    ```

3. Follow the prompts to enter the GitHub repository URL or the path to a local folder.

4. Interactively select the folders and files you wish to analyse. You can choose to include or exclude sub-folders.

5. If you want to count tokens in the files, use the --count-tokens flag when running the script.

   ```sh
   python repo2txt.py --count-tokens
   ```

6. The script will save the analysis, including the repository structure, file contents, and token statistics, to a text file in the current directory.
   

## Example

```sh
Enter the GitHub repository URL or the path to a local folder:
https://github.com/your-username/your-repo

Fetching README for: your-repo

Fetching repository structure for: your-repo

Contents of :
1. .git (dir)
2. .github (dir)
3. src (dir)
4. tests (dir)
5. README.md (file)

Enter the indices of the folders/files you want to extract (e.g., 1-5,7,9-12) or 'a' for all: 3,4,5
Do you want to select sub-folders in src? (y/n/a): a
Do you want to select sub-folders in tests? (y/n/a): n

Fetching contents of selected files for: your-repo

Repository contents saved to 'your-repo_contents.txt'.
```

# Notes
- The script skips binary files and certain file types by default.
- If a file cannot be read due to unsupported encoding, it will be skipped with a corresponding message in the output file.
- This repo is forked adjusted from - https://github.com/Doriandarko/RepoToTextForLLMs

# Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue to discuss any changes.

# License
This project is licensed under the MIT License. See the LICENSE file for details.
