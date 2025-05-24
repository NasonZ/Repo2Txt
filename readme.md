# Repo2Txt

`Repo2Txt` is an enhanced Python tool for converting GitHub repositories or local codebases into LLM-friendly text formats. It features interactive file selection, comprehensive token analysis, and smart navigation for efficient token budget management.

## 🚀 Key Features

### Core Functionality
- **Dual Mode Support**: Analyze both GitHub repositories and local directories
- **Interactive Selection**: Navigate through directories with granular file/folder selection
- **Smart Navigation**: Back navigation (`b`) and quit options (`q`) during selection
- **Token Analysis**: Comprehensive token counting and budget recommendations
- **Intelligent Output**: Organized outputs in repository-named directories

### Enhanced Selection Experience
- **Back Navigation**: Change your mind? Press `b` to go back to previous selections
- **Flexible Options**: Choose specific files, entire directories, or skip sections
- **Branch Selection**: For GitHub repos, select which branch to analyze
- **Progress Tracking**: Visual progress bars during file processing

### Advanced Token Analysis
- **Tree View**: Visual repository structure with token counts
- **Full Table View**: Detailed token breakdown for precise budgeting
- **Distribution Stats**: Min/median/max/std dev, file size buckets
- **Top Content**: Largest directories and files for quick decisions
- **Budget Recommendations**: Smart suggestions for token allocation

### Smart File Handling
- **Enhanced Binary Detection**: Multiple methods to identify binary files
- **Encoding Fallbacks**: Handles UTF-8, Latin-1, and other encodings
- **Size Limits**: Configurable max file size (default: 1MB)
- **Excluded Directories**: Automatically skips common non-source directories

## 📋 Prerequisites

- Python 3.6 or later
- Required packages:
  ```bash
  pip install PyGithub tqdm tiktoken
  ```
- GitHub Personal Access Token (for private repos)

## 🔧 Installation

```bash
git clone https://github.com/your-username/repo2txt.git
cd repo2txt
pip install -r requirements.txt
```

## 📖 Usage

### Basic Usage

```bash
# Analyze a GitHub repository
python repo2txt.py https://github.com/owner/repo

# Analyze a local directory
python repo2txt.py /path/to/local/repo

# Analyze current directory
python repo2txt.py .
```

### Command Line Options

```bash
python repo2txt.py [repo] [options]

Options:
  --no-tokens           Disable token counting
  --json               Export token data as JSON for analysis
  --output-dir DIR     Custom output directory (default: repo name)
  --max-file-size SIZE Maximum file size in bytes (default: 1MB)
  --debug              Enable debug logging
```

### Environment Setup

For GitHub repositories, set your Personal Access Token:
```bash
export GITHUB_TOKEN='your_github_token'
```

## 💡 Interactive Selection

### Navigation Commands
- **Number ranges**: `1-5,7,9-12` - Select specific items
- **All**: `a` - Select all items in current directory
- **Skip**: `s` - Skip current directory
- **Back**: `b` - Go back to previous selection
- **Quit**: `q` - Exit selection (with confirmation)

### Example Workflow
```
Contents of root:
  1. src (dir)
  2. tests (dir)
  3. docs (dir)
  4. README.md (file)

Options: Enter numbers (e.g., 1-5,7), 'a' for all, 's' to skip, 'b' to go back, 'q' to quit
Your choice: 1,3-4

Select items in 'src'?
Options: (y)es, (n)o, (a)ll, (b)ack: y

Contents of src:
  1. main.py (file, 1,234 bytes)
  2. utils.py (file, 567 bytes)
  3. config.py (file, 890 bytes)

Your choice: a
```

## 📊 Output Files

### Default Output (2 files)
```
repo-name/
├── repo-name_main_20240115_143022_analysis.txt  # Main analysis with file contents
└── repo-name_main_20240115_143022_tokens.txt    # Token report with tree & table
```

### With JSON Export (3 files)
```bash
python repo2txt.py https://github.com/owner/repo --json
```
Adds: `repo-name_main_20240115_143022_token_data.json`

## 📈 Token Report Structure

### 1. Directory Tree
```
📂 Directory Tree:
├── src/ (Tokens: 15,234, Files: 12)
│   ├── components/ (Tokens: 8,567, Files: 5)
│   │   ├── Button.tsx (Tokens: 1,234)
│   │   └── Card.tsx (Tokens: 2,345)
│   └── utils/ (Tokens: 3,456, Files: 3)
└── docs/ (Tokens: 5,678, Files: 8)
```

### 2. Token Count Table
```
📊 Token Count Summary:
File/Directory                    | Token Count | File Count
────────────────────────────────────────────────────────────
src                               |      15,234 |         12
  components                      |       8,567 |          5
    Button.tsx                    |       1,234 |
    Card.tsx                      |       2,345 |
```

### 3. Quick Stats
```
📈 Quick Stats:
Total: 45,678 tokens across 156 files
Average: 293 tokens/file

Distribution:
  Min: 12 | Median: 234 | Max: 5,678 | Std Dev: 456

File size distribution:
  ≤100 tokens:     23 files (14%)
  101-500:         89 files (57%)
  501-1000:        32 files (20%)
  1001-5000:       11 files ( 7%)
  >5000:            1 files ( 0%)

Top 5 largest directories:
    15,234 tokens: src/ (12 files, avg 1,269/file)
     8,567 tokens: src/components/ (5 files, avg 1,713/file)
```

## 🔄 Typical Workflow

1. **Initial Analysis**: Run the script and review the token report
2. **Budget Planning**: Use the distribution stats to plan token allocation
3. **Selective Extraction**: Re-run with specific selections
4. **Final Export**: Generate the analysis file with optimal content

## ⚙️ Configuration

### Excluded Directories (Default)
```python
__pycache__, .git, .hg, .svn, .idea, .vscode, node_modules,
.pytest_cache, .mypy_cache, venv, env, .env, dist, build
```

### Binary File Detection
- Extension-based detection (`.exe`, `.dll`, `.jpg`, etc.)
- MIME type checking
- Null byte detection
- Skip patterns (`package-lock.json`, `*.min.js`)

## 🐛 Troubleshooting

### Common Issues

1. **GitHub Rate Limiting**
   - Use a Personal Access Token
   - Implement delays between requests

2. **Large Repositories**
   - Use `--max-file-size` to limit file sizes
   - Select specific directories instead of entire repo

3. **Encoding Errors**
   - Script automatically tries multiple encodings
   - Binary files are detected and skipped

## 📝 Notes

- Token counting requires the `tiktoken` library
- Uses OpenAI's `cl100k_base` encoding by default
- Empty directories are automatically skipped
- Progress bars show real-time processing status

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Original concept from [Doriandarko/RepoToTextForLLMs](https://github.com/Doriandarko/RepoToTextForLLMs)
- Enhanced with interactive selection, token analysis, and navigation features
