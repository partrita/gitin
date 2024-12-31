# gitin - GitHub Repository Content Extractor

[![PyPI version](https://badge.fury.io/py/gitin.svg)](https://badge.fury.io/py/gitin)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Versions](https://img.shields.io/pypi/pyversions/gitin.svg)](https://pypi.org/project/gitin/)

A command-line tool designed to extract and format GitHub repository content for effective use with Large Language Models (LLMs). Perfect for providing codebases as context in LLM conversations.

## Features

- 🔍 **Smart Extraction**: Extract files based on patterns and content
- 📝 **LLM-Optimized**: Clean markdown output with token estimation
- 🚀 **Progress Tracking**: Visual progress bars for operations
- 🎯 **Flexible Filtering**: Include/exclude patterns and content search
- 📊 **Size Control**: File size limits and statistics
- 🔄 **Version Control**: Respects .gitignore rules

## Installation

### From PyPI (Recommended)
```bash
pip install gitin
```

### From Source
```bash
git clone https://github.com/unclecode/gitin
cd gitin
pip install -e .
```

## Quick Start

```bash
# Basic usage
gitin https://github.com/username/repo -o output.md

# Extract Python files only
gitin https://github.com/username/repo --include="*.py" -o python_files.md

# Search for specific content
gitin https://github.com/username/repo --search="TODO,FIXME" -o review.md
```

## Advanced Usage

### Code Review Workflow
```bash
# Extract implementation files, excluding tests
gitin https://github.com/username/repo \
  --include="src/*.py" \
  --exclude="test_*,*_test.py" \
  --search="TODO,FIXME,HACK" \
  -o code_review.md
```

### Feature Analysis
```bash
# Find async/await usage
gitin https://github.com/username/repo \
  --include="*.py,*.js" \
  --search="async def,await" \
  -o async_patterns.md
```

### Documentation Extraction
```bash
# Get all documentation files
gitin https://github.com/username/repo \
  --include="docs/*,*.md,*.rst" \
  --exclude="**/tests/*" \
  -o documentation.md
```

## Command-Line Options

```
Options:
  --version           Show the version and exit.
  --exclude TEXT      Comma-separated glob patterns to exclude
                     Example: --exclude="test_*,*.tmp,docs/*"
  --include TEXT      Comma-separated glob patterns to include
                     Example: --include="*.py,src/*.js,lib/*.rb"
  --search TEXT       Comma-separated strings to search in content
                     Example: --search="TODO,FIXME,HACK"
  --max-size INTEGER  Maximum file size in bytes (default: 1MB)
  -o, --output TEXT   Output markdown file path [required]
  --help             Show this message and exit.
```

## LLM Integration Tips

1. **Context Window Management**
   - Check the token count in the summary
   - Use `--max-size` to limit file sizes
   - Use `--search` to focus on relevant sections

2. **Effective Filtering**
   - Use `--include` for specific file types
   - Use `--exclude` to remove noise
   - Combine with `--search` for precise results

3. **Best Practices**
   ```bash
   # Extract core functionality
   gitin https://github.com/org/repo \
     --include="src/**/*.py" \
     --exclude="**/*test*" \
     --max-size=50000 \
     -o core_logic.md
   ```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for release history.
