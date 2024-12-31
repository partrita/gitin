# gitin

[![PyPI version](https://badge.fury.io/py/gitin.svg)](https://badge.fury.io/py/gitin)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Versions](https://img.shields.io/pypi/pyversions/gitin.svg)](https://pypi.org/project/gitin/)

👋 Hi there! I'm [unclecode](https://x.com/unclecode), the author of [Crawl4AI](https://github.com/unclecode/crawl4ai) - a no 1 trending GitHub repository that's crawl the web in LLms friendly way. While working with LLMs like Claude and GPT, I often need to provide codebase context efficiently. That's why I created `gitin` - a simple yet powerful tool that helps you extract and format GitHub repository content for LLM consumption.

## Why gitin?

When chatting with AI models about code, providing the right context is crucial. `gitin` helps you:
- Extract relevant code files from any GitHub repository
- Format them into a clean, token-efficient markdown file
- Filter files by type, size, and content
- Get token estimates for LLM context windows

## Installation

```bash
pip install gitin
```

## Quick Start

Basic usage - get all Python files from a repository:
```bash
gitin https://github.com/unclecode/crawl4ai -o output.md --include="*.py"
```

## Examples

### 1. Basic Repository Extraction
Extract Python files from Crawl4AI, excluding tests:
```bash
gitin https://github.com/unclecode/crawl4ai \
    --include="*.py" \
    --exclude="tests/*" \
    -o basic_example.md
```

### 2. Search for Specific Content
Find files containing async functions:
```bash
gitin https://github.com/unclecode/crawl4ai \
    --include="*.py" \
    --search="async def" \
    -o async_functions.md
```

### 3. Multiple File Types with Size Limit
Get both Python and Markdown files under 5KB:
```bash
gitin https://github.com/unclecode/crawl4ai \
    --include="*.py,*.md" \
    --exclude="tests/*,docs/*" \
    --max-size=5000 \
    -o small_files.md
```

### 4. Documentation Files Only
Extract markdown files for documentation:
```bash
gitin https://github.com/unclecode/crawl4ai \
    --include="docs/**/*.md" \
    -o documentation.md
```

## Output Format

The tool generates a clean markdown file with:
- Repository structure
- File contents with syntax highlighting
- Clear separators between files
- Token count estimation for LLMs

Example output structure:
```markdown
# Repository Content

## path/to/file1.py
```python
def hello():
    print("Hello, World!")
```

## path/to/file2.md
```markdown
# Documentation
This is a markdown file...
```

## Command-Line Options

```
Options:
  --version           Show the version and exit
  --exclude TEXT      Comma-separated glob patterns to exclude
                      Example: --exclude="test_*,*.tmp,docs/*"
  --include TEXT      Comma-separated glob patterns to include
                      Example: --include="*.py,src/*.js,lib/*.rb"
  --search TEXT       Comma-separated strings to search in file contents
                      Example: --search="TODO,FIXME,HACK"
  --max-size INTEGER  Maximum file size in bytes (default: 1MB)
  -o, --output TEXT   Output markdown file path [required]
  --help             Show this message and exit
```

## Use with LLMs

When using the output with AI models:

1. Generate the markdown file:
```bash
gitin https://github.com/your/repo -o context.md --include="*.py"
```

2. Copy the content to your conversation with the AI model

3. The AI model will now have context about your codebase and can help with:
   - Code review
   - Bug fixing
   - Feature implementation
   - Documentation
   - Refactoring suggestions

## Pro Tips

1. **Token Efficiency**: Use `--max-size` to limit file sizes and stay within context windows
2. **Relevant Context**: Use `--search` to find specific code patterns or TODO comments
3. **Multiple Patterns**: Combine patterns with commas: `--include="*.py,*.js,*.md"`
4. **Exclude Tests**: Use `--exclude="tests/*,*_test.py"` to focus on main code
5. **Documentation**: Include only docs with `--include="docs/**/*.md"`

## About the Author

I'm unclecode, and I love building tools that make AI development easier. Check out my other project [Crawl4AI](https://github.com/unclecode/crawl4ai) and follow me on X [@unclecode](https://x.com/unclecode).

## Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest features
- Submit pull requests

I'm extremely busy with Crawl4ai, so I may not be able to check this repository frequently. However, feel free to send your pull request, and I will try to approve it.

## License

MIT License - feel free to use in your projects!

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for release history.
