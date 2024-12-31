#!/usr/bin/env python3

import os
import subprocess
from pathlib import Path
from typing import List
import click
from fnmatch import fnmatch
import tempfile
import shutil
from tqdm import tqdm
from .__version__ import __version__

HELP_TEXT = """GitHub Repository Content Extractor

Extracts and formats repository content, optimized for use with Large Language Models.
Creates a single markdown file with clear separators between files.

Examples:

    # Basic usage - get all Python files
    gitin https://github.com/user/repo -o output.md --include="*.py"

    # Multiple file patterns with content search
    gitin https://github.com/user/repo \\
        --include="*.py,*.js" \\
        --search="TODO,FIXME" \\
        --exclude="test_*" \\
        -o code_review.md

    # Extract only specific file types with size limit
    gitin https://github.com/user/repo \\
        --include="src/*.py" \\
        --max-size=100000 \\
        -o small_py_files.md
"""

@click.command(help=HELP_TEXT)
@click.version_option(version=__version__)
@click.argument('github_url')
@click.option('--exclude', default='', 
              help="""Comma-separated glob patterns to exclude.
              Example: --exclude="test_*,*.tmp,docs/*" """)
@click.option('--include', default='', 
              help="""Comma-separated glob patterns to include.
              Example: --include="*.py,src/*.js,lib/*.rb" """)
@click.option('--search', default='', 
              help="""Comma-separated strings to search in file contents. Only files containing
              at least one of these strings will be included.
              Example: --search="TODO,FIXME,HACK" """)
@click.option('--max-size', default=1000000, 
              help="""Maximum file size in bytes to process. Files larger than this will be skipped.
              Default: 1MB""")
@click.option('-o', '--output', required=True,
              help="Output markdown file path")
def main(github_url: str, exclude: str, include: str, search: str, 
         max_size: int, output: str):
    """Extract and format repository content."""
    
    exclude_patterns = [p.strip() for p in exclude.split(',') if p.strip()]
    include_patterns = [p.strip() for p in include.split(',') if p.strip()]
    search_terms = [s.strip() for s in search.split(',') if s.strip()]
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Clone repository
        repo_dir = clone_repository(github_url, temp_dir)
        if not repo_dir:
            return
        
        # Process files
        process_repository(repo_dir, output, exclude_patterns, include_patterns,
                         search_terms, max_size)

def clone_repository(github_url: str, temp_dir: str) -> str:
    """Clone the repository and return the repo directory."""
    try:
        subprocess.run(['git', 'clone', '--depth=1', github_url, temp_dir], 
                      check=True, capture_output=True)
        return temp_dir
    except subprocess.CalledProcessError as e:
        click.echo(f"Error cloning repository: {e.stderr.decode()}", err=True)
        return None

def process_repository(repo_dir: str, output_file: str, 
                      exclude_patterns: List[str], include_patterns: List[str],
                      search_terms: List[str], max_size: int):
    """Process repository files and write to output markdown file."""
    processed_files = 0
    total_chars = 0
    
    # First, count total files for progress bar
    total_files = sum(1 for _ in os.walk(repo_dir))
    
    with open(output_file, 'w') as f:
        f.write(f"# Repository Content\n\n")
        
        # Create progress bar for directory scanning
        with tqdm(total=total_files, desc="Scanning directories", unit="dir") as pbar:
            for root, _, files in os.walk(repo_dir):
                if '.git' in root:
                    pbar.update(1)
                    continue
                
                # Create progress bar for files in current directory
                files_pbar = tqdm(files, desc=f"Processing {os.path.basename(root)}", 
                                leave=False, unit="file")
                
                for file in files_pbar:
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, repo_dir)
                    
                    # Update description with current file
                    files_pbar.set_description(f"Processing {rel_path}")
                    
                    # Skip files that match exclude patterns
                    if any(fnmatch(rel_path, pat) for pat in exclude_patterns):
                        continue
                        
                    # Skip files that don't match include patterns (if specified)
                    if include_patterns and not any(fnmatch(rel_path, pat) 
                                                  for pat in include_patterns):
                        continue
                    
                    # Skip files larger than max_size
                    if os.path.getsize(file_path) > max_size:
                        continue
                    
                    # Check file content for search terms
                    if search_terms:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as cf:
                            content = cf.read()
                            if not any(term.lower() in content.lower() 
                                     for term in search_terms):
                                continue
                    
                    # Write file content to markdown
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as cf:
                            content = cf.read()
                            f.write(f"\n## {rel_path}\n")
                            f.write("```\n")
                            f.write(content)
                            f.write("\n```\n")
                            processed_files += 1
                            total_chars += len(content)
                    except Exception as e:
                        click.echo(f"Error processing {rel_path}: {str(e)}", err=True)
                
                pbar.update(1)
    
    # Print summary statistics
    click.echo("\nSUMMARY:")
    click.echo(f"Files processed: {processed_files}")
    click.echo(f"Total characters: {total_chars}")
    click.echo(f"Estimated tokens: {total_chars // 4}")  # Rough estimate of tokens
    click.echo(f"Output written to: {output_file}")

if __name__ == '__main__':
    main()