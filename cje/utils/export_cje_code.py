#!/usr/bin/env python3
"""
Export CJE core code to a markdown file for easy sharing with chatbots.

This script collects all .py and README.md files from the core CJE directories
(calibration, estimators, data) and formats them into a single markdown file
with clear section headers and code blocks.
"""

import os
from pathlib import Path
from typing import List, Tuple
import datetime


def get_files_from_directory(base_path: Path, directory: str, extensions: List[str]) -> List[Tuple[Path, str]]:
    """Get all files with specified extensions from a directory.
    
    Returns list of (file_path, relative_path) tuples.
    """
    files = []
    dir_path = base_path / directory
    
    if not dir_path.exists():
        print(f"Warning: Directory {dir_path} does not exist")
        return files
    
    for ext in extensions:
        for file_path in dir_path.rglob(f"*{ext}"):
            # Skip __pycache__ directories
            if "__pycache__" in str(file_path):
                continue
            # Get relative path from base CJE directory
            relative_path = file_path.relative_to(base_path)
            files.append((file_path, str(relative_path)))
    
    return sorted(files, key=lambda x: x[1])


def format_file_content(file_path: Path, relative_path: str) -> str:
    """Format a single file's content for markdown output."""
    output = []
    
    # Add file header
    output.append(f"\n## 📄 {relative_path}\n")
    
    # Determine language for syntax highlighting
    if file_path.suffix == ".py":
        lang = "python"
    elif file_path.suffix == ".md":
        lang = "markdown"
    else:
        lang = ""
    
    # Read and add file content
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Add line count info
        line_count = len(content.splitlines())
        output.append(f"*({line_count} lines)*\n")
        
        # Add content in code block
        output.append(f"```{lang}")
        output.append(content)
        output.append("```\n")
        
    except Exception as e:
        output.append(f"*Error reading file: {e}*\n")
    
    return "\n".join(output)


def export_to_markdown(output_file: str = "cje_core_code.md"):
    """Export all core CJE code to a markdown file."""
    
    # Get the CJE base directory
    cje_base = Path(__file__).parent / "cje"
    
    # Directories to export
    directories = ["calibration", "estimators", "data"]
    
    # File extensions to include
    extensions = [".py", "README.md"]
    
    # Collect all files
    all_files = []
    for directory in directories:
        files = get_files_from_directory(cje_base, directory, extensions)
        all_files.extend([(directory, f[0], f[1]) for f in files])
    
    # Generate output
    output = []
    
    # Add header
    output.append("# CJE Core Code Export\n")
    output.append(f"*Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
    output.append(f"\nThis document contains all Python and README files from the CJE core directories: ")
    output.append(f"`{', '.join(directories)}`\n")
    
    # Add table of contents
    output.append("\n## Table of Contents\n")
    current_dir = None
    for directory, file_path, relative_path in all_files:
        if directory != current_dir:
            output.append(f"\n### {directory.title()} Module\n")
            current_dir = directory
        output.append(f"- [{relative_path}](#{relative_path.replace('/', '').replace('.', '').replace('_', '').lower()})")
    
    output.append("\n---\n")
    
    # Add file contents
    current_dir = None
    file_count = 0
    total_lines = 0
    
    for directory, file_path, relative_path in all_files:
        if directory != current_dir:
            output.append(f"\n# {directory.upper()} MODULE\n")
            current_dir = directory
        
        content = format_file_content(file_path, relative_path)
        output.append(content)
        file_count += 1
        
        # Count lines for stats
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                total_lines += len(f.readlines())
        except:
            pass
    
    # Add footer with stats
    output.append("\n---\n")
    output.append(f"\n## Summary Statistics\n")
    output.append(f"- Total files: {file_count}")
    output.append(f"- Total lines of code: {total_lines:,}")
    output.append(f"- Modules exported: {', '.join(directories)}")
    
    # Write to file
    output_path = Path(output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(output))
    
    print(f"✅ Successfully exported {file_count} files ({total_lines:,} lines) to {output_file}")
    print(f"📋 You can now copy the contents of {output_file} to your chatbot")
    
    # Also create a more compact version without empty lines
    compact_output_file = output_file.replace('.md', '_compact.md')
    create_compact_version(output_file, compact_output_file)
    print(f"📦 Also created compact version: {compact_output_file}")
    
    return output_path


def create_compact_version(input_file: str, output_file: str):
    """Create a more compact version with less whitespace."""
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Remove excessive blank lines (keep max 1 blank line)
    compact_lines = []
    prev_blank = False
    for line in lines:
        if line.strip() == "":
            if not prev_blank:
                compact_lines.append(line)
            prev_blank = True
        else:
            compact_lines.append(line)
            prev_blank = False
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(compact_lines)


def export_to_csv(output_file: str = "cje_core_code.csv"):
    """Export file metadata to CSV for analysis."""
    import csv
    
    # Get the CJE base directory
    cje_base = Path(__file__).parent / "cje"
    
    # Directories to export
    directories = ["calibration", "estimators", "data"]
    
    # File extensions to include
    extensions = [".py", "README.md"]
    
    # Collect all files
    all_files = []
    for directory in directories:
        files = get_files_from_directory(cje_base, directory, extensions)
        for file_path, relative_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = len(content.splitlines())
                    chars = len(content)
            except:
                lines = 0
                chars = 0
            
            all_files.append({
                'module': directory,
                'path': relative_path,
                'filename': file_path.name,
                'extension': file_path.suffix,
                'lines': lines,
                'characters': chars,
                'size_kb': file_path.stat().st_size / 1024
            })
    
    # Write CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        if all_files:
            writer = csv.DictWriter(f, fieldnames=all_files[0].keys())
            writer.writeheader()
            writer.writerows(all_files)
    
    print(f"📊 Also exported metadata to {output_file}")
    return output_file


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Export CJE core code for sharing with chatbots")
    parser.add_argument(
        "--output", "-o",
        default="cje_core_code.md",
        help="Output markdown file (default: cje_core_code.md)"
    )
    parser.add_argument(
        "--include-csv",
        action="store_true",
        help="Also generate a CSV file with file metadata"
    )
    parser.add_argument(
        "--modules",
        nargs="+",
        default=["calibration", "estimators", "data"],
        help="Modules to export (default: calibration estimators data)"
    )
    
    args = parser.parse_args()
    
    # Export to markdown
    md_file = export_to_markdown(args.output)
    
    # Optionally export to CSV
    if args.include_csv:
        csv_file = export_to_csv(args.output.replace('.md', '.csv'))
    
    print("\n💡 Tip: The markdown file is formatted for easy copying into ChatGPT, Claude, or other LLMs")
    print("   Just open the file and copy everything, or use: cat cje_core_code.md | pbcopy")