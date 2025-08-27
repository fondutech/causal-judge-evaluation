#!/bin/bash
# Concatenate all Python files in the CJE project for chatbot discussions

echo "================================"
echo "Concatenating Python files..."
echo "================================"

OUTPUT_FILE="cje_python_code.txt"

# Clear output file and write initial header
cat > "$OUTPUT_FILE" << EOF
CJE PYTHON CODE EXPORT
======================

This file contains all Python (.py) files from the CJE core library directory,
excluding test files and __pycache__ directories.

Generated on: $(date)

EOF

# Generate dynamic table of contents
echo "TABLE OF CONTENTS" >> "$OUTPUT_FILE"
echo "=================" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
echo "The following files are included:" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# Build TOC by scanning actual files
current_dir=""
find ./cje -name "*.py" \
    -not -path "*/__pycache__/*" \
    -not -path "*/test_*" \
    -not -name "test_*.py" \
    -not -path "*/tests/*" \
    -not -path "*/.mypy_cache/*" \
    -not -path "*/.pytest_cache/*" \
    -type f \
    2>/dev/null | \
    sort | \
    while read -r file; do
        # Get directory and filename
        file_dir=$(dirname "$file")
        file_name=$(basename "$file")
        clean_dir="${file_dir#./}"
        
        # Add directory header if changed
        if [ "$file_dir" != "$current_dir" ]; then
            if [ -n "$current_dir" ]; then
                echo "" >> "$OUTPUT_FILE"
            fi
            echo "$clean_dir/" >> "$OUTPUT_FILE"
            current_dir="$file_dir"
        fi
        
        # Add file entry with proper indentation
        echo "  ├── $file_name" >> "$OUTPUT_FILE"
    done

# Count files for summary
py_count=$(find ./cje -name "*.py" \
    -not -path "*/__pycache__/*" \
    -not -path "*/test_*" \
    -not -name "test_*.py" \
    -not -path "*/tests/*" \
    -not -path "*/.mypy_cache/*" \
    -not -path "*/.pytest_cache/*" \
    -type f 2>/dev/null | wc -l | tr -d ' ')

echo "" >> "$OUTPUT_FILE"
echo "Total: $py_count Python files from the cje/ directory" >> "$OUTPUT_FILE"
echo "Files are presented in alphabetical order within each directory." >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
echo "========================================" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# Function to add a file with proper formatting
add_file() {
    local file_path="$1"
    echo "" >> "$OUTPUT_FILE"
    echo "=== $file_path ===" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
    cat "$file_path" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
}

# Find and concatenate Python files only from the cje/ directory
find ./cje -name "*.py" \
    -not -path "*/__pycache__/*" \
    -not -path "*/test_*" \
    -not -name "test_*.py" \
    -not -path "*/tests/*" \
    -not -path "*/.mypy_cache/*" \
    -not -path "*/.pytest_cache/*" \
    -type f \
    2>/dev/null | \
    sort | \
    while read -r file; do
        # Clean up the path for display (remove leading ./)
        clean_path="${file#./}"
        add_file "$file"
    done

# Add footer with statistics
echo "" >> "$OUTPUT_FILE"
echo "========================================" >> "$OUTPUT_FILE"
echo "EXPORT SUMMARY" >> "$OUTPUT_FILE"
echo "========================================" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# Calculate statistics
total_lines=$(wc -l < "$OUTPUT_FILE")
py_files=$(find ./cje -name "*.py" \
    -not -path "*/__pycache__/*" \
    -not -path "*/test_*" \
    -not -name "test_*.py" \
    -not -path "*/tests/*" \
    -not -path "*/.mypy_cache/*" \
    -not -path "*/.pytest_cache/*" \
    -type f 2>/dev/null | wc -l | tr -d ' ')

file_size=$(du -h "$OUTPUT_FILE" | cut -f1)

echo "Total Python files included: $py_files" >> "$OUTPUT_FILE"
echo "Total lines in export: $total_lines" >> "$OUTPUT_FILE"
echo "File size: $file_size" >> "$OUTPUT_FILE"
echo "Generated: $(date)" >> "$OUTPUT_FILE"

# Output summary to terminal
echo ""
echo "✓ Created $OUTPUT_FILE"
echo "  Python files: $py_files"
echo "  Total lines: $total_lines"
echo "  File size: $file_size"
echo ""
echo "Ready to copy/paste into your chatbot!"
echo "Tip: Use 'cat $OUTPUT_FILE | pbcopy' to copy to clipboard on macOS"