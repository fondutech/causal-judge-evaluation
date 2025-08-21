#!/bin/bash

# Script to concatenate all README files in the CJE project
# Useful for providing context to LLMs or for documentation review

echo "# CJE Project Documentation - All README Files"
echo "Generated on: $(date)"
echo "=" 
echo ""

# Function to print a separator with the file path
print_separator() {
    local filepath=$1
    echo ""
    echo "# ============================================================================"
    echo "# FILE: $filepath"
    echo "# ============================================================================"
    echo ""
}

# Find and concatenate all README files (case-insensitive)
# Sort them with main README first, then by path
{
    # Main README first
    find . -maxdepth 1 -iname "readme*" -type f 2>/dev/null | head -1
    # Then all other READMEs sorted by path
    find . -path "./.git" -prune -o -path "./.*" -prune -o -mindepth 2 -iname "readme*" -type f -print 2>/dev/null | sort
} | while read -r readme_file; do
    if [ -f "$readme_file" ]; then
        # Clean up the path for display
        clean_path="${readme_file#./}"
        
        print_separator "$clean_path"
        
        # Output the file contents
        cat "$readme_file"
        
        # Add extra newline for separation
        echo ""
    fi
done

# Summary at the end
echo ""
echo "# ============================================================================"
echo "# END OF DOCUMENTATION"
echo "# ============================================================================"
echo ""

# Count statistics
readme_count=$(find . -path "./.git" -prune -o -path "./.*" -prune -o -iname "readme*" -type f -print 2>/dev/null | wc -l | tr -d ' ')
total_lines=$(find . -path "./.git" -prune -o -path "./.*" -prune -o -iname "readme*" -type f -exec wc -l {} \; 2>/dev/null | awk '{sum+=$1} END {print sum}')

echo "# Summary:"
echo "# - Total README files: $readme_count"
echo "# - Total lines of documentation: $total_lines"
echo "# - Modules documented: $(find . -path "./.git" -prune -o -path "./.*" -prune -o -iname "readme*" -type f -print 2>/dev/null | xargs dirname | sort -u | wc -l | tr -d ' ')"