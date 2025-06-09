#!/bin/bash

# Script to substitute {path} entries in debug payload files with actual file contents
# Usage: ./substitute_debug_payload.sh <input_file> [output_file]

set -euo pipefail

# Check arguments
if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    echo "Usage: $0 <input_file> [output_file]"
    echo "  input_file: File containing {path} template entries"
    echo "  output_file: Output file (optional, defaults to stdout)"
    exit 1
fi

INPUT_FILE="$1"
OUTPUT_FILE="${2:-}"

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file '$INPUT_FILE' not found" >&2
    exit 1
fi

# Function to substitute file contents
substitute_file_contents() {
    local content="$1"
    
    # Use a temporary file to process the content
    local temp_file=$(mktemp)
    echo "$content" > "$temp_file"
    
    # Find all {path} patterns using grep and process them
    local patterns=$(grep -o '{[^}]*}' "$temp_file" 2>/dev/null | sort -u || true)
    
    local result="$content"
    
    # Process each unique pattern
    while IFS= read -r pattern; do
        if [ -n "$pattern" ]; then
            # Extract file path by removing { and }
            local file_path="${pattern#\{}"
            file_path="${file_path%\}}"
            
            if [ -f "$file_path" ]; then
                # Create replacement with XML-wrapped file contents
                local replacement="<file path=\"$file_path\">
$(cat "$file_path")
</file>"
                # Replace all occurrences of this pattern
                result="${result//"$pattern"/"$replacement"}"
            else
                # Replace with warning comment
                local warning="<!-- Warning: File '$file_path' not found -->"
                result="${result//"$pattern"/"$warning"}"
            fi
        fi
    done <<< "$patterns"
    
    # Clean up temporary file
    rm -f "$temp_file"
    
    echo "$result"
}

# Read input file content
CONTENT=$(cat "$INPUT_FILE")

# Process the content
RESULT=$(substitute_file_contents "$CONTENT")

# Output result
if [ -n "$OUTPUT_FILE" ]; then
    echo "$RESULT" > "$OUTPUT_FILE"
    echo "Debug payload with substituted file contents written to: $OUTPUT_FILE" >&2
else
    echo "$RESULT"
fi
