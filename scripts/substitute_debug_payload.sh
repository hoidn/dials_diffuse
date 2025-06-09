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
    local result="$content"
    
    # Find all {path} patterns and replace them
    while [[ "$result" =~ \{([^}]+)\} ]]; do
        local file_path="${BASH_REMATCH[1]}"
        local pattern="{$file_path}"
        
        if [ -f "$file_path" ]; then
            # Create replacement with XML-wrapped file contents
            local replacement="<file path=\"$file_path\">
$(cat "$file_path")
</file>"
            # Replace the pattern with the file contents
            result="${result//"$pattern"/"$replacement"}"
        else
            # Replace with warning comment
            local warning="<!-- Warning: File '$file_path' not found -->"
            result="${result//"$pattern"/"$warning"}"
        fi
    done
    
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
