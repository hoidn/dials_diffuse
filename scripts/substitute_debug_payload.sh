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
    
    # Find all {path} patterns and process them
    while IFS= read -r line; do
        # Check if line contains {path} pattern
        if [[ "$line" =~ \{([^}]+)\} ]]; then
            # Extract the file path
            local file_path="${BASH_REMATCH[1]}"
            
            # Check if file exists
            if [ -f "$file_path" ]; then
                # Replace the {path} with XML-wrapped file contents
                echo "<file path=\"$file_path\">"
                cat "$file_path"
                echo "</file>"
            else
                echo "<!-- Warning: File '$file_path' not found -->"
                echo "$line"
            fi
        else
            # Line doesn't contain {path}, output as-is
            echo "$line"
        fi
    done <<< "$content"
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
