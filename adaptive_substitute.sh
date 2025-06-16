#!/bin/bash
# Script to substitute {path} entries in context packages with actual file contents
# while limiting output size based on importance scores
# Usage: ./adaptive_substitute.sh <input_file> <max_lines> [output_file]
set -euo pipefail

# Check arguments
if [ $# -lt 2 ] || [ $# -gt 3 ]; then
    echo "Usage: $0 <input_file> <max_lines> [output_file]"
    echo "  input_file: File containing JSON context package with {path} template entries"
    echo "  max_lines: Maximum number of lines allowed in the output file"
    echo "  output_file: Output file (optional, defaults to stdout)"
    exit 1
fi

INPUT_FILE="$1"
MAX_LINES="$2"
OUTPUT_FILE="${3:-}"

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file '$INPUT_FILE' not found" >&2
    exit 1
fi

# Check for required dependencies
for cmd in jq bc; do
    if ! command -v "$cmd" &> /dev/null; then
        echo "Error: $cmd is required but not installed. Please install $cmd." >&2
        exit 1
    fi
done

# Data structures to store file information
declare -A file_scores
declare -A file_lines
declare -A file_exists

# Extract file paths and scores from JSON
extract_file_info() {
    local json_content="$1"
    
    echo "Extracting file information..." >&2
    
    # Use jq to extract the included_files array
    local included_files=$(echo "$json_content" | jq -r '.context_package.included_files[]? | select(.path and .score) | "\(.path)|\(.score)"' 2>/dev/null || echo "")
    
    # Process each file path and score
    while IFS="|" read -r path score; do
        if [ -n "$path" ] && [ -n "$score" ]; then
            file_scores["$path"]="$score"
            
            # Check if file exists
            if [ -f "$path" ]; then
                file_exists["$path"]="true"
                
                # Count lines (including XML wrapper overhead)
                local actual_lines=$(wc -l < "$path")
                file_lines["$path"]=$((actual_lines + 2))  # +2 for XML tags
            else
                file_exists["$path"]="false"
                file_lines["$path"]=1  # One line for warning
            fi
        fi
    done <<< "$included_files"
    
    # Get the count of files safely
    set +u
    echo "Found ${#file_scores[@]} files to process" >&2
    set -u
}

# Determine the optimal threshold score to keep total lines under MAX_LINES
determine_threshold() {
    local total_lines=0
    local base_content_lines=$(echo "$CONTENT" | wc -l)
    local available_lines=$((MAX_LINES - base_content_lines))
    
    echo "Base content has $base_content_lines lines" >&2
    echo "Available lines for file content: $available_lines" >&2
    
    # Create a sorted list of files by score (highest to lowest)
    local sorted_files=$(for path in "${!file_scores[@]}"; do
        echo "${file_scores[$path]} $path"
    done | sort -rn)
    
    # If no files found, return minimum threshold
    if [ -z "$sorted_files" ]; then
        echo "No files found to process" >&2
        echo "0"
        return
    fi
    
    # Initialize threshold to the lowest score minus a small value
    local min_score=$(echo "$sorted_files" | tail -n1 | cut -d' ' -f1)
    local threshold=$(echo "$min_score - 0.1" | bc)
    
    echo "Calculating optimal threshold..." >&2
    
    # Track running total and previous valid threshold
    local current_total=0
    local prev_score=0
    
    # Calculate cumulative lines and find threshold
    while IFS=" " read -r score path; do
        # Skip non-existent files
        if [ "${file_exists[$path]:-false}" = "false" ]; then
            continue
        fi
        
        # Add this file's lines to the running total
        current_total=$((current_total + ${file_lines[$path]:-0}))
        
        echo "  Score $score: $path (${file_lines[$path]:-0} lines, cumulative: $current_total)" >&2
        
        # If we've exceeded the limit, use the previous score as threshold
        if [ "$current_total" -gt "$available_lines" ]; then
            if [ -n "$prev_score" ]; then
                # Use previous score as the threshold
                threshold="$prev_score"
                echo "Exceeded limit at score $score, using previous score $threshold as threshold" >&2
                break
            else
                # First file already exceeds limit, use its score
                threshold="$score"
                echo "First file already exceeds limit, using its score $threshold as threshold" >&2
                break
            fi
        fi
        
        # Store current score as previous for next iteration
        prev_score="$score"
        total_lines="$current_total"
    done <<< "$sorted_files"
    
    # If all files can be included, keep threshold below minimum score
    if [ "$total_lines" -le "$available_lines" ]; then
        echo "All files can be included. Threshold set to: $threshold (total $total_lines lines)" >&2
    else
        echo "Threshold score set to: $threshold (estimated $total_lines lines)" >&2
    fi
    
    echo "$threshold"
}

# Function to substitute file contents based on threshold
substitute_file_contents() {
    local content="$1"
    local threshold="$2"
    
    # Use a temporary file to process the content
    local temp_file=$(mktemp)
    echo "$content" > "$temp_file"
    
    # Find all {path} patterns using grep and process them
    local patterns=$(grep -o '{[^}]*}' "$temp_file" 2>/dev/null | sort -u || true)
    
    local result="$content"
    local included_files=0
    local excluded_files=0
    local total_lines=0
    
    echo "Processing file substitutions..." >&2
    
    # Process each unique pattern
    while IFS= read -r pattern; do
        if [ -n "$pattern" ]; then
            # Extract file path by removing { and }
            local file_path="${pattern#\{}"
            file_path="${file_path%\}}"
            
            # Check if file exists
            if [ -f "$file_path" ]; then
                local score="${file_scores[$file_path]:-0}"
                
                # Only include if score is above threshold
                if (( $(echo "$score > $threshold" | bc -l) )); then
                    if file "$file_path" | grep -q "text"; then
                        local file_line_count=$(wc -l < "$file_path")
                        total_lines=$((total_lines + file_line_count + 2))  # +2 for XML tags
                        included_files=$((included_files + 1))
                        
                        # Create replacement with XML-wrapped file contents
                        local replacement="<file path=\"$file_path\" score=\"$score\">
$(cat "$file_path")
</file>"
                        # Replace all occurrences of this pattern
                        result="${result//"$pattern"/"$replacement"}"
                        
                        echo "  Included: $file_path (score: $score, lines: $file_line_count)" >&2
                    else
                        # Replace with warning for binary files
                        local warning="<!-- Warning: File '$file_path' is binary and was skipped (score: $score) -->"
                        result="${result//"$pattern"/"$warning"}"
                        echo "  Skipped binary: $file_path" >&2
                    fi
                else
                    # Replace with score-based exclusion notice
                    local warning="<!-- File '$file_path' excluded: score $score below threshold $threshold -->"
                    result="${result//"$pattern"/"$warning"}"
                    excluded_files=$((excluded_files + 1))
                    echo "  Excluded: $file_path (score: $score)" >&2
                fi
            else
                # Replace with warning comment
                local warning="<!-- Warning: File '$file_path' not found -->"
                result="${result//"$pattern"/"$warning"}"
                echo "  Not found: $file_path" >&2
            fi
        fi
    done <<< "$patterns"
    
    # Clean up temporary file
    rm -f "$temp_file"
    
    echo "Substitution complete: $included_files files included, $excluded_files excluded" >&2
    echo "Estimated total content lines: $total_lines" >&2
    
    echo "$result"
}

# Main execution flow
echo "Processing context package: $INPUT_FILE" >&2
echo "Maximum output lines: $MAX_LINES" >&2

# Read input file content
CONTENT=$(cat "$INPUT_FILE")

# Check if content is wrapped in markdown code blocks and unwrap if necessary
if echo "$CONTENT" | head -1 | grep -q '```json'; then
    # Remove first and last lines (markdown code block markers)
    CONTENT=$(echo "$CONTENT" | sed '1d;$d')
    echo "Detected and removed markdown code block wrapper" >&2
fi

# Extract file information
extract_file_info "$CONTENT"

# Determine threshold score
THRESHOLD=$(determine_threshold)

# Process the content
RESULT=$(substitute_file_contents "$CONTENT" "$THRESHOLD")

# Output result
if [ -n "$OUTPUT_FILE" ]; then
    echo "$RESULT" > "$OUTPUT_FILE"
    echo "Context package with substituted file contents written to: $OUTPUT_FILE" >&2
    echo "Actual output size: $(wc -l < "$OUTPUT_FILE") lines" >&2
else
    echo "$RESULT"
fi

echo "Process completed successfully" >&2
