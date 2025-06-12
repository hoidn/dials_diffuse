// == BEGIN IDL ==
module src.diffusepipe.utils {

    # @depends_on_resource(type="ExternalLibrary:dxtbx", purpose="Primary CBF file parsing and scan object access")
    # @depends_on_resource(type="FileSystem", purpose="Reading CBF files for header analysis")
    # @depends_on_resource(type="Logging", purpose="Debug and warning messages for parsing results")

    // Utility interface for CBF header parsing and data type detection
    // Critical for Module 1.S.0: CBF Data Type Detection and Processing Route Selection
    interface CBFHeaderParser {

        // --- Method: get_angle_increment_from_cbf ---
        // Preconditions:
        // - `image_path` must point to a readable CBF file.
        // - The CBF file should have a valid header format (either dxtbx-parseable or standard text format).
        // Postconditions:
        // - Returns the Angle_increment value in degrees, or None if not determinable.
        // - Return value interpretation:
        //   - 0.0: True stills data (no oscillation) → route to DIALSStillsProcessAdapter
        //   - > 0.0: Sequence data (oscillation per frame) → route to DIALSSequenceProcessAdapter  
        //   - None: Could not determine → caller should default to sequence processing (safer)
        // Behavior:
        // - **Two-Phase Parsing Strategy for Robustness:**
        //   1. **Primary Method:** Uses dxtbx.load() to access scan object and get oscillation information.
        //      - More robust and handles various CBF formats correctly.
        //      - Accesses `scan.get_oscillation()[1]` for oscillation width per frame.
        //      - Handles missing scan objects gracefully (treats as stills).
        //   2. **Fallback Method:** Direct CBF header text parsing using regex.
        //      - Used when dxtbx is unavailable or fails.
        //      - Searches for "# Angle_increment <value> deg." pattern in header.
        //      - Handles case-insensitive matching and variable spacing.
        // - **Robust Error Handling:** Gracefully handles missing dxtbx, file read errors, and malformed headers.
        // - **Header-Only Processing:** Stops reading when binary data section is reached for efficiency.
        // - **Logging:** Provides detailed debug information for troubleshooting data type detection issues.
        // @raises_error(condition="IOError", description="When CBF file cannot be read or accessed")
        // @raises_error(condition="Exception", description="When both dxtbx and text parsing methods fail critically")
        static optional float get_angle_increment_from_cbf(
            string image_path    // Path to the CBF file to analyze
        );

        // --- Method: _parse_cbf_header_text (Internal) ---
        // Preconditions:
        // - `image_path` must point to a readable CBF file.
        // Postconditions:
        // - Returns the Angle_increment value parsed from header text, or None if not found.
        // Behavior:
        // - **Fallback Parsing Method:** Used when dxtbx is unavailable or primary method fails.
        // - **Regex Pattern Matching:** Uses flexible regex to match "Angle_increment" lines with variations:
        //   - Case-insensitive matching (Angle_increment, angle_increment, ANGLE_INCREMENT)
        //   - Variable whitespace handling
        //   - Optional "deg." suffix handling
        //   - Support for positive/negative values and decimals
        // - **Efficient File Reading:** Reads CBF files in chunks (16KB) to handle large files efficiently.
        // - **Binary Section Detection:** Stops parsing when reaching the binary data portion of CBF files.
        // - **Error Recovery:** Continues searching if individual line parsing fails.
        // @raises_error(condition="IOError", description="When CBF file cannot be read")
        static optional float _parse_cbf_header_text(
            string image_path    // Path to the CBF file to parse
        );
    }
}
// == END IDL ==