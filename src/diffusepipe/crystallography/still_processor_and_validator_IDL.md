// == BEGIN IDL ==
module src.diffusepipe.crystallography {

    # @depends_on([src.diffusepipe.adapters.DIALSStillsProcessAdapter])
    # @depends_on([src.diffusepipe.adapters.DIALSSequenceProcessAdapter])
    # @depends_on([src.diffusepipe.crystallography.ModelValidator])
    # @depends_on([src.diffusepipe.utils.cbf_utils]) // For CBF header parsing
    # @depends_on_resource(type="FileSystem", purpose="Reading CBF files and base experiment files")
    # @depends_on_type(src.diffusepipe.types.DIALSStillsProcessConfig)
    # @depends_on_type(src.diffusepipe.types.ExtractionConfig)
    # @depends_on_type(src.diffusepipe.types.OperationOutcome)
    # @depends_on_type(src.diffusepipe.exceptions.DIALSProcessingError) // For use in @raises_error

    interface StillProcessorAndValidatorComponent {

        // --- Method: _determine_processing_route ---
        // Preconditions:
        // - `image_path` must point to a readable CBF file.
        // - `config` must be a valid DIALSStillsProcessConfig object.
        // Postconditions:
        // - Returns tuple: (processing_route: string, selected_adapter: object).
        // - `processing_route` is either "stills" or "sequence".
        // - `selected_adapter` is the appropriate adapter instance for the determined route.
        // Behavior:
        // - **Implements Module 1.S.0: CBF Data Type Detection and Processing Route Selection.**
        // - **Step 1:** If `config.force_processing_mode` is set ("stills" or "sequence"), uses that override.
        // - **Step 2:** If no override, calls `get_angle_increment_from_cbf(image_path)` to parse CBF header.
        // - **Step 3:** Route selection logic:
        //   - If `Angle_increment == 0.0`: Returns ("stills", DIALSStillsProcessAdapter).
        //   - If `Angle_increment > 0.0`: Returns ("sequence", DIALSSequenceProcessAdapter).
        //   - If `Angle_increment` is negative, null, or parsing fails: Defaults to ("sequence", DIALSSequenceProcessAdapter) as safer option.
        // - **Logging:** Records detection results and routing decisions for debugging.
        tuple<string, object> _determine_processing_route(
            string image_path,                           // Path to CBF image file
            src.diffusepipe.types.DIALSStillsProcessConfig config // Configuration with optional processing mode override
        );

        // --- Method: process_and_validate_still ---
        // Preconditions:
        // - `image_path` must point to a readable CBF file.
        // - `config` and `extraction_config` must be valid configuration objects.
        // - `base_experiment_path` (if provided) must point to a readable DIALS experiment file.
        // - `external_pdb_path` (if provided) must point to a readable PDB file.
        // - `output_dir` (if provided) must be a writable directory path.
        // Postconditions:
        // - Returns an OperationOutcome with status indicating success or specific failure mode.
        // - On success, output_artifacts contains DIALS objects, validation results, and routing information.
        // - On failure, includes appropriate error codes and diagnostic information.
        // Behavior:
        // - **Step 1:** Calls `_determine_processing_route` to select appropriate DIALS adapter (Module 1.S.0).
        // - **Step 2:** Calls selected adapter's processing method to perform DIALS processing (either `process_still` for stills or `process_sequence` for sequences).
        // - **Step 3:** If DIALS processing succeeds, calls `ModelValidator.validate_geometry` with all validation checks.
        // - **Step 4:** Packages results into OperationOutcome with comprehensive output_artifacts:
        //   - `experiment`: DIALS Experiment object
        //   - `reflections`: DIALS reflection_table object  
        //   - `validation_passed`: Boolean result of geometric validation
        //   - `validation_metrics`: Detailed validation metrics from ModelValidator
        //   - `processing_route_used`: String indicating which processing pathway was used
        //   - `log_messages`: Processing logs for debugging
        // - **Status Codes:**
        //   - "SUCCESS": All processing and validation passed
        //   - "FAILURE_DIALS_PROCESSING": DIALS adapter failed
        //   - "FAILURE_GEOMETRY_VALIDATION": DIALS succeeded but validation failed
        // @raises_error(condition="DIALSProcessingError", description="If the selected DIALS adapter raises an unexpected exception during processing.")
        src.diffusepipe.types.OperationOutcome process_and_validate_still(
            string image_path,                                      // Path to CBF image file
            src.diffusepipe.types.DIALSStillsProcessConfig config,  // DIALS processing configuration
            src.diffusepipe.types.ExtractionConfig extraction_config, // Validation tolerance parameters
            optional string base_experiment_path,                   // Optional base experiment geometry
            optional string external_pdb_path,                     // Optional PDB file for validation
            optional string output_dir                              // Optional directory for diagnostic outputs
        );

        // --- Method: process_still (Legacy API) ---
        // Preconditions:
        // - `image_path` must point to a readable CBF file.
        // - `config` must be a valid DIALSStillsProcessConfig object.
        // - `base_experiment_path` (if provided) must point to a readable DIALS experiment file.
        // Postconditions:
        // - Returns an OperationOutcome from DIALS processing only (no validation).
        // - Uses the same routing logic as `process_and_validate_still` but skips validation.
        // Behavior:
        // - **Legacy API:** Provides backward compatibility for callers that only need DIALS processing.
        // - Calls `_determine_processing_route` to select appropriate adapter.
        // - Calls selected adapter's processing method (either `process_still` for stills or `process_sequence` for sequences).
        // - Returns OperationOutcome with DIALS results but without validation metrics.
        // - Maintains same routing logic and error handling as the full method.
        src.diffusepipe.types.OperationOutcome process_still(
            string image_path,                                      // Path to CBF image file
            src.diffusepipe.types.DIALSStillsProcessConfig config,  // DIALS processing configuration
            optional string base_experiment_path                    // Optional base experiment geometry
        );
    }
}
// == END IDL ==