// == BEGIN IDL ==
module src.diffusepipe.orchestration {

    # @depends_on_resource(type="ExternalTool:DIALS", purpose="Performing stills crystallographic processing via DIALS Python API (e.g., dials.stills_process)")
    # @depends_on_resource(type="FileSystem", purpose="Managing working directories, creating subdirectories, reading/writing logs and intermediate files")
    # @depends_on([src.diffusepipe.extraction.DataExtractor], [src.diffusepipe.diagnostics.ConsistencyChecker], [src.diffusepipe.diagnostics.QValueCalculator])
    # @depends_on_type(src.diffusepipe.types.StillsPipelineConfig)
    # @depends_on_type(src.diffusepipe.types.StillProcessingOutcome)
    # @depends_on_type(src.diffusepipe.types.ComponentInputFiles) // Used internally to prepare inputs for other components
    # @depends_on_type(src.diffusepipe.types.OperationOutcome)    // Used internally to interpret results from other components
    interface StillsPipelineOrchestrator {
        // Preconditions:
        // - Each path in `cbf_image_paths` must point to an existing, readable CBF file.
        // - `config` must be a valid `StillsPipelineConfig` object, including valid PHIL configuration for `dials.stills_process` within `config.dials_stills_process_config`.
        // - `root_output_directory` must be a path to a writable directory (it will be created if it doesn't exist).
        // - The DIALS Python environment must be correctly configured and accessible.
        // Postconditions:
        // - A subdirectory is created within `root_output_directory` for each successfully initiated CBF processing attempt.
        // - DIALS intermediate files (e.g., `integrated.expt`, `integrated.refl`, `shoeboxes.refl` if configured), extracted diffuse data (NPZ), and diagnostic outputs are stored within these respective subdirectories.
        // - A summary log file (e.g., `pipeline_summary.log`) is created in `root_output_directory` detailing the outcome for each image.
        // - Returns a list of `StillProcessingOutcome` objects, one for each input CBF file.
        // Behavior:
        // - Creates `root_output_directory` if it doesn't exist.
        // - Initializes a summary log file within `root_output_directory`.
        // - **Data Type Detection and Processing Route Selection:**
        //   1. Performs data type detection on the first image to determine `processing_route` ("stills" or "sequence") based on CBF header analysis and `config.dials_stills_process_config.force_processing_mode`.
        //   2. If the processing route is "sequence", calls `DIALSSequenceProcessAdapter.process_sequence()` once with the entire list of images.
        //   3. If the processing route is "stills", loops through each image, processing the first image normally and using its result as a reference for subsequent images.
        // - **True Sequence Processing (for sequence route):**
        //   1. Creates a single working subdirectory for the entire sequence.
        //   2. Calls `DIALSSequenceProcessAdapter.process_sequence()` with the full list of `cbf_image_paths`.
        //   3. The adapter processes all images as a cohesive dataset using DIALS's native scan-varying refinement, ensuring consistent crystal orientation across all frames.
        //   4. Returns a single composite ExperimentList (with one Experiment containing the entire scan) and reflection_table (with reflections from all images).
        //   5. Individual extraction outcomes are created for each image using the shared sequence processing results.
        // - **Reference-Based Stills Processing (for stills route):**
        //   1. For the first image in `cbf_image_paths`:
        //      a. Creates a unique working subdirectory inside `root_output_directory` (e.g., named using the CBF filename).
        //      b. Initializes a `StillProcessingOutcome` for the current image.
        //      c. Calls `DIALSStillsProcessAdapter.process_still()` with the individual image (no reference).
        //   2. For subsequent images:
        //      a. Uses the successfully processed result from the first image as a reference geometry.
        //      b. Calls `DIALSStillsProcessAdapter.process_still()` with the `base_expt_path` parameter set to the first image's experiment output.
        //   3. **DIALS Processing Stage:**
        //      a. Initializes the DIALS adapter with the PHIL parameters derived from `config.dials_stills_process_config`.
        //      b. The adapter imports the `cbf_path` into a DIALS `ExperimentList`.
        //      c. The adapter invokes its main processing method to perform spot finding, indexing, refinement, and integration.
        //      d. If the DIALS adapter reports failure, sets `StillProcessingOutcome.status` to "FAILURE_DIALS", records details in `dials_outcome`, logs to summary, and proceeds to the next CBF file.
        //      e. If successful, the adapter retrieves the `integrated.expt` (containing `Crystal_i`) and `integrated.refl` (containing Bragg spots, partialities, and optionally shoeboxes) as DIALS Python objects.
        //   5. **Data Extraction Stage:**
        //      a. If all DIALS steps succeeded:
        //         i. Prepares `ComponentInputFiles` (paths to `cbf_path`, `indexed_refined_detector.expt`, `indexed_refined_detector.refl`, `bragg_mask.pickle`, and `config.extraction_config.external_pdb_path` if specified in `config`).
        //         ii. Defines `output_npz_path` within the working directory.
        //         iii. Calls `DataExtractor.extract_from_still(inputs, config.extraction_config, output_npz_path)`.
        //         iv. Updates `StillProcessingOutcome.extraction_outcome` and `StillProcessingOutcome.status`.
        //   6. **Diagnostics Stage (Conditional):**
        //      a. If extraction was successful (`extraction_outcome.status == "SUCCESS"`) and `config.run_consistency_checker` is true:
        //         i. Calls `ConsistencyChecker.check_q_consistency(inputs_for_consistency, config.extraction_config.verbose, working_directory)`.
        //         ii. Updates `StillProcessingOutcome.consistency_outcome`.
        //      b. If extraction was successful and `config.run_q_calculator` is true:
        //         i. Calls `QValueCalculator.calculate_q_map(inputs_for_qcalc, output_prefix_in_working_dir)`.
        //         ii. Updates `StillProcessingOutcome.q_calc_outcome`.
        //   7. Appends the final `StillProcessingOutcome` for the current image to the list and writes a summary to the main log.
        // - Returns the list of all `StillProcessingOutcome` objects.
        // @raises_error(condition="InvalidConfiguration", description="The provided `StillsPipelineConfig` is invalid or essential global settings are missing.")
        // @raises_error(condition="DIALSEnvironmentError", description="DIALS command-line tools are not found or the DIALS environment is not correctly sourced.")
        // @raises_error(condition="FileSystemError", description="Cannot create the `root_output_directory` or its subdirectories, or cannot write the summary log file.")
        list<src.diffusepipe.types.StillProcessingOutcome> process_stills_batch(
            list<string> cbf_image_paths,
            src.diffusepipe.types.StillsPipelineConfig config,
            string root_output_directory
        );
    }
}
// == END IDL ==
