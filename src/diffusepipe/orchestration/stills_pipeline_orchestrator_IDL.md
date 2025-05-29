// == BEGIN IDL ==
module src.diffusepipe.orchestration {

    # @depends_on_resource(type="ExternalTool:DIALS", purpose="Executing DIALS CLI: import, find_spots, index, refine, generate_mask")
    # @depends_on_resource(type="FileSystem", purpose="Managing working directories, creating subdirectories, reading/writing logs and intermediate files")
    # @depends_on([src.diffusepipe.extraction.DataExtractor], [src.diffusepipe.diagnostics.ConsistencyChecker], [src.diffusepipe.diagnostics.QValueCalculator])
    # @depends_on_type(src.diffusepipe.types.StillsPipelineConfig)
    # @depends_on_type(src.diffusepipe.types.StillProcessingOutcome)
    # @depends_on_type(src.diffusepipe.types.ComponentInputFiles) // Used internally to prepare inputs for other components
    # @depends_on_type(src.diffusepipe.types.OperationOutcome)    // Used internally to interpret results from other components
    interface StillsPipelineOrchestrator {
        // Preconditions:
        // - Each path in `cbf_image_paths` must point to an existing, readable CBF file.
        // - `config` must be a valid `StillsPipelineConfig` object, including valid paths for PHIL files within `config.dials_exec_config`.
        // - `root_output_directory` must be a path to a writable directory (it will be created if it doesn't exist).
        // - The environment must be configured such that DIALS command-line tools are executable.
        // Postconditions:
        // - A subdirectory is created within `root_output_directory` for each successfully initiated CBF processing attempt.
        // - DIALS intermediate files, extracted diffuse data (NPZ), and diagnostic outputs are stored within these respective subdirectories.
        // - A summary log file (e.g., `pipeline_summary.log`) is created in `root_output_directory` detailing the outcome for each image.
        // - Returns a list of `StillProcessingOutcome` objects, one for each input CBF file.
        // Behavior:
        // - Creates `root_output_directory` if it doesn't exist.
        // - Initializes a summary log file within `root_output_directory`.
        // - For each `cbf_path` in `cbf_image_paths`:
        //   1. Creates a unique working subdirectory inside `root_output_directory` (e.g., named using the CBF filename).
        //   2. Initializes a `StillProcessingOutcome` for the current image.
        //   3. **DIALS Processing Stage:**
        //      a. Executes `dials.import` on `cbf_path` within the working directory. Logs command, stdout, stderr, exit code. Updates `dials_outcome`.
        //      b. If successful, executes `dials.find_spots` using `imported.expt`, `config.dials_exec_config.find_spots_phil_path`, and `config.dials_exec_config.min_spot_size`. Logs and updates outcome.
        //      c. If successful, executes `dials.index` using `imported.expt`, `strong.refl`, `config.dials_exec_config.unit_cell`, and `config.dials_exec_config.space_group`. Logs and updates outcome.
        //      d. If successful, executes `dials.refine` using `indexed_initial.expt`, `indexed_initial.refl`, and `config.dials_exec_config.refinement_phil_path` (if provided). Logs and updates outcome.
        //      e. If successful, executes `dials.generate_mask` using the refined experiment and reflection files. Logs and updates outcome.
        //      f. If any DIALS step fails, sets `StillProcessingOutcome.status` to "FAILURE_DIALS", records details in `dials_outcome`, logs to summary, and proceeds to the next CBF file.
        //   4. **Data Extraction Stage:**
        //      a. If all DIALS steps succeeded:
        //         i. Prepares `ComponentInputFiles` (paths to `cbf_path`, `indexed_refined_detector.expt`, `indexed_refined_detector.refl`, `bragg_mask.pickle`, and `config.extraction_config.external_pdb_path` if specified in `config`).
        //         ii. Defines `output_npz_path` within the working directory.
        //         iii. Calls `DataExtractor.extract_from_still(inputs, config.extraction_config, output_npz_path)`.
        //         iv. Updates `StillProcessingOutcome.extraction_outcome` and `StillProcessingOutcome.status`.
        //   5. **Diagnostics Stage (Conditional):**
        //      a. If extraction was successful (`extraction_outcome.status == "SUCCESS"`) and `config.run_consistency_checker` is true:
        //         i. Calls `ConsistencyChecker.check_q_consistency(inputs_for_consistency, config.extraction_config.verbose, working_directory)`.
        //         ii. Updates `StillProcessingOutcome.consistency_outcome`.
        //      b. If extraction was successful and `config.run_q_calculator` is true:
        //         i. Calls `QValueCalculator.calculate_q_map(inputs_for_qcalc, output_prefix_in_working_dir)`.
        //         ii. Updates `StillProcessingOutcome.q_calc_outcome`.
        //   6. Appends the final `StillProcessingOutcome` for the current image to the list and writes a summary to the main log.
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
