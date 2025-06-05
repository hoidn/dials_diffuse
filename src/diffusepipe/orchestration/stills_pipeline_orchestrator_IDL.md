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
        // - For each `cbf_path` in `cbf_image_paths`:
        //   1. Creates a unique working subdirectory inside `root_output_directory` (e.g., named using the CBF filename).
        //   2. Initializes a `StillProcessingOutcome` for the current image.
        //   3. **DIALS Processing Stage (using `dials.stills_process` Python API via an adapter):**
        //      a. Initializes a DIALS `Processor` (from `dials.command_line.stills_process`) adapter with the PHIL parameters derived from `config.dials_stills_process_config`.
        //      b. The adapter imports the `cbf_path` into a DIALS `ExperimentList`.
        //      c. The adapter invokes the `Processor`'s main method (e.g., `process_experiments`) on the imported `ExperimentList`. This internally handles spot finding, indexing, refinement, and integration (including partiality calculation and optional shoebox output).
        //      d. The adapter logs relevant command-equivalent information (e.g., PHIL parameters used), captures DIALS logs, and monitors for successful completion.
        //      e. If the `dials.stills_process` adapter reports failure, sets `StillProcessingOutcome.status` to "FAILURE_DIALS", records details in `dials_outcome`, logs to summary, and proceeds to the next CBF file.
        //      f. If successful, the adapter retrieves the `integrated.expt` (containing `Crystal_i`) and `integrated.refl` (containing Bragg spots, partialities, and optionally shoeboxes) as DIALS Python objects. Note: Partialities from `dials.stills_process` are used for DIALS integration quality but NOT as quantitative divisors in subsequent scaling (due to accuracy limitations for true stills).
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
