// == BEGIN IDL ==
module src.diffusepipe.orchestration {

    # @depends_on_resource(type="ExternalTool:DIALS", purpose="Executing DIALS command-line tools: import, find_spots, index, refine, generate_mask")
    # @depends_on_resource(type="FileSystem", purpose="Managing working directories, reading/writing intermediate files, logs")
    # @depends_on([src.diffusepipe.extraction.DataExtractor], [src.diffusepipe.diagnostics.ConsistencyChecker], [src.diffusepipe.diagnostics.QValueCalculator])
    # @depends_on_type(src.diffusepipe.types.PipelineConfig)
    # @depends_on_type(src.diffusepipe.types.ProcessingOutcome)
    # @depends_on_type(src.diffusepipe.types.FileSet)
    interface PipelineOrchestrator {
        // Preconditions:
        // - `config` contains valid pipeline settings, including paths to PHIL files.
        // - `cbf_files_to_process` is a list of existing, readable CBF file paths.
        // - `external_pdb_for_check` is a path to an existing, readable PDB file.
        // - DIALS environment must be configured and executables accessible from the execution environment.
        // - The root directory for processing (where subdirectories will be created) is implicitly the current working directory or managed by the implementation.
        // Postconditions:
        // - For each input CBF file, a corresponding `ProcessingOutcome` is returned.
        // - Processing attempts are made in dedicated subdirectories for each CBF file.
        // - DIALS intermediate files, diffuse data NPZ files (if successful), and diagnostic outputs are generated in respective subdirectories.
        // - A summary log file is created in the root processing directory detailing the outcomes for all images.
        // Behavior:
        // - Initializes a summary log.
        // - Iterates through each path in `cbf_files_to_process`:
        //   1. Validates the existence of the CBF file.
        //   2. Creates a unique working directory for the current CBF file (e.g., based on CBF filename).
        //   3. Executes the DIALS processing sequence within the working directory:
        //      a. `dials.import` using the CBF file.
        //      b. `dials.find_spots` using `imported.expt` and `config.dials_processing.find_spots_phil_file`, `config.dials_processing.min_spot_size`.
        //      c. `dials.index` using `imported.expt`, `strong.refl`, `config.dials_processing.unit_cell`, and `config.dials_processing.space_group`.
        //      d. `dials.refine` using `indexed_initial.expt`, `indexed_initial.refl`, and `config.dials_processing.refinement_phil_file` (or default refinement strategy if PHIL is not provided).
        //      e. `dials.generate_mask` using the refined experiment and reflection files.
        //   4. Checks for successful completion and existence of output files at each DIALS step. If a step fails, logs error and proceeds to next CBF file.
        //   5. If all DIALS steps are successful:
        //      a. Constructs `FileSet` and `ExtractionParameters` from `config` and DIALS outputs.
        //      b. Invokes `DataExtractor.extract_diffuse_data`.
        //      c. If extraction is successful and `config.run_diagnostics` is true:
        //         i. Invokes `ConsistencyChecker.check_q_consistency`.
        //         ii. Invokes `QValueCalculator.calculate_q_values_for_pixels`.
        //   6. Records the overall outcome for the current CBF file in the summary log and in the list of `ProcessingOutcome`.
        // - Finalizes and saves the summary log.
        // @raises_error(condition="InvalidPipelineConfiguration", description="Essential global configuration in `config` is missing or invalid (e.g., PHIL file paths).")
        // @raises_error(condition="DIALSEnvironmentError", description="DIALS executables are not found or DIALS environment is not correctly set up.")
        // @raises_error(condition="FileSystemError", description="Cannot create working directories or write log files.")
        list<src.diffusepipe.types.ProcessingOutcome> process_cbf_dataset(
            src.diffusepipe.types.PipelineConfig config,
            list<string> cbf_files_to_process,
            string external_pdb_for_check,
            string root_processing_directory // Directory where all processing subdirectories and summary log will be created.
        );
    }
}
// == END IDL ==
