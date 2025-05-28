// == BEGIN IDL ==
module src.diffusepipe.types {

    // Configuration for DIALS processing steps
    struct DIALSProcessingConfig {
        unit_cell: string; // Example: "27.424,32.134,34.513,88.66,108.46,111.88"
        space_group: string; // Example: "P1"
        find_spots_phil_file: string; // Path to spot finding PHIL file
        refinement_phil_file: optional string; // Path to refinement PHIL file
        min_spot_size: int; // Minimum spot size for dials.find_spots
    }

    // Parameters for the data extraction process
    struct ExtractionParameters {
        min_res: optional float; // d_max in Angstrom for diffuse data
        max_res: optional float; // d_min in Angstrom for diffuse data
        min_intensity: optional float; // Minimum pixel intensity for diffuse data
        max_intensity: optional float; // Maximum pixel intensity (saturation)
        gain: float; // Detector gain
        cell_length_tol: float; // Tolerance for cell length consistency check
        cell_angle_tol: float; // Tolerance for cell angle consistency check
        orient_tolerance_deg: float; // Orientation tolerance vs external PDB
        pixel_step: int; // Step for processing pixels (e.g., 1 for every pixel)
        lp_correction_enabled: boolean; // Enable/disable Lorentz-Polarization correction
        subtract_background_value: optional float; // Constant value to subtract from pixels
        plot_diagnostics: boolean; // Whether to generate diagnostic plots
        verbose_python: boolean; // Enable verbose output from Python scripts
    }

    // Overall configuration for the pipeline orchestrator
    struct PipelineConfig {
        dials_processing: DIALSProcessingConfig;
        extraction_params: ExtractionParameters;
        run_diagnostics: boolean; // Whether to run diagnostic scripts
        // Note: root_dir, log_summary_path etc. are typically handled by the orchestrator's runtime environment
        // or could be added here if they are truly configurable aspects of the *pipeline's behavior*
        // rather than runtime operational details.
    }

    // Represents a set of related input/output files for a processing step
    struct FileSet {
        experiment_file: optional string; // Path to DIALS .expt file
        reflection_file: optional string; // Path to DIALS .refl file
        image_file: optional string;      // Path to CBF image file
        bragg_mask_file: optional string; // Path to Bragg mask pickle file
        external_pdb_file: optional string; // Path to external PDB file for checks
    }

    // Outcome of processing a single CBF file through the main pipeline
    struct ProcessingOutcome {
        input_file_path: string; // Path to the original CBF file processed
        status: string; // e.g., "SUCCESS", "DIALS_FAILED", "EXTRACTION_FAILED", "DIAGNOSTICS_WARNED"
        message: optional string; // Details about success or failure
        working_directory: string; // Path to the dedicated working directory for this file
        output_npz_path: optional string; // Path to the generated NPZ file if extraction was successful
        // Add paths to specific log files if needed, e.g., dials_import_log, extraction_log
    }

    // Generic outcome for operations within components
    struct OperationOutcome {
        status: string; // e.g., "SUCCESS", "FAILURE"
        message: optional string; // Informative message
        error_details: optional map<string, string>; // Structured error information
        // Can include paths to generated files if applicable, e.g., plot_file_path
        output_artifact_paths: optional map<string, string>;
    }
}
// == END IDL ==
