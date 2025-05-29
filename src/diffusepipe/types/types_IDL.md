// == BEGIN IDL ==
module src.diffusepipe.types {

    // Configuration for DIALS command-line execution by the orchestrator
    // Behavior: Defines parameters passed to DIALS command-line tools.
    struct DIALSExecutionConfig {
        // Preconditions: Must be a valid DIALS unit cell string (e.g., "a,b,c,alpha,beta,gamma").
        unit_cell: string;

        // Preconditions: Must be a valid DIALS space group symbol (e.g., "P1", "P212121").
        space_group: string;

        // Preconditions: Path to an existing, readable PHIL file for dials.find_spots.
        find_spots_phil_path: string;

        // Preconditions: Path to an existing, readable PHIL file for dials.refine. If not provided, default DIALS refinement may be used.
        refinement_phil_path: optional string;

        // Preconditions: Must be a positive integer.
        min_spot_size: int;
    }

    // Parameters for the DataExtractor component
    // Behavior: Defines all settings controlling the diffuse data extraction process from a single image.
    struct ExtractionConfig {
        // Behavior: Low-resolution limit (maximum d-spacing in Angstroms). Data beyond this (smaller |q|) is excluded.
        min_res: optional float;

        // Behavior: High-resolution limit (minimum d-spacing in Angstroms). Data beyond this (larger |q|) is excluded.
        max_res: optional float;

        // Behavior: Minimum pixel intensity (after gain, corrections, background subtraction) to be included.
        min_intensity: optional float;

        // Behavior: Maximum pixel intensity (after gain, corrections, background subtraction) to be included (e.g., to filter saturated pixels).
        max_intensity: optional float;

        // Preconditions: Must be a positive float.
        // Behavior: Detector gain factor applied to raw pixel intensities.
        gain: float;

        // Preconditions: Must be a non-negative float (e.g., 0.01 for 1% tolerance).
        // Behavior: Fractional tolerance for comparing DIALS-derived cell lengths with an external PDB reference.
        cell_length_tol: float;

        // Preconditions: Must be a non-negative float (e.g., 0.1 for 0.1 degrees tolerance).
        // Behavior: Tolerance in degrees for comparing DIALS-derived cell angles with an external PDB reference.
        cell_angle_tol: float;

        // Preconditions: Must be a non-negative float.
        // Behavior: Tolerance in degrees for comparing DIALS-derived crystal orientation with an external PDB reference.
        orient_tolerance_deg: float;

        // Preconditions: Must be a positive integer.
        // Behavior: Process every Nth pixel (e.g., 1 for all pixels, 2 for every other).
        pixel_step: int;

        // Behavior: If true, a simplified Lorentz-Polarization correction is applied.
        lp_correction_enabled: boolean;

        // Behavior: Path to a pre-processed background image/map (e.g., NPZ or image format) to be subtracted pixel-wise.
        // This takes precedence over `subtract_constant_background_value` if both are provided.
        subtract_measured_background_path: optional string;

        // Behavior: A constant value to be subtracted from all pixels if `subtract_measured_background_path` is not used.
        subtract_constant_background_value: optional float;

        // Behavior: If true, diagnostic plots (e.g., q-distribution, intensity histograms) are generated.
        plot_diagnostics: boolean;

        // Behavior: If true, enables verbose logging output during extraction.
        verbose: boolean;
    }

    // Overall pipeline configuration for processing stills
    // Behavior: Encapsulates all settings for the StillsPipelineOrchestrator.
    struct StillsPipelineConfig {
        dials_exec_config: DIALSExecutionConfig;
        extraction_config: ExtractionConfig;
        run_consistency_checker: boolean; // If true, ConsistencyChecker is run after successful extraction.
        run_q_calculator: boolean;      // If true, QValueCalculator is run after successful extraction.
    }

    // Represents a set of related input file paths for a component
    // Behavior: Standardized way to pass file dependencies to components. Fields are optional to allow flexibility for different components.
    struct ComponentInputFiles {
        // Behavior: Path to the primary CBF image file being processed.
        cbf_image_path: optional string;

        // Behavior: Path to the DIALS experiment list JSON file (.expt) corresponding to the cbf_image_path.
        dials_expt_path: optional string;

        // Behavior: Path to the DIALS reflection table file (.refl) corresponding to the cbf_image_path.
        dials_refl_path: optional string;

        // Behavior: Path to the DIALS-generated Bragg mask pickle file.
        bragg_mask_path: optional string;

        // Behavior: Path to an external PDB file used for consistency checks (e.g., unit cell, orientation).
        external_pdb_path: optional string;
    }

    // Generic outcome for operations within components
    // Behavior: Standardized return type for component methods indicating success/failure and providing details.
    struct OperationOutcome {
        // Preconditions: Must be one of "SUCCESS", "FAILURE", "WARNING".
        status: string;
        message: optional string; // Human-readable message about the outcome.
        error_code: optional string; // A machine-readable code for specific error types.
        // Behavior: A map where keys are artifact names (e.g., "npz_file", "consistency_plot") and values are their file paths.
        output_artifacts: optional map<string, string>;
    }

    // Outcome for processing a single still image through the main pipeline
    // Behavior: Summarizes the results of all processing stages for a single input CBF image.
    struct StillProcessingOutcome {
        input_cbf_path: string; // Path to the original CBF file.
        // Preconditions: Must be one of "SUCCESS_ALL", "SUCCESS_DIALS_ONLY", "SUCCESS_EXTRACTION_ONLY", "FAILURE_DIALS", "FAILURE_EXTRACTION", "FAILURE_DIAGNOSTICS".
        status: string;
        message: optional string; // Overall message for this image's processing.
        working_directory: string; // Path to the dedicated working directory.
        dials_outcome: OperationOutcome; // Outcome of the DIALS processing steps.
        extraction_outcome: OperationOutcome; // Outcome of the DataExtractor.
        consistency_outcome: optional OperationOutcome; // Outcome of the ConsistencyChecker, if run.
        q_calc_outcome: optional OperationOutcome;      // Outcome of the QValueCalculator, if run.
    }
}
// == END IDL ==
