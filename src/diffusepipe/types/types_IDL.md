// == BEGIN IDL ==
module src.diffusepipe.types {

    // Configuration for DIALS stills_process Python API execution by the orchestrator
    // Behavior: Defines parameters to configure dials.stills_process.
    struct DIALSStillsProcessConfig {
        // Preconditions: Path to an existing, readable PHIL file containing comprehensive
        // parameters for dials.stills_process. This is the primary way to configure it.
        stills_process_phil_path: optional string;

        // Behavior: Known unit cell for indexing, e.g., "a,b,c,alpha,beta,gamma".
        // Overrides or supplements PHIL file if provided.
        known_unit_cell: optional string;

        // Behavior: Known space group for indexing, e.g., "P1", "C2".
        // Overrides or supplements PHIL file if provided.
        known_space_group: optional string;

        // Behavior: Spot finding algorithm, e.g., "dispersion".
        // Overrides or supplements PHIL file if provided.
        spotfinder_threshold_algorithm: optional string;

        // Behavior: Minimum spot area for spot finding.
        // Overrides or supplements PHIL file if provided.
        min_spot_area: optional int;

        // Behavior: If true, ensures shoeboxes are saved by dials.stills_process (needed for some Bragg mask generation methods).
        // Overrides or supplements PHIL file if provided.
        output_shoeboxes: optional boolean;

        // Behavior: If true, ensures partialities are calculated and output by dials.stills_process.
        // Note: While partialities are still useful for DIALS integration quality assessment,
        // they will NOT be used as quantitative divisors in this pipeline's scaling (due to
        // inaccuracies for true stills). Instead, P_spot serves as a threshold filter.
        // Overrides or supplements PHIL file if provided.
        calculate_partiality: optional boolean; // Should default to true in implementation

        // Add other critical, frequently tuned dials.stills_process parameters here as needed.
        // The adapter layer will be responsible for merging these with the PHIL file content.
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

        // Preconditions: Must be a positive float (e.g., 0.01 for 0.01 Å⁻¹ tolerance).
        // Behavior: Tolerance in Å⁻¹ for q-vector consistency checks in geometric model validation.
        // Used in Module 1.S.1.Validation to compare |q_bragg - q_pixel_recalculated|.
        q_consistency_tolerance_angstrom_inv: float;

        // Preconditions: Must be a positive integer.
        // Behavior: Process every Nth pixel (e.g., 1 for all pixels, 2 for every other).
        pixel_step: int;

        // Behavior: If true, Lorentz-Polarization correction is applied using DIALS Corrections API.
        // This leverages the robust, well-tested dials.algorithms.integration.Corrections class.
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

    // Configuration for the relative scaling model (future DiffuseDataMerger component)
    // Behavior: Controls the complexity and parameters of the custom scaling model used in Module 3.S.3.
    struct RelativeScalingConfig {
        // Behavior: If true, refines a per-still (or per-group) overall multiplicative scale factor.
        // This is part of the initial default multiplicative-only model.
        refine_per_still_scale: boolean; // Should default to true

        // Behavior: If true, refines a 1D resolution-dependent multiplicative scale factor.
        // This is optionally part of the initial default multiplicative-only model.
        refine_resolution_scale_multiplicative: boolean; // Should default to false initially

        // Behavior: Number of bins for resolution-dependent scaling if enabled.
        resolution_scale_bins: optional int;

        // Behavior: If true, refines additive offset components (e.g., background terms).
        // CRITICAL: This **must** be `false` for the initial v1 implementation to avoid parameter
        // correlation issues and simplify error propagation. Enabling this requires careful
        // validation and updates to error propagation logic (see plan.md, Module 3.S.4).
        // Only enable in future versions after the multiplicative model is stable and residuals clearly justify it.
        refine_additive_offset: boolean; // Should default to false and be hard-coded to false in v1.

        // Behavior: Minimum P_spot threshold for including Bragg reflections in reference generation.
        // Reflections below this threshold are excluded to avoid poor-quality data.
        min_partiality_threshold: float; // Should default to 0.1
    }

    // Overall pipeline configuration for processing stills
    // Behavior: Encapsulates all settings for the StillsPipelineOrchestrator.
    struct StillsPipelineConfig {
        dials_stills_process_config: DIALSStillsProcessConfig; // Updated
        extraction_config: ExtractionConfig;
        relative_scaling_config: RelativeScalingConfig; // Configuration for future scaling component
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
