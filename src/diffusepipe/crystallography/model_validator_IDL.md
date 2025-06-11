// == BEGIN IDL ==
module src.diffusepipe.crystallography {

    # @depends_on([src.diffusepipe.crystallography.QConsistencyChecker])
    # @depends_on_resource(type="DIALS/dxtbx", purpose="Using Experiment and reflection_table objects, crystal models")
    # @depends_on_resource(type="FileSystem", purpose="Reading PDB files, writing diagnostic plots")
    # @depends_on_resource(type="ExternalLibrary:iotbx", purpose="PDB file parsing and crystal symmetry analysis")
    # @depends_on_type(src.diffusepipe.types.ExtractionConfig)
    # @depends_on_type(src.diffusepipe.exceptions.PDBError) // For use in @raises_error

    // Container for validation metrics and results
    struct ValidationMetrics {
        // PDB consistency check results
        pdb_cell_passed: optional boolean;         // Result of unit cell comparison (null if not performed)
        pdb_orientation_passed: optional boolean;  // Result of orientation comparison (null if not performed)
        misorientation_angle_vs_pdb: optional float; // Misorientation angle in degrees (null if not calculated)
        
        // Q-vector consistency check results  
        q_consistency_passed: optional boolean;    // Result of Q-vector validation (null if not performed)
        mean_delta_q_mag: optional float;         // Mean |Δq| value in Å⁻¹ (null if not calculated)
        max_delta_q_mag: optional float;          // Maximum |Δq| value in Å⁻¹ (null if not calculated)
        median_delta_q_mag: optional float;       // Median |Δq| value in Å⁻¹ (null if not calculated)
        num_reflections_tested: int;              // Number of reflections used in Q-vector test
        
        // Diagnostic output information
        validation_plots_generated: boolean;      // Whether diagnostic plots were created
        plot_paths: map<string, string>;          // Map of plot type to file path
    }

    interface ModelValidator {

        // --- Method: validate_geometry ---
        // Preconditions:
        // - `experiment` is a valid DIALS Experiment object containing crystal model, beam, and detector.
        // - `reflections` is a valid DIALS reflection_table object.
        // - `external_pdb_path` (if provided) must point to a readable PDB file.
        // - `extraction_config` (if provided) must contain valid tolerance parameters.
        // - `output_dir` (if provided) must be a writable directory path for diagnostic plots.
        // Postconditions:
        // - Returns a tuple: (overall_validation_passed: bool, validation_metrics: ValidationMetrics).
        // - `overall_validation_passed` is True only if Q-vector consistency passes AND PDB checks pass (if performed).
        // - `validation_metrics` contains detailed results from all validation sub-checks.
        // - Diagnostic plots are generated in `output_dir` if specified and matplotlib is available.
        // Behavior:
        // - **Primary Validation:** Performs Q-vector consistency check using QConsistencyChecker with tolerance from `extraction_config.q_consistency_tolerance_angstrom_inv` (default: 0.01 Å⁻¹).
        // - **PDB Validation (if external_pdb_path provided):**
        //   1. Calls `_check_pdb_consistency` to compare unit cell parameters using `extraction_config.cell_length_tol` and `extraction_config.cell_angle_tol`.
        //   2. Compares crystal orientation using `extraction_config.orient_tolerance_deg`.
        // - **Diagnostic Output (if output_dir provided):** Calls `_generate_diagnostic_plots` to create visualization files.
        // - **Overall Pass Criteria:** Q-vector consistency must pass, num_reflections_tested > 0, and PDB checks must not fail (if performed).
        // @raises_error(condition="PDBError", description="If PDB file processing fails or contains invalid crystal symmetry information.")
        tuple<boolean, ValidationMetrics> validate_geometry(
            object experiment,                    // DIALS Experiment object
            object reflections,                   // DIALS reflection_table object
            optional string external_pdb_path,   // Path to reference PDB file for comparison
            optional src.diffusepipe.types.ExtractionConfig extraction_config, // Tolerance parameters
            optional string output_dir           // Directory for diagnostic plot output
        );

        // --- Method: _check_pdb_consistency (Internal) ---
        // Preconditions:
        // - `experiment` contains a valid crystal model.
        // - `pdb_path` points to a readable PDB file with crystal symmetry information.
        // - Tolerance parameters are positive values.
        // Postconditions:
        // - Returns tuple: (cell_similarity_passed: bool, orientation_similarity_passed: bool, misorientation_angle_degrees: optional float).
        // Behavior:
        // - Loads PDB crystal symmetry using iotbx.pdb.
        // - Compares unit cell parameters using CCTBX `is_similar_to` method with provided tolerances.
        // - Calculates misorientation angle between experiment A-matrix and PDB reference orientation.
        // - Handles both direct and inverted hand orientations, returning the smaller angle.
        // - Returns (True, True, None) if PDB lacks crystal symmetry information.
        // @raises_error(condition="PDBError", description="If PDB file cannot be read or processed.")
        static tuple<boolean, boolean, optional float> _check_pdb_consistency(
            object experiment,        // DIALS Experiment object
            string pdb_path,         // Path to PDB file
            float cell_length_tol,   // Relative tolerance for unit cell length comparison
            float cell_angle_tol,    // Absolute tolerance in degrees for unit cell angle comparison
            float orient_tolerance_deg // Tolerance in degrees for orientation comparison
        );

        // --- Method: _check_q_consistency (Internal) ---
        // Preconditions:
        // - `experiment` and `reflections` are valid DIALS objects.
        // - `tolerance` is a positive float in Å⁻¹.
        // Postconditions:
        // - Returns tuple: (q_consistency_passed: bool, statistics: map<string, float>).
        // Behavior:
        // - Delegates to QConsistencyChecker.check_q_consistency() method.
        // - Uses the primary Q-vector validation algorithm comparing q_model vs q_observed.
        tuple<boolean, map<string, float>> _check_q_consistency(
            object experiment,       // DIALS Experiment object
            object reflections,      // DIALS reflection_table object
            float tolerance         // Q-vector consistency tolerance in Å⁻¹
        );

        // --- Method: _generate_diagnostic_plots (Internal) ---
        // Preconditions:
        // - `q_stats` contains valid Q-vector consistency statistics.
        // - `output_dir` is a writable directory path.
        // - Matplotlib is available (graceful degradation if not).
        // Postconditions:
        // - Returns map of plot type names to generated file paths.
        // - Creates simple summary visualization plots for validation results.
        // Behavior:
        // - Generates diagnostic plots showing Q-vector consistency results.
        // - Uses matplotlib with 'Agg' backend for headless operation.
        // - Creates plots with validation statistics and pass/fail indicators.
        // - Handles missing matplotlib gracefully by logging warnings.
        static map<string, string> _generate_diagnostic_plots(
            map<string, float> q_stats,     // Q-vector consistency statistics
            optional string pdb_path,       // PDB path for plot annotations
            string output_dir               // Directory for plot output
        );
    }
}
// == END IDL ==