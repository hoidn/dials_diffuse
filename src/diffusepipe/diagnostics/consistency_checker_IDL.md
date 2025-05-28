// == BEGIN IDL ==
module src.diffusepipe.diagnostics {

    # @depends_on_resource(type="FileSystem", purpose="Reading EXPT and REFL files; Writing log and plot files")
    # @depends_on_type(src.diffusepipe.types.OperationOutcome)
    # @depends_on_type(src.diffusepipe.types.FileSet)
    interface ConsistencyChecker {
        // Preconditions:
        // - `input_files.experiment_file` (DIALS .expt) must exist and be readable.
        // - `input_files.reflection_file` (DIALS .refl) must exist and be readable.
        // - If `output_plot_directory` is provided, it must be a writable directory path.
        // Postconditions:
        // - Diagnostic information is printed to standard output (or logged if part of a larger system).
        // - If plotting is enabled and successful, diagnostic plot files (e.g., q_consistency_check_direct_recalc.png, q_difference_heatmap.png) are saved in `output_plot_directory` (or CWD if not specified).
        // - Returns an `OperationOutcome` detailing success or failure. `output_artifact_paths` may contain paths to generated plots.
        // Behavior:
        // - Logs verbose output if `verbose_output` is true.
        // - Loads the experiment list from `input_files.experiment_file`.
        // - Loads the reflection table from `input_files.reflection_file`.
        // - Verifies presence of essential columns in the reflection table.
        // - Selects indexed reflections.
        // - For each indexed reflection:
        //   1. Retrieves its Miller index, panel ID, and calibrated coordinates (xyzcal.mm).
        //   2. Calculates `q_bragg` using the crystal model (A-matrix and Miller index).
        //   3. Converts `xyzcal.mm` to pixel coordinates (fast_px, slow_px). If this fails, attempts to use `xyzobs.px.value`.
        //   4. Recalculates `q_pixel_recalculated` directly from the pixel coordinates, beam model, and panel model.
        //   5. Calculates the difference vector `q_bragg - q_pixel_recalculated` and its magnitude.
        //   6. Optionally, calculates `q_pred_dials = s1_vec - s0_vec` from the reflection table and compares it with `q_pixel_recalculated`.
        // - Aggregates statistics on the differences (mean, median, stddev, min, max).
        // - Prints a summary of these statistics.
        // - If plotting is enabled (implicitly, as this script often generates plots):
        //   a. Generates a histogram of q-vector difference magnitudes.
        //   b. Generates a scatter plot comparing |q_bragg| vs |q_pixel_recalculated|.
        //   c. Generates a heatmap of q-vector differences on the detector plane.
        //   d. Saves plots to `output_plot_directory` or current working directory.
        // @raises_error(condition="InputFileMissing", description="The .expt or .refl file was not found or is unreadable.")
        // @raises_error(condition="DIALSDataError", description="Error parsing DIALS files or essential data columns are missing.")
        // @raises_error(condition="PlotGenerationError", description="An error occurred while generating diagnostic plots (e.g., matplotlib not found or plotting failed).")
        // @raises_error(condition="FileSystemError", description="Cannot write plot files to the specified directory.")
        src.diffusepipe.types.OperationOutcome check_q_consistency(
            src.diffusepipe.types.FileSet input_files,
            boolean verbose_output,
            optional string output_plot_directory // Directory to save plots; defaults to CWD if not provided.
        );
    }
}
// == END IDL ==
