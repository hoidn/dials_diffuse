// == BEGIN IDL ==
module src.diffusepipe.diagnostics {

    # @depends_on_resource(type="FileSystem", purpose="Reading DIALS .expt and .refl files; Writing diagnostic plot files")
    # @depends_on_resource(type="DIALS/dxtbx", purpose="Parsing .expt for Experiment models and .refl for ReflectionTable")
    # @depends_on_resource(type="cctbx", purpose="Crystallographic math for q-vector calculations and transformations")
    # @depends_on_type(src.diffusepipe.types.ComponentInputFiles)
    # @depends_on_type(src.diffusepipe.types.OperationOutcome)
    interface ConsistencyChecker {
        // Preconditions:
        // - `inputs.dials_expt_path` must point to an existing, readable DIALS experiment list JSON file.
        // - `inputs.dials_refl_path` must point to an existing, readable DIALS reflection table file.
        // - If `output_plot_directory` is provided, it must be a path to a writable directory.
        // Postconditions:
        // - Diagnostic information comparing q-vectors is printed to standard output (or logged).
        // - If plotting is enabled (implicitly by this component's nature) and successful, diagnostic plot image files (e.g., "q_consistency_histogram.png", "q_magnitude_scatter.png", "q_difference_heatmap.png") are saved. Paths are reported in `output_artifacts` of the returned `OperationOutcome`.
        // - The returned `OperationOutcome` object details the success or failure of the consistency check.
        // Behavior:
        // 1. Logs verbose messages if `verbose` is true.
        // 2. Loads the `ExperimentList` from `inputs.dials_expt_path` using dxtbx.
        // 3. Loads the `ReflectionTable` from `inputs.dials_refl_path` using DIALS/cctbx tools.
        // 4. Verifies that essential columns (e.g., 'miller_index', 'id', 'panel', 'xyzcal.mm', 's1') are present in the reflection table. If not, returns "FAILURE".
        // 5. Filters for indexed reflections (e.g., using `flags.indexed`). If none, returns "WARNING" or "FAILURE".
        // 6. For each selected indexed reflection:
        //    a. Retrieves its `miller_index (hkl)`, `panel_id`, `id` (experiment_id), `xyzcal.mm`, and `s1` vector.
        //    b. Gets the corresponding `Experiment` object from the `ExperimentList` using `experiment_id`.
        //    c. **Calculate q_bragg:** Using the `Experiment.crystal` model (specifically `crystal.get_A()`, the Busing-Levy matrix) and the `hkl`, compute $\mathbf{q}_{Bragg} = \mathbf{A} \cdot \mathbf{hkl}$.
        //    d. **Calculate q_pixel_recalculated:**
        //       i. Get the specific `Panel` object from `Experiment.detector[panel_id]`.
        //       ii. Convert `xyzcal.mm` (calibrated centroid in mm) to pixel coordinates (fast_px, slow_px) using `Panel.millimeter_to_pixel()`.
        //       iii. If conversion fails (e.g., centroid outside panel), attempt to use `xyzobs.px.value` from the reflection table as fallback pixel coordinates. If neither is available, skip this reflection.
        //       iv. Get the laboratory coordinates of this pixel center using `Panel.get_pixel_lab_coord((fast_px, slow_px))`.
        //       v. Obtain the incident beam vector $\mathbf{k}_{in}$ from `Experiment.beam.get_s0()` (scaled by $2\pi/\lambda$).
        //       vi. Calculate the scattered beam vector $\mathbf{k}_{out}$ from the sample origin to the pixel's lab coordinate (normalized and scaled by $2\pi/\lambda$).
        //       vii. Compute $\mathbf{q}_{pixel\_recalc} = \mathbf{k}_{out} - \mathbf{k}_{in}$.
        //    e. Store the difference vector $\Delta\mathbf{q} = \mathbf{q}_{Bragg} - \mathbf{q}_{pixel\_recalc}$ and its magnitude.
        //    f. (Optional) Calculate $\mathbf{q}_{DIALS\_pred} = \mathbf{s1} - \mathbf{s0}$ (where $\mathbf{s0}$ is from `Experiment.beam.get_s0()`) and store difference $\mathbf{q}_{DIALS\_pred} - \mathbf{q}_{pixel\_recalc}$.
        // 7. Aggregates statistics for all calculated $|\Delta\mathbf{q}|$ values (mean, median, stddev, min, max).
        // 8. Prints a summary of these statistics. If `verbose` is true, prints details for reflections with large differences.
        // 9. **Generate Plots:**
        //    a. Histogram of $|\Delta\mathbf{q}|$ magnitudes.
        //    b. Scatter plot of $|\mathbf{q}_{Bragg}|$ vs. $|\mathbf{q}_{pixel\_recalc}|$.
        //    c. Heatmap of $|\Delta\mathbf{q}|$ projected onto the detector plane (using pixel coordinates).
        //    d. Saves plots to `output_plot_directory` (or CWD if null). Stores paths in `OperationOutcome.output_artifacts`.
        // 10. Returns `OperationOutcome` with status "SUCCESS" (or "WARNING" if issues like no indexed reflections were found but the script ran).
        // @raises_error(condition="InputFileError", description="Failure to read or parse the DIALS .expt or .refl file.")
        // @raises_error(condition="DIALSDataFormatError", description="Essential columns are missing from the reflection table, or DIALS models are incomplete.")
        // @raises_error(condition="PlottingLibraryError", description="An error occurred during plot generation, possibly due to matplotlib issues or missing dependencies.")
        // @raises_error(condition="FileSystemError", description="Cannot write plot files to the specified `output_plot_directory`.")
        src.diffusepipe.types.OperationOutcome check_q_consistency(
            src.diffusepipe.types.ComponentInputFiles inputs,
            boolean verbose,
            optional string output_plot_directory
        );
    }
}
// == END IDL ==
