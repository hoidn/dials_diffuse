// == BEGIN IDL ==
module src.diffusepipe.diagnostics {

    # @depends_on_resource(type="FileSystem", purpose="Reading EXPT file; Writing output NPZ/text files for q-values")
    # @depends_on_type(src.diffusepipe.types.OperationOutcome)
    interface QValueCalculator {
        // Preconditions:
        // - `experiment_file_path` (DIALS .expt) must exist and be readable.
        // - The directory for `output_file_prefix` must be writable.
        // Postconditions:
        // - Output files (e.g., `[output_file_prefix]_qx.npy`, `[output_file_prefix]_qy.npy`, `[output_file_prefix]_qz.npy`) containing the q-vector components for each pixel are created.
        // - Returns an `OperationOutcome` detailing success or failure. `output_artifact_paths` will map component names (e.g., "qx_map") to their file paths.
        // Behavior:
        // - Loads the DIALS experiment model (detector geometry, beam parameters) from `experiment_file_path`.
        // - For each panel in the detector model:
        //   - Creates arrays to store qx, qy, qz values, matching the panel's pixel dimensions.
        //   - Iterates through every pixel (fast_scan_idx, slow_scan_idx) on the panel.
        //   - For each pixel, calculates its laboratory coordinates.
        //   - Calculates the scattered wavevector $\mathbf{k}_{out}$ from the sample to the pixel.
        //   - Calculates the scattering vector $\mathbf{q} = \mathbf{k}_{out} - \mathbf{k}_{in}$ (where $\mathbf{k}_{in}$ is from the beam model).
        //   - Stores the components (qx, qy, qz) of $\mathbf{q}$ in the respective arrays.
        // - Saves the qx, qy, and qz arrays to separate files (e.g., NumPy .npy format) using `output_file_prefix`.
        // @raises_error(condition="InputFileMissing", description="The .expt file was not found or is unreadable.")
        // @raises_error(condition="DIALSModelError", description="Error parsing the DIALS experiment file or accessing detector/beam models.")
        // @raises_error(condition="OutputWriteError", description="Failed to write one or more output q-value files.")
        src.diffusepipe.types.OperationOutcome calculate_q_values_for_pixels(
            string experiment_file_path,
            string output_file_prefix // Prefix for output filenames, e.g., "/path/to/output/image01"
        );
    }
}
// == END IDL ==
