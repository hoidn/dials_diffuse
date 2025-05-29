// == BEGIN IDL ==
module src.diffusepipe.diagnostics {

    # @depends_on_resource(type="FileSystem", purpose="Reading DIALS .expt file; Writing output q-map NumPy files")
    # @depends_on_resource(type="DIALS/dxtbx", purpose="Parsing .expt file for Experiment model")
    # @depends_on_resource(type="cctbx", purpose="Underlying geometric calculations for q-vectors")
    # @depends_on_type(src.diffusepipe.types.ComponentInputFiles) // Specifically for dials_expt_path
    # @depends_on_type(src.diffusepipe.types.OperationOutcome)
    interface QValueCalculator {
        // Preconditions:
        // - `inputs.dials_expt_path` must point to an existing, readable DIALS experiment list JSON file.
        // - The directory implied by `output_prefix_for_q_maps` must be writable.
        // Postconditions:
        // - Three NumPy array files (e.g., `[prefix]_qx.npy`, `[prefix]_qy.npy`, `[prefix]_qz.npy`) are created. Each array has dimensions matching the detector, containing the respective q-component for each pixel.
        // - The returned `OperationOutcome` object details success or failure. `output_artifacts` will map "qx_map_path", "qy_map_path", "qz_map_path" to their file paths.
        // Behavior:
        // 1. Loads the `ExperimentList` from `inputs.dials_expt_path` (typically expecting one Experiment).
        // 2. Gets the `Beam` model and `Detector` model from the `Experiment`.
        // 3. For each `Panel` in the `Detector` model (though typically one for non-multi-panel detectors):
        //    a. Initializes NumPy arrays for $q_x, q_y, q_z$ with dimensions `(panel.get_image_size()[1], panel.get_image_size()[0])`.
        //    b. Iterates through all pixel indices `(slow_scan_idx, fast_scan_idx)` of the panel.
        //    c. For each pixel:
        //       i. Gets the laboratory coordinate of the pixel center using `Panel.get_pixel_lab_coord((fast_scan_idx, slow_scan_idx))`.
        //       ii. Gets the incident beam vector $\mathbf{k}_{in}$ from `Beam.get_s0()` (scaled by $2\pi/\lambda$, where $\lambda$ is `Beam.get_wavelength()`).
        //       iii. Calculates the scattered beam vector $\mathbf{k}_{out}$ from the sample origin (assumed [0,0,0] in lab frame) to the pixel's lab coordinate. This vector is normalized and then scaled by $2\pi/\lambda$.
        //       iv. Computes the scattering vector $\mathbf{q} = \mathbf{k}_{out} - \mathbf{k}_{in}$.
        //       v. Stores the components $q_x, q_y, q_z$ into the respective NumPy arrays at `[slow_scan_idx, fast_scan_idx]`.
        // 4. Saves the $q_x, q_y, q_z$ NumPy arrays to files using `output_prefix_for_q_maps` (e.g., `[prefix]_qx.npy`).
        // 5. Returns `OperationOutcome` with status "SUCCESS" and paths to output files in `output_artifacts`.
        // @raises_error(condition="InputFileError", description="The DIALS .expt file was not found or is unreadable.")
        // @raises_error(condition="DIALSModelError", description="Error parsing the DIALS experiment file, or critical beam/detector information is missing.")
        // @raises_error(condition="CalculationError", description="An unexpected error occurred during q-vector calculations for pixels.")
        // @raises_error(condition="OutputWriteError", description="Failed to write one or more of the output q-map NumPy files.")
        src.diffusepipe.types.OperationOutcome calculate_q_map(
            src.diffusepipe.types.ComponentInputFiles inputs,
            string output_prefix_for_q_maps
        );
    }
}
// == END IDL ==
