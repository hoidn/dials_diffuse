// == BEGIN IDL ==
module src.diffusepipe.extraction {

    # @depends_on_resource(type="FileSystem", purpose="Reading CBF, DIALS .expt, DIALS .refl (for consistency check), PDB, Bragg Mask files; Writing NPZ and diagnostic plot files")
    # @depends_on_resource(type="DIALS/dxtbx", purpose="Parsing .expt file for geometric model (beam, detector, crystal)")
    # @depends_on_resource(type="cctbx", purpose="Optional: PDB parsing, advanced geometric calculations, crystallographic math primitives")
    # @depends_on_type(src.diffusepipe.types.ExtractionConfig)
    # @depends_on_type(src.diffusepipe.types.ComponentInputFiles)
    # @depends_on_type(src.diffusepipe.types.OperationOutcome)
    interface DataExtractor {
        // Preconditions:
        // - `inputs.cbf_image_path` must point to an existing, readable CBF image file.
        // - `inputs.dials_expt_path` must point to an existing, readable DIALS experiment list JSON file.
        // - `inputs.bragg_mask_path` must point to an existing, readable pickle file containing the Bragg mask (typically a NumPy boolean array).
        // - If `inputs.external_pdb_path` is provided, it must be an existing, readable PDB file.
        // - `config` must be a valid `ExtractionConfig` object.
        // - The directory for `output_npz_path` must be writable.
        // Postconditions:
        // - If `status` in the returned `OperationOutcome` is "SUCCESS":
        //   - An NPZ file is created at `output_npz_path`. This file minimally contains 'q_vectors' (Nx3 array), 'intensities' (N array), and 'sigmas' (N array).
        //   - If `config.plot_diagnostics` is true, diagnostic plot image files may be created (paths reported in `output_artifacts`).
        // - The returned `OperationOutcome` object details the success or failure of the operation.
        // Behavior:
        // 1. Logs verbose messages if `config.verbose` is true.
        // 2. **Load Data:**
        //    a. Parses `inputs.dials_expt_path` using dxtbx to obtain the DIALS `Experiment` object (containing beam, detector, crystal models).
        //    b. Reads the raw pixel data from `inputs.cbf_image_path`.
        //    c. Loads the Bragg mask (boolean NumPy array) from `inputs.bragg_mask_path`.
        //    d. If `inputs.external_pdb_path` is provided, parses it to extract reference unit cell parameters and potentially orientation information.
        // 3. **Consistency Checks (if `inputs.external_pdb_path` provided):**
        //    a. Compares unit cell parameters (lengths and angles) from the DIALS crystal model against the reference PDB, using `config.cell_length_tol` and `config.cell_angle_tol`.
        //    b. Compares crystal orientation from the DIALS crystal model against the reference PDB, using `config.orient_tolerance_deg`.
        //    c. If any consistency check fails, returns an `OperationOutcome` with status "FAILURE" and appropriate error message/code.
        // 4. **Pixel Processing Loop (or equivalent vectorized operations):**
        //    a. Iterates through each pixel of the image, respecting `config.pixel_step`.
        //    b. Skips pixels that are set to true in the Bragg mask.
        //    c. For each valid pixel:
        //       i. **Q-Vector Calculation:** Calculates the scattering vector `q` (components qx, qy, qz) for the pixel center using the DIALS `Experiment` geometry (detector panel, beam vector).
        //       ii. **Intensity & Initial Error:** Reads raw intensity $I_{raw}$. Applies `config.gain`. Initial sigma $\sigma_{raw} = \sqrt{I_{raw} \cdot \text{gain}}$ (assuming Poisson statistics).
        //       iii. **Corrections (and error propagation) using DIALS API and Custom Calculations:**
        //            - **Lorentz-Polarization (LP) correction:** Obtained via adapter to DIALS `dials.algorithms.integration.Corrections` class using the Experiment model for the current still. Handles complex effects like parallax internally.
        //            - **Detector Quantum Efficiency (QE) correction:** Obtained via adapter to DIALS `dials.algorithms.integration.Corrections` class.
        //            - **Solid angle correction:** Custom calculation based on pixel geometry and detector properties (not available from DIALS Corrections for arbitrary diffuse pixels).
        //            - **Air attenuation correction:** Custom calculation based on beam path length and wavelength (if parameters provided and significant).
        //            All correction factors are converted to multipliers and combined into `TotalCorrection_mult(p)`. Each correction factor is applied to $I$ and its effect propagated to $\sigma_I$.
        //       iv. **Background Subtraction (and error propagation):**
        //            - If `config.subtract_measured_background_path` is provided: Load background map, subtract value for current pixel. Add variance of background to current $\sigma_I^2$.
        //            - Else if `config.subtract_constant_background_value` is provided: Subtract constant. (Error propagation for constant subtraction is typically zero unless constant has uncertainty).
        //       v. **Filtering:**
        //            - Resolution Filter: Exclude if d-spacing (derived from $|q|$) is outside [`config.max_res`, `config.min_res`].
        //            - Intensity Filter: Exclude if current corrected intensity is outside [`config.min_intensity`, `config.max_intensity`].
        //       vi. If pixel passes all filters, store its $(q_x, q_y, q_z)$, final corrected $I_{corr}$, and final propagated $\sigma_{I_{corr}}$.
        // 5. **Output Generation:**
        //    a. If no pixels pass filters, return `OperationOutcome` with status "FAILURE" or "WARNING".
        //    b. Saves the collected q-vectors, intensities, and sigmas as arrays in an NPZ file at `output_npz_path`.
        //    c. If `config.plot_diagnostics` is true, generates diagnostic plots (e.g., q-space coverage, intensity distributions) and saves them. Adds paths to `OperationOutcome.output_artifacts`.
        // 6. Returns `OperationOutcome` with status "SUCCESS".
        // @raises_error(condition="InputFileError", description="Failure to read or parse one of the essential input files (CBF, EXPT, Bragg Mask, PDB).")
        // @raises_error(condition="DIALSModelIncomplete", description="The DIALS .expt file is missing critical geometry information (e.g., beam, detector, or crystal model).")
        // @raises_error(condition="ConsistencyCheckFailed", description="Geometric parameters derived from DIALS data failed consistency checks against the external PDB within the specified tolerances.")
        // @raises_error(condition="BackgroundFileError", description="Failed to read or apply the `subtract_measured_background_path` file.")
        // @raises_error(condition="ProcessingError", description="An unexpected error occurred during pixel processing, q-vector calculation, or application of corrections.")
        // @raises_error(condition="OutputWriteError", description="Failed to write the output NPZ file or any diagnostic plot files.")
        src.diffusepipe.types.OperationOutcome extract_from_still(
            src.diffusepipe.types.ComponentInputFiles inputs,
            src.diffusepipe.types.ExtractionConfig config,
            string output_npz_path
        );
    }
}
// == END IDL ==
