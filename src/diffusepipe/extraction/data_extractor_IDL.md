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
        // - `start_angle` must be a valid angle that corresponds to a frame in the DIALS scan (for sequence data).
        // - If `mask_total_2d` is not provided, `inputs.bragg_mask_path` must point to an existing, readable pickle file containing the Bragg mask.
        // - If `mask_total_2d` is provided, it must be a tuple of boolean arrays (one per detector panel) representing the combined mask (Mask_pixel AND NOT BraggMask_2D_raw_i).
        // - If `inputs.external_pdb_path` is provided, it must be an existing, readable PDB file.
        // - `config` must be a valid `ExtractionConfig` object.
        // - The directory for `output_npz_path` must be writable.
        // Postconditions:
        // - If `status` in the returned `OperationOutcome` is "SUCCESS":
        //   - An NPZ file is created at `output_npz_path`. This file minimally contains 'q_vectors' (Nx3 array), 'intensities' (N array), and 'sigmas' (N array).
        //   - If processing input that is part of a sequence (scan-varying data), **must also include 'frame_indices' (N array of 0-based integers)**, representing the DIALS scan point index for each observation. This is required for correct voxelization in Phase 3.
        //   - If `config.save_original_pixel_coordinates` is true, also includes original pixel coordinate arrays.
        //   - If `config.plot_diagnostics` is true, diagnostic plot image files may be created (paths reported in `output_artifacts`).
        // - The returned `OperationOutcome` object details the success or failure of the operation.
        // Behavior:
        // 1. Logs verbose messages if `config.verbose` is true.
        // 2. **Load Data:**
        //    a. Parses `inputs.dials_expt_path` using dxtbx to obtain the DIALS `ExperimentList` object from sequence processing.
        //    b. Uses the scan object's `get_image_index_from_angle(start_angle)` method to determine the correct frame index for this specific image.
        //    c. Selects the frame-specific experiment using the resolved frame index and gets scan-varying geometry if needed.
        //    d. Gets the corresponding imageset directly from the experiment object and reads raw pixel data for the correct frame.
        //    e. Loads the Bragg mask (boolean NumPy array) from `inputs.bragg_mask_path`.
        //    f. If `inputs.external_pdb_path` is provided, parses it to extract reference unit cell parameters and potentially orientation information.
        // 3. **Consistency Checks (if `inputs.external_pdb_path` provided):**
        //    a. Compares unit cell parameters (lengths and angles) from the DIALS crystal model against the reference PDB, using `config.cell_length_tol` and `config.cell_angle_tol`.
        //    b. Compares crystal orientation from the DIALS crystal model against the reference PDB, using `config.orient_tolerance_deg`.
        //    c. If any consistency check fails, returns an `OperationOutcome` with status "FAILURE" and appropriate error message/code.
        // 4. **Pixel Processing (High-Performance Vectorized Implementation):**
        //    Processing is performed using efficient, vectorized batch operations on all valid pixels simultaneously, avoiding slow per-pixel loops.
        //    a. Identifies all unmasked pixels respecting `config.pixel_step` using vectorized coordinate generation and boolean masking.
        //    b. Batch calculates lab coordinates for all valid pixels using DIALS vectorized API (`panel.get_lab_coord(flex.vec2_double)`).
        //    c. For all valid pixels simultaneously:
        //       i. **Q-Vector Calculation:** Calculates the scattering vector `q` (components qx, qy, qz) for the pixel center using the DIALS `Experiment` geometry (detector panel, beam vector).
        //       ii. **Intensity & Initial Error:** Reads raw intensity $I_{raw}$. Applies `config.gain`. Initial sigma $\sigma_{raw} = \sqrt{I_{raw} \cdot \text{gain}}$ (assuming Poisson statistics).
        //       iii. **Corrections (and error propagation) using DIALS API and Custom Calculations:**
        //            - **Lorentz-Polarization (LP) correction:** Obtained via DIALS `dials.algorithms.integration.Corrections` class. LP correction can be enabled/disabled via `config.lp_correction_enabled`. Returns divisors which are converted to multipliers (LP_mult = 1/LP_divisor).
        //            - **Detector Quantum Efficiency (QE) correction:** Obtained via DIALS `dials.algorithms.integration.Corrections` class. Returns multipliers directly.
        //            - **Solid angle correction:** Custom calculation: SA_mult = 1/((pixel_area × cos_θ) / r²) where θ is angle between panel normal and scattered beam direction.
        //            - **Air attenuation correction:** Custom calculation using Beer-Lambert law: Air_mult = 1/exp(-μ_air × path_length). Uses configurable air temperature and pressure via `config.air_temperature_k` and `config.air_pressure_atm`.
        //            All correction factors are combined as multipliers: `TotalCorrection_mult = LP_mult × QE_mult × SA_mult × Air_mult`. Error propagation assumes correction factors have negligible uncertainty: σ_corrected = σ_initial × TotalCorrection_mult.
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
            string output_npz_path,
            float start_angle,
            optional tuple mask_total_2d
        );
    }
}
// == END IDL ==
