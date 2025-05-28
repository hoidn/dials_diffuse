// == BEGIN IDL ==
module src.diffusepipe.extraction {

    # @depends_on_resource(type="FileSystem", purpose="Reading CBF, EXPT, PDB, Bragg Mask files; Writing NPZ and log files")
    # @depends_on_type(src.diffusepipe.types.ExtractionParameters)
    # @depends_on_type(src.diffusepipe.types.OperationOutcome)
    # @depends_on_type(src.diffusepipe.types.FileSet)
    interface DataExtractor {
        // Preconditions:
        // - All file paths in `input_files` (experiment_file, image_file, bragg_mask_file, external_pdb_file) must exist and be readable.
        // - `params` contains valid settings for extraction.
        // - `output_npz_path` specifies a writable file path.
        // Postconditions:
        // - If successful, an NPZ file is created at `output_npz_path`. This file contains at least 'q_vectors' and 'intensities' arrays.
        // - Returns an `OperationOutcome` detailing success or failure. If `params.plot_diagnostics` is true, `output_artifact_paths` in the outcome may contain paths to generated plots.
        // Behavior:
        // - Logs verbose output if `params.verbose_python` is true.
        // - Loads the DIALS experiment model from `input_files.experiment_file`.
        // - Loads the raw CBF image data from `input_files.image_file`.
        // - Loads the Bragg reflection mask from `input_files.bragg_mask_file`.
        // - Performs consistency checks using the geometry from the DIALS experiment model against the unit cell and orientation derived from `input_files.external_pdb_file`, respecting `params.cell_length_tol`, `params.cell_angle_tol`, and `params.orient_tolerance_deg`. If checks fail, returns a failure `OperationOutcome`.
        // - Iterates through pixels of the image (stepping by `params.pixel_step`):
        //   1. If a pixel is not masked by the Bragg mask:
        //      a. Calculates its q-vector using the DIALS experiment geometry.
        //      b. Reads the pixel intensity from the CBF image.
        //      c. Applies `params.gain` to the intensity.
        //      d. Filters the pixel based on resolution limits (`params.min_res`, `params.max_res`).
        //      e. Filters the pixel based on intensity limits (`params.min_intensity`, `params.max_intensity`).
        //      f. If `params.lp_correction_enabled` is true, applies a simplified Lorentz-Polarization correction.
        //      g. If `params.subtract_background_value` is provided, subtracts this constant value.
        //      h. If the pixel passes all filters, its q-vector and processed intensity are stored.
        // - If data is successfully extracted, saves the collected q-vectors and intensities (and any other relevant data) to an NPZ file at `output_npz_path`.
        // - If `params.plot_diagnostics` is true, generates and saves diagnostic plots (e.g., resolution distribution, intensity histogram).
        // @raises_error(condition="InputFileMissing", description="One or more essential input files specified in `input_files` were not found or are unreadable.")
        // @raises_error(condition="DIALSModelError", description="Failed to parse or interpret the DIALS experiment file.")
        // @raises_error(condition="PDBFileError", description="Failed to parse or interpret the external PDB file.")
        // @raises_error(condition="ConsistencyCheckFailed", description="Geometric consistency checks against the external PDB failed based on provided tolerances.")
        // @raises_error(condition="ExtractionError", description="An unexpected error occurred during pixel data processing or q-vector calculation.")
        // @raises_error(condition="OutputWriteError", description="Failed to write the output NPZ file or diagnostic plots.")
        src.diffusepipe.types.OperationOutcome extract_diffuse_data(
            src.diffusepipe.types.FileSet input_files,
            src.diffusepipe.types.ExtractionParameters params,
            string output_npz_path
        );
    }
}
// == END IDL ==
