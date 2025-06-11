// == BEGIN IDL ==
module src.diffusepipe.adapters {

    # @depends_on_resource(type="ExternalTool:DIALS", purpose="Using DIALS CLI tools (dials.import, dials.find_spots, dials.index, dials.integrate) for sequence data processing")
    # @depends_on_resource(type="FileSystem", purpose="Creating temporary directories, executing CLI commands, loading result files")
    # @depends_on_type(src.diffusepipe.types.DIALSStillsProcessConfig)
    # @depends_on_type(dxtbx.model.Experiment)
    # @depends_on_type(dials.array_family.flex.reflection_table)

    interface DIALSSequenceProcessAdapter {

        // --- Method: process_still ---
        // Preconditions:
        // - `image_path` must be a valid path to a readable CBF file containing sequence/oscillation data (typically Angle_increment > 0.0°).
        // - `config` must be a valid `DIALSStillsProcessConfig` object with parameters compatible with sequence processing.
        // - The DIALS command-line tools must be available in the system PATH.
        // - The CBF file should contain valid crystallographic data with oscillation.
        // Postconditions:
        // - If successful (third tuple element is True), returns valid DIALS Experiment and reflection_table objects.
        // - The Experiment object contains an indexed and refined crystal model.
        // - The reflection_table contains integrated reflections with a 'partiality' column.
        // - The fourth tuple element contains human-readable log messages detailing the processing steps.
        // - If failed (third tuple element is False), the first two elements are None.
        // - All temporary files are cleaned up after processing (temporary directory is removed).
        // Behavior:
        // - Executes a sequence of DIALS CLI tools optimized for oscillation/sequence data processing.
        // - Creates a temporary directory and changes to it for isolated processing.
        // - Workflow steps:
        //   1. `dials.import`: Imports the CBF file to create imported.expt.
        //   2. `dials.find_spots`: Finds spots with critical parameters:
        //      - spotfinder.filter.min_spot_size=3 (not default 2)
        //      - spotfinder.threshold.algorithm=dispersion (not default)
        //   3. `dials.index`: Indexes with critical parameters:
        //      - indexing.method=fft3d (not fft1d)
        //      - geometry.convert_sequences_to_stills=false (preserve oscillation)
        //      - Applies known unit cell and space group if provided in config.
        //   4. `dials.integrate`: Integrates the indexed reflections.
        // - Loads the final integrated.expt and integrated.refl files using DIALS Python API.
        // - Validates that the reflection table contains the required 'partiality' column.
        // - Returns to the original working directory after processing.
        // - Note: The `base_expt_path` parameter is ignored as this adapter always imports from scratch.
        // @raises_error(condition="DIALSError", description="When any DIALS CLI command fails or result files are missing")
        // @raises_error(condition="ConfigurationError", description="When configuration is invalid (missing image file)")
        // @raises_error(condition="DataValidationError", description="When required partiality data is missing from results")
        tuple<object, object, boolean, string> process_still(
            string image_path,
            src.diffusepipe.types.DIALSStillsProcessConfig config,
            optional string base_expt_path
        );
    }
}
// == END IDL ==