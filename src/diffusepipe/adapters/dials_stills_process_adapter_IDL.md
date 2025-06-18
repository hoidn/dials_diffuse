// == BEGIN IDL ==
module src.diffusepipe.adapters {

    # @depends_on_resource(type="ExternalTool:DIALS", purpose="Using dials.stills_process.Processor Python API for true still image processing")
    # @depends_on_resource(type="FileSystem", purpose="Loading PHIL configuration files and managing temporary data")
    # @depends_on_type(src.diffusepipe.types.DIALSStillsProcessConfig)
    # @depends_on_type(dxtbx.model.Experiment)
    # @depends_on_type(dials.array_family.flex.reflection_table)

    interface DIALSStillsProcessAdapter {

        // --- Method: process_still ---
        // Preconditions:
        // - `image_path` must be a valid path to a readable CBF file containing a true still image (Angle_increment = 0.0°).
        // - `config` must be a valid `DIALSStillsProcessConfig` object with appropriate parameters for stills processing.
        // - If provided, `base_expt_path` must be a valid path to a DIALS experiment JSON file.
        // - The DIALS Python environment must be properly configured and accessible.
        // - The CBF file should contain valid crystallographic data suitable for stills processing.
        // Postconditions:
        // - If successful (third tuple element is True), returns valid DIALS Experiment and reflection_table objects.
        // - The Experiment object contains a refined crystal model (Crystal_i) with orientation and unit cell.
        // - The reflection_table contains indexed and integrated Bragg reflections with a 'partiality' column.
        // - The fourth tuple element contains human-readable log messages detailing the processing steps.
        // - If failed (third tuple element is False), the first two elements are None.
        // Behavior:
        // - Wraps the `dials.command_line.stills_process.Processor` Python API for processing true still images.
        // - **Reference-Based Indexing:** If `base_expt_path` is provided, it is loaded, and its crystal models are injected into the DIALS parameters (`params.indexing.known_symmetry.crystal_models`) to constrain the indexing search.
        // - Generates appropriate PHIL parameters from the provided configuration:
        //   1. Starts with the default dials.stills_process PHIL scope.
        //   2. Merges any user-provided PHIL file specified in config.stills_process_phil_path.
        //   3. Applies configuration overrides (unit cell, space group, spotfinder settings, etc.).
        //   4. **Reference Constraint:** If `base_expt_path` provided, sets `params.indexing.known_symmetry.crystal_models` to reference crystal models.
        // - Imports the CBF image to create a DIALS ExperimentList.
        // - Instantiates a Processor with the extracted PHIL parameters (including reference crystal models if provided).
        // - Runs the Processor which internally performs: spot finding, indexing, refinement, and integration.
        // - Extracts the first experiment and its associated reflections from the Processor results.
        // - Validates that the reflection table contains the required 'partiality' column.
        // - Returns the processed data with success status and detailed log messages.
        // @raises_error(condition="DIALSError", description="When DIALS operations fail (import, processing, or result extraction)")
        // @raises_error(condition="ConfigurationError", description="When configuration is invalid (missing files, invalid parameters)")
        // @raises_error(condition="DataValidationError", description="When required partiality data is missing from results")
        tuple<object, object, boolean, string> process_still(
            string image_path,
            src.diffusepipe.types.DIALSStillsProcessConfig config,
            optional string base_expt_path,
            optional string output_dir_final
        );
    }
}
// == END IDL ==