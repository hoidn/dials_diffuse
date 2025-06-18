// == BEGIN IDL ==
module src.diffusepipe.adapters {

    # @depends_on_resource(type="ExternalTool:DIALS", purpose="Using DIALS CLI tools (dials.import, dials.find_spots, dials.index, dials.integrate) for sequence data processing")
    # @depends_on_resource(type="FileSystem", purpose="Creating temporary directories, executing CLI commands, loading result files")
    # @depends_on_type(src.diffusepipe.types.DIALSStillsProcessConfig)
    # @depends_on_type(dxtbx.model.Experiment)
    # @depends_on_type(dials.array_family.flex.reflection_table)

    interface DIALSSequenceProcessAdapter {

        // --- Method: process_sequence ---
        // Preconditions:
        // - `image_paths` must be a list of valid paths to readable CBF files containing sequence/oscillation data (typically Angle_increment > 0.0°).
        // - All images in the sequence must be from the same crystal and experimental setup.
        // - `config` must be a valid `DIALSStillsProcessConfig` object with parameters compatible with sequence processing.
        // - The DIALS command-line tools must be available in the system PATH.
        // - The CBF files should contain valid crystallographic data with consistent oscillation parameters.
        // Postconditions:
        // - If successful (third tuple element is True), returns valid DIALS ExperimentList and reflection_table objects.
        // - The ExperimentList contains one Experiment with a scan object covering all images in the sequence.
        // - The Experiment contains a single, consistent, scan-varying crystal model that maintains orientation consistency across all frames.
        // - The reflection_table contains integrated reflections from all images with a 'partiality' column.
        // - **Note on Filtering:** To select reflections for a specific frame `i` (0-indexed) from the returned table,
        //   filter on the `z` coordinate of the `xyzobs.px.value` column (e.g., `table.select(table['xyzobs.px.value'].parts()[2].iround() == i)`).
        //   The `imageset_id` column will be 0 for all reflections.
        // - The fourth tuple element contains human-readable log messages detailing the processing steps.
        // - If failed (third tuple element is False), the first two elements are None.
        // - All temporary files are cleaned up after processing (temporary directory is removed).
        // Behavior:
        // - Processes the entire list of images as a single sequence in one DIALS run to ensure a consistent scan-varying model.
        // - Creates a temporary directory and changes to it for isolated processing.
        // - **True Sequence Processing:** Processes all images as a single sequence, leveraging DIALS's native scan-varying refinement for consistent crystal orientation.
        // - Workflow steps:
        //   1. `dials.import`: Imports all CBF files in the sequence to create a single imported.expt representing the entire scan.
        //   2. `dials.find_spots`: Finds spots across the entire sequence with critical parameters:
        //      - spotfinder.filter.min_spot_size=3 (not default 2)
        //      - spotfinder.threshold.algorithm=dispersion (not default)
        //   3. `dials.index`: Indexes the sequence with critical parameters:
        //      - indexing.method=fft3d (not fft1d)
        //      - geometry.convert_sequences_to_stills=false (preserve oscillation)
        //      - Applies known unit cell and space group if provided in config.
        //   4. `dials.integrate`: Integrates all reflections across the sequence with scan-varying refinement.
        // - Loads the final integrated.expt and integrated.refl files using DIALS Python API.
        // - The output represents a single, cohesive sequence with consistent crystal model across all frames.
        // - Validates that the reflection table contains the required 'partiality' column.  
        // - Returns to the original working directory after processing.
        // @raises_error(condition="DIALSError", description="When any DIALS CLI command fails or result files are missing")
        // @raises_error(condition="ConfigurationError", description="When configuration is invalid (missing image files)")
        // @raises_error(condition="DataValidationError", description="When required partiality data is missing from results")
        tuple<object, object, boolean, string> process_sequence(
            list<string> image_paths,
            src.diffusepipe.types.DIALSStillsProcessConfig config,
            optional string output_dir_final
        );
    }
}
// == END IDL ==