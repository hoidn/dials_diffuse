// == BEGIN IDL ==
module src.diffusepipe.adapters {

    # @depends_on_resource(type="ExternalTool:DIALS", purpose="Using DXTBX/DIALS Python API for loading and saving experiment/reflection data")
    # @depends_on_resource(type="FileSystem", purpose="Reading and writing experiment (.expt) and reflection (.refl) files")
    # @depends_on_type(dxtbx.model.experiment_list.ExperimentList)
    # @depends_on_type(dials.array_family.flex.reflection_table)

    interface DXTBXIOAdapter {

        // --- Method: load_experiment_list ---
        // Preconditions:
        // - `expt_path` must be a valid path to a readable DIALS experiment JSON file (.expt).
        // - The file must contain valid experiment data compatible with DXTBX format.
        // - The DXTBX Python libraries must be available and properly configured.
        // Postconditions:
        // - Returns a valid DIALS ExperimentList object containing one or more experiments.
        // - The returned object provides access to detector, beam, crystal, and other experimental models.
        // Behavior:
        // - Wraps `dxtbx.model.experiment_list.ExperimentListFactory.from_json_file()`.
        // - Loads the experiment file and parses it into DIALS/DXTBX data structures.
        // - Validates that the loaded data is accessible and well-formed.
        // @raises_error(condition="FileSystemError", description="When the experiment file does not exist, is not readable, or is corrupted")
        // @raises_error(condition="DIALSError", description="When DXTBX fails to parse the experiment file or the format is invalid")
        object load_experiment_list(
            string expt_path
        );

        // --- Method: load_reflection_table ---
        // Preconditions:
        // - `refl_path` must be a valid path to a readable DIALS reflection file (.refl).
        // - The file must contain valid reflection data compatible with DIALS flex format.
        // - The DIALS Python libraries must be available and properly configured.
        // Postconditions:
        // - Returns a valid DIALS reflection_table object containing reflection data.
        // - The returned object provides access to reflection properties (positions, intensities, etc.).
        // Behavior:
        // - Wraps `dials.array_family.flex.reflection_table.from_file()`.
        // - Loads the reflection file and parses it into a DIALS reflection_table data structure.
        // - Validates that the loaded data is accessible and contains expected columns.
        // @raises_error(condition="FileSystemError", description="When the reflection file does not exist, is not readable, or is corrupted")
        // @raises_error(condition="DIALSError", description="When DIALS fails to parse the reflection file or the format is invalid")
        object load_reflection_table(
            string refl_path
        );

        // --- Method: save_experiment_list ---
        // Preconditions:
        // - `experiments` must be a valid DIALS ExperimentList object.
        // - `expt_path` must be a valid file path where the experiment file will be written.
        // - The parent directory of `expt_path` must be writable.
        // - The DXTBX Python libraries must be available and properly configured.
        // Postconditions:
        // - A valid DIALS experiment JSON file is created at `expt_path`.
        // - The file contains all experiment data from the input ExperimentList.
        // - The parent directory is created if it does not exist.
        // Behavior:
        // - Wraps `dxtbx.model.experiment_list.ExperimentListDumper.as_json()`.
        // - Ensures the parent directory exists, creating it if necessary.
        // - Serializes the ExperimentList to JSON format and writes to the specified path.
        // - Validates that the file was written successfully and is readable.
        // @raises_error(condition="DIALSError", description="When DXTBX fails to serialize the experiment data or write the file")
        // @raises_error(condition="FileSystemError", description="When the target path is not writable or directory creation fails")
        void save_experiment_list(
            object experiments,
            string expt_path
        );

        // --- Method: save_reflection_table ---
        // Preconditions:
        // - `reflections` must be a valid DIALS reflection_table object.
        // - `refl_path` must be a valid file path where the reflection file will be written.
        // - The parent directory of `refl_path` must be writable.
        // - The DIALS Python libraries must be available and properly configured.
        // Postconditions:
        // - A valid DIALS reflection file is created at `refl_path`.
        // - The file contains all reflection data from the input reflection_table.
        // - The parent directory is created if it does not exist.
        // Behavior:
        // - Calls the `as_file()` method on the reflection_table object.
        // - Ensures the parent directory exists, creating it if necessary.
        // - Writes the reflection data to the specified path in DIALS native format.
        // - Validates that the file was written successfully and is readable.
        // @raises_error(condition="DIALSError", description="When DIALS fails to serialize the reflection data or write the file")
        // @raises_error(condition="FileSystemError", description="When the target path is not writable or directory creation fails")
        void save_reflection_table(
            object reflections,
            string refl_path
        );
    }
}
// == END IDL ==