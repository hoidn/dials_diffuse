// == BEGIN IDL ==
module src.diffusepipe.adapters {

    # @depends_on_resource(type="ExternalTool:DIALS", purpose="Using DIALS masking utilities (dials.util.masking.generate_mask or equivalent)")
    # @depends_on_resource(type="FileSystem", purpose="Writing temporary mask files if needed")
    # @depends_on_type(dxtbx.model.Experiment)
    # @depends_on_type(dials.array_family.flex.reflection_table)
    # @depends_on_type(dials.array_family.flex.bool)

    interface DIALSGenerateMaskAdapter {

        // --- Method: generate_bragg_mask ---
        // Preconditions:
        // - `experiment` must be a valid DIALS Experiment object containing detector geometry and beam information.
        // - `reflections` must be a valid DIALS reflection_table object containing indexed Bragg spots with position data.
        // - If provided, `mask_generation_params` must contain valid key-value pairs for mask generation settings.
        // - The DIALS masking utilities must be available in the Python environment.
        // Postconditions:
        // - If successful (second tuple element is True), returns a tuple of panel masks where each element is a flex.bool array.
        // - The number of masks in the tuple corresponds to the number of detector panels in the experiment.
        // - Each mask array has the same dimensions as its corresponding detector panel.
        // - Pixels marked as True in the mask represent regions occupied by Bragg peaks (to be excluded from diffuse processing).
        // - The third tuple element contains human-readable log messages detailing the mask generation process.
        // - If failed (second tuple element is False), the first element is None.
        // Behavior:
        // - Wraps the DIALS masking functionality to generate Bragg peak exclusion masks.
        // - Uses the `dials.util.masking.generate_mask` utility function or equivalent DIALS masking logic.
        // - Default mask generation parameters are applied if `mask_generation_params` is None:
        //   - Uses reflection positions from the reflection table to define Bragg regions.
        //   - Applies appropriate margin around each reflection to account for peak shape.
        //   - Handles multiple detector panels correctly.
        // - If `mask_generation_params` is provided, merges these with defaults:
        //   - Supports parameters like: border, resolution_range, ice_rings, shadow_angles, etc.
        //   - Parameter names and values should match DIALS masking utility expectations.
        // - Validates the generated mask result:
        //   - Ensures mask dimensions match detector panel dimensions.
        //   - Verifies that reasonable regions are masked (not entire panels, not zero coverage).
        //   - Logs statistics about masked pixel counts per panel.
        // - Returns per-panel boolean masks suitable for downstream diffuse processing exclusion.
        // @raises_error(condition="DIALSError", description="When DIALS masking operations fail or produce invalid results")
        // @raises_error(condition="BraggMaskError", description="When mask generation parameters are invalid or mask validation fails")
        tuple<tuple<object>, boolean, string> generate_bragg_mask(
            object experiment,
            object reflections,
            optional map<string, any> mask_generation_params
        );
    }
}
// == END IDL ==