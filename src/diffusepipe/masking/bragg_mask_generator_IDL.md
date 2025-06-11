// == BEGIN IDL ==
module src.diffusepipe.masking {

    # @depends_on_resource(type="DIALS/dxtbx", purpose="Using Experiment, reflection_table, Detector objects, flex arrays, MaskCode")
    # @depends_on([src.diffusepipe.adapters.DIALSGenerateMaskAdapter]) // For Option A
    # @depends_on_type(src.diffusepipe.exceptions.BraggMaskError) // For use in @raises_error

    interface BraggMaskGenerator {

        // --- Method: generate_bragg_mask_from_spots (Option A for Bragg masking) ---
        // Preconditions:
        // - `experiment` is a valid DIALS Experiment object containing the crystal model and geometry.
        // - `reflections` is a valid DIALS reflection_table containing indexed spot positions.
        // - `config` (if provided) is a dictionary containing valid parameters for `dials.util.masking.generate_mask` (e.g., 'border', 'algorithm').
        // Postconditions:
        // - Returns a tuple of `flex.bool` Bragg mask arrays, one for each panel in the detector.
        // - In the returned Bragg masks, `True` indicates a pixel considered part of a Bragg reflection region.
        // Behavior:
        // - This method implements Option A for Bragg masking as defined in `plan.md`.
        // - It utilizes the `DIALSGenerateMaskAdapter` to invoke `dials.util.masking.generate_mask` or equivalent DIALS functionality.
        // - Generates a mask that covers regions occupied by the indexed Bragg spots found in `reflections`.
        // @raises_error(condition="BraggMaskError", description="If DIALS mask generation via the adapter fails or input objects are invalid.")
        tuple<object> generate_bragg_mask_from_spots( // conceptual return: tuple<flex.bool, ...>
            object experiment, // DIALS Experiment object
            object reflections, // DIALS reflection_table object
            optional map<string, any> config // Parameters for DIALSGenerateMaskAdapter
        );

        // --- Method: generate_bragg_mask_from_shoeboxes (Option B for Bragg masking) ---
        // Preconditions:
        // - `reflections` is a valid DIALS reflection_table that includes a 'shoebox' column. Each shoebox must contain 3D mask data.
        // - `detector` is a valid DIALS Detector object, used to determine panel dimensions for the output masks.
        // Postconditions:
        // - Returns a tuple of `flex.bool` Bragg mask arrays, one for each panel.
        // - `True` in the mask indicates a pixel that falls within the foreground or strong region of any reflection's shoebox.
        // Behavior:
        // - This method implements Option B for Bragg masking as defined in `plan.md`.
        // - Initializes per-panel 2D masks to all False.
        // - Iterates through each reflection in the `reflections` table.
        // - For each reflection, accesses its 3D shoebox mask.
        // - Pixels within the shoebox marked with `MaskCode.Foreground` or `MaskCode.Strong` are projected onto the corresponding 2D panel mask, setting those pixels to True.
        // @raises_error(condition="BraggMaskError", description="If the 'shoebox' column is missing from `reflections`, or if shoebox data is invalid or processing fails.")
        tuple<object> generate_bragg_mask_from_shoeboxes( // conceptual return: tuple<flex.bool, ...>
            object reflections, // DIALS reflection_table with shoeboxes
            object detector     // DIALS Detector object
        );

        // --- Method: get_total_mask_for_still ---
        // Preconditions:
        // - `bragg_mask` is a tuple of `flex.bool` arrays (one per panel), where `True` indicates a Bragg-masked pixel.
        // - `global_pixel_mask` is a tuple of `flex.bool` arrays (one per panel), where `True` indicates a good pixel (passed static/dynamic checks).
        // - Both `bragg_mask` and `global_pixel_mask` must have the same number of panels, and corresponding panels must have compatible dimensions.
        // Postconditions:
        // - Returns a tuple of `flex.bool` total mask arrays, one for each panel.
        // - `True` in the total mask indicates a pixel suitable for diffuse scattering analysis (i.e., it's a good pixel and not part of a Bragg region).
        // Behavior:
        // - For each panel, calculates the total mask using the formula:
        //   `Mask_total_panel = global_pixel_mask_panel AND (NOT bragg_mask_panel)`.
        // @raises_error(condition="BraggMaskError", description="If `bragg_mask` and `global_pixel_mask` have incompatible panel counts or dimensions, or if the logical combination fails.")
        tuple<object> get_total_mask_for_still( // conceptual return: tuple<flex.bool, ...>
            tuple<object> bragg_mask,          // tuple<flex.bool, ...>, Bragg regions = True
            tuple<object> global_pixel_mask    // tuple<flex.bool, ...>, Good pixels = True
        );
    }
}
// == END IDL ==