// == BEGIN IDL ==
module src.diffusepipe.masking {

    # @depends_on_resource(type="DIALS/dxtbx", purpose="Using Detector and ImageSet objects, flex arrays for mask representation")
    # @depends_on_type(src.diffusepipe.exceptions.MaskGenerationError) // For use in @raises_error

    // Defines a circular region for masking
    struct Circle {
        center_x: float; // Center X coordinate in pixels
        center_y: float; // Center Y coordinate in pixels
        radius: float;   // Radius in pixels
    }

    // Defines a rectangular region for masking
    struct Rectangle {
        min_x: float; // Minimum X coordinate in pixels
        max_x: float; // Maximum X coordinate in pixels
        min_y: float; // Minimum Y coordinate in pixels
        max_y: float; // Maximum Y coordinate in pixels
    }

    // Parameters for static mask generation
    struct StaticMaskParams {
        // Behavior: Defines a beamstop region to be masked. Can be a Circle or Rectangle.
        beamstop: optional union<Circle, Rectangle>;
        // Behavior: A list of rectangular regions to be explicitly masked.
        untrusted_rects: optional list<Rectangle>;
        // Behavior: A list of panel indices (0-based) that should be entirely masked.
        untrusted_panels: optional list<int>;
    }

    // Parameters for dynamic mask generation
    struct DynamicMaskParams {
        // Behavior: Pixels with intensity consistently above this threshold in representative images are masked as hot.
        hot_pixel_thresh: optional float;
        // Behavior: Tolerance for considering a pixel value as negative. Pixels below -abs(negative_pixel_tolerance) are masked.
        negative_pixel_tolerance: float;
        // Behavior: If the fraction of pixels flagged as bad by dynamic checks exceeds this, a warning may be issued.
        max_fraction_bad_pixels: float;
    }

    interface PixelMaskGenerator {

        // --- Method: generate_combined_pixel_mask ---
        // Preconditions:
        // - `detector` is a valid DIALS Detector object.
        // - `static_params` and `dynamic_params` are valid parameter objects.
        // - `representative_images` is a list of valid DIALS ImageSet objects. This list can be empty, in which case the dynamic mask component will effectively be an all-pass mask.
        // Postconditions:
        // - Returns a tuple of `flex.bool` mask arrays, one for each panel in the `detector`.
        // - In the returned masks, `True` indicates a good (unmasked) pixel, and `False` indicates a bad (masked) pixel.
        // Behavior:
        // - Orchestrates the generation of a global pixel mask.
        // - Internally calls `generate_static_mask` to create a mask based on detector geometry and `static_params`.
        // - Internally calls `generate_dynamic_mask` to create a mask based on analysis of `representative_images` and `dynamic_params`.
        // - Combines the static and dynamic masks using a logical AND operation (a pixel is good only if it's good in both static and dynamic masks).
        // @raises_error(condition="MaskGenerationError", description="If any step in the static, dynamic, or combined mask generation process fails.")
        tuple<object> generate_combined_pixel_mask( // Conceptual return: tuple<flex.bool, ...>
            object detector, // DIALS Detector object
            StaticMaskParams static_params,
            list<object> representative_images, // List of DIALS ImageSet objects
            DynamicMaskParams dynamic_params
        );

        // --- Method: generate_static_mask (Potentially a helper, but part of the conceptual interface) ---
        // Preconditions:
        // - `detector` is a valid DIALS Detector object.
        // - `static_params` is a valid StaticMaskParams object.
        // Postconditions:
        // - Returns a tuple of `flex.bool` static mask arrays.
        // Behavior:
        // - Creates a mask based on detector's trusted pixel value range.
        // - Applies masks for any defined `beamstop` in `static_params`.
        // - Applies masks for any `untrusted_rects` in `static_params`.
        // - Applies masks for any `untrusted_panels` in `static_params`.
        // @raises_error(condition="MaskGenerationError", description="If static mask generation fails.")
        tuple<object> generate_static_mask( // conceptual return: tuple<flex.bool, ...>
            object detector, // DIALS Detector object
            StaticMaskParams static_params
        );

        // --- Method: generate_dynamic_mask (Potentially a helper, but part of the conceptual interface) ---
        // Preconditions:
        // - `detector` is a valid DIALS Detector object.
        // - `representative_images` is a list of DIALS ImageSet objects.
        // - `dynamic_params` is a valid DynamicMaskParams object.
        // Postconditions:
        // - Returns a tuple of `flex.bool` dynamic mask arrays.
        // Behavior:
        // - If `representative_images` is empty, returns an all-True mask (all pixels good).
        // - Otherwise, analyzes pixel statistics (e.g., identifying hot pixels above `dynamic_params.hot_pixel_thresh`, negative pixels below `dynamic_params.negative_pixel_tolerance`) across the `representative_images`.
        // - Creates a mask where pixels exhibiting anomalous behavior are flagged as bad.
        // @raises_error(condition="MaskGenerationError", description="If dynamic mask generation fails.")
        tuple<object> generate_dynamic_mask( // conceptual return: tuple<flex.bool, ...>
            object detector, // DIALS Detector object
            list<object> representative_images, // List of DIALS ImageSet objects
            DynamicMaskParams dynamic_params
        );
    }
}
// == END IDL ==