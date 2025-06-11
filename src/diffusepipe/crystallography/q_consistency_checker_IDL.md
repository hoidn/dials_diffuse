// == BEGIN IDL ==
module src.diffusepipe.crystallography {

    # @depends_on_resource(type="DIALS/dxtbx", purpose="Using Experiment and reflection_table objects for q-vector calculation and validation")
    # @depends_on_type(src.diffusepipe.exceptions.MissingReflectionColumns) // For use in @raises_error
    # @depends_on_type(src.diffusepipe.exceptions.NoValidReflections) // For use in @raises_error

    // Statistics returned by q-vector consistency check
    struct QConsistencyStatistics {
        count: int;          // Number of reflections successfully processed
        mean: optional float;   // Mean |Δq| value in Å⁻¹ (null if no valid reflections)
        median: optional float; // Median |Δq| value in Å⁻¹ (null if no valid reflections)
        max: optional float;    // Maximum |Δq| value in Å⁻¹ (null if no valid reflections)
    }

    interface QConsistencyChecker {

        // --- Method: check_q_consistency ---
        // Preconditions:
        // - `experiment` is a valid DIALS Experiment object containing beam and detector models.
        // - `reflections` is a valid DIALS reflection_table object.
        // - `tolerance` is a positive float representing the acceptable mean |Δq| tolerance in Å⁻¹.
        // - `max_reflections` is a positive integer limiting the number of reflections to sample for performance.
        // Postconditions:
        // - Returns a tuple: (validation_passed: bool, statistics: QConsistencyStatistics).
        // - `validation_passed` is True if both mean |Δq| ≤ tolerance AND max |Δq| ≤ 5 × tolerance.
        // - `statistics.count` indicates the number of reflections successfully processed.
        // Behavior:
        // - Validates that required columns ("miller_index", "panel", "s1") exist in the reflection table.
        // - Selects pixel centroid column in priority order: "xyzobs.px.value" > "xyzcal.px".
        // - Randomly samples up to `max_reflections` reflections from the table for performance.
        // - For each sampled reflection:
        //   1. Calculates `q_model = s1 - s0` using the reflection's s1 vector and beam's s0.
        //   2. Obtains observed pixel coordinates (px, py) from the selected centroid column.
        //   3. Converts pixel coordinates to lab frame using `detector[panel_id].get_pixel_lab_coord()`.
        //   4. Calculates observed scatter vector `s1_obs` as unit direction vector scaled by |s0|.
        //   5. Calculates `q_observed = s1_obs - s0`.
        //   6. Computes `|Δq| = |q_model - q_observed|`.
        // - Aggregates statistics (mean, median, max) of all |Δq| values.
        // - Applies pass/fail criteria: mean ≤ tolerance AND max ≤ 5 × tolerance.
        // @raises_error(condition="MissingReflectionColumns", description="If essential columns ('miller_index', 'panel', 's1') are missing from the reflection table.")
        // @raises_error(condition="NoValidReflections", description="If no reflections can be processed (empty table or all processing attempts fail).")
        tuple<boolean, QConsistencyStatistics> check_q_consistency(
            object experiment,        // DIALS Experiment object
            object reflections,       // DIALS reflection_table object  
            float tolerance,          // Acceptable mean |Δq| tolerance in Å⁻¹ (default: 0.01)
            int max_reflections      // Maximum reflections to sample (default: 500)
        );
    }
}
// == END IDL ==