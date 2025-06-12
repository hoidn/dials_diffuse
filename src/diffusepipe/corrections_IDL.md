// == BEGIN IDL ==
module src.diffusepipe.corrections {

    # @depends_on_resource(type="ExternalLibrary:numpy", purpose="Numerical calculations for correction factors")
    # @depends_on_resource(type="Logging", purpose="Debug logging of correction calculations")

    // Utility module providing helper functions for pixel correction factors
    // as specified in Module 2.S.2 of the plan (rule 0.7: all corrections as multipliers)
    interface CorrectionsHelper {

        // --- Method: apply_corrections ---
        // Preconditions:
        // - All correction factor parameters must be positive finite numbers.
        // - `raw_I` must be a non-negative intensity value.
        // Postconditions:
        // - Returns the corrected intensity with all factors applied as multipliers.
        // - The result equals `raw_I * lp_mult * qe_mult * sa_mult * air_mult`.
        // Behavior:
        // - **Centralizes Correction Logic:** Implements the multiplicative correction formula from Module 2.S.2.
        // - **All Multipliers Convention:** Enforces the project convention that all corrections are expressed as multipliers (rule 0.7).
        // - **Debug Logging:** Logs individual correction values for debugging and validation.
        // - **Mathematical Formula:** TotalCorrection_mult = LP_mult × QE_mult × SA_mult × Air_mult
        // @raises_error(condition="ValueError", description="When any correction factor is non-positive or non-finite")
        static float apply_corrections(
            float raw_I,        // Raw intensity value
            float lp_mult,      // Lorentz-Polarization correction multiplier (1/LP_divisor)
            float qe_mult,      // Quantum Efficiency correction multiplier
            float sa_mult,      // Solid Angle correction multiplier (1/solid_angle)
            float air_mult      // Air Attenuation correction multiplier (1/attenuation)
        );

        // --- Method: calculate_analytic_pixel_corrections_45deg ---
        // Preconditions:
        // - `raw_intensity` must be non-negative.
        // - `wavelength_angstrom` must be positive (typical X-ray wavelengths: 0.5-2.0 Å).
        // - `detector_distance_mm` must be positive.
        // - `pixel_size_mm` must be positive.
        // - `air_path_length_mm` (if provided) must be non-negative.
        // Postconditions:
        // - Returns tuple: (corrected_intensity, correction_factors_dict).
        // - `correction_factors_dict` contains individual correction factors for validation.
        // - The corrected intensity should recover analytic intensity within 1% (regression test requirement).
        // Behavior:
        // - **Regression Test Function:** Provides known analytic values for testing the correction pipeline.
        // - **45° Scattering Geometry:** Calculates corrections for a pixel at exactly 45° scattering angle.
        // - **Lorentz-Polarization:** Uses formula LP = 1/(sin²θ × (1 + cos²(2θ))) for unpolarized X-rays.
        // - **Solid Angle:** Calculates geometric solid angle considering pixel position and size.
        // - **Air Attenuation:** Applies Beer-Lambert law with approximate μ_air for testing purposes.
        // - **QE Correction:** Assumes ideal detector (QE = 1.0) for analytical simplicity.
        // - **Validation Target:** This function must recover analytic intensity to <1% accuracy for regression testing.
        static tuple<float, map<string, float>> calculate_analytic_pixel_corrections_45deg(
            float raw_intensity,              // Raw pixel intensity
            float wavelength_angstrom,        // X-ray wavelength in Angstroms
            float detector_distance_mm,       // Sample-to-detector distance in mm
            float pixel_size_mm,              // Pixel size in mm (assumed square)
            optional float air_path_length_mm // Air path length in mm for attenuation calculation
        );

        // --- Method: validate_correction_factors ---
        // Preconditions:
        // - `correction_factors` must be a map containing correction factor values.
        // - `tolerance` must be a positive float.
        // Postconditions:
        // - Returns tuple: (is_valid, error_message).
        // - `is_valid` is true only if all correction factors pass reasonableness checks.
        // - `error_message` provides specific details about any validation failures.
        // Behavior:
        // - **Reasonableness Checks:** Validates that correction factors fall within expected physical ranges.
        // - **Positivity Check:** All factors must be positive.
        // - **LP Range Check:** LP correction should be between 0.1 and 10.0 (geometric limits).
        // - **QE Range Check:** QE correction should be between 0.1 and 2.0 (realistic detector efficiency range).
        // - **SA Range Check:** Solid angle correction should not be extreme (1e-6 to 1e6).
        // - **Air Range Check:** Air attenuation should be close to 1.0 (0.9 to 1.2 for typical geometries).
        // - **Error Reporting:** Provides specific error messages for debugging.
        static tuple<boolean, string> validate_correction_factors(
            map<string, float> correction_factors, // Dictionary of correction factors to validate
            optional float tolerance               // Tolerance for validation checks (default: 0.01)
        );

        // --- Method: create_synthetic_experiment_for_testing ---
        // Preconditions:
        // - Mock testing framework must be available.
        // Postconditions:
        // - Returns a mock DIALS Experiment object with proper geometric parameters.
        // - The mock object supports necessary methods for correction calculations.
        // Behavior:
        // - **Test Fixture Creation:** Creates synthetic DIALS-like objects for unit testing.
        // - **45° Geometry Setup:** Positions detector and beam for 45° scattering angle testing.
        // - **Proper Mock Structure:** Implements necessary DIALS API methods with realistic values.
        // - **Regression Test Support:** Provides controlled geometry for validating correction calculations.
        // - **Beam Configuration:** Sets wavelength and direction vector appropriate for testing.
        // - **Detector Configuration:** Sets pixel size, position, and axes for realistic geometry.
        static object create_synthetic_experiment_for_testing();
    }
}
// == END IDL ==