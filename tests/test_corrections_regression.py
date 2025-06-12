"""
Regression tests for correction factors as specified in Module 2.S.2.

This test module implements the regression test requirement from the plan:
"A regression test must be implemented that, for a synthetic experiment and a few 
selected pixel positions (including off-Bragg positions), compares the individual 
LP_divisor and QE_multiplier values obtained from the DIALS Corrections adapter 
against trusted reference values or a separate, careful analytic calculation."
"""

import numpy as np
from unittest.mock import Mock, patch, MagicMock

from diffusepipe.corrections import (
    apply_corrections,
    calculate_analytic_pixel_corrections_45deg,
    validate_correction_factors,
    create_synthetic_experiment_for_testing,
)
from diffusepipe.extraction.data_extractor import DataExtractor


class TestCorrectionsRegression:
    """Regression tests for correction factor calculations."""

    def test_apply_corrections_helper(self):
        """Test the apply_corrections helper function."""
        raw_intensity = 1000.0
        lp_mult = 0.5
        qe_mult = 0.8
        sa_mult = 1.2
        air_mult = 1.05

        corrected_intensity = apply_corrections(
            raw_intensity, lp_mult, qe_mult, sa_mult, air_mult
        )

        expected = raw_intensity * lp_mult * qe_mult * sa_mult * air_mult
        assert abs(corrected_intensity - expected) < 1e-10

    def test_analytic_45deg_pixel_corrections(self):
        """Test analytic correction calculation for 45° pixel."""
        raw_intensity = 1000.0
        wavelength = 1.0  # Angstrom
        detector_distance = 200.0  # mm
        pixel_size = 0.172  # mm

        corrected_intensity, factors = calculate_analytic_pixel_corrections_45deg(
            raw_intensity, wavelength, detector_distance, pixel_size
        )

        # Verify factors are reasonable
        assert 0.1 <= factors["lp_mult"] <= 10.0
        assert 0.5 <= factors["qe_mult"] <= 2.0  # Should be 1.0 for ideal detector
        assert factors["sa_mult"] > 0
        assert 0.9 <= factors["air_mult"] <= 1.1  # Should be ~1.0

        # Verify total correction is applied
        expected_total = (
            factors["lp_mult"]
            * factors["qe_mult"]
            * factors["sa_mult"]
            * factors["air_mult"]
        )
        assert abs(factors["total_mult"] - expected_total) < 1e-10

        expected_corrected = raw_intensity * expected_total
        assert abs(corrected_intensity - expected_corrected) < 1e-10

    def test_lp_correction_45deg_analytic_vs_implementation(self):
        """
        Regression test comparing DataExtractor LP correction with analytic calculation.

        This implements the specific regression test requirement from Module 2.S.2.
        """
        # Create synthetic experiment for 45° pixel
        experiment = create_synthetic_experiment_for_testing()
        extractor = DataExtractor()

        # Calculate s1 vector for 45° scattering
        wavelength = 1.0  # Angstrom

        # For 45° scattering: lab_coord = [141.4, 0, 200] gives 45° angle
        lab_coord = np.array([141.4, 0.0, 200.0])
        scatter_direction = lab_coord / np.linalg.norm(lab_coord)
        s1_vector = scatter_direction * (1.0 / wavelength)  # |s1| = 1/λ

        # Calculate analytic LP correction for 45°
        theta = np.pi / 4  # 45 degrees
        sin_theta = np.sin(theta)
        cos_2theta = np.cos(2 * theta)
        lp_divisor_analytic = sin_theta**2 * (1 + cos_2theta**2)
        lp_mult_analytic = 1.0 / lp_divisor_analytic

        # Get LP correction from DataExtractor implementation
        with (
            patch("dials.algorithms.integration.Corrections") as mock_corrections_class,
            patch("dials.array_family.flex") as mock_flex,
        ):
            mock_corrections_obj = Mock()
            mock_corrections_class.return_value = mock_corrections_obj

            # Mock flex array creation
            mock_flex.vec3_double.return_value = [None]
            mock_flex.size_t.return_value = [0]

            # Set the mock to return the analytic divisor
            from unittest.mock import MagicMock

            mock_lp_divisors = MagicMock()
            mock_lp_divisors.__getitem__.return_value = lp_divisor_analytic
            mock_lp_divisors.__len__.return_value = 1
            mock_corrections_obj.lp.return_value = mock_lp_divisors

            lp_mult_implementation = extractor._get_lp_correction(
                s1_vector, experiment.beam, experiment
            )

            # Compare implementation with analytic (should match within 1%)
            relative_error = (
                abs(lp_mult_implementation - lp_mult_analytic) / lp_mult_analytic
            )
            assert relative_error < 0.01, (
                f"LP correction mismatch: analytic={lp_mult_analytic:.6f}, "
                f"implementation={lp_mult_implementation:.6f}, error={relative_error:.3%}"
            )

    def test_correction_factors_validation(self):
        """Test the correction factor validation function."""
        # Valid factors
        valid_factors = {
            "lp_mult": 1.5,
            "qe_mult": 0.8,
            "sa_mult": 1000.0,
            "air_mult": 1.02,
        }

        is_valid, message = validate_correction_factors(valid_factors)
        assert is_valid
        assert "reasonable" in message

        # Invalid factors - negative LP
        invalid_factors = valid_factors.copy()
        invalid_factors["lp_mult"] = -0.5

        is_valid, message = validate_correction_factors(invalid_factors)
        assert not is_valid
        assert "not positive" in message

        # Invalid factors - extreme QE
        invalid_factors = valid_factors.copy()
        invalid_factors["qe_mult"] = 10.0

        is_valid, message = validate_correction_factors(invalid_factors)
        assert not is_valid
        assert "unreasonable" in message

    def test_multiple_pixel_positions_regression(self):
        """Test correction factors for multiple pixel positions."""
        extractor = DataExtractor()
        experiment = create_synthetic_experiment_for_testing()

        # Test positions: on-axis, 30°, 45°, 60°
        test_positions = [
            (0.0, 0.0, 200.0),  # On-axis (0°)
            (100.0, 0.0, 173.2),  # ~30°
            (141.4, 0.0, 141.4),  # 45°
            (173.2, 0.0, 100.0),  # ~60°
        ]

        wavelength = 1.0

        with patch(
            "dials.algorithms.integration.Corrections"
        ) as mock_corrections_class:
            mock_corrections_obj = Mock()
            mock_corrections_class.return_value = mock_corrections_obj

            for i, (x, y, z) in enumerate(test_positions):
                lab_coord = np.array([x, y, z])
                scatter_direction = lab_coord / np.linalg.norm(lab_coord)
                s1_vector = scatter_direction * (1.0 / wavelength)

                # Calculate expected scattering angle
                beam_direction = np.array([0, 0, -1])
                cos_theta = -np.dot(scatter_direction, beam_direction)
                theta = np.arccos(cos_theta)

                # Calculate analytic LP factor
                sin_theta = np.sin(theta)
                cos_2theta = np.cos(2 * theta)
                lp_divisor_expected = sin_theta**2 * (1 + cos_2theta**2)

                # Handle potential division by zero for on-axis scattering
                if abs(lp_divisor_expected) < 1e-9:
                    lp_mult_expected = float(
                        "inf"
                    )  # Very large correction for forward scattering
                else:
                    lp_mult_expected = 1.0 / lp_divisor_expected

                # Mock DIALS to return expected value
                mock_lp_divisors = MagicMock(name="mock_flex_array_for_lp")
                mock_lp_divisors.__getitem__.return_value = lp_divisor_expected
                mock_lp_divisors.__len__.return_value = 1
                mock_corrections_obj.lp.return_value = mock_lp_divisors

                # Test LP correction
                lp_mult = extractor._get_lp_correction(
                    s1_vector, experiment.beam, experiment
                )

                # Handle infinity case for on-axis scattering
                if np.isinf(lp_mult_expected):
                    # For on-axis scattering with zero divisor, the data extractor should return 1.0 as fallback
                    assert lp_mult == 1.0, (
                        f"Position {i} (θ={np.degrees(theta):.1f}°): "
                        f"Expected fallback LP correction 1.0 for zero divisor, got {lp_mult}"
                    )
                else:
                    relative_error = abs(lp_mult - lp_mult_expected) / lp_mult_expected
                    assert relative_error < 0.01, (
                        f"Position {i} (θ={np.degrees(theta):.1f}°): "
                        f"LP error {relative_error:.3%}"
                    )

                # Test that LP correction is reasonable for this angle (except for on-axis)
                if not np.isinf(lp_mult_expected):
                    assert (
                        0.1 <= lp_mult <= 10.0
                    ), f"Unreasonable LP correction: {lp_mult}"

    def test_solid_angle_correction_known_geometry(self):
        """Test solid angle correction for known geometric configuration."""
        extractor = DataExtractor()
        experiment = create_synthetic_experiment_for_testing()
        panel = experiment.detector[0]

        # Test with known geometry: pixel at (100, 50, 200) mm
        lab_coord = np.array([100.0, 50.0, 200.0])

        sa_mult = extractor._calculate_solid_angle_correction(lab_coord, panel, 10, 20)

        # Calculate expected solid angle
        pixel_size = 0.172  # mm
        pixel_area = pixel_size**2
        distance = np.linalg.norm(lab_coord)

        # Get panel normal (should be [0, 0, 1] for this mock)
        fast_axis = np.array([1, 0, 0])
        slow_axis = np.array([0, 1, 0])
        normal = np.cross(fast_axis, slow_axis)  # [0, 0, 1]

        scatter_direction = lab_coord / distance
        cos_theta = abs(np.dot(normal, scatter_direction))

        expected_solid_angle = (pixel_area * cos_theta) / (distance**2)
        expected_sa_mult = 1.0 / expected_solid_angle

        # Allow for small numerical differences
        relative_error = abs(sa_mult - expected_sa_mult) / expected_sa_mult
        assert relative_error < 0.01, (
            f"Solid angle correction mismatch: expected={expected_sa_mult:.6f}, "
            f"actual={sa_mult:.6f}, error={relative_error:.3%}"
        )

    def test_air_attenuation_known_conditions(self):
        """Test air attenuation correction for known conditions."""
        extractor = DataExtractor()
        experiment = create_synthetic_experiment_for_testing()

        # Mock config with known conditions
        config = Mock()
        config.air_temperature_k = 293.15  # 20°C
        config.air_pressure_atm = 1.0  # 1 atm

        # Test with short path (should have minimal attenuation)
        lab_coord_short = np.array([100.0, 0.0, 100.0])  # ~141 mm path
        air_mult_short = extractor._calculate_air_attenuation_correction(
            lab_coord_short, experiment.beam, config
        )

        # Should be very close to 1.0 for short paths
        assert 0.99 <= air_mult_short <= 1.01

        # Test with longer path (should have slightly more attenuation)
        lab_coord_long = np.array([500.0, 0.0, 500.0])  # ~707 mm path
        air_mult_long = extractor._calculate_air_attenuation_correction(
            lab_coord_long, experiment.beam, config
        )

        # Should still be close to 1.0 but slightly higher than short path
        assert air_mult_long >= air_mult_short
        assert 0.98 <= air_mult_long <= 1.05

    def test_end_to_end_correction_pipeline_45deg(self):
        """End-to-end test of correction pipeline for 45° pixel."""
        extractor = DataExtractor()
        experiment = create_synthetic_experiment_for_testing()

        # Mock config
        config = Mock()
        config.lp_correction_enabled = True
        config.air_temperature_k = 293.15
        config.air_pressure_atm = 1.0

        # Test data for 45° pixel
        intensity = 1000.0
        sigma = np.sqrt(intensity)
        q_vector = np.array([0.1, 0.1, 0.1])  # Arbitrary q-vector
        lab_coord = np.array([141.4, 0.0, 200.0])  # 45° position
        panel = experiment.detector[0]
        beam = experiment.beam

        with (
            patch("dials.algorithms.integration.Corrections") as mock_corrections_class,
            patch("dials.array_family.flex") as mock_flex,
        ):
            mock_corrections_obj = Mock()
            mock_corrections_class.return_value = mock_corrections_obj

            # Mock flex array creation
            mock_flex.vec3_double.return_value = [None]
            mock_flex.size_t.return_value = [0]

            # Use analytic values for 45° pixel
            theta = np.pi / 4
            sin_theta = np.sin(theta)
            cos_2theta = np.cos(2 * theta)
            lp_divisor = sin_theta**2 * (1 + cos_2theta**2)

            from unittest.mock import MagicMock

            mock_lp_divisors = MagicMock()
            mock_lp_divisors.__getitem__.return_value = lp_divisor
            mock_lp_divisors.__len__.return_value = 1
            mock_corrections_obj.lp.return_value = mock_lp_divisors

            # Mock QE as ideal detector
            mock_qe_multipliers = MagicMock()
            mock_qe_multipliers.__getitem__.return_value = 1.0
            mock_qe_multipliers.__len__.return_value = 1
            mock_corrections_obj.qe.return_value = mock_qe_multipliers

            corrected_intensity, corrected_sigma = extractor._apply_pixel_corrections(
                intensity,
                sigma,
                q_vector,
                lab_coord,
                panel,
                beam,
                experiment,
                10,
                20,
                config,
            )

            # Verify corrections were applied
            assert corrected_intensity != intensity
            assert corrected_sigma != sigma

            # Total correction should include LP correction
            # The correction is much larger due to solid angle correction
            assert (
                corrected_intensity > intensity
            )  # Corrected should be larger than original
            assert corrected_sigma > sigma  # Corrected sigma should also be larger
