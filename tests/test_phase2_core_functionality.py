"""
Core functionality tests for Phase 2 implementation.

Tests the essential Phase 2 functionality without requiring DIALS dependencies,
focusing on the correction calculation logic and data flow.
"""

import pytest
import numpy as np
from unittest.mock import Mock

from diffusepipe.extraction.data_extractor import DataExtractor
from diffusepipe.types.types_IDL import ExtractionConfig
from diffusepipe.corrections import (
    apply_corrections,
    calculate_analytic_pixel_corrections_45deg,
    validate_correction_factors,
)


class TestPhase2CoreFunctionality:
    """Test core Phase 2 functionality."""

    def test_correction_helper_function(self):
        """Test the apply_corrections helper function."""
        raw_intensity = 1000.0
        lp_mult = 0.5
        qe_mult = 0.8
        sa_mult = 1.2
        air_mult = 1.05
        
        corrected = apply_corrections(raw_intensity, lp_mult, qe_mult, sa_mult, air_mult)
        expected = raw_intensity * lp_mult * qe_mult * sa_mult * air_mult
        
        assert abs(corrected - expected) < 1e-10
        assert corrected == 1000.0 * 0.5 * 0.8 * 1.2 * 1.05  # 504.0

    def test_solid_angle_correction_calculation(self):
        """Test solid angle correction calculation with mock panel."""
        extractor = DataExtractor()
        
        # Create mock panel
        panel = Mock()
        panel.get_pixel_size.return_value = (0.172, 0.172)  # mm
        panel.get_fast_axis.return_value = [1, 0, 0]
        panel.get_slow_axis.return_value = [0, 1, 0]
        
        # Test pixel position
        lab_coord = np.array([100.0, 50.0, 200.0])  # mm
        
        sa_mult = extractor._calculate_solid_angle_correction(lab_coord, panel, 10, 20)
        
        # Should return reasonable correction factor
        assert sa_mult > 0
        assert sa_mult < 1e7  # Sanity check - can be large for small pixels
        
        # Manual calculation to verify
        pixel_area = 0.172 * 0.172  # mm²
        r = np.linalg.norm(lab_coord)
        normal = np.array([0, 0, 1])  # z-axis from cross product of x and y
        scatter_direction = lab_coord / r
        cos_theta = abs(np.dot(normal, scatter_direction))
        expected_solid_angle = (pixel_area * cos_theta) / (r * r)
        expected_sa_mult = 1.0 / expected_solid_angle
        
        assert abs(sa_mult - expected_sa_mult) / expected_sa_mult < 0.01

    def test_air_attenuation_coefficient_with_parameters(self):
        """Test air attenuation coefficient calculation with different parameters."""
        extractor = DataExtractor()
        
        energy_ev = 12000  # 12 keV
        
        # Standard conditions
        mu_standard = extractor._calculate_air_attenuation_coefficient(energy_ev, 293.15, 1.0)
        
        # Higher pressure should increase attenuation
        mu_high_pressure = extractor._calculate_air_attenuation_coefficient(energy_ev, 293.15, 2.0)
        assert mu_high_pressure > mu_standard
        
        # Higher temperature should decrease attenuation  
        mu_high_temp = extractor._calculate_air_attenuation_coefficient(energy_ev, 350.0, 1.0)
        assert mu_high_temp < mu_standard
        
        # All should be reasonable values
        assert 0.0001 < mu_standard < 1.0
        assert 0.0001 < mu_high_pressure < 2.0
        assert 0.0001 < mu_high_temp < 1.0

    def test_air_attenuation_correction_calculation(self):
        """Test air attenuation correction calculation with mock beam."""
        extractor = DataExtractor()
        
        # Create mock beam
        beam = Mock()
        beam.get_wavelength.return_value = 1.0  # 1 Angstrom
        
        # Create mock config
        config = Mock()
        config.air_temperature_k = 293.15
        config.air_pressure_atm = 1.0
        
        # Test short path (minimal attenuation)
        lab_coord_short = np.array([100.0, 0.0, 100.0])
        air_mult_short = extractor._calculate_air_attenuation_correction(lab_coord_short, beam, config)
        
        # Should be close to 1.0 for short paths
        assert 0.99 <= air_mult_short <= 1.01
        
        # Test longer path
        lab_coord_long = np.array([500.0, 0.0, 500.0])
        air_mult_long = extractor._calculate_air_attenuation_correction(lab_coord_long, beam, config)
        
        # Should be slightly higher (more correction) for longer path
        assert air_mult_long >= air_mult_short
        assert 0.98 <= air_mult_long <= 1.05

    def test_mask_total_2d_parameter_handling(self):
        """Test that DataExtractor properly handles the new mask_total_2d parameter."""
        extractor = DataExtractor()
        
        # Test validation with mask_total_2d provided
        from diffusepipe.types.types_IDL import ComponentInputFiles
        
        inputs = ComponentInputFiles(
            cbf_image_path="/fake/path.cbf",
            dials_expt_path="/fake/path.expt"
            # Note: no bragg_mask_path
        )
        
        config = ExtractionConfig(
            gain=1.0,
            cell_length_tol=0.01,
            cell_angle_tol=0.1,
            orient_tolerance_deg=1.0,
            q_consistency_tolerance_angstrom_inv=0.01,
            pixel_step=1,
            lp_correction_enabled=True,
            plot_diagnostics=False,
            verbose=False
        )
        
        # Create mock mask
        mock_mask = np.zeros((100, 100), dtype=bool)
        mask_total_2d = (mock_mask,)
        
        # This should pass validation since mask_total_2d is provided
        result = extractor._validate_inputs(inputs, config, "/tmp/output.npz", mask_total_2d)
        
        # Note: This will fail due to file existence checks, but the mask validation should pass
        # The important thing is it doesn't fail due to missing bragg_mask_path
        assert "mask_total_2d not passed" not in result.message

    def test_analytic_45deg_correction_calculation(self):
        """Test the analytic 45° correction calculation."""
        raw_intensity = 1000.0
        wavelength = 1.0  # Angstrom
        detector_distance = 200.0  # mm
        pixel_size = 0.172  # mm
        
        corrected_intensity, factors = calculate_analytic_pixel_corrections_45deg(
            raw_intensity, wavelength, detector_distance, pixel_size
        )
        
        # Verify factors are reasonable
        assert 0.1 <= factors['lp_mult'] <= 10.0
        assert factors['qe_mult'] == 1.0  # Ideal detector
        assert factors['sa_mult'] > 0
        assert factors['air_mult'] == 1.0  # No air path specified
        
        # Verify correction was applied
        expected_total = factors['lp_mult'] * factors['qe_mult'] * factors['sa_mult'] * factors['air_mult']
        assert abs(factors['total_mult'] - expected_total) < 1e-10
        assert abs(corrected_intensity - raw_intensity * expected_total) < 1e-10

    def test_correction_factors_validation(self):
        """Test correction factor validation logic."""
        # Valid factors
        valid_factors = {
            'lp_mult': 1.5,
            'qe_mult': 0.8,
            'sa_mult': 1000.0,
            'air_mult': 1.02
        }
        
        is_valid, message = validate_correction_factors(valid_factors)
        assert is_valid
        assert "reasonable" in message
        
        # Test various invalid cases
        invalid_cases = [
            ({'lp_mult': -0.5, 'qe_mult': 0.8, 'sa_mult': 1.0, 'air_mult': 1.0}, "not positive"),
            ({'lp_mult': 15.0, 'qe_mult': 0.8, 'sa_mult': 1.0, 'air_mult': 1.0}, "unreasonable"),
            ({'lp_mult': 1.0, 'qe_mult': 5.0, 'sa_mult': 1.0, 'air_mult': 1.0}, "unreasonable"),
            ({'lp_mult': 1.0, 'qe_mult': 0.8, 'sa_mult': 1e7, 'air_mult': 1.0}, "unreasonable"),
            ({'lp_mult': 1.0, 'qe_mult': 0.8, 'sa_mult': 1.0, 'air_mult': 2.0}, "unreasonable"),
        ]
        
        for invalid_factors, expected_error in invalid_cases:
            is_valid, message = validate_correction_factors(invalid_factors)
            assert not is_valid
            assert expected_error in message

    def test_configuration_parameters_added(self):
        """Test that new configuration parameters are properly handled."""
        # Test that ExtractionConfig accepts the new air parameters
        config = ExtractionConfig(
            gain=1.0,
            cell_length_tol=0.01,
            cell_angle_tol=0.1,
            orient_tolerance_deg=1.0,
            q_consistency_tolerance_angstrom_inv=0.01,
            pixel_step=1,
            lp_correction_enabled=True,
            plot_diagnostics=False,
            verbose=False,
            air_temperature_k=300.0,  # Custom temperature
            air_pressure_atm=0.8      # Custom pressure
        )
        
        assert config.air_temperature_k == 300.0
        assert config.air_pressure_atm == 0.8
        
        # Test defaults
        config_default = ExtractionConfig(
            gain=1.0,
            cell_length_tol=0.01,
            cell_angle_tol=0.1,
            orient_tolerance_deg=1.0,
            q_consistency_tolerance_angstrom_inv=0.01,
            pixel_step=1,
            lp_correction_enabled=True,
            plot_diagnostics=False,
            verbose=False
        )
        
        assert config_default.air_temperature_k == 293.15  # Default 20°C
        assert config_default.air_pressure_atm == 1.0       # Default 1 atm