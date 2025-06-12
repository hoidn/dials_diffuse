"""
Corrections helper module for diffuse scattering data processing.

This module provides helper functions for applying and testing pixel correction factors
as specified in Module 2.S.2 of the plan. It centralizes correction logic and provides
regression test capabilities.
"""

import numpy as np
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


def apply_corrections(
    raw_I: float, lp_mult: float, qe_mult: float, sa_mult: float, air_mult: float
) -> float:
    """
    Apply all pixel correction factors to raw intensity.

    This helper function centralizes the correction logic and ensures
    all corrections are applied as multipliers as per Module 0.7.

    Args:
        raw_I: Raw intensity value
        lp_mult: Lorentz-Polarization correction multiplier (1/LP_divisor)
        qe_mult: Quantum Efficiency correction multiplier
        sa_mult: Solid Angle correction multiplier (1/solid_angle)
        air_mult: Air Attenuation correction multiplier (1/attenuation)

    Returns:
        Corrected intensity value
    """
    total_correction_mult = lp_mult * qe_mult * sa_mult * air_mult
    corrected_I = raw_I * total_correction_mult

    logger.debug(
        f"Corrections applied: LP={lp_mult:.4f}, QE={qe_mult:.4f}, "
        f"SA={sa_mult:.4f}, Air={air_mult:.4f}, Total={total_correction_mult:.4f}"
    )

    return corrected_I


def calculate_analytic_pixel_corrections_45deg(
    raw_intensity: float,
    wavelength_angstrom: float,
    detector_distance_mm: float,
    pixel_size_mm: float,
    air_path_length_mm: float = None,
) -> Tuple[float, dict]:
    """
    Calculate analytic correction factors for a pixel at 45° scattering angle.

    This function provides known analytic values for regression testing
    of the correction pipeline as required by Module 2.S.2.

    Args:
        raw_intensity: Raw pixel intensity
        wavelength_angstrom: X-ray wavelength in Angstroms
        detector_distance_mm: Sample-to-detector distance in mm
        pixel_size_mm: Pixel size in mm (assumed square)
        air_path_length_mm: Air path length in mm (optional)

    Returns:
        Tuple of (corrected_intensity, correction_factors_dict)
    """

    # For 45° scattering angle
    theta = np.pi / 4  # 45 degrees
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    # 1. Lorentz-Polarization correction for 45° scattering
    # LP = 1 / (sin²θ * (1 + cos²(2θ)) for unpolarized X-rays
    sin2_theta = sin_theta**2
    cos_2theta = np.cos(2 * theta)
    lp_divisor = sin2_theta * (1 + cos_2theta**2)
    lp_mult = 1.0 / lp_divisor

    # 2. Quantum Efficiency correction (assume ideal detector)
    qe_mult = 1.0

    # 3. Solid Angle correction
    # For 45° scattering, pixel is at distance d/cos(45°) from sample
    actual_distance = detector_distance_mm / cos_theta
    pixel_area = pixel_size_mm**2
    solid_angle = pixel_area / (actual_distance**2)
    sa_mult = 1.0 / solid_angle

    # 4. Air Attenuation correction (if specified)
    air_mult = 1.0
    if air_path_length_mm is not None:
        # Simple approximation for air attenuation
        # Very rough approximation: μ ≈ 0.01 cm⁻¹ for ~10 keV X-rays in air
        mu_air_per_cm = 0.01
        path_length_cm = air_path_length_mm / 10.0
        attenuation = np.exp(-mu_air_per_cm * path_length_cm)
        air_mult = 1.0 / attenuation

    # Apply all corrections
    corrected_intensity = apply_corrections(
        raw_intensity, lp_mult, qe_mult, sa_mult, air_mult
    )

    correction_factors = {
        "lp_mult": lp_mult,
        "qe_mult": qe_mult,
        "sa_mult": sa_mult,
        "air_mult": air_mult,
        "total_mult": lp_mult * qe_mult * sa_mult * air_mult,
    }

    return corrected_intensity, correction_factors


def validate_correction_factors(
    correction_factors: dict, tolerance: float = 0.01
) -> Tuple[bool, str]:
    """
    Validate that correction factors are reasonable.

    Args:
        correction_factors: Dictionary of correction factors
        tolerance: Tolerance for validation checks

    Returns:
        Tuple of (is_valid, error_message)
    """

    # Check that all factors are positive
    for name, value in correction_factors.items():
        if value <= 0:
            return False, f"Correction factor {name} is not positive: {value}"

    # Check that LP correction is reasonable (should be between 0.1 and 10)
    lp_mult = correction_factors.get("lp_mult", 1.0)
    if lp_mult < 0.1 or lp_mult > 10.0:
        return False, f"LP correction factor unreasonable: {lp_mult}"

    # Check that QE correction is reasonable (should be between 0.1 and 2.0)
    qe_mult = correction_factors.get("qe_mult", 1.0)
    if qe_mult < 0.1 or qe_mult > 2.0:
        return False, f"QE correction factor unreasonable: {qe_mult}"

    # Check that SA correction is reasonable (depends on geometry but shouldn't be extreme)
    sa_mult = correction_factors.get("sa_mult", 1.0)
    if sa_mult < 1e-6 or sa_mult > 1e6:
        return False, f"Solid angle correction factor unreasonable: {sa_mult}"

    # Check that Air correction is close to 1 (small correction for typical distances)
    air_mult = correction_factors.get("air_mult", 1.0)
    if air_mult < 0.9 or air_mult > 1.2:
        return False, f"Air attenuation correction factor unreasonable: {air_mult}"

    return True, "All correction factors are reasonable"


def create_synthetic_experiment_for_testing():
    """
    Create a synthetic DIALS-like experiment object for testing.

    Returns:
        Mock experiment object with proper geometric parameters
    """
    from unittest.mock import Mock

    # Create mock experiment
    experiment = Mock()

    # Mock beam at 45° geometry
    beam = Mock()
    beam.get_wavelength.return_value = 1.0  # 1 Angstrom
    beam.get_s0.return_value = [0, 0, -1]  # Beam along -z
    experiment.beam = beam

    # Mock detector panel positioned for 45° scattering
    panel = Mock()
    panel.get_pixel_size.return_value = (0.172, 0.172)  # mm
    # Position pixel at 45° scattering angle
    # For 200mm detector distance, 45° pixel is at (141.4, 0, 200)
    panel.get_pixel_lab_coord.return_value = [141.4, 0.0, 200.0]  # mm
    panel.get_fast_axis.return_value = [1, 0, 0]
    panel.get_slow_axis.return_value = [0, 1, 0]

    # Mock detector with proper magic method support
    detector = Mock()
    detector.__getitem__ = Mock(return_value=panel)
    detector.__len__ = Mock(return_value=1)
    experiment.detector = detector

    # Mock crystal
    crystal = Mock()
    experiment.crystal = crystal

    # Mock goniometer (None for stills)
    experiment.goniometer = None

    return experiment
