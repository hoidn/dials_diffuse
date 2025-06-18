"""
Tests for DiffuseScalingModel and components.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from dials.algorithms.scaling.active_parameter_managers import active_parameter_manager
from dials.array_family import flex

from diffusepipe.scaling.diffuse_scaling_model import DiffuseScalingModel
from diffusepipe.scaling.components.per_still_multiplier import (
    PerStillMultiplierComponent,
)
from diffusepipe.scaling.components.resolution_smoother import (
    ResolutionSmootherComponent,
)


class TestPerStillMultiplierComponent:
    """Test PerStillMultiplierComponent functionality."""

    @pytest.fixture
    def parameter_manager(self):
        """Create active parameter manager for testing."""
        # Create mock components and selection list for the parameter manager
        mock_components = {}  # Dictionary of components
        selection_list = []
        return active_parameter_manager(mock_components, selection_list)

    @pytest.fixture
    def component(self):
        """Create component for testing."""
        still_ids = [1, 2, 3]
        return PerStillMultiplierComponent(still_ids)

    def test_initialization(self, component):
        """Test component initialization."""
        assert component.n_params == 3
        assert component.n_stills == 3
        assert len(component.still_ids) == 3

        # Check initial parameters (should be 1.0)
        params = component.parameters
        assert len(params) == 3
        assert all(abs(p - 1.0) < 1e-6 for p in params)

    def test_initialization_with_custom_values(self):
        """Test initialization with custom initial values."""
        still_ids = [1, 2]
        initial_values = {1: 1.5, 2: 0.8}

        component = PerStillMultiplierComponent(still_ids, initial_values)

        assert component.get_scale_for_still(1) == 1.5
        assert component.get_scale_for_still(2) == 0.8

    def test_scale_calculation_flex_table(self, component):
        """Test scale calculation with DIALS-like reflection table."""
        # Mock reflection table
        reflection_table = Mock()
        still_ids = flex.int([1, 2, 3, 1, 2])
        reflection_table.get.return_value = still_ids

        scales, derivatives = component.calculate_scales_and_derivatives(
            reflection_table
        )

        assert len(scales) == 5
        assert len(derivatives) == 5
        assert all(abs(s - 1.0) < 1e-6 for s in scales)  # Initial scales
        assert all(abs(d - 1.0) < 1e-6 for d in derivatives)  # Unit derivatives

    def test_scale_calculation_dict_format(self, component):
        """Test scale calculation with dictionary format."""
        data_dict = {
            "still_ids": np.array([1, 2, 3, 1]),
            "intensities": np.array([100, 200, 150, 120]),
        }

        scales, derivatives = component.calculate_scales_and_derivatives(data_dict)

        assert len(scales) == 4
        assert all(abs(s - 1.0) < 1e-6 for s in scales)

    def test_unknown_still_handling(self, component):
        """Test handling of unknown still IDs."""
        data_dict = {
            "still_ids": np.array([1, 999]),  # 999 is unknown
            "intensities": np.array([100, 200]),
        }

        with patch(
            "diffusepipe.scaling.components.per_still_multiplier.logger"
        ) as mock_logger:
            scales, derivatives = component.calculate_scales_and_derivatives(data_dict)

            assert len(scales) == 2
            assert abs(scales[0] - 1.0) < 1e-6  # Known still
            assert abs(scales[1] - 1.0) < 1e-6  # Unknown still (unity scale)

            # Should have logged warning
            mock_logger.warning.assert_called()

    def test_get_set_scale_for_still(self, component):
        """Test getting and setting scales for individual stills."""
        # Test getting initial scale
        assert component.get_scale_for_still(1) == 1.0

        # Test setting new scale
        component.set_scale_for_still(1, 1.5)
        assert component.get_scale_for_still(1) == 1.5

        # Test error for unknown still
        assert component.get_scale_for_still(999) == 1.0  # Default

        with pytest.raises(ValueError):
            component.set_scale_for_still(999, 1.5)

        # Test error for negative scale
        with pytest.raises(ValueError):
            component.set_scale_for_still(1, -0.5)

    def test_component_info(self, component):
        """Test component information retrieval."""
        info = component.get_component_info()

        required_keys = [
            "component_type",
            "n_parameters",
            "n_stills",
            "still_ids",
            "current_scales",
            "scale_statistics",
        ]

        for key in required_keys:
            assert key in info

        assert info["component_type"] == "PerStillMultiplier"
        assert info["n_parameters"] == 3
        assert len(info["current_scales"]) == 3


class TestResolutionSmootherComponent:
    """Test ResolutionSmootherComponent functionality."""

    @pytest.fixture
    def parameter_manager(self):
        """Create active parameter manager for testing."""
        # Create mock components and selection list for the parameter manager
        mock_components = {}  # Dictionary of components
        selection_list = []
        return active_parameter_manager(mock_components, selection_list)

    @pytest.fixture
    def component(self):
        """Create component for testing."""
        return ResolutionSmootherComponent(
            n_control_points=3, resolution_range=(0.1, 1.0)
        )

    def test_initialization(self, component):
        """Test component initialization."""
        assert component.n_params == 3
        assert component.n_control_points == 3
        assert component.q_min == 0.1
        assert component.q_max == 1.0

        # Check initial parameters (should be 1.0)
        params = component.parameters
        assert len(params) == 3
        assert all(abs(p - 1.0) < 1e-6 for p in params)

    def test_v1_constraints(self):
        """Test v1 parameter constraints."""
        # Test exceeding control point limit
        with pytest.raises(ValueError, match="v1 model allows ≤5 control points"):
            ResolutionSmootherComponent(n_control_points=6, resolution_range=(0.1, 1.0))

        # Test invalid resolution range
        with pytest.raises(ValueError, match="Invalid resolution range"):
            ResolutionSmootherComponent(
                n_control_points=3,
                resolution_range=(1.0, 0.1),  # Invalid: min >= max
            )

    def test_q_magnitude_extraction_dict(self, component):
        """Test q-magnitude extraction from dictionary format."""
        data_dict = {
            "q_vectors_lab": np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
            "still_ids": np.array([1, 2]),
        }

        q_mags = component._extract_q_magnitudes(data_dict)

        assert len(q_mags) == 2
        expected_mag_1 = np.linalg.norm([0.1, 0.2, 0.3])
        expected_mag_2 = np.linalg.norm([0.4, 0.5, 0.6])
        assert abs(q_mags[0] - expected_mag_1) < 1e-6
        assert abs(q_mags[1] - expected_mag_2) < 1e-6

    def test_q_magnitude_extraction_direct(self, component):
        """Test q-magnitude extraction from direct q_magnitudes."""
        data_dict = {"q_magnitudes": [0.5, 0.8, 1.2], "still_ids": np.array([1, 2, 3])}

        q_mags = component._extract_q_magnitudes(data_dict)

        assert q_mags == [0.5, 0.8, 1.2]

    def test_scale_calculation(self, component):
        """Test resolution-dependent scale calculation."""
        data_dict = {
            "q_vectors_lab": np.array([[0.2, 0.0, 0.0], [0.5, 0.0, 0.0]]),
            "still_ids": np.array([1, 2]),
        }

        scales, derivatives = component.calculate_scales_and_derivatives(data_dict)

        assert len(scales) == 2
        assert all(s > 0 for s in scales)  # Should be positive

        # Verify derivatives array has correct shape (n_obs x n_params) and is not all zeros
        n_obs = len(scales)
        n_params = component.n_params
        
        # The derivatives should be a 2D structure when used, but may be flattened in flex array
        # Check that we have the right total number of elements
        expected_size = n_obs * n_params
        assert derivatives.size() == expected_size, f"Expected {expected_size} derivatives, got {derivatives.size()}"
        
        # Verify that analytical derivatives are being calculated (not all zeros)
        assert not all(abs(d) < 1e-12 for d in derivatives), "Analytical derivatives should not be all zeros"
        
        # Additional verification: the derivatives should have meaningful values
        # Since we're using the DIALS smoother, we expect non-trivial derivative calculations
        non_zero_derivatives = [d for d in derivatives if abs(d) > 1e-12]
        assert len(non_zero_derivatives) > 0, "Should have some non-zero analytical derivatives"

    def test_get_scale_for_q(self, component):
        """Test getting scale for specific q-value."""
        scale_1 = component.get_scale_for_q(0.5)
        scale_2 = component.get_scale_for_q(0.8)

        assert scale_1 > 0
        assert scale_2 > 0

        # Test edge cases
        assert component.get_scale_for_q(0.0) == 1.0  # Invalid q
        assert component.get_scale_for_q(-0.1) == 1.0  # Negative q

    def test_component_info(self, component):
        """Test component information retrieval."""
        info = component.get_component_info()

        required_keys = [
            "component_type",
            "n_parameters",
            "n_control_points",
            "q_range",
            "control_point_values",
            "scale_statistics",
        ]

        for key in required_keys:
            assert key in info

        assert info["component_type"] == "ResolutionSmoother"
        assert info["n_control_points"] == 3


class TestDiffuseScalingModel:
    """Test DiffuseScalingModel functionality."""

    @pytest.fixture
    def basic_config(self):
        """Create basic configuration for testing."""
        return {
            "still_ids": [1, 2, 3],
            "per_still_scale": {"enabled": True},
            "resolution_smoother": {"enabled": False},
            "experimental_components": {
                "panel_scale": {"enabled": False},
                "spatial_scale": {"enabled": False},
                "additive_offset": {"enabled": False},
            },
            "partiality_threshold": 0.1,
        }

    @pytest.fixture
    def model(self, basic_config):
        """Create model for testing."""
        return DiffuseScalingModel(basic_config)

    def test_basic_initialization(self, model):
        """Test basic model initialization."""
        assert model.n_total_params == 3  # One per still
        assert "per_still" in model.diffuse_components
        assert "resolution" not in model.diffuse_components
        assert model.partiality_threshold == 0.1

    def test_v1_constraint_validation(self):
        """Test v1 constraint validation."""
        # Test forbidden component
        invalid_config = {
            "still_ids": [1, 2],
            "experimental_components": {
                "additive_offset": {"enabled": True}  # Forbidden in v1
            },
        }

        with pytest.raises(
            ValueError, match="additive_offset component is hard-disabled"
        ):
            DiffuseScalingModel(invalid_config)

        # Test resolution smoother constraint
        invalid_config = {
            "still_ids": [1, 2],
            "resolution_smoother": {
                "enabled": True,
                "n_control_points": 6,  # Exceeds limit
            },
        }

        with pytest.raises(
            ValueError, match="resolution smoother limited to ≤5 points"
        ):
            DiffuseScalingModel(invalid_config)

    def test_parameter_limit_enforcement(self):
        """Test parameter limit enforcement."""
        # Create config that would exceed parameter limit
        many_stills = list(range(100))  # 100 stills
        config = {
            "still_ids": many_stills,
            "per_still_scale": {"enabled": True},
            "resolution_smoother": {
                "enabled": True,
                "n_control_points": 5,  # Max allowed
            },
        }

        # Should exceed MAX_FREE_PARAMS_BASE + n_stills = 5 + 100 = 105
        # Total would be 100 (stills) + 5 (resolution) = 105, which should be OK
        model = DiffuseScalingModel(config)
        assert model.n_total_params == 105

    def test_get_scales_for_observation(self, model):
        """Test getting scales for individual observations."""
        mult_scale, add_offset = model.get_scales_for_observation(1, 0.5)

        assert mult_scale > 0
        assert add_offset == 0.0  # Always 0 in v1

    def test_model_with_resolution_smoother(self):
        """Test model with resolution smoother enabled."""
        config = {
            "still_ids": [1, 2],
            "per_still_scale": {"enabled": True},
            "resolution_smoother": {
                "enabled": True,
                "n_control_points": 3,
                "resolution_range": (0.1, 1.0),
            },
        }

        model = DiffuseScalingModel(config)

        assert model.n_total_params == 5  # 2 stills + 3 resolution points
        assert "resolution" in model.diffuse_components

    def test_generate_references(self, model):
        """Test reference generation."""
        # Create mock binned data
        binned_data = {
            0: {
                "observations": [
                    {
                        "intensity": 100.0,
                        "sigma": 10.0,
                        "still_id": 1,
                        "q_vector_lab": np.array([0.1, 0.1, 0.1]),
                    },
                    {
                        "intensity": 120.0,
                        "sigma": 12.0,
                        "still_id": 2,
                        "q_vector_lab": np.array([0.1, 0.1, 0.1]),
                    },
                ]
            }
        }

        bragg_refs, diffuse_refs = model.generate_references(binned_data, {})

        assert isinstance(diffuse_refs, dict)
        assert 0 in diffuse_refs
        assert "intensity" in diffuse_refs[0]
        assert "sigma" in diffuse_refs[0]
        assert "n_observations" in diffuse_refs[0]

    def test_model_info(self, model):
        """Test model information retrieval."""
        info = model.get_model_info()

        required_keys = [
            "model_type",
            "n_total_params",
            "components",
            "refinement_statistics",
            "partiality_threshold",
        ]

        for key in required_keys:
            assert key in info

        assert info["model_type"] == "DiffuseScalingModel_v1"
        assert info["n_total_params"] == 3

    def test_refinement_actually_refines(self):
        """Test that the refinement process actually improves scale factors."""
        # Create synthetic binned pixel data with known scale factors applied
        # Still 0 has scale factor 0.8, Still 1 has scale factor 1.2
        synthetic_scales = {0: 0.8, 1: 1.2}

        # Create configuration for model
        config = {
            "still_ids": [0, 1],
            "per_still_scale": {"enabled": True},
            "resolution_smoother": {"enabled": False},
            "experimental_components": {
                "panel_scale": {"enabled": False},
                "spatial_scale": {"enabled": False},
                "additive_offset": {"enabled": False},
            },
            "partiality_threshold": 0.1,
        }

        # Create model
        model = DiffuseScalingModel(config)

        # Create synthetic binned pixel data
        # Base intensity for reference
        base_intensity = 100.0

        binned_pixel_data = {
            0: {  # Voxel 0
                "observations": [
                    {
                        "intensity": base_intensity
                        * synthetic_scales[0],  # Still 0 scaled down
                        "sigma": 10.0,
                        "still_id": 0,
                        "q_vector_lab": np.array([0.1, 0.1, 0.1]),
                    },
                    {
                        "intensity": base_intensity
                        * synthetic_scales[1],  # Still 1 scaled up
                        "sigma": 10.0,
                        "still_id": 1,
                        "q_vector_lab": np.array([0.1, 0.1, 0.1]),
                    },
                ]
            },
            1: {  # Voxel 1
                "observations": [
                    {
                        "intensity": 150.0 * synthetic_scales[0],  # Still 0 scaled down
                        "sigma": 12.0,
                        "still_id": 0,
                        "q_vector_lab": np.array([0.2, 0.2, 0.2]),
                    },
                    {
                        "intensity": 150.0 * synthetic_scales[1],  # Still 1 scaled up
                        "sigma": 12.0,
                        "still_id": 1,
                        "q_vector_lab": np.array([0.2, 0.2, 0.2]),
                    },
                ]
            },
        }

        # Empty bragg reflections for this test
        bragg_reflections = {}

        # Refinement configuration
        refinement_config = {
            "max_iterations": 20,
            "convergence_tolerance": 1e-4,
            "parameter_shift_tolerance": 1e-4,
        }

        # Calculate initial R-factor
        bragg_refs, diffuse_refs = model.generate_references(
            binned_pixel_data, bragg_reflections
        )
        initial_r_factor = model._calculate_r_factor(binned_pixel_data, diffuse_refs)

        # Run refinement
        refined_params, refinement_stats = model.refine_parameters(
            binned_pixel_data, bragg_reflections, refinement_config
        )

        # Verify that refinement improved R-factor
        final_r_factor = refinement_stats["final_r_factor"]
        assert (
            final_r_factor < initial_r_factor
        ), f"R-factor should improve: {final_r_factor} < {initial_r_factor}"

        # Most importantly, verify that refinement actually changed the parameters from initial values
        # and improved the R-factor
        print(
            f"Initial R-factor: {initial_r_factor:.6f}, Final R-factor: {final_r_factor:.6f}"
        )
        print(
            f"Refined scales: Still 0 = {refined_params[0]['multiplicative_scale']:.3f}, Still 1 = {refined_params[1]['multiplicative_scale']:.3f}"
        )

        # Verify parameters changed from initial values (1.0)
        for still_id in [0, 1]:
            refined_scale = refined_params[still_id]["multiplicative_scale"]
            assert (
                abs(refined_scale - 1.0) > 0.01
            ), f"Still {still_id} scale should change from initial 1.0, got {refined_scale:.3f}"

        # Verify convergence was achieved
        assert refinement_stats["convergence_achieved"], "Refinement should converge"

        # Verify additive offsets are zero (v1 model constraint)
        for still_id in [0, 1]:
            assert refined_params[still_id]["additive_offset"] == 0.0
