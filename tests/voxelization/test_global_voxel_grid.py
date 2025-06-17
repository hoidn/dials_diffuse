"""
Tests for GlobalVoxelGrid implementation.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from scitbx import matrix
from cctbx import uctbx
from dxtbx.model import Experiment, Crystal

from diffusepipe.voxelization.global_voxel_grid import (
    GlobalVoxelGrid,
    GlobalVoxelGridConfig,
    CorrectedDiffusePixelData,
)


class TestGlobalVoxelGrid:
    """Test GlobalVoxelGrid functionality."""

    @pytest.fixture
    def mock_crystal_models(self):
        """Create mock crystal models for testing."""
        crystals = []
        for i in range(3):
            crystal = Mock(spec=Crystal)
            # Mock unit cell with slight variations
            unit_cell = uctbx.unit_cell(
                (
                    10.0 + i * 0.1,  # a
                    15.0 + i * 0.1,  # b
                    20.0 + i * 0.1,  # c
                    90.0,  # alpha
                    90.0,  # beta
                    90.0,  # gamma
                )
            )
            crystal.get_unit_cell.return_value = unit_cell

            # Mock U matrix (identity with small rotations)
            u_matrix = matrix.rec(
                (1.0, 0.01 * i, 0.0, -0.01 * i, 1.0, 0.0, 0.0, 0.0, 1.0), (3, 3)
            )
            crystal.get_U.return_value = u_matrix

            # Mock space group
            from cctbx.sgtbx import space_group_info

            crystal.get_space_group.return_value = space_group_info("P1").group()

            crystals.append(crystal)

        return crystals

    @pytest.fixture
    def mock_experiments(self, mock_crystal_models):
        """Create mock experiments with crystal models."""
        experiments = []
        for crystal in mock_crystal_models:
            exp = Mock(spec=Experiment)
            exp.crystal = crystal
            experiments.append(exp)
        return experiments

    @pytest.fixture
    def sample_diffuse_data(self):
        """Create sample corrected diffuse pixel data."""
        # Generate sample q-vectors in a reasonable range
        n_points = 100
        q_vectors = np.random.normal(0, 0.5, (n_points, 3))  # Around origin
        intensities = np.random.exponential(100, n_points)
        sigmas = np.sqrt(intensities) + 1.0
        still_ids = np.random.randint(0, 3, n_points)

        return [
            CorrectedDiffusePixelData(
                q_vectors=q_vectors,
                intensities=intensities,
                sigmas=sigmas,
                still_ids=still_ids,
            )
        ]

    @pytest.fixture
    def grid_config(self):
        """Create test grid configuration."""
        return GlobalVoxelGridConfig(
            d_min_target=1.0,
            d_max_target=10.0,
            ndiv_h=2,
            ndiv_k=2,
            ndiv_l=2,
            max_rms_delta_hkl=0.1,
        )

    def test_grid_initialization(
        self, mock_experiments, sample_diffuse_data, grid_config
    ):
        """Test basic grid initialization."""
        grid = GlobalVoxelGrid(mock_experiments, sample_diffuse_data, grid_config)

        assert grid.crystal_avg_ref is not None
        assert grid.A_avg_ref is not None
        assert hasattr(grid, "hkl_min")
        assert hasattr(grid, "hkl_max")
        assert grid.total_voxels > 0

    def test_config_validation(self, mock_experiments, sample_diffuse_data):
        """Test grid configuration validation."""
        # Test invalid d_min
        with pytest.raises(ValueError, match="d_min_target must be positive"):
            invalid_config = GlobalVoxelGridConfig(
                d_min_target=-1.0, d_max_target=10.0, ndiv_h=2, ndiv_k=2, ndiv_l=2
            )
            GlobalVoxelGrid(mock_experiments, sample_diffuse_data, invalid_config)

        # Test invalid d_max
        with pytest.raises(
            ValueError, match="d_max_target must be greater than d_min_target"
        ):
            invalid_config = GlobalVoxelGridConfig(
                d_min_target=10.0, d_max_target=5.0, ndiv_h=2, ndiv_k=2, ndiv_l=2
            )
            GlobalVoxelGrid(mock_experiments, sample_diffuse_data, invalid_config)

        # Test invalid subdivisions
        with pytest.raises(ValueError, match="Grid subdivisions must be positive"):
            invalid_config = GlobalVoxelGridConfig(
                d_min_target=1.0, d_max_target=10.0, ndiv_h=0, ndiv_k=2, ndiv_l=2
            )
            GlobalVoxelGrid(mock_experiments, sample_diffuse_data, invalid_config)

    def test_input_validation(self, sample_diffuse_data, grid_config):
        """Test input validation."""
        # Test empty experiment list
        with pytest.raises(ValueError, match="experiment_list cannot be empty"):
            GlobalVoxelGrid([], sample_diffuse_data, grid_config)

        # Test empty diffuse data
        with pytest.raises(
            ValueError, match="corrected_diffuse_pixel_data cannot be empty"
        ):
            mock_exp = Mock(spec=Experiment)
            mock_exp.crystal = Mock(spec=Crystal)
            GlobalVoxelGrid([mock_exp], [], grid_config)

    def test_hkl_voxel_conversion(
        self, mock_experiments, sample_diffuse_data, grid_config
    ):
        """Test HKL to voxel index conversion and back."""
        grid = GlobalVoxelGrid(mock_experiments, sample_diffuse_data, grid_config)

        # Test conversion consistency
        test_hkl = (0.5, -0.3, 1.2)
        voxel_idx = grid.hkl_to_voxel_idx(*test_hkl)
        recovered_hkl = grid.voxel_idx_to_hkl_center(voxel_idx)

        # Should be close due to rounding
        assert abs(recovered_hkl[0] - test_hkl[0]) < 1.0
        assert abs(recovered_hkl[1] - test_hkl[1]) < 1.0
        assert abs(recovered_hkl[2] - test_hkl[2]) < 1.0

    def test_q_vector_calculation(
        self, mock_experiments, sample_diffuse_data, grid_config
    ):
        """Test q-vector calculation for voxel centers."""
        grid = GlobalVoxelGrid(mock_experiments, sample_diffuse_data, grid_config)

        # Test first voxel
        voxel_idx = 0
        q_vector = grid.get_q_vector_for_voxel_center(voxel_idx)

        assert hasattr(q_vector, "length")
        assert q_vector.length() >= 0

    def test_crystal_averaging_diagnostics(
        self, mock_experiments, sample_diffuse_data, grid_config
    ):
        """Test crystal averaging diagnostic calculations."""
        grid = GlobalVoxelGrid(mock_experiments, sample_diffuse_data, grid_config)

        diagnostics = grid.get_crystal_averaging_diagnostics()

        required_keys = [
            "rms_misorientation_deg",
            "n_crystals_averaged",
            "hkl_range_min",
            "hkl_range_max",
            "total_voxels",
        ]

        for key in required_keys:
            assert key in diagnostics

        assert diagnostics["n_crystals_averaged"] == 3
        assert diagnostics["total_voxels"] > 0

    def test_unit_cell_averaging(self, mock_crystal_models):
        """Test unit cell averaging functionality."""
        # Create a minimal grid to test unit cell averaging
        experiments = []
        for crystal in mock_crystal_models:
            exp = Mock(spec=Experiment)
            exp.crystal = crystal
            experiments.append(exp)

        # Simple diffuse data
        diffuse_data = [
            CorrectedDiffusePixelData(
                q_vectors=np.array([[0.1, 0.1, 0.1]]),
                intensities=np.array([100.0]),
                sigmas=np.array([10.0]),
                still_ids=np.array([0]),
            )
        ]

        config = GlobalVoxelGridConfig(
            d_min_target=1.0, d_max_target=10.0, ndiv_h=1, ndiv_k=1, ndiv_l=1
        )

        grid = GlobalVoxelGrid(experiments, diffuse_data, config)

        # Check that averaging happened
        assert grid.crystal_avg_ref is not None
        assert grid.diagnostics["n_crystals_averaged"] == 3

    def test_orientation_spread_warning(
        self, mock_experiments, sample_diffuse_data, grid_config
    ):
        """Test orientation spread warning for large misorientations."""
        # Modify one crystal to have moderate misorientation (1.5° rotation)
        # This should trigger warning but not ValueError (threshold is 2.0°)
        angle_rad = np.radians(1.5)  # 1.5 degrees - enough for warning, less than 2° error threshold
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        moderate_rotation = matrix.rec(
            (cos_a, -sin_a, 0.0, sin_a, cos_a, 0.0, 0.0, 0.0, 1.0), (3, 3)
        )
        mock_experiments[2].crystal.get_U.return_value = moderate_rotation

        with patch("diffusepipe.voxelization.global_voxel_grid.logger") as mock_logger:
            grid = GlobalVoxelGrid(mock_experiments, sample_diffuse_data, grid_config)

            # Check if warning was logged (misorientation should be > 1.0° but < 2.0°)
            warning_calls = [
                call
                for call in mock_logger.warning.call_args_list
                if "misorientation" in str(call)
            ]
            assert len(warning_calls) > 0 or grid.rms_misorientation_deg > 1.0

    def test_grid_initialization_raises_error_on_high_misorientation(
        self, mock_experiments, sample_diffuse_data, grid_config
    ):
        """Test that GlobalVoxelGrid raises ValueError when RMS misorientation exceeds threshold."""
        # Create crystal models with very high misorientation (>2°)
        # Use larger rotation angles to ensure we exceed the 2.0° threshold
        high_misorientation_crystals = []

        for i, exp in enumerate(mock_experiments):
            crystal = Mock(spec=Crystal)

            # Mock unit cell (same for all)
            unit_cell = uctbx.unit_cell((10.0, 15.0, 20.0, 90.0, 90.0, 90.0))
            crystal.get_unit_cell.return_value = unit_cell

            # Create rotations with increasing misorientation
            if i == 0:
                # Reference crystal (identity)
                u_matrix = matrix.rec(
                    (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0), (3, 3)
                )
            elif i == 1:
                # 15° rotation around z-axis (should exceed 2° threshold)
                angle_rad = np.radians(15.0)
                cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
                u_matrix = matrix.rec(
                    (cos_a, -sin_a, 0.0, sin_a, cos_a, 0.0, 0.0, 0.0, 1.0), (3, 3)
                )
            else:
                # 20° rotation around z-axis
                angle_rad = np.radians(20.0)
                cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
                u_matrix = matrix.rec(
                    (cos_a, -sin_a, 0.0, sin_a, cos_a, 0.0, 0.0, 0.0, 1.0), (3, 3)
                )

            crystal.get_U.return_value = u_matrix

            # Mock space group
            from cctbx.sgtbx import space_group_info

            crystal.get_space_group.return_value = space_group_info("P1").group()

            exp.crystal = crystal
            high_misorientation_crystals.append(exp)

        # Should raise ValueError due to high misorientation
        with pytest.raises(ValueError) as exc_info:
            GlobalVoxelGrid(
                high_misorientation_crystals, sample_diffuse_data, grid_config
            )

        # Check error message contains expected text
        assert "exceeds threshold" in str(exc_info.value)
        assert "not suitable for merging" in str(exc_info.value)

    def test_grid_initialization_succeeds_on_low_misorientation(
        self, sample_diffuse_data, grid_config
    ):
        """Test that GlobalVoxelGrid initializes successfully with low misorientation crystals."""
        # Create crystal models with very small misorientation (<2°)
        low_misorientation_experiments = []

        for i in range(3):
            exp = Mock(spec=Experiment)
            crystal = Mock(spec=Crystal)

            # Mock unit cell (same for all)
            unit_cell = uctbx.unit_cell((10.0, 15.0, 20.0, 90.0, 90.0, 90.0))
            crystal.get_unit_cell.return_value = unit_cell

            # Create very small rotations (< 1°)
            small_angle_rad = np.radians(0.5 * i)  # 0°, 0.5°, 1.0° rotations
            cos_a, sin_a = np.cos(small_angle_rad), np.sin(small_angle_rad)
            u_matrix = matrix.rec(
                (cos_a, -sin_a, 0.0, sin_a, cos_a, 0.0, 0.0, 0.0, 1.0), (3, 3)
            )
            crystal.get_U.return_value = u_matrix

            # Mock space group
            from cctbx.sgtbx import space_group_info

            crystal.get_space_group.return_value = space_group_info("P1").group()

            exp.crystal = crystal
            low_misorientation_experiments.append(exp)

        # Should initialize successfully without raising an exception
        grid = GlobalVoxelGrid(
            low_misorientation_experiments, sample_diffuse_data, grid_config
        )

        # Verify the grid was created successfully
        assert grid.crystal_avg_ref is not None
        assert grid.A_avg_ref is not None
        assert hasattr(grid, "rms_misorientation_deg")
        assert grid.rms_misorientation_deg < 2.0

    def test_hkl_transformation_vectorization_equivalence(
        self, mock_experiments, grid_config
    ):
        """Test that vectorized HKL transformation produces same results as loop-based approach."""
        import time
        from scitbx import matrix
        
        # Create small sample of q-vectors for equivalence testing
        np.random.seed(42)  # For reproducible results
        n_vectors = 100
        sample_q_vectors = np.random.uniform(-2.0, 2.0, (n_vectors, 3))
        
        # Create sample diffuse data
        sample_diffuse_data = [
            CorrectedDiffusePixelData(
                q_vectors=sample_q_vectors,
                intensities=np.random.uniform(0, 1000, n_vectors),
                sigmas=np.random.uniform(1, 100, n_vectors),
                still_ids=np.zeros(n_vectors, dtype=int),
            )
        ]
        
        # Create a minimal grid to get A_inv matrix
        grid = GlobalVoxelGrid(mock_experiments, sample_diffuse_data, grid_config)
        A_inv = grid.A_avg_ref.inverse()
        
        # Test old loop-based logic
        hkl_fractional_old = []
        for q_vec in sample_q_vectors:
            q_matrix = matrix.col(q_vec)
            hkl_frac = A_inv * q_matrix
            hkl_fractional_old.append(hkl_frac.elems)
        hkl_expected = np.array(hkl_fractional_old)
        
        # Test new vectorized logic
        A_inv_np = np.array(A_inv.elems).reshape(3, 3)
        hkl_actual = (A_inv_np @ sample_q_vectors.T).T
        
        # Assert numerical equivalence
        assert np.allclose(hkl_expected, hkl_actual, rtol=1e-10, atol=1e-12), \
            "Vectorized HKL transformation does not match loop-based approach"

    @pytest.mark.slow
    def test_hkl_transformation_performance(self, mock_experiments, grid_config):
        """Test that vectorized HKL transformation is significantly faster than loop-based approach."""
        import time
        from scitbx import matrix
        
        # Create large sample for performance testing
        np.random.seed(42)
        n_vectors = 100000  # Large dataset for performance comparison
        large_q_vectors = np.random.uniform(-2.0, 2.0, (n_vectors, 3))
        
        # Create sample diffuse data
        large_diffuse_data = [
            CorrectedDiffusePixelData(
                q_vectors=large_q_vectors,
                intensities=np.random.uniform(0, 1000, n_vectors),
                sigmas=np.random.uniform(1, 100, n_vectors),
                still_ids=np.zeros(n_vectors, dtype=int),
            )
        ]
        
        # Create a minimal grid to get A_inv matrix
        grid = GlobalVoxelGrid(mock_experiments, large_diffuse_data, grid_config)
        A_inv = grid.A_avg_ref.inverse()
        
        # Measure old loop-based approach (on subset to avoid timeout)
        subset_size = 10000
        subset_q_vectors = large_q_vectors[:subset_size]
        
        start_time = time.perf_counter()
        hkl_fractional_old = []
        for q_vec in subset_q_vectors:
            q_matrix = matrix.col(q_vec)
            hkl_frac = A_inv * q_matrix
            hkl_fractional_old.append(hkl_frac.elems)
        loop_time = time.perf_counter() - start_time
        
        # Measure new vectorized approach (full dataset)
        A_inv_np = np.array(A_inv.elems).reshape(3, 3)
        start_time = time.perf_counter()
        hkl_vectorized = (A_inv_np @ large_q_vectors.T).T
        vectorized_time = time.perf_counter() - start_time
        
        # Log the timings
        print(f"Loop approach time (10k vectors): {loop_time:.4f}s")
        print(f"Vectorized approach time (100k vectors): {vectorized_time:.4f}s")
        print(f"Performance improvement factor: {(loop_time * 10) / vectorized_time:.1f}x")
        
        # Assert significant speedup (even accounting for 10x more data in vectorized test)
        # This is a conservative test - actual speedup should be much higher
        expected_min_speedup = 5  # Expect at least 5x speedup
        actual_speedup = (loop_time * 10) / vectorized_time  # Account for 10x more data
        assert actual_speedup > expected_min_speedup, \
            f"Vectorized approach only {actual_speedup:.1f}x faster, expected >{expected_min_speedup}x"
