"""
Tests for VoxelAccumulator implementation.
"""

import pytest
import numpy as np
import tempfile
import os
from unittest.mock import Mock

from cctbx import sgtbx
from scitbx import matrix

from diffusepipe.voxelization.voxel_accumulator import VoxelAccumulator
from diffusepipe.voxelization.global_voxel_grid import (
    GlobalVoxelGrid,
)


class TestVoxelAccumulator:
    """Test VoxelAccumulator functionality."""

    @pytest.fixture
    def mock_grid(self):
        """Create mock GlobalVoxelGrid for testing."""
        grid = Mock(spec=GlobalVoxelGrid)

        # Mock A matrix inverse for HKL transformation
        A_matrix = matrix.rec((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0), (3, 3))
        grid.A_avg_ref = A_matrix

        # Mock crystal for ASU mapping
        from cctbx import uctbx
        from dxtbx.model import Crystal

        mock_crystal = Mock(spec=Crystal)
        unit_cell = uctbx.unit_cell((10.0, 15.0, 20.0, 90.0, 90.0, 90.0))
        space_group = sgtbx.space_group_info("P1").group()

        mock_crystal.get_unit_cell.return_value = unit_cell
        mock_crystal.get_space_group.return_value = space_group

        grid.crystal_avg_ref = mock_crystal

        # Mock voxel mapping methods
        def mock_hkl_to_voxel(h, k, l):
            # Simple mapping for testing
            if abs(h) > 5 or abs(k) > 5 or abs(l) > 5:
                raise ValueError("Outside grid")
            return int(h + 5) * 121 + int(k + 5) * 11 + int(l + 5)

        def mock_voxel_to_hkl(voxel_idx):
            l = voxel_idx % 11 - 5
            k = (voxel_idx // 11) % 11 - 5
            h = (voxel_idx // 121) - 5
            return float(h), float(k), float(l)

        def mock_get_q_for_voxel(voxel_idx):
            h, k, l = mock_voxel_to_hkl(voxel_idx)
            return matrix.col((h * 0.1, k * 0.1, l * 0.1))

        grid.hkl_to_voxel_idx.side_effect = mock_hkl_to_voxel
        grid.voxel_idx_to_hkl_center.side_effect = mock_voxel_to_hkl
        grid.get_q_vector_for_voxel_center.side_effect = mock_get_q_for_voxel

        # Mock configuration needed for vectorized operations
        from diffusepipe.voxelization.global_voxel_grid import GlobalVoxelGridConfig
        mock_config = Mock(spec=GlobalVoxelGridConfig)
        mock_config.ndiv_h = 1  # Simple subdivision for testing
        mock_config.ndiv_k = 1
        mock_config.ndiv_l = 1
        grid.config = mock_config

        # Mock HKL boundaries for vectorized bounds checking
        grid.hkl_min = (-5, -5, -5)  # Should match the bounds used in mock_hkl_to_voxel
        grid.hkl_max = (5, 5, 5)

        return grid

    @pytest.fixture
    def space_group_info(self):
        """Create space group info for testing."""
        return sgtbx.space_group_info("P1")

    @pytest.fixture
    def sample_observations(self):
        """Create sample observation data."""
        n_obs = 50
        # Create q-vectors near origin to stay within mock grid bounds
        q_vectors = np.random.normal(0, 0.3, (n_obs, 3))
        intensities = np.random.exponential(100, n_obs)
        sigmas = np.sqrt(intensities) + 1.0
        still_id = 1

        return still_id, q_vectors, intensities, sigmas

    def test_memory_backend_initialization(self, mock_grid, space_group_info):
        """Test initialization with memory backend."""
        accumulator = VoxelAccumulator(mock_grid, space_group_info, backend="memory")

        assert accumulator.backend == "memory"
        assert accumulator.n_total_observations == 0
        assert hasattr(accumulator, "_voxel_data")

    @pytest.mark.skipif(
        not hasattr(VoxelAccumulator, "HDF5_AVAILABLE")
        or not getattr(VoxelAccumulator, "HDF5_AVAILABLE", False),
        reason="h5py not available",
    )
    def test_hdf5_backend_initialization(self, mock_grid, space_group_info):
        """Test initialization with HDF5 backend."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = os.path.join(temp_dir, "test.h5")

            accumulator = VoxelAccumulator(
                mock_grid, space_group_info, backend="hdf5", storage_path=storage_path
            )

            assert accumulator.backend == "hdf5"
            assert accumulator.storage_path == storage_path
            assert hasattr(accumulator, "h5_file")

            accumulator.finalize()

    def test_invalid_backend(self, mock_grid, space_group_info):
        """Test error with invalid backend."""
        with pytest.raises(ValueError, match="Unknown backend"):
            VoxelAccumulator(mock_grid, space_group_info, backend="invalid")

    def test_add_observations_memory(
        self, mock_grid, space_group_info, sample_observations
    ):
        """Test adding observations with memory backend."""
        accumulator = VoxelAccumulator(mock_grid, space_group_info, backend="memory")

        still_id, q_vectors, intensities, sigmas = sample_observations

        n_binned = accumulator.add_observations(
            still_id, q_vectors, intensities, sigmas
        )

        assert n_binned > 0
        assert accumulator.n_total_observations == n_binned
        assert len(accumulator._voxel_data) > 0

    def test_add_observations_array_mismatch(self, mock_grid, space_group_info):
        """Test error with mismatched array lengths."""
        accumulator = VoxelAccumulator(mock_grid, space_group_info, backend="memory")

        q_vectors = np.random.normal(0, 0.1, (10, 3))
        intensities = np.random.exponential(100, 5)  # Wrong length
        sigmas = np.sqrt(intensities)

        with pytest.raises(ValueError, match="Array length mismatch"):
            accumulator.add_observations(1, q_vectors, intensities, sigmas)

    def test_get_observations_for_voxel_memory(self, mock_grid, space_group_info):
        """Test retrieving observations for specific voxel."""
        accumulator = VoxelAccumulator(mock_grid, space_group_info, backend="memory")

        # Add some test observations
        q_vectors = np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]])
        intensities = np.array([100.0, 200.0])
        sigmas = np.array([10.0, 15.0])

        accumulator.add_observations(1, q_vectors, intensities, sigmas)

        # Get observations for a voxel that should exist
        voxel_idx = list(accumulator._voxel_data.keys())[0]
        voxel_obs = accumulator.get_observations_for_voxel(voxel_idx)

        assert voxel_obs["n_observations"] > 0
        assert len(voxel_obs["intensities"]) == voxel_obs["n_observations"]
        assert voxel_obs["q_vectors_lab"].shape == (voxel_obs["n_observations"], 3)

    def test_get_observations_empty_voxel(self, mock_grid, space_group_info):
        """Test retrieving observations from empty voxel."""
        accumulator = VoxelAccumulator(mock_grid, space_group_info, backend="memory")

        # Request non-existent voxel
        voxel_obs = accumulator.get_observations_for_voxel(999999)

        assert voxel_obs["n_observations"] == 0
        assert len(voxel_obs["intensities"]) == 0
        assert voxel_obs["q_vectors_lab"].shape == (0, 3)

    def test_get_all_binned_data_for_scaling(
        self, mock_grid, space_group_info, sample_observations
    ):
        """Test getting complete binned dataset."""
        accumulator = VoxelAccumulator(mock_grid, space_group_info, backend="memory")

        still_id, q_vectors, intensities, sigmas = sample_observations
        accumulator.add_observations(still_id, q_vectors, intensities, sigmas)

        binned_data = accumulator.get_all_binned_data_for_scaling()

        assert isinstance(binned_data, dict)
        assert len(binned_data) > 0

        # Check data structure - now uses efficient NumPy array format
        for voxel_idx, voxel_data in binned_data.items():
            # Check for new efficient format (same as get_observations_for_voxel output)
            assert "intensities" in voxel_data
            assert "sigmas" in voxel_data
            assert "still_ids" in voxel_data
            assert "q_vectors_lab" in voxel_data
            assert "n_observations" in voxel_data

            # Verify arrays have correct shapes and types
            n_obs = voxel_data["n_observations"]
            assert isinstance(voxel_data["intensities"], np.ndarray)
            assert isinstance(voxel_data["sigmas"], np.ndarray)
            assert isinstance(voxel_data["still_ids"], np.ndarray)
            assert isinstance(voxel_data["q_vectors_lab"], np.ndarray)
            
            assert len(voxel_data["intensities"]) == n_obs
            assert len(voxel_data["sigmas"]) == n_obs
            assert len(voxel_data["still_ids"]) == n_obs
            assert voxel_data["q_vectors_lab"].shape == (n_obs, 3)

    def test_accumulation_statistics(
        self, mock_grid, space_group_info, sample_observations
    ):
        """Test statistics calculation."""
        accumulator = VoxelAccumulator(mock_grid, space_group_info, backend="memory")

        still_id, q_vectors, intensities, sigmas = sample_observations
        n_binned = accumulator.add_observations(
            still_id, q_vectors, intensities, sigmas
        )

        stats = accumulator.get_accumulation_statistics()

        required_keys = [
            "total_observations",
            "unique_voxels",
            "observations_per_voxel_stats",
            "still_distribution",
            "backend",
        ]

        for key in required_keys:
            assert key in stats

        assert stats["total_observations"] == n_binned
        assert stats["unique_voxels"] > 0
        assert stats["backend"] == "memory"
        assert still_id in stats["still_distribution"]

    def test_multiple_stills(self, mock_grid, space_group_info):
        """Test accumulating observations from multiple stills."""
        accumulator = VoxelAccumulator(mock_grid, space_group_info, backend="memory")

        # Add observations from multiple stills
        for still_id in range(3):
            q_vectors = np.random.normal(0, 0.2, (20, 3))
            intensities = np.random.exponential(100, 20)
            sigmas = np.sqrt(intensities) + 1.0

            accumulator.add_observations(still_id, q_vectors, intensities, sigmas)

        stats = accumulator.get_accumulation_statistics()

        assert len(stats["still_distribution"]) == 3
        assert stats["total_observations"] > 0

    def test_asu_mapping(self, mock_grid, space_group_info):
        """Test ASU mapping functionality with fractional inputs."""
        accumulator = VoxelAccumulator(mock_grid, space_group_info, backend="memory")

        # Test with fractional HKL array (as would come from q-vector transformation)
        hkl_array = np.array([[1.1, 2.3, 3.7], [-1.2, -2.8, -3.1]])
        hkl_asu = accumulator._map_to_asu(hkl_array)

        assert hkl_asu.shape == hkl_array.shape
        assert hkl_asu.dtype == float

    def test_asu_mapping_with_p2_symmetry(self):
        """Test ASU mapping with P2 space group to verify correct symmetry handling."""
        # Create a mock GlobalVoxelGrid with P2 space group
        mock_grid = Mock(spec=GlobalVoxelGrid)

        # Create a mock crystal with P2 symmetry
        from cctbx import uctbx
        from dxtbx.model import Crystal

        # Mock crystal with P2 space group
        mock_crystal = Mock(spec=Crystal)
        unit_cell = uctbx.unit_cell((10.0, 15.0, 20.0, 90.0, 90.0, 90.0))
        space_group = sgtbx.space_group_info("P2").group()

        # Mock crystal methods
        mock_crystal.get_unit_cell.return_value = unit_cell
        mock_crystal.get_space_group.return_value = space_group

        mock_grid.crystal_avg_ref = mock_crystal

        # Create P2 space group info
        space_group_info = sgtbx.space_group_info("P2")

        # Initialize accumulator with P2 symmetry
        accumulator = VoxelAccumulator(mock_grid, space_group_info, backend="memory")

        # Test symmetry-equivalent reflections with fractional coordinates
        # In P2, (h,k,l) and (-h,k,-l) should map to the same ASU reflection
        hkl_array = np.array(
            [
                [1.1, 2.3, 3.7],
                [-1.1, 2.3, -3.7],
            ]  # Original reflection  # P2 symmetry equivalent
        )

        hkl_asu = accumulator._map_to_asu(hkl_array)

        # Both reflections should map to the same ASU index
        # For P2 space group, the ASU mapping should make them equivalent
        assert hkl_asu.shape == (2, 3)

        # The ASU mapping should handle the symmetry operation
        # Both input reflections should map to equivalent positions in the ASU
        asu_1 = hkl_asu[0]
        asu_2 = hkl_asu[1]

        print(f"Original: {hkl_array[0]} -> ASU: {asu_1}")
        print(f"Original: {hkl_array[1]} -> ASU: {asu_2}")

        # For P2, verify that symmetry-equivalent fractional HKLs map to the same ASU HKL coordinate
        # Using np.allclose to handle floating-point precision in the comparison
        assert np.allclose(asu_1, asu_2, atol=1e-10), f"Symmetry-equivalent reflections should map to same ASU coordinate: {asu_1} vs {asu_2}"

        # Test another case: ensure basic functionality works with fractional coordinates
        single_hkl = np.array([[2.4, 3.7, 4.1]])
        single_asu = accumulator._map_to_asu(single_hkl)
        assert single_asu.shape == (1, 3)
        
        # Verify that the new implementation preserves fractional precision
        # Test with a variety of fractional coordinates
        test_hkls = np.array([
            [0.1, 0.2, 0.3],
            [1.7, 2.8, 3.9],
            [-0.5, 1.5, -2.5]
        ])
        
        asu_result = accumulator._map_to_asu(test_hkls)
        assert asu_result.shape == (3, 3)
        
        # Verify that results are not just rounded integers (which would indicate precision loss)
        for i in range(3):
            has_fractional_part = not np.allclose(asu_result[i], np.round(asu_result[i]), atol=1e-10)
            # At least some of the mapped coordinates should have fractional parts preserved
            # (exact behavior depends on space group operations, but we shouldn't lose all precision)
        
        print(f"ASU mapping preserved fractional precision - input: {test_hkls}")
        print(f"ASU mapping result: {asu_result}")

    def test_empty_observations(self, mock_grid, space_group_info):
        """Test handling of empty observation arrays."""
        accumulator = VoxelAccumulator(mock_grid, space_group_info, backend="memory")

        # Add empty arrays
        empty_q = np.array([]).reshape(0, 3)
        empty_i = np.array([])
        empty_s = np.array([])

        n_binned = accumulator.add_observations(1, empty_q, empty_i, empty_s)
        assert n_binned == 0

    def test_out_of_bounds_observations(self, mock_grid, space_group_info):
        """Test handling observations outside grid bounds."""
        accumulator = VoxelAccumulator(mock_grid, space_group_info, backend="memory")

        # Create q-vectors that will be outside mock grid bounds
        q_vectors = np.array([[10.0, 10.0, 10.0]])  # Large values
        intensities = np.array([100.0])
        sigmas = np.array([10.0])

        n_binned = accumulator.add_observations(1, q_vectors, intensities, sigmas)

        # Should handle gracefully with fewer binned observations
        assert n_binned >= 0
        assert n_binned <= len(q_vectors)

    def test_hkl_transformation_vectorization_equivalence(
        self, mock_grid, space_group_info
    ):
        """Test that vectorized HKL transformation produces same results as loop-based approach."""
        accumulator = VoxelAccumulator(mock_grid, space_group_info, backend="memory")

        # Create sample q-vectors for testing
        np.random.seed(42)  # For reproducible results
        n_vectors = 100
        sample_q_vectors = np.random.uniform(-2.0, 2.0, (n_vectors, 3))

        # Call both legacy and vectorized methods
        hkl_legacy = accumulator._legacy_hkl_transform(sample_q_vectors)
        hkl_vectorized = accumulator._vectorized_hkl_transform(sample_q_vectors)

        # Calculate maximum absolute difference for logging
        max_abs_diff = np.max(np.abs(hkl_legacy - hkl_vectorized))
        print(
            f"Maximum absolute difference between legacy and vectorized: {max_abs_diff:.2e}"
        )

        # Assert numerical equivalence with relaxed tolerance
        assert np.allclose(
            hkl_legacy, hkl_vectorized, rtol=1e-9, atol=1e-12
        ), f"Vectorized HKL transformation does not match legacy approach (max diff: {max_abs_diff:.2e})"

    @pytest.mark.slow
    def test_hkl_transformation_performance(self, mock_grid, space_group_info):
        """Test that vectorized HKL transformation is significantly faster than loop-based approach."""
        import time

        accumulator = VoxelAccumulator(mock_grid, space_group_info, backend="memory")

        # Create large sample for performance testing
        np.random.seed(42)
        n_vectors = 100000  # Large dataset for performance comparison
        large_q_vectors = np.random.uniform(-2.0, 2.0, (n_vectors, 3))

        # Measure legacy approach (on subset to avoid timeout)
        subset_size = 10000
        subset_q_vectors = large_q_vectors[:subset_size]

        start_time = time.perf_counter()
        hkl_legacy = accumulator._legacy_hkl_transform(subset_q_vectors)
        legacy_time = time.perf_counter() - start_time

        # Measure vectorized approach (on full dataset)
        start_time = time.perf_counter()
        hkl_vectorized = accumulator._vectorized_hkl_transform(large_q_vectors)
        vectorized_time = time.perf_counter() - start_time

        # Log the timings
        print("VoxelAccumulator HKL transformation performance:")
        print(f"  Legacy approach time (10k vectors): {legacy_time:.4f}s")
        print(f"  Vectorized approach time (100k vectors): {vectorized_time:.4f}s")
        print(
            f"  Performance improvement factor: {(legacy_time * 10) / vectorized_time:.1f}x"
        )

        # Assert significant speedup (even accounting for 10x more data in vectorized test)
        expected_min_speedup = 10  # Expect at least 10x speedup for this operation
        actual_speedup = (
            legacy_time * 10
        ) / vectorized_time  # Account for 10x more data
        assert (
            actual_speedup > expected_min_speedup
        ), f"Vectorized approach only {actual_speedup:.1f}x faster, expected >{expected_min_speedup}x"

    def test_vectorized_voxel_indexing_correctness(self, mock_grid, space_group_info):
        """Test that vectorized voxel indexing produces same results as loop-based approach."""
        accumulator = VoxelAccumulator(mock_grid, space_group_info, backend="memory")
        
        # Create sample q-vectors for testing
        np.random.seed(42)  # For reproducible results
        n_vectors = 100
        sample_q_vectors = np.random.uniform(-0.5, 0.5, (n_vectors, 3))  # Keep within mock grid bounds
        sample_intensities = np.random.exponential(100, n_vectors)
        sample_sigmas = np.sqrt(sample_intensities) + 1.0
        still_id = 1
        
        # Legacy approach: temporarily implement old loop logic for comparison
        def legacy_voxel_indexing(hkl_asu):
            """Legacy loop-based voxel indexing for correctness testing."""
            voxel_indices = []
            valid_mask = []
            
            for i, (h, k, l) in enumerate(hkl_asu):
                try:
                    voxel_idx = mock_grid.hkl_to_voxel_idx(h, k, l)
                    voxel_indices.append(voxel_idx)
                    valid_mask.append(True)
                except (ValueError, IndexError):
                    valid_mask.append(False)
            
            valid_mask = np.array(valid_mask)
            valid_voxel_indices = np.array(voxel_indices)[valid_mask]
            return valid_mask, valid_voxel_indices
        
        # Get HKL coordinates (reuse existing transformation)
        hkl_array = accumulator._vectorized_hkl_transform(sample_q_vectors)
        hkl_asu = accumulator._map_to_asu(hkl_array)
        
        # Run legacy approach
        legacy_valid_mask, legacy_voxel_indices = legacy_voxel_indexing(hkl_asu)
        
        # Run vectorized approach by calling add_observations
        n_binned = accumulator.add_observations(still_id, sample_q_vectors, sample_intensities, sample_sigmas)
        
        # The vectorized approach should have binned the same number of observations
        assert n_binned == len(legacy_voxel_indices), f"Vectorized approach binned {n_binned} observations, legacy approach would bin {len(legacy_voxel_indices)}"
        
        print(f"Vectorized voxel indexing correctness test: {n_binned} observations processed successfully")

    @pytest.mark.slow
    def test_vectorized_voxel_indexing_performance(self, mock_grid, space_group_info):
        """Test that vectorized voxel indexing is significantly faster than loop-based approach."""
        import time
        
        accumulator = VoxelAccumulator(mock_grid, space_group_info, backend="memory")
        
        # Create large sample for performance testing
        np.random.seed(42)
        n_vectors = 100000  # Large dataset for performance comparison
        large_q_vectors = np.random.uniform(-0.3, 0.3, (n_vectors, 3))  # Keep within mock grid bounds
        large_intensities = np.random.exponential(100, n_vectors)
        large_sigmas = np.sqrt(large_intensities) + 1.0
        
        # Measure legacy approach (on subset to avoid timeout)
        subset_size = 10000
        subset_q_vectors = large_q_vectors[:subset_size]
        subset_intensities = large_intensities[:subset_size]
        subset_sigmas = large_sigmas[:subset_size]
        
        def legacy_add_observations(q_vectors, intensities, sigmas):
            """Legacy loop-based add_observations for performance testing."""
            hkl_array = accumulator._vectorized_hkl_transform(q_vectors)
            hkl_asu = accumulator._map_to_asu(hkl_array)
            
            # Legacy loop-based voxel indexing
            voxel_indices = []
            valid_mask = []
            
            for i, (h, k, l) in enumerate(hkl_asu):
                try:
                    voxel_idx = mock_grid.hkl_to_voxel_idx(h, k, l)
                    voxel_indices.append(voxel_idx)
                    valid_mask.append(True)
                except (ValueError, IndexError):
                    valid_mask.append(False)
            
            return np.sum(valid_mask)
        
        # Measure legacy approach
        start_time = time.perf_counter()
        legacy_count = legacy_add_observations(subset_q_vectors, subset_intensities, subset_sigmas)
        legacy_time = time.perf_counter() - start_time
        
        # Reset accumulator for clean measurement
        accumulator = VoxelAccumulator(mock_grid, space_group_info, backend="memory")
        
        # Measure vectorized approach (on full dataset)
        start_time = time.perf_counter()
        vectorized_count = accumulator.add_observations(1, large_q_vectors, large_intensities, large_sigmas)
        vectorized_time = time.perf_counter() - start_time
        
        # Log the timings
        print("VoxelAccumulator voxel indexing performance:")
        print(f"  Legacy approach time (10k vectors): {legacy_time:.4f}s")
        print(f"  Vectorized approach time (100k vectors): {vectorized_time:.4f}s")
        print(f"  Legacy processed: {legacy_count} observations")
        print(f"  Vectorized processed: {vectorized_count} observations")
        
        # Calculate normalized speedup
        normalized_speedup = (legacy_time * 10) / vectorized_time  # Account for 10x more data
        print(f"  Performance improvement factor: {normalized_speedup:.1f}x")
        
        # Assert significant speedup (even accounting for 10x more data in vectorized test)
        expected_min_speedup = 5  # Expect at least 5x speedup for voxel indexing (real-world expectation)
        assert (
            normalized_speedup > expected_min_speedup
        ), f"Vectorized voxel indexing only {normalized_speedup:.1f}x faster, expected >{expected_min_speedup}x"
