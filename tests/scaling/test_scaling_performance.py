"""
Performance tests for vectorized scaling and merging implementation.

Demonstrates significant speedup over legacy loop-based approaches.
"""

import pytest
import numpy as np
import time
import logging
from unittest.mock import Mock

from diffusepipe.scaling.diffuse_scaling_model import DiffuseScalingModel
from diffusepipe.merging.merger import DiffuseDataMerger
from diffusepipe.voxelization.global_voxel_grid import GlobalVoxelGrid

# Reduce logging during performance tests
logging.getLogger().setLevel(logging.WARNING)


class TestScalingPerformance:
    """Performance tests for vectorized scaling implementation."""

    @pytest.fixture
    def performance_scaling_model(self):
        """Create a scaling model for performance testing."""
        config = {
            "still_ids": [0, 1, 2, 3, 4],  # More stills for realistic test
            "per_still_scale": {"enabled": True},
            "resolution_smoother": {"enabled": False},
        }
        return DiffuseScalingModel(config)

    @pytest.fixture
    def large_binned_data(self):
        """Create large synthetic dataset for performance testing."""
        np.random.seed(456)  # Different seed for performance tests
        
        # Create realistic large dataset: 1000 voxels with varying observation counts
        n_voxels = 1000
        binned_data = {}
        
        total_observations = 0
        for voxel_idx in range(n_voxels):
            # Realistic distribution: most voxels have few observations, some have many
            n_obs = np.random.choice([1, 2, 3, 5, 8, 12, 20, 50], p=[0.3, 0.25, 0.2, 0.1, 0.08, 0.05, 0.015, 0.005])
            
            binned_data[voxel_idx] = {
                "n_observations": n_obs,
                "intensities": np.random.exponential(100, n_obs),
                "sigmas": np.random.exponential(10, n_obs),
                "still_ids": np.random.choice([0, 1, 2, 3, 4], n_obs),
                "q_vectors_lab": np.random.normal(0, 0.5, (n_obs, 3)),
            }
            total_observations += n_obs
        
        print(f"Created large dataset: {n_voxels} voxels, {total_observations} total observations")
        return binned_data

    @pytest.mark.slow
    def test_scaling_vectorization_performance(self, performance_scaling_model, large_binned_data):
        """Test performance improvement of vectorized scaling operations."""
        # Aggregate data once for vectorized operations
        aggregated_data = performance_scaling_model._aggregate_all_observations(large_binned_data)
        total_obs = aggregated_data["total_observations"]
        
        print(f"\nPerformance test: {total_obs} observations across {len(large_binned_data)} voxels")
        
        # Test 1: Reference generation performance
        print("\n1. Reference Generation Performance:")
        
        # Vectorized approach
        start_time = time.perf_counter()
        bragg_refs_vec, diffuse_refs_vec = performance_scaling_model.vectorized_generate_references(
            aggregated_data, {}
        )
        vectorized_ref_time = time.perf_counter() - start_time
        
        # Legacy approach
        start_time = time.perf_counter()
        bragg_refs_legacy, diffuse_refs_legacy = performance_scaling_model.generate_references(
            large_binned_data, {}
        )
        legacy_ref_time = time.perf_counter() - start_time
        
        ref_speedup = legacy_ref_time / vectorized_ref_time
        print(f"  Legacy approach time: {legacy_ref_time:.4f}s")
        print(f"  Vectorized approach time: {vectorized_ref_time:.4f}s")
        print(f"  Speedup factor: {ref_speedup:.1f}x")
        
        # Test 2: R-factor calculation performance
        print("\n2. R-factor Calculation Performance:")
        
        # Vectorized approach
        start_time = time.perf_counter()
        r_factor_vec = performance_scaling_model.vectorized_calculate_r_factor(aggregated_data, diffuse_refs_vec)
        vectorized_rfactor_time = time.perf_counter() - start_time
        
        # Legacy approach  
        start_time = time.perf_counter()
        r_factor_legacy = performance_scaling_model._calculate_r_factor(large_binned_data, diffuse_refs_legacy)
        legacy_rfactor_time = time.perf_counter() - start_time
        
        rfactor_speedup = legacy_rfactor_time / vectorized_rfactor_time
        print(f"  Legacy approach time: {legacy_rfactor_time:.4f}s")
        print(f"  Vectorized approach time: {vectorized_rfactor_time:.4f}s")
        print(f"  Speedup factor: {rfactor_speedup:.1f}x")
        
        # Verify results are equivalent
        assert np.isclose(r_factor_vec, r_factor_legacy, rtol=1e-8), \
            "R-factors should be equivalent between methods"
        
        # Assert significant speedup
        min_expected_speedup = 3.0  # Expect at least 3x speedup for realistic datasets
        assert ref_speedup > min_expected_speedup, \
            f"Reference generation speedup ({ref_speedup:.1f}x) below expected minimum ({min_expected_speedup}x)"
        assert rfactor_speedup > min_expected_speedup, \
            f"R-factor calculation speedup ({rfactor_speedup:.1f}x) below expected minimum ({min_expected_speedup}x)"
        
        print(f"\n✅ Performance test passed - achieved {ref_speedup:.1f}x and {rfactor_speedup:.1f}x speedups")

    @pytest.mark.slow
    def test_end_to_end_refinement_performance(self, performance_scaling_model, large_binned_data):
        """Test performance of complete refinement workflow with vectorization."""
        print(f"\nE2E Refinement Performance Test:")
        
        # Test refinement with limited iterations for performance comparison
        refinement_config = {
            "max_iterations": 3,  # Limited iterations for performance test
            "convergence_tolerance": 1e-6,
            "parameter_shift_tolerance": 1e-6,
        }
        
        # Measure vectorized refinement time
        start_time = time.perf_counter()
        refined_params, stats = performance_scaling_model.refine_parameters(
            large_binned_data, {}, refinement_config
        )
        vectorized_time = time.perf_counter() - start_time
        
        print(f"  Vectorized refinement time: {vectorized_time:.4f}s")
        print(f"  Completed {stats['n_iterations']} iterations")
        print(f"  Final R-factor: {stats.get('final_r_factor', 'N/A')}")
        
        # Basic verification that refinement completed
        assert stats['n_iterations'] > 0, "Refinement should complete at least one iteration"
        assert len(refined_params) > 0, "Should produce refined parameters"
        
        # Check that processing rate is reasonable
        total_obs = sum(voxel_data["n_observations"] for voxel_data in large_binned_data.values())
        obs_per_second = (total_obs * stats['n_iterations']) / vectorized_time
        
        print(f"  Processing rate: {obs_per_second:.0f} observations/second per iteration")
        
        # Expect reasonable processing rate (should handle thousands per second)
        min_expected_rate = 5000  # observations per second per iteration
        assert obs_per_second > min_expected_rate, \
            f"Processing rate ({obs_per_second:.0f}/s) below expected minimum ({min_expected_rate}/s)"
        
        print(f"✅ E2E refinement performance test passed - {obs_per_second:.0f} obs/s per iteration")

    @pytest.fixture
    def mock_voxel_grid(self):
        """Create mock voxel grid for merger testing."""
        grid = Mock(spec=GlobalVoxelGrid)
        
        def mock_voxel_to_hkl(voxel_idx):
            # Simple deterministic mapping for testing
            h = (voxel_idx % 10) - 5
            k = ((voxel_idx // 10) % 10) - 5  
            l = ((voxel_idx // 100) % 10) - 5
            return float(h), float(k), float(l)
        
        def mock_get_q_for_voxel(voxel_idx):
            h, k, l = mock_voxel_to_hkl(voxel_idx)
            from scitbx import matrix
            return matrix.col((h * 0.1, k * 0.1, l * 0.1))
        
        grid.voxel_idx_to_hkl_center.side_effect = mock_voxel_to_hkl
        grid.get_q_vector_for_voxel_center.side_effect = mock_get_q_for_voxel
        
        return grid

    @pytest.mark.slow 
    def test_merging_vectorization_performance(self, performance_scaling_model, large_binned_data, mock_voxel_grid):
        """Test performance improvement of vectorized merging operations."""
        print(f"\nMerging Vectorization Performance Test:")
        
        merger = DiffuseDataMerger(mock_voxel_grid)
        merge_config = {
            "minimum_observations": 1,
            "weight_method": "inverse_variance",
            "outlier_rejection": {"enabled": False}
        }
        
        # Test vectorized merging performance
        start_time = time.perf_counter()
        vectorized_result = merger.vectorized_merge_scaled_data(
            large_binned_data, performance_scaling_model, merge_config
        )
        vectorized_merge_time = time.perf_counter() - start_time
        
        # Test legacy merging performance  
        merge_config_legacy = merge_config.copy()
        merge_config_legacy["use_legacy"] = True
        
        start_time = time.perf_counter()
        legacy_result = merger.merge_scaled_data(
            large_binned_data, performance_scaling_model, merge_config_legacy
        )
        legacy_merge_time = time.perf_counter() - start_time
        
        merge_speedup = legacy_merge_time / vectorized_merge_time
        print(f"  Legacy merging time: {legacy_merge_time:.4f}s")
        print(f"  Vectorized merging time: {vectorized_merge_time:.4f}s")
        print(f"  Speedup factor: {merge_speedup:.1f}x")
        
        # Verify results are equivalent
        assert len(vectorized_result.voxel_indices) == len(legacy_result.voxel_indices), \
            "Should produce same number of merged voxels"
        
        # Assert significant speedup
        min_expected_speedup = 2.0  # Expect at least 2x speedup for merging
        assert merge_speedup > min_expected_speedup, \
            f"Merging speedup ({merge_speedup:.1f}x) below expected minimum ({min_expected_speedup}x)"
        
        print(f"✅ Merging performance test passed - achieved {merge_speedup:.1f}x speedup")

    @pytest.mark.slow
    def test_memory_efficiency(self, performance_scaling_model, large_binned_data):
        """Test that vectorized approach doesn't use excessive memory."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Measure memory before operations
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Perform vectorized operations
        aggregated_data = performance_scaling_model._aggregate_all_observations(large_binned_data)
        bragg_refs, diffuse_refs = performance_scaling_model.vectorized_generate_references(aggregated_data, {})
        r_factor = performance_scaling_model.vectorized_calculate_r_factor(aggregated_data, diffuse_refs)
        
        # Measure memory after operations
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        total_obs = aggregated_data["total_observations"]
        memory_per_obs = memory_increase / total_obs * 1024  # KB per observation
        
        print(f"\nMemory Efficiency Test:")
        print(f"  Initial memory: {initial_memory:.1f} MB")
        print(f"  Final memory: {final_memory:.1f} MB")
        print(f"  Memory increase: {memory_increase:.1f} MB")
        print(f"  Memory per observation: {memory_per_obs:.2f} KB")
        
        # Memory usage should be reasonable (< 1KB per observation is good)
        max_expected_memory_per_obs = 1.0  # KB
        assert memory_per_obs < max_expected_memory_per_obs, \
            f"Memory usage per observation ({memory_per_obs:.2f} KB) exceeds limit ({max_expected_memory_per_obs} KB)"
        
        print(f"✅ Memory efficiency test passed - {memory_per_obs:.2f} KB per observation")