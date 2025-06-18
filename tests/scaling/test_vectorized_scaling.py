"""
Tests for vectorized scaling implementation correctness.

Verifies that the new vectorized approach produces identical results to the legacy loop-based approach.
"""

import pytest
import numpy as np
import tempfile
import os
from unittest.mock import Mock

from diffusepipe.scaling.diffuse_scaling_model import DiffuseScalingModel
from diffusepipe.scaling.components.per_still_multiplier import PerStillMultiplierComponent


class TestVectorizedScaling:
    """Test vectorized scaling correctness against legacy implementation."""

    @pytest.fixture
    def sample_scaling_model(self):
        """Create a sample scaling model for testing."""
        # Create configuration for a simple 2-still model
        config = {
            "still_ids": [0, 1],
            "per_still_scale": {"enabled": True, "initial_values": {0: 1.2, 1: 0.8}},
            "resolution_smoother": {"enabled": False},
        }
        
        model = DiffuseScalingModel(config)
        return model

    @pytest.fixture
    def sample_binned_data(self):
        """Create sample binned data for testing."""
        np.random.seed(42)  # For reproducible results
        
        # Create synthetic voxel data with the new efficient format
        binned_data = {}
        
        # Voxel 1: 10 observations
        n_obs_1 = 10
        binned_data[1001] = {
            "n_observations": n_obs_1,
            "intensities": np.random.exponential(100, n_obs_1),
            "sigmas": np.random.exponential(10, n_obs_1),
            "still_ids": np.random.choice([0, 1], n_obs_1),
            "q_vectors_lab": np.random.normal(0, 0.5, (n_obs_1, 3)),
        }
        
        # Voxel 2: 15 observations
        n_obs_2 = 15
        binned_data[1002] = {
            "n_observations": n_obs_2,
            "intensities": np.random.exponential(150, n_obs_2),
            "sigmas": np.random.exponential(15, n_obs_2),
            "still_ids": np.random.choice([0, 1], n_obs_2),
            "q_vectors_lab": np.random.normal(0, 0.8, (n_obs_2, 3)),
        }
        
        # Voxel 3: 5 observations
        n_obs_3 = 5
        binned_data[1003] = {
            "n_observations": n_obs_3,
            "intensities": np.random.exponential(75, n_obs_3),
            "sigmas": np.random.exponential(8, n_obs_3),
            "still_ids": np.random.choice([0, 1], n_obs_3),
            "q_vectors_lab": np.random.normal(0, 0.3, (n_obs_3, 3)),
        }
        
        return binned_data

    def test_vectorized_calculate_scales_and_derivatives_correctness(self, sample_scaling_model):
        """Test that vectorized scaling calculation produces same results as individual calls."""
        # Create test data
        still_ids = np.array([0, 1, 0, 1, 0])
        q_magnitudes = np.array([0.5, 0.8, 1.2, 0.3, 1.0])
        
        # Calculate using individual method calls (legacy approach)
        legacy_scales = []
        legacy_derivatives = []
        
        for i in range(len(still_ids)):
            scale, derivs = sample_scaling_model.calculate_scales_and_derivatives(
                still_ids[i], q_magnitudes[i]
            )
            legacy_scales.append(scale)
            legacy_derivatives.append(np.array(derivs))
        
        legacy_scales = np.array(legacy_scales)
        legacy_derivatives = np.array(legacy_derivatives)
        
        # Calculate using vectorized method
        vectorized_scales, vectorized_derivatives = sample_scaling_model.vectorized_calculate_scales_and_derivatives(
            still_ids, q_magnitudes
        )
        
        # Compare results
        print(f"Legacy scales: {legacy_scales}")
        print(f"Vectorized scales: {vectorized_scales}")
        print(f"Max scale difference: {np.max(np.abs(legacy_scales - vectorized_scales))}")
        
        # Scales should be identical
        assert np.allclose(legacy_scales, vectorized_scales, rtol=1e-10, atol=1e-12), \
            f"Scales differ: legacy={legacy_scales}, vectorized={vectorized_scales}"
        
        # Derivatives should be identical
        assert np.allclose(legacy_derivatives, vectorized_derivatives, rtol=1e-10, atol=1e-12), \
            f"Derivatives differ significantly"
        
        print("✅ Vectorized scaling calculation produces identical results to legacy approach")

    def test_vectorized_generate_references_correctness(self, sample_scaling_model, sample_binned_data):
        """Test that vectorized reference generation produces same results as legacy approach."""
        # First, aggregate data for vectorized approach
        aggregated_data = sample_scaling_model._aggregate_all_observations(sample_binned_data)
        
        # Generate references using vectorized method
        bragg_refs_vec, diffuse_refs_vec = sample_scaling_model.vectorized_generate_references(
            aggregated_data, {}
        )
        
        # Generate references using legacy method
        bragg_refs_legacy, diffuse_refs_legacy = sample_scaling_model.generate_references(
            sample_binned_data, {}
        )
        
        # Compare results - should have same voxels
        assert set(diffuse_refs_vec.keys()) == set(diffuse_refs_legacy.keys()), \
            "Different voxels in vectorized vs legacy references"
        
        # Compare reference intensities for each voxel
        for voxel_idx in diffuse_refs_vec.keys():
            vec_intensity = diffuse_refs_vec[voxel_idx]["intensity"]
            legacy_intensity = diffuse_refs_legacy[voxel_idx]["intensity"]
            
            print(f"Voxel {voxel_idx}: vectorized={vec_intensity:.6f}, legacy={legacy_intensity:.6f}")
            
            # Allow small numerical differences due to different calculation order
            assert np.isclose(vec_intensity, legacy_intensity, rtol=1e-8, atol=1e-10), \
                f"Reference intensities differ for voxel {voxel_idx}: {vec_intensity} vs {legacy_intensity}"
        
        print("✅ Vectorized reference generation produces equivalent results to legacy approach")

    def test_vectorized_r_factor_correctness(self, sample_scaling_model, sample_binned_data):
        """Test that vectorized R-factor calculation produces same results as legacy approach."""
        # Generate references first
        aggregated_data = sample_scaling_model._aggregate_all_observations(sample_binned_data)
        _, diffuse_refs = sample_scaling_model.vectorized_generate_references(aggregated_data, {})
        
        # Calculate R-factor using vectorized method
        r_factor_vec = sample_scaling_model.vectorized_calculate_r_factor(aggregated_data, diffuse_refs)
        
        # Calculate R-factor using legacy method
        r_factor_legacy = sample_scaling_model._calculate_r_factor(sample_binned_data, diffuse_refs)
        
        print(f"Vectorized R-factor: {r_factor_vec:.8f}")
        print(f"Legacy R-factor: {r_factor_legacy:.8f}")
        print(f"Absolute difference: {abs(r_factor_vec - r_factor_legacy):.2e}")
        
        # R-factors should be very close (allow small numerical differences)
        assert np.isclose(r_factor_vec, r_factor_legacy, rtol=1e-8, atol=1e-10), \
            f"R-factors differ significantly: vectorized={r_factor_vec}, legacy={r_factor_legacy}"
        
        print("✅ Vectorized R-factor calculation produces equivalent results to legacy approach")

    def test_aggregated_data_structure(self, sample_scaling_model, sample_binned_data):
        """Test that aggregated data structure contains all observations correctly."""
        aggregated_data = sample_scaling_model._aggregate_all_observations(sample_binned_data)
        
        # Count expected observations
        expected_total = sum(voxel_data["n_observations"] for voxel_data in sample_binned_data.values())
        actual_total = aggregated_data["total_observations"]
        
        assert actual_total == expected_total, \
            f"Total observations mismatch: expected {expected_total}, got {actual_total}"
        
        # Check array lengths are consistent
        assert len(aggregated_data["all_intensities"]) == actual_total
        assert len(aggregated_data["all_sigmas"]) == actual_total
        assert len(aggregated_data["all_still_ids"]) == actual_total
        assert len(aggregated_data["all_q_mags"]) == actual_total
        assert len(aggregated_data["voxel_idx_map"]) == actual_total
        
        # Check that voxel indices are correct
        unique_voxels = set(aggregated_data["voxel_idx_map"])
        expected_voxels = set(sample_binned_data.keys())
        assert unique_voxels == expected_voxels, \
            f"Voxel indices mismatch: expected {expected_voxels}, got {unique_voxels}"
        
        print("✅ Aggregated data structure is correct and complete")

    @pytest.mark.slow
    def test_large_dataset_correctness(self, sample_scaling_model):
        """Test vectorization correctness on a larger dataset."""
        np.random.seed(123)  # For reproducible results
        
        # Create larger dataset
        n_voxels = 100
        large_binned_data = {}
        
        for voxel_idx in range(n_voxels):
            n_obs = np.random.randint(1, 20)  # Variable observations per voxel
            large_binned_data[voxel_idx] = {
                "n_observations": n_obs,
                "intensities": np.random.exponential(100, n_obs),
                "sigmas": np.random.exponential(10, n_obs),
                "still_ids": np.random.choice([0, 1], n_obs),
                "q_vectors_lab": np.random.normal(0, 0.5, (n_obs, 3)),
            }
        
        # Test aggregation
        aggregated_data = sample_scaling_model._aggregate_all_observations(large_binned_data)
        
        # Test vectorized vs legacy reference generation
        bragg_refs_vec, diffuse_refs_vec = sample_scaling_model.vectorized_generate_references(
            aggregated_data, {}
        )
        bragg_refs_legacy, diffuse_refs_legacy = sample_scaling_model.generate_references(
            large_binned_data, {}
        )
        
        # Compare all reference intensities
        assert set(diffuse_refs_vec.keys()) == set(diffuse_refs_legacy.keys())
        
        max_diff = 0.0
        for voxel_idx in diffuse_refs_vec.keys():
            vec_intensity = diffuse_refs_vec[voxel_idx]["intensity"]
            legacy_intensity = diffuse_refs_legacy[voxel_idx]["intensity"]
            diff = abs(vec_intensity - legacy_intensity)
            max_diff = max(max_diff, diff)
            
            assert np.isclose(vec_intensity, legacy_intensity, rtol=1e-8, atol=1e-10), \
                f"Large dataset reference mismatch for voxel {voxel_idx}"
        
        print(f"✅ Large dataset test passed - max reference difference: {max_diff:.2e}")