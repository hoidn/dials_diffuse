"""
Tests for DiffuseDataMerger implementation.
"""

import pytest
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch

from diffusepipe.merging.merger import DiffuseDataMerger, VoxelDataRelative
from diffusepipe.voxelization.global_voxel_grid import GlobalVoxelGrid
from diffusepipe.scaling.diffuse_scaling_model import DiffuseScalingModel


class TestDiffuseDataMerger:
    """Test DiffuseDataMerger functionality."""

    @pytest.fixture
    def mock_grid(self):
        """Create mock GlobalVoxelGrid for testing."""
        grid = Mock(spec=GlobalVoxelGrid)
        
        # Mock voxel coordinate methods
        def mock_voxel_to_hkl(voxel_idx):
            # Simple mapping for testing
            return float(voxel_idx % 3), float((voxel_idx // 3) % 3), float(voxel_idx // 9)
        
        def mock_get_q_for_voxel(voxel_idx):
            h, k, l = mock_voxel_to_hkl(voxel_idx)
            from scitbx import matrix
            return matrix.col((h * 0.1, k * 0.1, l * 0.1))
        
        grid.voxel_idx_to_hkl_center.side_effect = mock_voxel_to_hkl
        grid.get_q_vector_for_voxel_center.side_effect = mock_get_q_for_voxel
        
        return grid

    @pytest.fixture
    def mock_scaling_model(self):
        """Create mock DiffuseScalingModel for testing."""
        model = Mock(spec=DiffuseScalingModel)
        
        # Mock scaling method to return simple scales
        def mock_get_scales(still_id, q_magnitude):
            # Return different scales for different stills
            multiplicative_scale = 1.0 + still_id * 0.1  # 1.1, 1.2, 1.3, etc.
            additive_offset = 0.0  # v1 model
            return multiplicative_scale, additive_offset
        
        model.get_scales_for_observation.side_effect = mock_get_scales
        return model

    @pytest.fixture
    def merger(self, mock_grid):
        """Create merger for testing."""
        return DiffuseDataMerger(mock_grid)

    @pytest.fixture
    def sample_binned_data(self):
        """Create sample binned pixel data."""
        return {
            0: {
                'observations': [
                    {
                        'intensity': 100.0,
                        'sigma': 10.0,
                        'still_id': 1,
                        'q_vector_lab': np.array([0.1, 0.1, 0.1])
                    },
                    {
                        'intensity': 120.0,
                        'sigma': 12.0,
                        'still_id': 2,
                        'q_vector_lab': np.array([0.1, 0.1, 0.1])
                    }
                ]
            },
            1: {
                'observations': [
                    {
                        'intensity': 80.0,
                        'sigma': 8.0,
                        'still_id': 1,
                        'q_vector_lab': np.array([0.2, 0.2, 0.2])
                    }
                ]
            }
        }

    @pytest.fixture
    def merge_config(self):
        """Create merge configuration."""
        return {
            'outlier_rejection': {'enabled': False},
            'minimum_observations': 1,
            'weight_method': 'inverse_variance'
        }

    def test_initialization(self, mock_grid):
        """Test merger initialization."""
        merger = DiffuseDataMerger(mock_grid)
        assert merger.global_voxel_grid == mock_grid

    def test_apply_scaling_to_observation(self, merger, mock_scaling_model):
        """Test scaling application to single observation."""
        observation = {
            'intensity': 100.0,
            'sigma': 10.0,
            'still_id': 1,
            'q_vector_lab': np.array([0.1, 0.1, 0.1])
        }
        
        scaled_intensity, scaled_sigma = merger.apply_scaling_to_observation(
            observation, mock_scaling_model
        )
        
        # With still_id=1, scale should be 1.1, so 100/1.1 ≈ 90.91
        expected_intensity = 100.0 / 1.1
        expected_sigma = 10.0 / 1.1
        
        assert abs(scaled_intensity - expected_intensity) < 1e-10
        assert abs(scaled_sigma - expected_sigma) < 1e-10

    def test_apply_scaling_v1_violation_warning(self, merger, caplog):
        """Test warning for v1 model violation."""
        import logging
        
        # Create custom mock model to return non-zero additive offset
        mock_scaling_model = Mock()
        mock_scaling_model.get_scales_for_observation.return_value = (1.1, 0.01)
        
        observation = {
            'intensity': 100.0,
            'sigma': 10.0,
            'still_id': 1,
            'q_vector_lab': np.array([0.1, 0.1, 0.1])
        }
        
        with caplog.at_level(logging.WARNING):
            merger.apply_scaling_to_observation(observation, mock_scaling_model)
            
        # Should log warning about v1 violation
        assert "v1 model violation" in caplog.text

    def test_weighted_merge_single_observation(self, merger):
        """Test merging with single observation."""
        scaled_observations = [(100.0, 10.0)]
        
        merged_intensity, merged_sigma, n_obs = merger.weighted_merge_voxel(
            scaled_observations, "inverse_variance"
        )
        
        assert merged_intensity == 100.0
        assert merged_sigma == 10.0
        assert n_obs == 1

    def test_weighted_merge_multiple_observations(self, merger):
        """Test merging with multiple observations."""
        scaled_observations = [
            (100.0, 10.0),  # weight = 1/100 = 0.01
            (120.0, 5.0),   # weight = 1/25 = 0.04
            (80.0, 20.0)    # weight = 1/400 = 0.0025
        ]
        
        merged_intensity, merged_sigma, n_obs = merger.weighted_merge_voxel(
            scaled_observations, "inverse_variance"
        )
        
        assert n_obs == 3
        assert merged_intensity > 0
        assert merged_sigma > 0
        
        # Higher weight observations should dominate
        assert 100 < merged_intensity < 120  # Should be closer to 120 (higher weight)

    def test_weighted_merge_uniform_weights(self, merger):
        """Test merging with uniform weighting."""
        scaled_observations = [
            (100.0, 10.0),
            (120.0, 5.0),
            (80.0, 20.0)
        ]
        
        merged_intensity, merged_sigma, n_obs = merger.weighted_merge_voxel(
            scaled_observations, "uniform"
        )
        
        assert n_obs == 3
        # Should be simple average
        expected_intensity = (100.0 + 120.0 + 80.0) / 3
        assert abs(merged_intensity - expected_intensity) < 1e-10

    def test_weighted_merge_invalid_method(self, merger):
        """Test error with invalid weighting method."""
        scaled_observations = [(100.0, 10.0)]
        
        with pytest.raises(ValueError, match="Unknown weight method"):
            merger.weighted_merge_voxel(scaled_observations, "invalid_method")

    def test_weighted_merge_empty_observations(self, merger):
        """Test error with empty observation list."""
        with pytest.raises(ValueError, match="No observations to merge"):
            merger.weighted_merge_voxel([], "inverse_variance")

    def test_calculate_voxel_coordinates(self, merger):
        """Test voxel coordinate calculation."""
        voxel_indices = [0, 1, 5]
        
        coordinates = merger.calculate_voxel_coordinates(voxel_indices)
        
        required_keys = [
            'H_center', 'K_center', 'L_center',
            'q_center_x', 'q_center_y', 'q_center_z', 'q_magnitude_center'
        ]
        
        for key in required_keys:
            assert key in coordinates
            assert len(coordinates[key]) == 3

    def test_merge_scaled_data_basic(self, merger, mock_scaling_model, 
                                   sample_binned_data, merge_config):
        """Test basic data merging functionality."""
        voxel_data = merger.merge_scaled_data(
            sample_binned_data, mock_scaling_model, merge_config
        )
        
        assert isinstance(voxel_data, VoxelDataRelative)
        assert len(voxel_data.voxel_indices) == 2  # Two voxels in sample data
        assert len(voxel_data.I_merged_relative) == 2
        assert len(voxel_data.Sigma_merged_relative) == 2
        assert all(voxel_data.num_observations > 0)

    def test_merge_scaled_data_minimum_observations(self, merger, mock_scaling_model, 
                                                  sample_binned_data):
        """Test minimum observations filtering."""
        config = {
            'minimum_observations': 2,  # Require at least 2 observations
            'weight_method': 'inverse_variance'
        }
        
        voxel_data = merger.merge_scaled_data(
            sample_binned_data, mock_scaling_model, config
        )
        
        # Only voxel 0 has 2 observations, voxel 1 has only 1
        assert len(voxel_data.voxel_indices) == 1
        assert voxel_data.voxel_indices[0] == 0

    def test_merge_scaled_data_outlier_rejection(self, merger, mock_scaling_model):
        """Test outlier rejection functionality."""
        # Create data with obvious outlier
        binned_data = {
            0: {
                'observations': [
                    {
                        'intensity': 100.0,
                        'sigma': 10.0,
                        'still_id': 1,
                        'q_vector_lab': np.array([0.1, 0.1, 0.1])
                    },
                    {
                        'intensity': 110.0,
                        'sigma': 11.0,
                        'still_id': 1,
                        'q_vector_lab': np.array([0.1, 0.1, 0.1])
                    },
                    {
                        'intensity': 1000.0,  # Outlier
                        'sigma': 100.0,
                        'still_id': 1,
                        'q_vector_lab': np.array([0.1, 0.1, 0.1])
                    }
                ]
            }
        }
        
        config = {
            'outlier_rejection': {
                'enabled': True,
                'sigma_threshold': 2.0
            },
            'minimum_observations': 1,
            'weight_method': 'inverse_variance'
        }
        
        voxel_data = merger.merge_scaled_data(
            binned_data, mock_scaling_model, config
        )
        
        # Should have removed the outlier and merged remaining observations
        assert len(voxel_data.voxel_indices) == 1
        # Merged intensity should be much less than 1000 (outlier excluded)
        assert voxel_data.I_merged_relative[0] < 200

    def test_get_merge_statistics(self, merger):
        """Test statistics calculation."""
        # Create sample voxel data
        voxel_data = VoxelDataRelative(
            voxel_indices=np.array([0, 1, 2]),
            H_center=np.array([0.0, 1.0, 2.0]),
            K_center=np.array([0.0, 1.0, 2.0]),
            L_center=np.array([0.0, 1.0, 2.0]),
            q_center_x=np.array([0.1, 0.2, 0.3]),
            q_center_y=np.array([0.1, 0.2, 0.3]),
            q_center_z=np.array([0.1, 0.2, 0.3]),
            q_magnitude_center=np.array([0.17, 0.35, 0.52]),
            I_merged_relative=np.array([100.0, 200.0, 150.0]),
            Sigma_merged_relative=np.array([10.0, 20.0, 15.0]),
            num_observations=np.array([3, 5, 2])
        )
        
        stats = merger.get_merge_statistics(voxel_data)
        
        required_keys = [
            'total_voxels', 'intensity_statistics', 'observation_statistics',
            'resolution_coverage', 'data_quality'
        ]
        
        for key in required_keys:
            assert key in stats
        
        assert stats['total_voxels'] == 3
        assert stats['observation_statistics']['total_observations'] == 10
        assert stats['intensity_statistics']['mean'] == 150.0

    def test_save_voxel_data_npz(self, merger):
        """Test saving voxel data to NPZ format."""
        voxel_data = VoxelDataRelative(
            voxel_indices=np.array([0, 1]),
            H_center=np.array([0.0, 1.0]),
            K_center=np.array([0.0, 1.0]),
            L_center=np.array([0.0, 1.0]),
            q_center_x=np.array([0.1, 0.2]),
            q_center_y=np.array([0.1, 0.2]),
            q_center_z=np.array([0.1, 0.2]),
            q_magnitude_center=np.array([0.17, 0.35]),
            I_merged_relative=np.array([100.0, 200.0]),
            Sigma_merged_relative=np.array([10.0, 20.0]),
            num_observations=np.array([3, 5])
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "test_voxel_data.npz")
            
            merger.save_voxel_data(voxel_data, output_path, format="npz")
            
            assert os.path.exists(output_path)
            
            # Verify data can be loaded
            loaded_data = np.load(output_path)
            assert 'voxel_indices' in loaded_data
            assert 'I_merged_relative' in loaded_data
            assert len(loaded_data['voxel_indices']) == 2

    def test_save_voxel_data_invalid_format(self, merger):
        """Test error with invalid save format."""
        voxel_data = VoxelDataRelative(
            voxel_indices=np.array([0]),
            H_center=np.array([0.0]),
            K_center=np.array([0.0]),
            L_center=np.array([0.0]),
            q_center_x=np.array([0.1]),
            q_center_y=np.array([0.1]),
            q_center_z=np.array([0.1]),
            q_magnitude_center=np.array([0.17]),
            I_merged_relative=np.array([100.0]),
            Sigma_merged_relative=np.array([10.0]),
            num_observations=np.array([3])
        )
        
        with pytest.raises(ValueError, match="Unsupported format"):
            merger.save_voxel_data(voxel_data, "test.xyz", format="invalid")