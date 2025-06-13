"""
Integration tests for Phase 3 workflow.
"""

import pytest
import numpy as np
from unittest.mock import Mock
from cctbx import sgtbx

from diffusepipe.voxelization.global_voxel_grid import (
    GlobalVoxelGrid, GlobalVoxelGridConfig, CorrectedDiffusePixelData
)
from diffusepipe.voxelization.voxel_accumulator import VoxelAccumulator
from diffusepipe.scaling.diffuse_scaling_model import DiffuseScalingModel
from diffusepipe.merging.merger import DiffuseDataMerger


class TestPhase3Workflow:
    """Test complete Phase 3 workflow integration."""

    @pytest.fixture
    def mock_experiments(self):
        """Create mock experiments for testing."""
        from dxtbx.model import Experiment, Crystal, Beam, Detector
        from cctbx import uctbx
        from scitbx import matrix
        
        experiments = []
        for i in range(3):
            exp = Mock(spec=Experiment)
            
            # Mock crystal
            crystal = Mock(spec=Crystal)
            unit_cell = uctbx.unit_cell((10.0, 15.0, 20.0, 90.0, 90.0, 90.0))
            crystal.get_unit_cell.return_value = unit_cell
            
            u_matrix = matrix.rec((
                1.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 1.0
            ), (3, 3))
            crystal.get_U.return_value = u_matrix
            
            space_group_info = sgtbx.space_group_info("P1")
            crystal.get_space_group.return_value = space_group_info.group()
            
            exp.crystal = crystal
            experiments.append(exp)
        
        return experiments

    @pytest.fixture
    def sample_diffuse_data(self):
        """Create sample corrected diffuse data."""
        n_points = 100
        q_vectors = np.random.normal(0, 0.3, (n_points, 3))
        intensities = np.random.exponential(100, n_points)
        sigmas = np.sqrt(intensities) + 1.0
        still_ids = np.random.randint(0, 3, n_points)
        
        return [CorrectedDiffusePixelData(
            q_vectors=q_vectors,
            intensities=intensities,
            sigmas=sigmas,
            still_ids=still_ids
        )]

    @pytest.fixture
    def grid_config(self):
        """Create grid configuration."""
        return GlobalVoxelGridConfig(
            d_min_target=1.0,
            d_max_target=10.0,
            ndiv_h=2,
            ndiv_k=2,
            ndiv_l=2
        )

    @pytest.fixture
    def scaling_model_config(self):
        """Create scaling model configuration."""
        return {
            'still_ids': [0, 1, 2],
            'per_still_scale': {'enabled': True},
            'resolution_smoother': {'enabled': False},
            'experimental_components': {
                'panel_scale': {'enabled': False},
                'spatial_scale': {'enabled': False},
                'additive_offset': {'enabled': False}
            }
        }

    def test_end_to_end_phase3_workflow(self, mock_experiments, sample_diffuse_data, 
                                       grid_config, scaling_model_config):
        """Test complete Phase 3 workflow from grid creation to merging."""
        
        # Step 1: Create GlobalVoxelGrid
        global_grid = GlobalVoxelGrid(
            mock_experiments, sample_diffuse_data, grid_config
        )
        
        assert global_grid.total_voxels > 0
        assert global_grid.crystal_avg_ref is not None
        
        # Step 2: Initialize VoxelAccumulator
        space_group_info = sgtbx.space_group_info("P1")
        accumulator = VoxelAccumulator(
            global_grid, space_group_info, backend="memory"
        )
        
        # Add observations from diffuse data
        diffuse_data = sample_diffuse_data[0]
        n_binned = accumulator.add_observations(
            0,  # still_id
            diffuse_data.q_vectors,
            diffuse_data.intensities,
            diffuse_data.sigmas
        )
        
        assert n_binned > 0
        
        # Step 3: Get binned data for scaling
        binned_data = accumulator.get_all_binned_data_for_scaling()
        assert len(binned_data) > 0
        
        # Step 4: Initialize and configure scaling model
        scaling_model = DiffuseScalingModel(scaling_model_config)
        assert scaling_model.n_total_params == 3  # One per still
        
        # Step 5: Perform scaling refinement (simplified)
        refinement_config = {
            'max_iterations': 3,
            'convergence_tolerance': 1e-3
        }
        
        refined_params, refinement_stats = scaling_model.refine_parameters(
            binned_data, {}, refinement_config
        )
        
        assert 'final_r_factor' in refinement_stats
        assert len(refined_params) == 3  # One per still
        
        # Step 6: Merge scaled data
        merger = DiffuseDataMerger(global_grid)
        
        merge_config = {
            'outlier_rejection': {'enabled': False},
            'minimum_observations': 1,
            'weight_method': 'inverse_variance'
        }
        
        voxel_data = merger.merge_scaled_data(
            binned_data, scaling_model, merge_config
        )
        
        # Verify final output
        assert len(voxel_data.voxel_indices) > 0
        assert len(voxel_data.I_merged_relative) == len(voxel_data.voxel_indices)
        assert len(voxel_data.Sigma_merged_relative) == len(voxel_data.voxel_indices)
        assert all(voxel_data.num_observations > 0)
        
        # Step 7: Get final statistics
        stats = merger.get_merge_statistics(voxel_data)
        
        assert stats['total_voxels'] > 0
        assert stats['observation_statistics']['total_observations'] > 0
        
        print(f"Phase 3 workflow completed successfully:")
        print(f"  - Grid voxels: {global_grid.total_voxels}")
        print(f"  - Binned observations: {accumulator.n_total_observations}")
        print(f"  - Final voxels: {stats['total_voxels']}")
        print(f"  - Final R-factor: {refinement_stats['final_r_factor']:.6f}")

    def test_workflow_with_resolution_smoother(self, mock_experiments, sample_diffuse_data, 
                                             grid_config):
        """Test workflow with resolution smoother enabled."""
        
        # Configure with resolution smoother
        scaling_config = {
            'still_ids': [0, 1, 2],
            'per_still_scale': {'enabled': True},
            'resolution_smoother': {
                'enabled': True,
                'n_control_points': 3,
                'resolution_range': (0.1, 1.0)
            },
            'experimental_components': {
                'panel_scale': {'enabled': False},
                'spatial_scale': {'enabled': False},
                'additive_offset': {'enabled': False}
            }
        }
        
        # Create grid and accumulate data
        global_grid = GlobalVoxelGrid(
            mock_experiments, sample_diffuse_data, grid_config
        )
        
        space_group_info = sgtbx.space_group_info("P1")
        accumulator = VoxelAccumulator(
            global_grid, space_group_info, backend="memory"
        )
        
        diffuse_data = sample_diffuse_data[0]
        accumulator.add_observations(
            0, diffuse_data.q_vectors, diffuse_data.intensities, diffuse_data.sigmas
        )
        
        binned_data = accumulator.get_all_binned_data_for_scaling()
        
        # Test scaling model with resolution smoother
        scaling_model = DiffuseScalingModel(scaling_config)
        assert scaling_model.n_total_params == 6  # 3 stills + 3 resolution points
        assert 'resolution' in scaling_model.components
        
        # Test that scaling works with resolution component
        test_scale, test_offset = scaling_model.get_scales_for_observation(0, 0.5)
        assert test_scale > 0
        assert test_offset == 0.0  # v1 model

    def test_workflow_error_handling(self, mock_experiments, grid_config):
        """Test workflow error handling with various edge cases."""
        
        # Test with empty diffuse data
        empty_diffuse_data = [CorrectedDiffusePixelData(
            q_vectors=np.array([]).reshape(0, 3),
            intensities=np.array([]),
            sigmas=np.array([]),
            still_ids=np.array([])
        )]
        
        with pytest.raises(ValueError):
            GlobalVoxelGrid(mock_experiments, empty_diffuse_data, grid_config)
        
        # Test with invalid scaling configuration
        invalid_scaling_config = {
            'still_ids': [0, 1],
            'experimental_components': {
                'additive_offset': {'enabled': True}  # Forbidden in v1
            }
        }
        
        with pytest.raises(ValueError, match="additive_offset component is hard-disabled"):
            DiffuseScalingModel(invalid_scaling_config)

    def test_memory_vs_hdf5_backends(self, mock_experiments, sample_diffuse_data, grid_config):
        """Test both memory and HDF5 backends produce consistent results."""
        
        global_grid = GlobalVoxelGrid(
            mock_experiments, sample_diffuse_data, grid_config
        )
        space_group_info = sgtbx.space_group_info("P1")
        
        # Test with memory backend
        accumulator_memory = VoxelAccumulator(
            global_grid, space_group_info, backend="memory"
        )
        
        diffuse_data = sample_diffuse_data[0]
        n_binned_memory = accumulator_memory.add_observations(
            0, diffuse_data.q_vectors, diffuse_data.intensities, diffuse_data.sigmas
        )
        
        binned_data_memory = accumulator_memory.get_all_binned_data_for_scaling()
        
        # Both backends should bin the same number of observations
        assert n_binned_memory > 0
        assert len(binned_data_memory) > 0
        
        # Memory backend statistics
        stats_memory = accumulator_memory.get_accumulation_statistics()
        assert stats_memory['total_observations'] == n_binned_memory