"""
Tests for StillsPipelineOrchestrator including Phase 3 integration.
"""

import pytest
import tempfile
import os
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from diffusepipe.orchestration.stills_pipeline_orchestrator import StillsPipelineOrchestrator
from diffusepipe.types.types_IDL import (
    StillsPipelineConfig,
    DIALSStillsProcessConfig,
    ExtractionConfig,
    RelativeScalingConfig,
    StillProcessingOutcome,
    OperationOutcome
)
from diffusepipe.exceptions import FileSystemError, ConfigurationError


class TestStillsPipelineOrchestrator:
    """Test StillsPipelineOrchestrator functionality."""
    
    @pytest.fixture
    def basic_config(self):
        """Create basic pipeline configuration."""
        return StillsPipelineConfig(
            dials_stills_process_config=DIALSStillsProcessConfig(
                stills_process_phil_path=None,
                force_processing_mode='stills',
                calculate_partiality=True
            ),
            extraction_config=ExtractionConfig(
                gain=1.0,
                cell_length_tol=0.05,
                cell_angle_tol=2.0,
                orient_tolerance_deg=2.0,
                q_consistency_tolerance_angstrom_inv=0.01,
                pixel_step=1,
                lp_correction_enabled=True,
                plot_diagnostics=False,
                verbose=False,
                external_pdb_path=None
            ),
            relative_scaling_config=RelativeScalingConfig(
                refine_per_still_scale=True,
                refine_resolution_scale_multiplicative=False,
                refine_additive_offset=False,
                min_partiality_threshold=0.1,
                grid_config={
                    'd_min_target': 1.0,
                    'd_max_target': 10.0,
                    'ndiv_h': 3,
                    'ndiv_k': 3,
                    'ndiv_l': 3
                }
            ),
            run_consistency_checker=False,
            run_q_calculator=False
        )
    
    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator instance."""
        return StillsPipelineOrchestrator()
    
    def test_initialization(self, orchestrator):
        """Test orchestrator initialization."""
        assert orchestrator.summary_log_entries == []
        assert orchestrator.phase2_outputs == []
        assert orchestrator.successful_experiments == []
        assert orchestrator.pixel_mask is None
    
    def test_validate_inputs_missing_cbf(self, orchestrator, basic_config):
        """Test input validation with missing CBF file."""
        cbf_paths = ['/nonexistent/file.cbf']
        
        with pytest.raises(FileSystemError, match="CBF file not found"):
            orchestrator._validate_inputs(cbf_paths, basic_config, '/tmp')
    
    def test_validate_inputs_none_config(self, orchestrator):
        """Test input validation with None configuration."""
        cbf_paths = []
        
        with pytest.raises(ConfigurationError, match="Configuration cannot be None"):
            orchestrator._validate_inputs(cbf_paths, None, '/tmp')
    
    def test_determine_processing_route_forced(self, orchestrator, basic_config):
        """Test processing route determination with forced mode."""
        # Test forced stills mode
        basic_config.dials_stills_process_config.force_processing_mode = 'stills'
        route = orchestrator._determine_processing_route('dummy.cbf', basic_config)
        assert route == 'stills'
        
        # Test forced sequence mode
        basic_config.dials_stills_process_config.force_processing_mode = 'sequence'
        route = orchestrator._determine_processing_route('dummy.cbf', basic_config)
        assert route == 'sequence'
    
    @patch('diffusepipe.orchestration.stills_pipeline_orchestrator.CBFUtils')
    def test_determine_processing_route_auto(self, mock_cbf_utils, orchestrator, basic_config):
        """Test automatic processing route determination."""
        basic_config.dials_stills_process_config.force_processing_mode = None
        
        # Mock CBF utils
        mock_utils_instance = Mock()
        mock_cbf_utils.return_value = mock_utils_instance
        
        # Test stills detection (angle_increment = 0)
        mock_utils_instance.get_angle_increment.return_value = 0.0
        route = orchestrator._determine_processing_route('dummy.cbf', basic_config)
        assert route == 'stills'
        
        # Test sequence detection (angle_increment > 0)
        mock_utils_instance.get_angle_increment.return_value = 0.1
        route = orchestrator._determine_processing_route('dummy.cbf', basic_config)
        assert route == 'sequence'
    
    def test_should_run_phase3_insufficient_data(self, orchestrator, basic_config):
        """Test Phase 3 skip with insufficient data."""
        # Only one successful output
        orchestrator.phase2_outputs = [{'dummy': 'data'}]
        
        assert not orchestrator._should_run_phase3(basic_config)
    
    def test_should_run_phase3_missing_config(self, orchestrator):
        """Test Phase 3 skip with missing configuration."""
        # Enough data but no config
        orchestrator.phase2_outputs = [{'dummy': 'data1'}, {'dummy': 'data2'}]
        
        config = StillsPipelineConfig(
            dials_stills_process_config=DIALSStillsProcessConfig(),
            extraction_config=ExtractionConfig(gain=1.0, cell_length_tol=0.05, 
                                             cell_angle_tol=2.0, orient_tolerance_deg=2.0,
                                             q_consistency_tolerance_angstrom_inv=0.01,
                                             pixel_step=1, lp_correction_enabled=True,
                                             plot_diagnostics=False, verbose=False,
                                             external_pdb_path=None),
            relative_scaling_config=None,  # Missing Phase 3 config
            run_consistency_checker=False,
            run_q_calculator=False
        )
        
        assert not orchestrator._should_run_phase3(config)
    
    def test_should_run_phase3_ready(self, orchestrator, basic_config):
        """Test Phase 3 readiness check with valid conditions."""
        # Sufficient data and config
        orchestrator.phase2_outputs = [
            {'dummy': 'data1'}, 
            {'dummy': 'data2'},
            {'dummy': 'data3'}
        ]
        
        assert orchestrator._should_run_phase3(basic_config)
    
    def test_collect_phase2_outputs(self, orchestrator):
        """Test collection of Phase 2 outputs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock NPZ file
            npz_path = os.path.join(temp_dir, 'diffuse_data.npz')
            np.savez(npz_path,
                    q_vectors=np.random.randn(100, 3),
                    intensities=np.random.exponential(100, 100),
                    sigmas=np.random.uniform(1, 10, 100))
            
            # Create mock outcome
            outcome = StillProcessingOutcome(
                input_cbf_path='/dummy/path.cbf',
                status='SUCCESS_ALL',
                working_directory=temp_dir,
                dials_outcome=OperationOutcome(status='SUCCESS'),
                extraction_outcome=OperationOutcome(
                    status='SUCCESS',
                    output_artifacts={'diffuse_data_path': npz_path}
                )
            )
            
            # Collect outputs
            orchestrator._collect_phase2_outputs(outcome)
            
            assert len(orchestrator.phase2_outputs) == 1
            assert 'diffuse_data' in orchestrator.phase2_outputs[0]
            assert orchestrator.phase2_outputs[0]['still_id'] == 0
    
    @patch('diffusepipe.orchestration.stills_pipeline_orchestrator.DIALSStillsProcessAdapter')
    @patch('diffusepipe.orchestration.stills_pipeline_orchestrator.DataExtractor')
    @patch.object(StillsPipelineOrchestrator, '_generate_bragg_mask', return_value="/fake/mask.pickle")
    def test_process_single_still_success(self, mock_generate_mask, mock_extractor, 
                                        mock_adapter, orchestrator, basic_config):
        """Test successful processing of single still."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock DIALS adapter
            mock_adapter_instance = Mock()
            mock_adapter.return_value = mock_adapter_instance
            
            # Mock experiment and reflections
            mock_experiment = Mock()
            mock_reflections = Mock()
            mock_adapter_instance.process_still.return_value = (
                mock_experiment, mock_reflections
            )
            
            # Mock data extractor
            mock_extractor_instance = Mock()
            mock_extractor.return_value = mock_extractor_instance
            
            # Process still
            outcome = orchestrator._process_single_still(
                'dummy.cbf', basic_config, Path(temp_dir)
            )
            
            assert outcome.status == 'SUCCESS_ALL'
            assert outcome.dials_outcome.status == 'SUCCESS'
            assert outcome.extraction_outcome.status == 'SUCCESS'
    
    @patch('os.path.exists', return_value=True)
    @patch('diffusepipe.orchestration.stills_pipeline_orchestrator.Path.mkdir')
    def test_process_stills_batch_no_phase3(self, mock_mkdir, mock_exists, orchestrator, basic_config):
        """Test batch processing without Phase 3."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Disable Phase 3
            basic_config.relative_scaling_config = None
            
            # Mock successful processing
            with patch.object(orchestrator, '_process_single_still') as mock_process:
                mock_outcome = StillProcessingOutcome(
                    input_cbf_path='dummy.cbf',
                    status='FAILURE_DIALS',
                    working_directory=temp_dir,
                    dials_outcome=OperationOutcome(status='FAILURE'),
                    extraction_outcome=OperationOutcome(status='FAILURE')
                )
                mock_process.return_value = mock_outcome
                
                # Process batch
                outcomes = orchestrator.process_stills_batch(
                    ['dummy.cbf'], basic_config, temp_dir
                )
                
                assert len(outcomes) == 1
                assert outcomes[0].status == 'FAILURE_DIALS'
    
    def test_integration_with_phase3_components(self):
        """Test that Phase 3 components can be imported and instantiated."""
        # This test verifies that all Phase 3 imports work correctly
        from diffusepipe.voxelization.global_voxel_grid import (
            GlobalVoxelGrid, GlobalVoxelGridConfig, CorrectedDiffusePixelData
        )
        from diffusepipe.voxelization.voxel_accumulator import VoxelAccumulator
        from diffusepipe.scaling.diffuse_scaling_model import DiffuseScalingModel
        from diffusepipe.merging.merger import DiffuseDataMerger
        
        # Verify classes can be referenced
        assert GlobalVoxelGrid is not None
        assert VoxelAccumulator is not None
        assert DiffuseScalingModel is not None
        assert DiffuseDataMerger is not None