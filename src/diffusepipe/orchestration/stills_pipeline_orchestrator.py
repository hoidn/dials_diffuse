"""
StillsPipelineOrchestrator implementation for managing full pipeline workflow.

Orchestrates all phases of diffuse scattering processing including:
- Phase 1: Per-still geometry, indexing, and masking
- Phase 2: Diffuse intensity extraction and correction
- Phase 3: Voxelization, relative scaling, and merging
"""

import os
import logging
import multiprocessing
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
import json
import numpy as np

from dxtbx.model import ExperimentList
from dials.array_family import flex
from cctbx import sgtbx

from ..types.types_IDL import (
    StillsPipelineConfig,
    StillProcessingOutcome,
    ComponentInputFiles,
    OperationOutcome
)
from ..adapters.dials_stills_process_adapter import DIALSStillsProcessAdapter
from ..adapters.dials_sequence_process_adapter import DIALSSequenceProcessAdapter
from ..adapters.dials_generate_mask_adapter import DIALSGenerateMaskAdapter
from ..extraction.data_extractor import DataExtractor
from ..diagnostics.q_consistency_checker import ConsistencyChecker
from ..diagnostics.q_calculator import QValueCalculator
from ..masking.pixel_mask_generator import PixelMaskGenerator
from ..masking.bragg_mask_generator import BraggMaskGenerator
from ..voxelization.global_voxel_grid import (
    GlobalVoxelGrid, GlobalVoxelGridConfig, CorrectedDiffusePixelData
)
from ..voxelization.voxel_accumulator import VoxelAccumulator
from ..scaling.diffuse_scaling_model import DiffuseScalingModel
from ..merging.merger import DiffuseDataMerger
from ..utils.cbf_utils import CBFUtils
from ..exceptions import (
    ConfigurationError as InvalidConfigurationError,
    DIALSError as DIALSEnvironmentError,
    FileSystemError
)

logger = logging.getLogger(__name__)


class StillsPipelineOrchestrator:
    """
    Orchestrates the complete diffuse scattering processing pipeline.
    
    Manages workflow from raw CBF files through all processing phases:
    - Phase 1: DIALS processing and masking
    - Phase 2: Diffuse extraction and correction
    - Phase 3: Voxelization, scaling, and merging
    """
    
    def __init__(self):
        """Initialize the orchestrator."""
        self.summary_log_entries = []
        self.phase2_outputs = []  # Collect Phase 2 outputs for Phase 3
        self.successful_experiments = []  # Collect successful Experiment objects
        self.pixel_mask = None  # Global pixel mask
        
    def process_stills_batch(self,
                           cbf_image_paths: List[str],
                           config: StillsPipelineConfig,
                           root_output_directory: str) -> List[StillProcessingOutcome]:
        """
        Process a batch of CBF files through the complete pipeline.
        
        Args:
            cbf_image_paths: List of paths to CBF files
            config: Pipeline configuration
            root_output_directory: Output directory for all results
            
        Returns:
            List of StillProcessingOutcome objects
        """
        # Validate inputs
        self._validate_inputs(cbf_image_paths, config, root_output_directory)
        
        # Create output directory
        output_path = Path(root_output_directory)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize summary log
        summary_log_path = output_path / "pipeline_summary.log"
        self._initialize_summary_log(summary_log_path)
        
        # Phase 1 & 2: Process individual stills
        logger.info(f"Starting Phase 1 & 2 processing for {len(cbf_image_paths)} images")
        outcomes = []
        
        # Generate global pixel mask first
        logger.info("Generating global pixel mask")
        self.pixel_mask = self._generate_global_pixel_mask(cbf_image_paths, config, output_path)
        
        # Process stills (can be parallelized)
        for cbf_path in cbf_image_paths:
            outcome = self._process_single_still(cbf_path, config, output_path)
            outcomes.append(outcome)
            self._update_summary_log(summary_log_path, outcome)
            
            # Collect successful results for Phase 3
            if outcome.status == "SUCCESS_ALL":
                self._collect_phase2_outputs(outcome)
        
        # Phase 3: Voxelization, scaling, and merging
        if len(self.phase2_outputs) > 0 and self._should_run_phase3(config):
            logger.info(f"Starting Phase 3 processing with {len(self.phase2_outputs)} successful stills")
            phase3_output_dir = output_path / "phase3_merged"
            phase3_output_dir.mkdir(exist_ok=True)
            
            try:
                self._run_phase3(config, phase3_output_dir)
                logger.info("Phase 3 completed successfully")
            except Exception as e:
                logger.error(f"Phase 3 failed: {e}")
                # Phase 3 failure doesn't affect individual still outcomes
        else:
            logger.info("Skipping Phase 3 (insufficient data or disabled)")
        
        # Finalize summary
        self._finalize_summary_log(summary_log_path)
        
        return outcomes
    
    def _validate_inputs(self, cbf_paths: List[str], config: StillsPipelineConfig, 
                        output_dir: str):
        """Validate input parameters."""
        # Check CBF files exist
        for cbf_path in cbf_paths:
            if not os.path.exists(cbf_path):
                raise FileSystemError(f"CBF file not found: {cbf_path}")
        
        # Validate configuration
        if config is None:
            raise InvalidConfigurationError("Configuration cannot be None")
        
        # Check output directory is writable
        output_path = Path(output_dir)
        if output_path.exists() and not os.access(output_path, os.W_OK):
            raise FileSystemError(f"Output directory not writable: {output_dir}")
    
    def _initialize_summary_log(self, log_path: Path):
        """Initialize the summary log file."""
        with open(log_path, 'w') as f:
            f.write("DiffusePipe Processing Summary\n")
            f.write("=" * 80 + "\n\n")
    
    def _generate_global_pixel_mask(self, cbf_paths: List[str], 
                                   config: StillsPipelineConfig,
                                   output_path: Path) -> Optional[Any]:
        """Generate global pixel mask from all images."""
        try:
            # Use PixelMaskGenerator to create mask
            # This is a simplified version - actual implementation would process all images
            # For now, return None to use per-image masks
            return None
        except Exception as e:
            logger.warning(f"Failed to generate global pixel mask: {e}")
            return None
    
    def _process_single_still(self, cbf_path: str, config: StillsPipelineConfig,
                            root_output_dir: Path) -> StillProcessingOutcome:
        """Process a single CBF file through Phase 1 & 2."""
        # Create working directory
        cbf_name = Path(cbf_path).stem
        working_dir = root_output_dir / cbf_name
        working_dir.mkdir(exist_ok=True)
        
        # Initialize outcome
        outcome = StillProcessingOutcome(
            input_cbf_path=cbf_path,
            status="FAILURE_DIALS",  # Default to failure
            working_directory=str(working_dir),
            dials_outcome=OperationOutcome(status="FAILURE"),
            extraction_outcome=OperationOutcome(status="FAILURE")
        )
        
        try:
            # Module 1.S.0: Data type detection
            processing_route = self._determine_processing_route(cbf_path, config)
            logger.info(f"Processing {cbf_name} using {processing_route} route")
            
            # Module 1.S.1: DIALS processing
            dials_result = self._run_dials_processing(
                cbf_path, config, working_dir, processing_route
            )
            
            if dials_result['success']:
                outcome.dials_outcome = OperationOutcome(
                    status="SUCCESS",
                    output_artifacts=dials_result['artifacts']
                )
                
                # Module 1.S.3: Bragg mask generation
                bragg_mask_path = self._generate_bragg_mask(
                    dials_result['experiment'],
                    dials_result['reflections'],
                    working_dir
                )
                
                # Module 2.S.1 & 2.S.2: Data extraction
                extraction_result = self._run_data_extraction(
                    cbf_path, dials_result, config, working_dir, bragg_mask_path
                )
                
                if extraction_result['success']:
                    outcome.extraction_outcome = OperationOutcome(
                        status="SUCCESS",
                        output_artifacts=extraction_result['artifacts']
                    )
                    outcome.status = "SUCCESS_ALL"
                    
                    # Store successful results
                    self.successful_experiments.append(dials_result['experiment'])
                else:
                    outcome.status = "FAILURE_EXTRACTION"
                    outcome.extraction_outcome = OperationOutcome(
                        status="FAILURE",
                        message=extraction_result.get('error', 'Unknown extraction error')
                    )
            else:
                outcome.dials_outcome = OperationOutcome(
                    status="FAILURE",
                    message=dials_result.get('error', 'DIALS processing failed')
                )
                
        except Exception as e:
            logger.error(f"Error processing {cbf_name}: {e}")
            outcome.message = str(e)
        
        return outcome
    
    def _determine_processing_route(self, cbf_path: str, 
                                  config: StillsPipelineConfig) -> str:
        """Determine processing route based on CBF data type."""
        # Check forced mode
        forced_mode = config.dials_stills_process_config.force_processing_mode
        if forced_mode in ['stills', 'sequence']:
            return forced_mode
        
        # Auto-detect from CBF header
        try:
            cbf_utils = CBFUtils()
            angle_increment = cbf_utils.get_angle_increment(cbf_path)
            
            if angle_increment is None or abs(angle_increment) < 1e-6:
                return 'stills'
            else:
                return 'sequence'
        except Exception as e:
            logger.warning(f"Failed to detect data type, defaulting to sequence: {e}")
            return 'sequence'
    
    def _run_dials_processing(self, cbf_path: str, config: StillsPipelineConfig,
                            working_dir: Path, processing_route: str) -> Dict:
        """Run DIALS processing using appropriate adapter."""
        try:
            if processing_route == 'stills':
                adapter = DIALSStillsProcessAdapter()
            else:
                adapter = DIALSSequenceProcessAdapter()
            
            # Process the image
            experiment, reflections = adapter.process_still(
                cbf_path, 
                config.dials_stills_process_config,
                str(working_dir)
            )
            
            # Save outputs
            expt_path = working_dir / "integrated.expt"
            refl_path = working_dir / "integrated.refl"
            
            experiment.as_file(str(expt_path))
            reflections.as_file(str(refl_path))
            
            return {
                'success': True,
                'experiment': experiment,
                'reflections': reflections,
                'artifacts': {
                    'experiment_path': str(expt_path),
                    'reflections_path': str(refl_path)
                }
            }
            
        except Exception as e:
            logger.error(f"DIALS processing failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _generate_bragg_mask(self, experiment: ExperimentList, 
                           reflections: flex.reflection_table,
                           working_dir: Path) -> str:
        """Generate Bragg mask for the still."""
        try:
            generator = BraggMaskGenerator()
            mask = generator.generate_bragg_mask_from_spots(
                experiment[0], reflections
            )
            
            # Save mask
            mask_path = working_dir / "bragg_mask.pickle"
            import pickle
            with open(mask_path, 'wb') as f:
                pickle.dump(mask, f)
            
            return str(mask_path)
            
        except Exception as e:
            logger.error(f"Bragg mask generation failed: {e}")
            raise
    
    def _run_data_extraction(self, cbf_path: str, dials_result: Dict,
                           config: StillsPipelineConfig, working_dir: Path,
                           bragg_mask_path: str) -> Dict:
        """Run diffuse data extraction."""
        try:
            # Prepare inputs
            inputs = ComponentInputFiles(
                cbf_image_path=cbf_path,
                dials_expt_path=dials_result['artifacts']['experiment_path'],
                dials_refl_path=dials_result['artifacts']['reflections_path'],
                bragg_mask_path=bragg_mask_path,
                external_pdb_path=config.extraction_config.external_pdb_path
            )
            
            # Run extraction
            output_npz_path = str(working_dir / "diffuse_data.npz")
            extractor = DataExtractor()
            extractor.extract_from_still(
                inputs,
                config.extraction_config,
                output_npz_path
            )
            
            return {
                'success': True,
                'artifacts': {
                    'diffuse_data_path': output_npz_path
                }
            }
            
        except Exception as e:
            logger.error(f"Data extraction failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _collect_phase2_outputs(self, outcome: StillProcessingOutcome):
        """Collect successful Phase 2 outputs for Phase 3 processing."""
        if outcome.extraction_outcome.output_artifacts:
            npz_path = outcome.extraction_outcome.output_artifacts.get('diffuse_data_path')
            if npz_path and os.path.exists(npz_path):
                # Load the NPZ file
                data = np.load(npz_path)
                
                # Create CorrectedDiffusePixelData object
                diffuse_data = CorrectedDiffusePixelData(
                    q_vectors=data['q_vectors'],
                    intensities=data['intensities'], 
                    sigmas=data['sigmas'],
                    still_ids=data.get('still_ids', np.zeros(len(data['intensities'])))
                )
                
                self.phase2_outputs.append({
                    'cbf_path': outcome.input_cbf_path,
                    'working_dir': outcome.working_directory,
                    'diffuse_data': diffuse_data,
                    'still_id': len(self.phase2_outputs)  # Sequential ID
                })
    
    def _should_run_phase3(self, config: StillsPipelineConfig) -> bool:
        """Check if Phase 3 should be run."""
        # Phase 3 requires at least some successful Phase 2 outputs
        if len(self.phase2_outputs) < 2:
            logger.warning("Insufficient data for Phase 3 (need at least 2 successful stills)")
            return False
        
        # Check if Phase 3 is configured
        if not hasattr(config, 'relative_scaling_config') or config.relative_scaling_config is None:
            logger.warning("Phase 3 configuration missing")
            return False
        
        return True
    
    def _run_phase3(self, config: StillsPipelineConfig, output_dir: Path):
        """Run Phase 3: Voxelization, scaling, and merging."""
        logger.info("Starting Phase 3: Voxelization, scaling, and merging")
        
        # Extract configuration
        scaling_config = config.relative_scaling_config
        grid_config_dict = scaling_config.grid_config or {}
        
        # Create grid configuration
        grid_config = GlobalVoxelGridConfig(
            d_min_target=grid_config_dict.get('d_min_target', 1.0),
            d_max_target=grid_config_dict.get('d_max_target', 10.0),
            ndiv_h=grid_config_dict.get('ndiv_h', 5),
            ndiv_k=grid_config_dict.get('ndiv_k', 5),
            ndiv_l=grid_config_dict.get('ndiv_l', 5)
        )
        
        # Module 3.S.1: Create global voxel grid
        logger.info("Creating global voxel grid")
        diffuse_data_list = [item['diffuse_data'] for item in self.phase2_outputs]
        
        global_grid = GlobalVoxelGrid(
            self.successful_experiments,
            diffuse_data_list,
            grid_config
        )
        
        diagnostics = global_grid.get_crystal_averaging_diagnostics()
        logger.info(f"Grid created: {diagnostics['total_voxels']} voxels, "
                   f"RMS misorientation: {diagnostics['rms_misorientation_deg']:.2f}°")
        
        # Module 3.S.2: Bin observations into voxels
        logger.info("Binning observations into voxels")
        space_group = self.successful_experiments[0].crystal.get_space_group()
        space_group_info = sgtbx.space_group_info(group=space_group)
        
        accumulator = VoxelAccumulator(
            global_grid,
            space_group_info,
            backend=scaling_config.voxel_accumulator_backend
        )
        
        # Add all observations
        for i, phase2_data in enumerate(self.phase2_outputs):
            diffuse_data = phase2_data['diffuse_data']
            n_binned = accumulator.add_observations(
                phase2_data['still_id'],
                diffuse_data.q_vectors,
                diffuse_data.intensities,
                diffuse_data.sigmas
            )
            logger.debug(f"Still {i}: binned {n_binned} observations")
        
        accumulator_stats = accumulator.get_accumulation_statistics()
        logger.info(f"Total observations binned: {accumulator_stats['total_observations']} "
                   f"in {accumulator_stats['unique_voxels']} voxels")
        
        # Get binned data for scaling
        binned_data = accumulator.get_all_binned_data_for_scaling()
        
        # Module 3.S.3: Relative scaling
        logger.info("Performing relative scaling")
        
        # Create scaling model configuration
        still_ids = [item['still_id'] for item in self.phase2_outputs]
        scaling_model_config = {
            'still_ids': still_ids,
            'per_still_scale': {'enabled': scaling_config.refine_per_still_scale},
            'resolution_smoother': {
                'enabled': scaling_config.refine_resolution_scale_multiplicative,
                'n_control_points': scaling_config.resolution_scale_bins or 3,
                'resolution_range': (0.1, 2.0)  # Default range
            },
            'experimental_components': {
                'panel_scale': {'enabled': False},
                'spatial_scale': {'enabled': False},
                'additive_offset': {'enabled': scaling_config.refine_additive_offset}
            },
            'partiality_threshold': scaling_config.min_partiality_threshold
        }
        
        scaling_model = DiffuseScalingModel(scaling_model_config)
        
        # Perform refinement
        refinement_config = scaling_config.refinement_config or {
            'max_iterations': 10,
            'convergence_tolerance': 1e-4
        }
        
        refined_params, refinement_stats = scaling_model.refine_parameters(
            binned_data,
            {},  # No Bragg reflections for now
            refinement_config
        )
        
        logger.info(f"Scaling refinement completed: {refinement_stats['n_iterations']} iterations, "
                   f"final R-factor: {refinement_stats['final_r_factor']:.6f}")
        
        # Module 3.S.4: Merge scaled data
        logger.info("Merging scaled data")
        merger = DiffuseDataMerger(global_grid)
        
        merge_config = scaling_config.merge_config or {
            'outlier_rejection': {'enabled': True, 'sigma_threshold': 3.0},
            'minimum_observations': 2,
            'weight_method': 'inverse_variance'
        }
        
        voxel_data = merger.merge_scaled_data(
            binned_data,
            scaling_model,
            merge_config
        )
        
        merge_stats = merger.get_merge_statistics(voxel_data)
        logger.info(f"Merging completed: {merge_stats['total_voxels']} voxels, "
                   f"mean intensity: {merge_stats['intensity_statistics']['mean']:.2f}")
        
        # Save merged data
        output_path = output_dir / "merged_diffuse_data.npz"
        merger.save_voxel_data(voxel_data, str(output_path))
        logger.info(f"Saved merged data to {output_path}")
        
        # Save scaling model information
        model_info_path = output_dir / "scaling_model_info.json"
        with open(model_info_path, 'w') as f:
            json.dump(scaling_model.get_model_info(), f, indent=2)
        
        # Save merge statistics
        stats_path = output_dir / "merge_statistics.json"
        with open(stats_path, 'w') as f:
            json.dump(merge_stats, f, indent=2)
        
        logger.info("Phase 3 completed successfully")
    
    def _update_summary_log(self, log_path: Path, outcome: StillProcessingOutcome):
        """Update the summary log with processing outcome."""
        entry = f"{Path(outcome.input_cbf_path).name}: {outcome.status}\n"
        if outcome.message:
            entry += f"  Message: {outcome.message}\n"
        
        with open(log_path, 'a') as f:
            f.write(entry)
    
    def _finalize_summary_log(self, log_path: Path):
        """Finalize the summary log with statistics."""
        with open(log_path, 'a') as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write("Processing Complete\n")
            f.write(f"Total Phase 2 outputs: {len(self.phase2_outputs)}\n")
            if hasattr(self, 'phase3_completed'):
                f.write("Phase 3: Completed\n")