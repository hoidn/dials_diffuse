"""
DiffuseScalingModel implementation for Phase 3 relative scaling.

Custom scaling model using DIALS components with v1 parameter constraints.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

from dials.algorithms.scaling.model.model import ScalingModelBase
from dials.algorithms.scaling.active_parameter_managers import active_parameter_manager
from dials.array_family import flex

from .components.per_still_multiplier import PerStillMultiplierComponent
from .components.resolution_smoother import ResolutionSmootherComponent

logger = logging.getLogger(__name__)

# v1 model constraints
MAX_FREE_PARAMS_BASE = 5  # Maximum additional parameters beyond per-still scales


class DiffuseScalingModel(ScalingModelBase):
    """
    Custom scaling model for relative scaling of diffuse scattering data.
    
    Implements v1 parameter-guarded model with per-still multiplicative scales
    and optional resolution smoothing. Enforces hard limits on model complexity.
    """
    
    def __init__(self, configdict: Dict, is_scaled: bool = False):
        """
        Initialize diffuse scaling model.
        
        Args:
            configdict: Configuration dictionary for scaling components
            is_scaled: Whether model has been applied to data
        """
        # Add required keys for DIALS base class
        dials_configdict = dict(configdict)
        dials_configdict.setdefault('corrections', ['scale'])  # Basic scale correction
        
        super().__init__(dials_configdict, is_scaled)
        
        self.diffuse_configdict = configdict
        self._validate_v1_config()
        
        # Initialize parameter manager with empty components
        self.active_parameter_manager = active_parameter_manager({}, [])
        
        # Track components and their parameter counts
        self.diffuse_components = {}
        self.component_param_counts = {}
        
        # Initialize components based on configuration
        self._initialize_components()
        
        # Store refinement state
        self.refinement_statistics = {}
        self.partiality_threshold = self.diffuse_configdict.get('partiality_threshold', 0.1)
        
        logger.info(f"DiffuseScalingModel initialized with {self.n_total_params} parameters")
    
    def _validate_v1_config(self):
        """Validate configuration against v1 constraints."""
        # Check that advanced components are disabled
        experimental = self.diffuse_configdict.get('experimental_components', {})
        
        forbidden_components = ['panel_scale', 'spatial_scale', 'additive_offset']
        for comp in forbidden_components:
            if experimental.get(comp, {}).get('enabled', False):
                raise ValueError(f"v1 model: {comp} component is hard-disabled")
        
        # Check resolution smoother constraints
        res_config = self.diffuse_configdict.get('resolution_smoother', {})
        if res_config.get('enabled', False):
            n_points = res_config.get('n_control_points', 0)
            if n_points > 5:
                raise ValueError(f"v1 model: resolution smoother limited to ≤5 points, got {n_points}")
    
    def _initialize_components(self):
        """Initialize scaling components based on configuration."""
        # Always initialize per-still multiplier (required for v1)
        per_still_config = self.diffuse_configdict.get('per_still_scale', {'enabled': True})
        if per_still_config.get('enabled', True):
            still_ids = self._get_still_ids_from_config()
            initial_values = per_still_config.get('initial_values', None)
            
            self.per_still_component = PerStillMultiplierComponent(
                self.active_parameter_manager, 
                still_ids, 
                initial_values
            )
            self.diffuse_components['per_still'] = self.per_still_component
            self.component_param_counts['per_still'] = len(still_ids)
        
        # Optionally initialize resolution smoother
        res_config = self.diffuse_configdict.get('resolution_smoother', {})
        if res_config.get('enabled', False):
            n_points = res_config.get('n_control_points', 3)
            q_range = res_config.get('resolution_range', (0.1, 2.0))
            
            self.resolution_component = ResolutionSmootherComponent(
                self.active_parameter_manager,
                n_points,
                q_range
            )
            self.diffuse_components['resolution'] = self.resolution_component
            self.component_param_counts['resolution'] = n_points
        
        # Verify total parameter count
        self.n_total_params = sum(self.component_param_counts.values())
        n_stills = self.component_param_counts.get('per_still', 0)
        max_allowed = MAX_FREE_PARAMS_BASE + n_stills
        
        if self.n_total_params > max_allowed:
            raise ValueError(f"v1 model: {self.n_total_params} parameters exceeds "
                           f"limit of {max_allowed} ({MAX_FREE_PARAMS_BASE} + {n_stills} stills)")
    
    def _get_still_ids_from_config(self) -> List[int]:
        """Extract still IDs from configuration or data."""
        # This would normally come from the actual data
        # For now, use configuration or reasonable defaults
        still_ids = self.diffuse_configdict.get('still_ids', [])
        if not still_ids:
            # Default to small number for testing
            still_ids = list(range(3))
            logger.warning(f"No still_ids in config, using default: {still_ids}")
        return still_ids
    
    def configure_components(self, reflection_table, experiment, params):
        """Configure components based on actual data."""
        # This method is called by DIALS framework
        # Update component configurations if needed
        pass
    
    def get_scales(self, data_dict: Dict) -> np.ndarray:
        """
        Get scale factors for observations.
        
        Args:
            data_dict: Dictionary containing observation data
            
        Returns:
            Array of scale factors
        """
        n_obs = len(data_dict.get('still_ids', []))
        if n_obs == 0:
            return np.array([])
        
        # Start with per-still scales
        per_still_scales, _ = self.per_still_component.calculate_scales_and_derivatives(data_dict)
        total_scales = np.array(per_still_scales)
        
        # Apply resolution-dependent corrections if enabled
        if 'resolution' in self.diffuse_components:
            res_scales, _ = self.resolution_component.calculate_scales_and_derivatives(data_dict)
            total_scales *= np.array(res_scales)
        
        return total_scales
    
    def get_scales_for_observation(self, 
                                  still_id: int,
                                  q_magnitude: float,
                                  **kwargs) -> Tuple[float, float]:
        """
        Get multiplicative scale and additive offset for single observation.
        
        Args:
            still_id: Still identifier
            q_magnitude: Magnitude of scattering vector
            **kwargs: Additional parameters (ignored)
            
        Returns:
            Tuple of (multiplicative_scale, additive_offset)
        """
        # Get per-still scale
        multiplicative_scale = self.per_still_component.get_scale_for_still(still_id)
        
        # Apply resolution correction if enabled
        if 'resolution' in self.diffuse_components:
            res_scale = self.resolution_component.get_scale_for_q(q_magnitude)
            multiplicative_scale *= res_scale
        
        # Additive offset is always 0.0 in v1
        additive_offset = 0.0
        
        return multiplicative_scale, additive_offset
    
    def refine_parameters(self, 
                         binned_pixel_data: Dict,
                         bragg_reflections: Dict,
                         refinement_config: Dict) -> Tuple[Dict, Dict]:
        """
        Refine model parameters using iterative minimization.
        
        Args:
            binned_pixel_data: Voxel-organized diffuse observations
            bragg_reflections: Bragg reflection data with partiality
            refinement_config: Refinement settings
            
        Returns:
            Tuple of (refined_parameters, refinement_statistics)
        """
        max_iterations = refinement_config.get('max_iterations', 10)
        convergence_tol = refinement_config.get('convergence_tolerance', 1e-4)
        
        logger.info(f"Starting parameter refinement: {max_iterations} max iterations")
        
        prev_r_factor = float('inf')
        
        for iteration in range(max_iterations):
            logger.debug(f"Refinement iteration {iteration + 1}")
            
            # Generate references with current parameters
            bragg_refs, diffuse_refs = self.generate_references(
                binned_pixel_data, bragg_reflections
            )
            
            # Calculate current R-factor
            r_factor = self._calculate_r_factor(binned_pixel_data, diffuse_refs)
            
            logger.debug(f"Iteration {iteration + 1}: R-factor = {r_factor:.6f}")
            
            # Check convergence
            if abs(prev_r_factor - r_factor) < convergence_tol:
                logger.info(f"Converged after {iteration + 1} iterations")
                break
            
            # Perform one step of parameter refinement
            self._refine_step(binned_pixel_data, diffuse_refs)
            
            prev_r_factor = r_factor
        
        # Extract refined parameters
        refined_params = self._extract_refined_parameters()
        
        # Calculate final statistics
        final_stats = {
            'n_iterations': iteration + 1,
            'final_r_factor': r_factor,
            'convergence_achieved': iteration < max_iterations - 1,
            'parameter_shifts': self._calculate_parameter_shifts()
        }
        
        self.refinement_statistics = final_stats
        logger.info(f"Refinement completed: R-factor = {r_factor:.6f}")
        
        return refined_params, final_stats
    
    def generate_references(self, 
                           binned_pixel_data: Dict,
                           bragg_reflections: Dict) -> Tuple[Dict, Dict]:
        """
        Generate Bragg and diffuse reference intensities.
        
        Args:
            binned_pixel_data: Voxel-organized diffuse data
            bragg_reflections: Bragg reflection data
            
        Returns:
            Tuple of (bragg_references, diffuse_references)
        """
        # Generate diffuse references (main focus for diffuse scaling)
        diffuse_refs = {}
        
        for voxel_idx, voxel_data in binned_pixel_data.items():
            observations = voxel_data['observations']
            if len(observations) == 0:
                continue
            
            # Apply current scaling to observations
            scaled_intensities = []
            weights = []
            
            for obs in observations:
                still_id = obs['still_id']
                q_mag = np.linalg.norm(obs['q_vector_lab'])
                
                mult_scale, add_offset = self.get_scales_for_observation(still_id, q_mag)
                
                # Apply scaling: I_scaled = (I_obs - C) / M
                # For v1: C = 0, so I_scaled = I_obs / M
                scaled_intensity = obs['intensity'] / mult_scale
                weight = 1.0 / (obs['sigma'] / mult_scale)**2
                
                scaled_intensities.append(scaled_intensity)
                weights.append(weight)
            
            # Weighted average for reference
            if weights:
                weighted_intensities = np.array(scaled_intensities) * np.array(weights)
                total_weight = np.sum(weights)
                
                if total_weight > 0:
                    ref_intensity = np.sum(weighted_intensities) / total_weight
                    ref_sigma = 1.0 / np.sqrt(total_weight)
                    
                    diffuse_refs[voxel_idx] = {
                        'intensity': ref_intensity,
                        'sigma': ref_sigma,
                        'n_observations': len(observations)
                    }
        
        # Bragg references (simplified for now)
        bragg_refs = {}
        if bragg_reflections:
            # Implementation would filter by partiality and create references
            # For now, return empty dict
            pass
        
        logger.debug(f"Generated {len(diffuse_refs)} diffuse references")
        
        return bragg_refs, diffuse_refs
    
    def _calculate_r_factor(self, binned_pixel_data: Dict, diffuse_refs: Dict) -> float:
        """Calculate R-factor for current model."""
        numerator = 0.0
        denominator = 0.0
        
        for voxel_idx, voxel_data in binned_pixel_data.items():
            if voxel_idx not in diffuse_refs:
                continue
            
            ref_intensity = diffuse_refs[voxel_idx]['intensity']
            
            for obs in voxel_data['observations']:
                still_id = obs['still_id']
                q_mag = np.linalg.norm(obs['q_vector_lab'])
                
                mult_scale, _ = self.get_scales_for_observation(still_id, q_mag)
                scaled_obs = obs['intensity'] / mult_scale
                
                numerator += abs(scaled_obs - ref_intensity)
                denominator += scaled_obs
        
        if denominator > 0:
            return numerator / denominator
        else:
            return float('inf')
    
    def _refine_step(self, binned_pixel_data: Dict, diffuse_refs: Dict):
        """Perform one step of parameter refinement."""
        # Simplified refinement step
        # Real implementation would use DIALS minimizer
        
        # For now, implement simple gradient descent
        learning_rate = 0.01
        
        # Calculate gradients for per-still parameters
        for still_id in self.per_still_component.still_ids:
            current_scale = self.per_still_component.get_scale_for_still(still_id)
            
            # Calculate approximate gradient
            gradient = self._calculate_gradient_for_still(
                still_id, binned_pixel_data, diffuse_refs
            )
            
            # Update parameter
            new_scale = current_scale - learning_rate * gradient
            new_scale = max(0.1, min(10.0, new_scale))  # Clamp to reasonable range
            
            self.per_still_component.set_scale_for_still(still_id, new_scale)
    
    def _calculate_gradient_for_still(self, still_id: int, 
                                    binned_pixel_data: Dict, 
                                    diffuse_refs: Dict) -> float:
        """Calculate gradient for a still's scale parameter."""
        gradient = 0.0
        count = 0
        
        for voxel_idx, voxel_data in binned_pixel_data.items():
            if voxel_idx not in diffuse_refs:
                continue
            
            ref_intensity = diffuse_refs[voxel_idx]['intensity']
            
            for obs in voxel_data['observations']:
                if obs['still_id'] != still_id:
                    continue
                
                q_mag = np.linalg.norm(obs['q_vector_lab'])
                mult_scale, _ = self.get_scales_for_observation(still_id, q_mag)
                
                scaled_obs = obs['intensity'] / mult_scale
                residual = scaled_obs - ref_intensity
                
                # Gradient of residual w.r.t. scale parameter
                # d/dM [(I/M) - Iref] = -I/M^2
                grad_contrib = -obs['intensity'] / (mult_scale**2)
                gradient += residual * grad_contrib
                count += 1
        
        if count > 0:
            gradient /= count
        
        return gradient
    
    def _extract_refined_parameters(self) -> Dict:
        """Extract refined parameters from components."""
        refined_params = {}
        
        for still_id in self.per_still_component.still_ids:
            refined_params[still_id] = {
                'multiplicative_scale': self.per_still_component.get_scale_for_still(still_id),
                'additive_offset': 0.0  # Always 0.0 in v1
            }
        
        return refined_params
    
    def _calculate_parameter_shifts(self) -> Dict:
        """Calculate parameter shifts from last iteration."""
        # Simplified - would track actual shifts in real implementation
        return {'per_still': 0.001}
    
    def get_model_info(self) -> Dict:
        """Get comprehensive model information."""
        info = {
            'model_type': 'DiffuseScalingModel_v1',
            'n_total_params': self.n_total_params,
            'components': {},
            'refinement_statistics': self.refinement_statistics,
            'partiality_threshold': self.partiality_threshold
        }
        
        for comp_name, component in self.diffuse_components.items():
            info['components'][comp_name] = component.get_component_info()
        
        return info
    
    def __str__(self) -> str:
        """String representation of model."""
        comp_names = list(self.diffuse_components.keys())
        return f"DiffuseScalingModel({self.n_total_params} params, components: {comp_names})"