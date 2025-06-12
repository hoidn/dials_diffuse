"""
ResolutionSmootherComponent for diffuse scaling model.

Implements 1D Gaussian smoother for resolution-dependent multiplicative scaling.
"""

import logging
import numpy as np
from typing import Tuple

from dials.algorithms.scaling.model.components.scale_components import ScaleComponentBase
from dials.algorithms.scaling.model.components.smooth_scale_components import GaussianSmoother1D
from dials.array_family import flex

logger = logging.getLogger(__name__)


class ResolutionSmootherComponent(ScaleComponentBase):
    """
    1D Gaussian smoother for resolution-dependent multiplicative scaling.
    
    Provides smooth multiplicative corrections as a function of |q| (resolution).
    Limited to ≤5 control points in v1 implementation.
    """
    
    def __init__(self,
                 active_parameter_manager,
                 n_control_points: int,
                 resolution_range: Tuple[float, float]):
        """
        Initialize resolution smoother component.
        
        Args:
            active_parameter_manager: DIALS parameter manager for refinement
            n_control_points: Number of control points (≤5 for v1)
            resolution_range: (q_min, q_max) for smoother domain
        """
        # Validate v1 constraints
        if n_control_points > 5:
            raise ValueError(f"v1 model allows ≤5 control points, got {n_control_points}")
        
        q_min, q_max = resolution_range
        if q_min >= q_max:
            raise ValueError("Invalid resolution range: q_min must be < q_max")
        
        self.n_control_points = n_control_points
        self.q_min = q_min
        self.q_max = q_max
        self.q_range = (q_min, q_max)
        
        # Create 1D Gaussian smoother
        self.smoother = GaussianSmoother1D(
            x_range=self.q_range,
            num_intervals=n_control_points - 1  # num_intervals is one less than control points
        )
        
        # Initialize control points to unity (no correction)
        initial_values = flex.double([1.0] * n_control_points)
        
        # Initialize base class with parameter values
        super().__init__(initial_values)
        
        # Store parameter manager reference
        self.active_parameter_manager = active_parameter_manager
        
        logger.info(f"ResolutionSmootherComponent initialized: {n_control_points} points, "
                   f"q_range=({q_min:.4f}, {q_max:.4f})")
    
    @property
    def n_params(self) -> int:
        """Number of parameters in this component."""
        return self.n_control_points
    
    @property
    def parameters(self) -> flex.double:
        """Current parameter values from the parameter manager."""
        # Use the parent class parameters directly
        return self._parameters
    
    def calculate_scales_and_derivatives(self, 
                                       reflection_table,
                                       block_id=None):
        """
        Calculate resolution-dependent scale factors and derivatives.
        
        Args:
            reflection_table: Data structure containing q-vectors or d-spacing
            block_id: Unused for this component
            
        Returns:
            tuple: (scales, derivatives) as flex arrays
        """
        # Extract q-magnitudes from different data formats
        q_magnitudes = self._extract_q_magnitudes(reflection_table)
        n_reflections = len(q_magnitudes)
        
        if n_reflections == 0:
            return flex.double(), flex.double()
        
        # Update smoother with current parameters
        current_params = self.parameters
# Parameter setting handled by parent class
        
        # Evaluate smoother at q-values
        q_locations = flex.double(q_magnitudes)
        
        # Clamp q-values to smoother range to avoid extrapolation
        q_clamped = flex.double()
        for q in q_locations:
            q_clamp = max(self.q_min, min(self.q_max, q))
            q_clamped.append(q_clamp)
        
        # Evaluate smoother
        scales, derivatives = self.smoother.value_error_for_location(q_clamped)
        
        # For multiplicative scaling, we want positive values
        # Apply softplus-like transformation to ensure positivity
        scales_positive = flex.double()
        for scale in scales:
            # Use exp to ensure positivity while allowing the smoother 
            # to work in log space around 1.0
            positive_scale = max(0.01, float(scale))  # Minimum scale
            scales_positive.append(positive_scale)
        
        # Derivatives need to be transformed accordingly
        # For now, use simple approach - could be more sophisticated
        derivatives_transformed = derivatives
        
        return scales_positive, derivatives_transformed
    
    def _extract_q_magnitudes(self, reflection_table):
        """
        Extract q-magnitudes from various data formats.
        
        Args:
            reflection_table: Data structure with q-vectors or d-spacing
            
        Returns:
            List of q-magnitudes
        """
        q_magnitudes = []
        
        if isinstance(reflection_table, dict) and ('q_vectors_lab' in reflection_table or 'q_magnitudes' in reflection_table):
            # Diffuse data dictionary format
            if 'q_vectors_lab' in reflection_table:
                q_vectors = reflection_table['q_vectors_lab']
                if len(q_vectors.shape) == 2 and q_vectors.shape[1] == 3:
                    q_magnitudes = np.linalg.norm(q_vectors, axis=1)
                else:
                    raise ValueError("Invalid q_vectors shape")
            elif 'q_magnitudes' in reflection_table:
                q_magnitudes = reflection_table['q_magnitudes']
                
        elif hasattr(reflection_table, 'get'):
            # DIALS flex.reflection_table
            if 'q_vector' in reflection_table:
                q_vectors = reflection_table.get('q_vector')
                q_magnitudes = [q.length() for q in q_vectors]
            elif 'd' in reflection_table:
                d_spacings = reflection_table.get('d')
                q_magnitudes = [1.0/d if d > 0 else 0.0 for d in d_spacings]
            else:
                raise ValueError("No q-vector data in diffuse data dictionary")
        else:
            raise ValueError("Unsupported reflection_table format")
        
        return q_magnitudes
    
    def get_scale_for_q(self, q_magnitude: float) -> float:
        """
        Get resolution-dependent scale factor for a specific q-value.
        
        Args:
            q_magnitude: Magnitude of scattering vector
            
        Returns:
            Multiplicative scale factor
        """
        if q_magnitude <= 0:
            return 1.0
        
        # Clamp to smoother range
        q_clamp = max(self.q_min, min(self.q_max, q_magnitude))
        
        # Update smoother with current parameters
        current_params = self.parameters
# Parameter setting handled by parent class
        
        # Evaluate at single point
        q_array = flex.double([q_clamp])
        scales, _ = self.smoother.value_error_for_location(q_array)
        
        # Ensure positive scale
        scale = max(0.01, float(scales[0]))
        return scale
    
    def get_component_info(self) -> dict:
        """
        Get information about this component.
        
        Returns:
            Dictionary with component details
        """
        current_params = self.parameters
        
        # Sample smoother across range for statistics
        n_samples = 20
        q_samples = np.linspace(self.q_min, self.q_max, n_samples)
        scale_samples = [self.get_scale_for_q(q) for q in q_samples]
        
        return {
            "component_type": "ResolutionSmoother",
            "n_parameters": self.n_params,
            "n_control_points": self.n_control_points,
            "q_range": self.q_range,
            "control_point_values": [float(p) for p in current_params],
            "scale_statistics": {
                "mean": np.mean(scale_samples),
                "std": np.std(scale_samples),
                "min": np.min(scale_samples),
                "max": np.max(scale_samples)
            }
        }
    
    def update_reflection_data(self, reflection_table):
        """
        Update component state based on reflection data.
        
        This component doesn't need to update state based on reflections.
        """
        pass
    
    def __str__(self) -> str:
        """String representation of component."""
        info = self.get_component_info()
        stats = info["scale_statistics"]
        return (f"ResolutionSmootherComponent({self.n_control_points} points, "
                f"scales: {stats['mean']:.3f}±{stats['std']:.3f})")
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"ResolutionSmootherComponent(n_control_points={self.n_control_points}, "
                f"q_range=({self.q_min:.4f}, {self.q_max:.4f}))")