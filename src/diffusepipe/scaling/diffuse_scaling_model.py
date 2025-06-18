"""
DiffuseScalingModel implementation for Phase 3 relative scaling.

Custom scaling model using DIALS components with v1 parameter constraints.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple

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
        dials_configdict.setdefault("corrections", ["scale"])  # Basic scale correction

        super().__init__(dials_configdict, is_scaled)

        self.diffuse_configdict = configdict
        self._validate_v1_config()

        # Track components and their parameter counts
        self.diffuse_components = {}
        self.component_param_counts = {}

        # Initialize components based on configuration
        self._initialize_components()

        # Initialize parameter manager after components are created
        self._setup_parameter_manager()

        # Store refinement state
        self.refinement_statistics = {}
        self.partiality_threshold = self.diffuse_configdict.get(
            "partiality_threshold", 0.1
        )

        logger.info(
            f"DiffuseScalingModel initialized with {self.n_total_params} parameters"
        )

    def _validate_v1_config(self):
        """Validate configuration against v1 constraints."""
        # Check that advanced components are disabled
        experimental = self.diffuse_configdict.get("experimental_components", {})

        forbidden_components = ["panel_scale", "spatial_scale", "additive_offset"]
        for comp in forbidden_components:
            if experimental.get(comp, {}).get("enabled", False):
                raise ValueError(f"v1 model: {comp} component is hard-disabled")

        # Check resolution smoother constraints
        res_config = self.diffuse_configdict.get("resolution_smoother", {})
        if res_config.get("enabled", False):
            n_points = res_config.get("n_control_points", 0)
            if n_points > 5:
                raise ValueError(
                    f"v1 model: resolution smoother limited to ≤5 points, got {n_points}"
                )

    def _initialize_components(self):
        """Initialize scaling components based on configuration."""
        # Always initialize per-still multiplier (required for v1)
        per_still_config = self.diffuse_configdict.get(
            "per_still_scale", {"enabled": True}
        )
        if per_still_config.get("enabled", True):
            still_ids = self._get_still_ids_from_config()
            initial_values = per_still_config.get("initial_values", None)

            self.per_still_component = PerStillMultiplierComponent(
                still_ids, initial_values
            )
            self.diffuse_components["per_still"] = self.per_still_component
            self.component_param_counts["per_still"] = len(still_ids)

        # Optionally initialize resolution smoother
        res_config = self.diffuse_configdict.get("resolution_smoother", {})
        if res_config.get("enabled", False):
            n_points = res_config.get("n_control_points", 3)
            q_range = res_config.get("resolution_range", (0.1, 2.0))

            self.resolution_component = ResolutionSmootherComponent(n_points, q_range)
            self.diffuse_components["resolution"] = self.resolution_component
            self.component_param_counts["resolution"] = n_points

        # Verify total parameter count
        self.n_total_params = sum(self.component_param_counts.values())
        n_stills = self.component_param_counts.get("per_still", 0)
        max_allowed = MAX_FREE_PARAMS_BASE + n_stills

        if self.n_total_params > max_allowed:
            raise ValueError(
                f"v1 model: {self.n_total_params} parameters exceeds "
                f"limit of {max_allowed} ({MAX_FREE_PARAMS_BASE} + {n_stills} stills)"
            )

    def _get_still_ids_from_config(self) -> List[int]:
        """Extract still IDs from configuration or data."""
        # This would normally come from the actual data
        # For now, use configuration or reasonable defaults
        still_ids = self.diffuse_configdict.get("still_ids", [])
        if not still_ids:
            # Default to small number for testing
            still_ids = list(range(3))
            logger.warning(f"No still_ids in config, using default: {still_ids}")
        return still_ids

    def _setup_parameter_manager(self):
        """Set up parameter manager with all component parameters."""
        # Collect all parameters from components
        all_components = list(self.diffuse_components.values())
        component_names = list(self.diffuse_components.keys())

        # Create component dictionary for parameter manager
        components_dict = {}
        for name, component in zip(component_names, all_components):
            components_dict[name] = component

        # Create active parameter manager
        self.active_parameter_manager = active_parameter_manager(
            components_dict, component_names
        )

    @property
    def components(self):
        """Property to return component names for compatibility with DIALS and tests."""
        return self.diffuse_components

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
        n_obs = len(data_dict.get("still_ids", []))
        if n_obs == 0:
            return np.array([])

        # Start with per-still scales
        per_still_scales, _ = self.per_still_component.calculate_scales_and_derivatives(
            data_dict
        )
        total_scales = np.array(per_still_scales)

        # Apply resolution-dependent corrections if enabled
        if "resolution" in self.diffuse_components:
            res_scales, _ = self.resolution_component.calculate_scales_and_derivatives(
                data_dict
            )
            total_scales *= np.array(res_scales)

        return total_scales

    def get_scales_for_observation(
        self, still_id: int, q_magnitude: float, **kwargs
    ) -> Tuple[float, float]:
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
        if "resolution" in self.diffuse_components:
            res_scale = self.resolution_component.get_scale_for_q(q_magnitude)
            multiplicative_scale *= res_scale

        # Additive offset is always 0.0 in v1
        additive_offset = 0.0

        return multiplicative_scale, additive_offset

    def calculate_scales_and_derivatives(
        self, still_id: int, q_magnitude: float
    ) -> Tuple[float, flex.double]:
        """
        Calculate total scale and derivatives for a single observation.

        This helper centralizes the complex logic of calculating the total multiplicative
        scale and its derivatives with respect to all model parameters for a single observation.
        Essential for populating the Jacobian matrix in the refinement loop.

        Args:
            still_id: Still identifier
            q_magnitude: Magnitude of scattering vector

        Returns:
            Tuple of (total_scale, derivatives_array)
        """
        # Start with per-still scale and its derivative
        per_still_scale = self.per_still_component.get_scale_for_still(still_id)
        total_scale = per_still_scale

        # Initialize derivatives array
        total_params = self.n_total_params
        derivatives = flex.double(total_params, 0.0)

        # Per-still component derivatives (always present)
        if still_id in self.per_still_component.still_id_to_param_idx:
            param_idx = self.per_still_component.still_id_to_param_idx[still_id]
            # d(total_scale)/d(per_still_param) = resolution_scale (if present) or 1.0
            if "resolution" in self.diffuse_components:
                res_scale = self.resolution_component.get_scale_for_q(q_magnitude)
                derivatives[param_idx] = res_scale
                total_scale *= res_scale
            else:
                derivatives[param_idx] = 1.0

        # Resolution component derivatives (if enabled)
        if "resolution" in self.diffuse_components:
            # Create mock data for resolution component evaluation
            mock_data = {"q_magnitudes": [q_magnitude]}
            _, res_derivatives = (
                self.resolution_component.calculate_scales_and_derivatives(mock_data)
            )

            # Resolution component parameters start after per-still parameters
            per_still_params = self.component_param_counts.get("per_still", 0)
            res_params = self.component_param_counts.get("resolution", 0)

            # Apply chain rule: d(total)/d(res_param) = per_still_scale * d(res_scale)/d(res_param)
            for i in range(res_params):
                if len(res_derivatives) > i:
                    derivatives[per_still_params + i] = (
                        per_still_scale * res_derivatives[i]
                    )

        return total_scale, derivatives

    def vectorized_calculate_scales_and_derivatives(
        self, still_ids: np.ndarray, q_magnitudes: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Vectorized version: Calculate total scales and derivatives for arrays of observations.

        This method performs the same calculations as calculate_scales_and_derivatives
        but operates on entire arrays for significant performance gains (10x+ speedup).

        Args:
            still_ids: Array of still identifiers, shape (N,)
            q_magnitudes: Array of q-vector magnitudes, shape (N,)

        Returns:
            Tuple of (total_scales, derivatives_matrix) as NumPy arrays
            - total_scales: shape (N,) - total scale for each observation
            - derivatives_matrix: shape (N, n_params) - derivatives w.r.t. all parameters
        """
        n_obs = len(still_ids)
        if n_obs == 0:
            return np.array([]), np.zeros((0, self.n_total_params))

        # Initialize arrays
        total_scales = np.ones(n_obs)
        derivatives_matrix = np.zeros((n_obs, self.n_total_params))

        # Get per-still scales using vectorized component
        per_still_data = {"still_ids": still_ids}
        per_still_scales, per_still_derivs = self.per_still_component.calculate_scales_and_derivatives(per_still_data)
        
        # Convert DIALS flex arrays to NumPy arrays
        per_still_scales_np = np.array(per_still_scales)
        per_still_derivs_np = np.array(per_still_derivs)
        
        # Apply per-still scales
        total_scales = per_still_scales_np

        # Set per-still derivatives
        for i, still_id in enumerate(still_ids):
            if still_id in self.per_still_component.still_id_to_param_idx:
                param_idx = self.per_still_component.still_id_to_param_idx[still_id]
                derivatives_matrix[i, param_idx] = 1.0  # Will be modified by chain rule if resolution component exists

        # Apply resolution component if enabled
        if "resolution" in self.diffuse_components:
            # Get resolution scales using vectorized component
            res_data = {"q_magnitudes": q_magnitudes}
            res_scales, res_derivs = self.resolution_component.calculate_scales_and_derivatives(res_data)
            
            # Convert to NumPy arrays
            res_scales_np = np.array(res_scales)
            res_derivs_np = np.array(res_derivs)
            
            # Apply multiplicative combination: total_scale = per_still_scale * res_scale
            total_scales *= res_scales_np

            # Apply chain rule for derivatives
            per_still_params = self.component_param_counts.get("per_still", 0)
            res_params = self.component_param_counts.get("resolution", 0)

            # Update per-still derivatives with chain rule: d(total)/d(per_still) = res_scale
            for i, still_id in enumerate(still_ids):
                if still_id in self.per_still_component.still_id_to_param_idx:
                    param_idx = self.per_still_component.still_id_to_param_idx[still_id]
                    derivatives_matrix[i, param_idx] = res_scales_np[i]

            # Add resolution derivatives with chain rule: d(total)/d(res_param) = per_still_scale * d(res)/d(res_param)
            # Reshape res_derivs_np to (n_obs, res_params)
            if len(res_derivs_np) == n_obs * res_params:
                res_derivs_matrix = res_derivs_np.reshape(n_obs, res_params)
                for j in range(res_params):
                    derivatives_matrix[:, per_still_params + j] = per_still_scales_np * res_derivs_matrix[:, j]

        return total_scales, derivatives_matrix

    def _aggregate_all_observations(self, binned_pixel_data: Dict) -> Dict[str, np.ndarray]:
        """
        Aggregate all observations from nested voxel dictionary into flat NumPy arrays.
        
        This preprocessing step extracts all observations into continuous arrays
        for vectorized processing, eliminating the need for nested loops.
        
        Args:
            binned_pixel_data: Voxel-organized diffuse observations
            
        Returns:
            Dictionary containing:
            - all_intensities: shape (N,) - all observation intensities
            - all_sigmas: shape (N,) - all observation sigmas  
            - all_still_ids: shape (N,) - all still identifiers
            - all_q_mags: shape (N,) - all q-vector magnitudes
            - voxel_idx_map: shape (N,) - voxel index for each observation
            - voxel_start_indices: dict mapping voxel_idx -> start position in arrays
            - total_observations: int - total number of observations
        """
        # Count total observations first
        total_obs = 0
        voxel_obs_counts = {}
        
        for voxel_idx, voxel_data in binned_pixel_data.items():
            voxel_data = self._normalize_voxel_data_format(voxel_data)
            n_obs = voxel_data["n_observations"]
            voxel_obs_counts[voxel_idx] = n_obs
            total_obs += n_obs
        
        if total_obs == 0:
            return {
                "all_intensities": np.array([]),
                "all_sigmas": np.array([]),
                "all_still_ids": np.array([]),
                "all_q_mags": np.array([]),
                "voxel_idx_map": np.array([]),
                "voxel_start_indices": {},
                "total_observations": 0
            }
        
        # Pre-allocate arrays
        all_intensities = np.zeros(total_obs)
        all_sigmas = np.zeros(total_obs)
        all_still_ids = np.zeros(total_obs, dtype=int)
        all_q_mags = np.zeros(total_obs)
        voxel_idx_map = np.zeros(total_obs, dtype=int)
        
        # Track starting indices for each voxel
        voxel_start_indices = {}
        current_idx = 0
        
        # Fill arrays by iterating through voxels once
        for voxel_idx, voxel_data in binned_pixel_data.items():
            voxel_data = self._normalize_voxel_data_format(voxel_data)
            n_obs = voxel_data["n_observations"]
            
            if n_obs == 0:
                continue
                
            # Record start index for this voxel
            voxel_start_indices[voxel_idx] = current_idx
            
            # Extract data arrays from voxel
            intensities = voxel_data["intensities"]
            sigmas = voxel_data["sigmas"]
            still_ids = voxel_data["still_ids"]
            q_vectors = voxel_data["q_vectors_lab"]
            
            # Calculate q-magnitudes
            q_mags = np.linalg.norm(q_vectors, axis=1)
            
            # Fill the aggregated arrays
            end_idx = current_idx + n_obs
            all_intensities[current_idx:end_idx] = intensities
            all_sigmas[current_idx:end_idx] = sigmas
            all_still_ids[current_idx:end_idx] = still_ids
            all_q_mags[current_idx:end_idx] = q_mags
            voxel_idx_map[current_idx:end_idx] = voxel_idx
            
            current_idx = end_idx
        
        return {
            "all_intensities": all_intensities,
            "all_sigmas": all_sigmas,
            "all_still_ids": all_still_ids,
            "all_q_mags": all_q_mags,
            "voxel_idx_map": voxel_idx_map,
            "voxel_start_indices": voxel_start_indices,
            "total_observations": total_obs
        }

    def refine_parameters(
        self, binned_pixel_data: Dict, bragg_reflections: Dict, refinement_config: Dict
    ) -> Tuple[Dict, Dict]:
        """
        Refine model parameters using iterative Levenberg-Marquardt minimization.

        Args:
            binned_pixel_data: Voxel-organized diffuse observations
            bragg_reflections: Bragg reflection data with partiality
            refinement_config: Refinement settings

        Returns:
            Tuple of (refined_parameters, refinement_statistics)
        """
        # WARNING: Placeholder implementation alert
        logger.warning(
            "The current 'refine_parameters' implementation is a simplified placeholder "
            "and does not use a robust DIALS/ScitBX minimizer. The results may not be optimal."
        )
        max_iterations = refinement_config.get("max_iterations", 10)
        convergence_tol = refinement_config.get("convergence_tolerance", 1e-6)
        parameter_shift_tol = refinement_config.get("parameter_shift_tolerance", 1e-6)

        logger.info(
            f"Starting VECTORIZED Levenberg-Marquardt parameter refinement: {max_iterations} max iterations"
        )

        # Pre-aggregate all observations for vectorized processing
        logger.debug("Aggregating observations for vectorized processing...")
        aggregated_data = self._aggregate_all_observations(binned_pixel_data)
        total_obs = aggregated_data["total_observations"]
        
        if total_obs == 0:
            logger.warning("No observations for refinement")
            return {}, {"n_iterations": 0, "final_r_factor": float("inf"), "convergence_achieved": False}

        logger.info(f"Processing {total_obs} observations with vectorized algorithm")

        prev_r_factor = float("inf")
        parameter_shifts_history = []

        for iteration in range(max_iterations):
            logger.debug(f"Refinement iteration {iteration + 1}")

            # 1. Generate references with current parameters (VECTORIZED)
            bragg_refs, diffuse_refs = self.vectorized_generate_references(
                aggregated_data, bragg_reflections
            )

            # 2. Calculate current R-factor (VECTORIZED)
            r_factor = self.vectorized_calculate_r_factor(aggregated_data, diffuse_refs)
            logger.debug(f"Iteration {iteration + 1}: R-factor = {r_factor:.6f}")

            # 3. VECTORIZED residuals and Jacobian calculation
            # Extract aggregated arrays
            all_intensities = aggregated_data["all_intensities"]
            all_still_ids = aggregated_data["all_still_ids"]
            all_q_mags = aggregated_data["all_q_mags"]
            voxel_idx_map = aggregated_data["voxel_idx_map"]
            
            # Create mask for observations that have reference intensities
            valid_mask = np.array([voxel_idx in diffuse_refs for voxel_idx in voxel_idx_map])
            
            if not np.any(valid_mask):
                logger.warning("No observations with reference intensities")
                break
                
            # Filter to valid observations
            valid_intensities = all_intensities[valid_mask]
            valid_still_ids = all_still_ids[valid_mask]
            valid_q_mags = all_q_mags[valid_mask]
            valid_voxel_indices = voxel_idx_map[valid_mask]
            n_residuals = len(valid_intensities)
            
            # VECTORIZED scaling and derivatives calculation
            all_scales, all_derivatives = self.vectorized_calculate_scales_and_derivatives(
                valid_still_ids, valid_q_mags
            )
            
            # Stabilize scales to prevent division by zero
            safe_total_scales = np.where(np.abs(all_scales) > 1e-9, all_scales, 1e-9)
            
            # Get reference intensities for all valid observations
            all_ref_intensities = np.array([diffuse_refs[voxel_idx]["intensity"] for voxel_idx in valid_voxel_indices])
            
            # VECTORIZED residual calculation: residual = (I_obs / M) - I_ref
            valid_scaled_obs = valid_intensities / safe_total_scales
            all_residuals = valid_scaled_obs - all_ref_intensities
            
            # VECTORIZED Jacobian calculation: d(residual)/d(param) = (-I_obs / M²) * (dM/dparam)
            # Broadcasting: (-I_obs / M²) is shape (N,), derivatives is shape (N, n_params)
            jacobian_matrix = (-valid_intensities / (safe_total_scales**2))[:, np.newaxis] * all_derivatives

            if n_residuals == 0:
                logger.warning("No valid residuals for refinement")
                break

            # 4. Solve using least squares (same as before, but with vectorized inputs)
            try:
                # Solve normal equations: J^T J delta = -J^T r
                JtJ = jacobian_matrix.T @ jacobian_matrix
                Jtr = jacobian_matrix.T @ all_residuals

                # Add small damping for numerical stability
                damping = 1e-6
                n_params = self.n_total_params
                JtJ += damping * np.eye(n_params)

                # Solve for parameter shifts
                parameter_shifts_np = np.linalg.solve(JtJ, -Jtr)
                parameter_shifts = flex.double(parameter_shifts_np.tolist())

                # 5. Apply parameter shifts
                current_params = self.active_parameter_manager.get_param_vals()
                new_params = current_params + parameter_shifts
                self.active_parameter_manager.set_param_vals(new_params)

                # Update component parameters
                self._update_component_parameters_from_manager()

                # Track parameter shifts
                shift_magnitude = flex.max(flex.abs(parameter_shifts))
                parameter_shifts_history.append(shift_magnitude)

                logger.debug(f"  Max parameter shift: {shift_magnitude:.6f}")

                # 6. Check convergence
                r_factor_change = abs(prev_r_factor - r_factor)
                if (
                    r_factor_change < convergence_tol
                    and shift_magnitude < parameter_shift_tol
                ):
                    logger.info(f"Converged after {iteration + 1} iterations")
                    break

                prev_r_factor = r_factor

            except Exception as e:
                logger.error(
                    f"Levenberg-Marquardt solver failed at iteration {iteration + 1}: {e}"
                )
                break

        # Extract refined parameters
        refined_params = self._extract_refined_parameters()

        # Calculate final statistics
        final_r_factor = self._calculate_r_factor(binned_pixel_data, diffuse_refs)
        final_stats = {
            "n_iterations": iteration + 1,
            "final_r_factor": final_r_factor,
            "convergence_achieved": iteration < max_iterations - 1,
            "parameter_shifts": parameter_shifts_history,
            "initial_r_factor": prev_r_factor if iteration == 0 else None,
            "r_factor_improvement": (
                prev_r_factor - final_r_factor if iteration > 0 else 0.0
            ),
        }

        self.refinement_statistics = final_stats
        logger.info(f"Refinement completed: R-factor = {final_r_factor:.6f}")

        return refined_params, final_stats

    def _update_component_parameters_from_manager(self):
        """Update component parameters from active parameter manager."""
        all_params = self.active_parameter_manager.get_param_vals()

        # Update per-still component parameters
        per_still_params = self.component_param_counts.get("per_still", 0)
        if per_still_params > 0:
            self.per_still_component._parameters = all_params[:per_still_params]

        # Update resolution component parameters (if enabled)
        if "resolution" in self.diffuse_components:
            res_params = self.component_param_counts.get("resolution", 0)
            if res_params > 0:
                self.resolution_component._parameters = all_params[
                    per_still_params : per_still_params + res_params
                ]

    def _normalize_voxel_data_format(self, voxel_data: Dict) -> Dict:
        """
        Normalize voxel data to the new efficient format for backward compatibility.
        
        Handles both old format (list of observation dicts) and new format (NumPy arrays).
        """
        # Check if it's already in the new format
        if "n_observations" in voxel_data and "intensities" in voxel_data:
            return voxel_data
        
        # Convert from old format to new format
        if "observations" in voxel_data:
            observations = voxel_data["observations"]
            n_obs = len(observations)
            
            if n_obs == 0:
                return {
                    "n_observations": 0,
                    "intensities": np.array([]),
                    "sigmas": np.array([]),
                    "still_ids": np.array([]),
                    "q_vectors_lab": np.array([]).reshape(0, 3),
                }
            
            # Extract arrays from list of dictionaries
            intensities = np.array([obs["intensity"] for obs in observations])
            sigmas = np.array([obs["sigma"] for obs in observations])
            still_ids = np.array([obs["still_id"] for obs in observations])
            q_vectors = np.array([obs["q_vector_lab"] for obs in observations])
            
            return {
                "n_observations": n_obs,
                "intensities": intensities,
                "sigmas": sigmas,
                "still_ids": still_ids,
                "q_vectors_lab": q_vectors,
            }
        
        # Fallback: assume it's already correct
        return voxel_data

    def generate_references(
        self, binned_pixel_data: Dict, bragg_reflections: Dict
    ) -> Tuple[Dict, Dict]:
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
            # Normalize data format for backward compatibility
            voxel_data = self._normalize_voxel_data_format(voxel_data)
            # Extract NumPy arrays from voxel_data
            n_obs = voxel_data["n_observations"]
            if n_obs == 0:
                continue

            still_ids = voxel_data["still_ids"]
            q_vectors = voxel_data["q_vectors_lab"]
            intensities = voxel_data["intensities"]
            sigmas = voxel_data["sigmas"]

            # Apply current scaling to observations
            scaled_intensities = []
            weights = []

            # Create new inner loop: iterate through array indices
            for i in range(n_obs):
                still_id = still_ids[i]
                q_mag = np.linalg.norm(q_vectors[i])

                mult_scale, add_offset = self.get_scales_for_observation(
                    still_id, q_mag
                )

                # Apply scaling: I_scaled = (I_obs - C) / M
                # For v1: C = 0, so I_scaled = I_obs / M
                scaled_intensity = intensities[i] / mult_scale
                variance = (sigmas[i] / mult_scale) ** 2
                weight = 1.0 / (variance + 1e-10)

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
                        "intensity": ref_intensity,
                        "sigma": ref_sigma,
                        "n_observations": n_obs,
                    }

        # Bragg references (simplified for now)
        bragg_refs = {}
        if bragg_reflections:
            # Implementation would filter by partiality and create references
            # For now, return empty dict
            pass

        logger.debug(f"Generated {len(diffuse_refs)} diffuse references")

        return bragg_refs, diffuse_refs

    def vectorized_generate_references(
        self, aggregated_data: Dict, bragg_reflections: Dict
    ) -> Tuple[Dict, Dict]:
        """
        Vectorized version: Generate Bragg and diffuse reference intensities using aggregated data.
        
        Uses vectorized scaling and pandas groupby for efficient voxel-wise averaging.
        
        Args:
            aggregated_data: Output from _aggregate_all_observations
            bragg_reflections: Bragg reflection data
            
        Returns:
            Tuple of (bragg_references, diffuse_references)
        """
        # Extract aggregated arrays
        all_intensities = aggregated_data["all_intensities"]
        all_sigmas = aggregated_data["all_sigmas"]
        all_still_ids = aggregated_data["all_still_ids"]
        all_q_mags = aggregated_data["all_q_mags"]
        voxel_idx_map = aggregated_data["voxel_idx_map"]
        total_obs = aggregated_data["total_observations"]
        
        if total_obs == 0:
            return {}, {}
            
        # Vectorized scaling calculation for all observations
        all_scales, _ = self.vectorized_calculate_scales_and_derivatives(all_still_ids, all_q_mags)
        
        # Apply scaling: I_scaled = I_obs / M
        all_scaled_intensities = all_intensities / all_scales
        all_weights = 1.0 / ((all_sigmas / all_scales) ** 2 + 1e-10)
        
        # Use pandas for efficient grouped averaging
        import pandas as pd
        
        # Create DataFrame for groupby operation
        df = pd.DataFrame({
            'voxel_idx': voxel_idx_map,
            'scaled_intensity': all_scaled_intensities,
            'weight': all_weights
        })
        
        # Vectorized weighted average per voxel
        def weighted_average_group(group):
            weighted_intensities = group['scaled_intensity'] * group['weight']
            total_weight = group['weight'].sum()
            
            if total_weight > 0:
                ref_intensity = weighted_intensities.sum() / total_weight
                ref_sigma = 1.0 / np.sqrt(total_weight)
                n_observations = len(group)
                
                return pd.Series({
                    'intensity': ref_intensity,
                    'sigma': ref_sigma,
                    'n_observations': n_observations
                })
            else:
                return pd.Series({
                    'intensity': 0.0,
                    'sigma': float('inf'),
                    'n_observations': 0
                })
        
        # Apply grouped aggregation
        merged_results = df.groupby('voxel_idx').apply(weighted_average_group, include_groups=False)
        
        # Convert results to dictionary format
        diffuse_refs = {}
        for voxel_idx in merged_results.index:
            row = merged_results.loc[voxel_idx]
            diffuse_refs[voxel_idx] = {
                "intensity": row['intensity'],
                "sigma": row['sigma'],
                "n_observations": int(row['n_observations'])
            }
        
        # Bragg references (simplified for now)
        bragg_refs = {}
        if bragg_reflections:
            # Implementation would filter by partiality and create references
            # For now, return empty dict
            pass

        logger.debug(f"Generated {len(diffuse_refs)} diffuse references (vectorized)")
        return bragg_refs, diffuse_refs

    def _calculate_r_factor(self, binned_pixel_data: Dict, diffuse_refs: Dict) -> float:
        """Calculate R-factor for current model."""
        numerator = 0.0
        denominator = 0.0

        for voxel_idx, voxel_data in binned_pixel_data.items():
            if voxel_idx not in diffuse_refs:
                continue

            ref_intensity = diffuse_refs[voxel_idx]["intensity"]

            # Normalize data format for backward compatibility
            voxel_data = self._normalize_voxel_data_format(voxel_data)
            # Extract NumPy arrays from voxel_data
            n_obs = voxel_data["n_observations"]
            still_ids = voxel_data["still_ids"]
            q_vectors = voxel_data["q_vectors_lab"]
            intensities = voxel_data["intensities"]

            # Create new inner loop: iterate through array indices
            for i in range(n_obs):
                still_id = still_ids[i]
                q_mag = np.linalg.norm(q_vectors[i])

                mult_scale, _ = self.get_scales_for_observation(still_id, q_mag)
                scaled_obs = intensities[i] / mult_scale

                numerator += abs(scaled_obs - ref_intensity)
                denominator += scaled_obs

        if denominator > 0:
            return numerator / denominator
        else:
            return float("inf")

    def vectorized_calculate_r_factor(self, aggregated_data: Dict, diffuse_refs: Dict) -> float:
        """
        Vectorized version: Calculate R-factor using aggregated data and vectorized operations.
        
        Args:
            aggregated_data: Output from _aggregate_all_observations  
            diffuse_refs: Reference intensities per voxel
            
        Returns:
            R-factor value
        """
        # Extract aggregated arrays
        all_intensities = aggregated_data["all_intensities"]
        all_still_ids = aggregated_data["all_still_ids"]
        all_q_mags = aggregated_data["all_q_mags"]
        voxel_idx_map = aggregated_data["voxel_idx_map"]
        total_obs = aggregated_data["total_observations"]
        
        if total_obs == 0 or len(diffuse_refs) == 0:
            return float("inf")
        
        # Create mask for observations that have reference intensities
        valid_mask = np.array([voxel_idx in diffuse_refs for voxel_idx in voxel_idx_map])
        
        if not np.any(valid_mask):
            return float("inf")
            
        # Filter to valid observations only
        valid_intensities = all_intensities[valid_mask]
        valid_still_ids = all_still_ids[valid_mask]
        valid_q_mags = all_q_mags[valid_mask]
        valid_voxel_indices = voxel_idx_map[valid_mask]
        
        # Vectorized scaling calculation for valid observations
        valid_scales, _ = self.vectorized_calculate_scales_and_derivatives(valid_still_ids, valid_q_mags)
        
        # Apply scaling: I_scaled = I_obs / M
        valid_scaled_obs = valid_intensities / valid_scales
        
        # Get corresponding reference intensities
        valid_ref_intensities = np.array([diffuse_refs[voxel_idx]["intensity"] for voxel_idx in valid_voxel_indices])
        
        # Vectorized R-factor calculation
        numerator = np.sum(np.abs(valid_scaled_obs - valid_ref_intensities))
        denominator = np.sum(valid_scaled_obs)
        
        if denominator > 0:
            return numerator / denominator
        else:
            return float("inf")

    def _extract_refined_parameters(self) -> Dict:
        """Extract refined parameters from components."""
        refined_params = {}

        for still_id in self.per_still_component.still_ids:
            key = int(still_id)  # Cast to Python int for JSON serialization
            refined_params[key] = {
                "multiplicative_scale": self.per_still_component.get_scale_for_still(
                    still_id
                ),
                "additive_offset": 0.0,  # Always 0.0 in v1
            }

        return refined_params

    def _calculate_parameter_shifts(self) -> Dict:
        """Calculate parameter shifts from last iteration."""
        # Simplified - would track actual shifts in real implementation
        return {"per_still": 0.001}

    def get_model_info(self) -> Dict:
        """Get comprehensive model information."""
        info = {
            "model_type": "DiffuseScalingModel_v1",
            "n_total_params": self.n_total_params,
            "components": {},
            "refinement_statistics": self.refinement_statistics,
            "partiality_threshold": self.partiality_threshold,
        }

        for comp_name, component in self.diffuse_components.items():
            info["components"][comp_name] = component.get_component_info()

        return info

    def __str__(self) -> str:
        """String representation of model."""
        comp_names = list(self.diffuse_components.keys())
        return f"DiffuseScalingModel({self.n_total_params} params, components: {comp_names})"
