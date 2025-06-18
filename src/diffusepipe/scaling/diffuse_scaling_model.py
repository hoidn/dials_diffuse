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
            f"Starting Levenberg-Marquardt parameter refinement: {max_iterations} max iterations"
        )

        prev_r_factor = float("inf")
        parameter_shifts_history = []

        for iteration in range(max_iterations):
            logger.debug(f"Refinement iteration {iteration + 1}")

            # 1. Generate references with current parameters
            bragg_refs, diffuse_refs = self.generate_references(
                binned_pixel_data, bragg_reflections
            )

            # 2. Calculate current R-factor
            r_factor = self._calculate_r_factor(binned_pixel_data, diffuse_refs)
            logger.debug(f"Iteration {iteration + 1}: R-factor = {r_factor:.6f}")

            # 3. Compute residuals and Jacobian matrix
            residuals = flex.double()
            jacobian_rows = []

            for voxel_idx, voxel_data in binned_pixel_data.items():
                if voxel_idx not in diffuse_refs:
                    continue

                ref_intensity = diffuse_refs[voxel_idx]["intensity"]

                for obs in voxel_data["observations"]:
                    still_id = obs["still_id"]
                    q_mag = np.linalg.norm(obs["q_vector_lab"])
                    obs_intensity = obs["intensity"]

                    # Get current scale and derivatives
                    total_scale, derivatives = self.calculate_scales_and_derivatives(
                        still_id, q_mag
                    )

                    # Calculate residual: (I_obs / M) - I_ref
                    scaled_obs = (
                        obs_intensity / total_scale
                        if total_scale > 0
                        else obs_intensity
                    )
                    residual = scaled_obs - ref_intensity
                    residuals.append(residual)

                    # Calculate Jacobian row: d(residual)/d(param) = (-I_obs / M²) * (dM/dparam)
                    jacobian_row = flex.double()
                    for j in range(self.n_total_params):
                        if total_scale > 0:
                            grad = (
                                -obs_intensity / (total_scale * total_scale)
                            ) * derivatives[j]
                        else:
                            grad = 0.0
                        jacobian_row.append(grad)

                    jacobian_rows.append(jacobian_row)

            if len(residuals) == 0:
                logger.warning("No observations for refinement")
                break

            # Convert jacobian to matrix format
            n_residuals = len(residuals)
            n_params = self.n_total_params
            jacobian = flex.double(flex.grid(n_residuals, n_params))

            for i, row in enumerate(jacobian_rows):
                for j in range(n_params):
                    jacobian[i, j] = row[j]

            # 4. Solve using least squares (simplified approach)
            try:
                # Convert to numpy for easier matrix operations
                residuals_np = np.array(residuals)
                jacobian_np = np.zeros((n_residuals, n_params))

                for i in range(n_residuals):
                    for j in range(n_params):
                        jacobian_np[i, j] = jacobian[i, j]

                # Solve normal equations: J^T J delta = -J^T r
                JtJ = jacobian_np.T @ jacobian_np
                Jtr = jacobian_np.T @ residuals_np

                # Add small damping for numerical stability
                damping = 1e-6
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
            observations = voxel_data["observations"]
            if len(observations) == 0:
                continue

            # Apply current scaling to observations
            scaled_intensities = []
            weights = []

            for obs in observations:
                still_id = obs["still_id"]
                q_mag = np.linalg.norm(obs["q_vector_lab"])

                mult_scale, add_offset = self.get_scales_for_observation(
                    still_id, q_mag
                )

                # Apply scaling: I_scaled = (I_obs - C) / M
                # For v1: C = 0, so I_scaled = I_obs / M
                scaled_intensity = obs["intensity"] / mult_scale
                weight = 1.0 / (obs["sigma"] / mult_scale) ** 2

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
                        "n_observations": len(observations),
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

            ref_intensity = diffuse_refs[voxel_idx]["intensity"]

            for obs in voxel_data["observations"]:
                still_id = obs["still_id"]
                q_mag = np.linalg.norm(obs["q_vector_lab"])

                mult_scale, _ = self.get_scales_for_observation(still_id, q_mag)
                scaled_obs = obs["intensity"] / mult_scale

                numerator += abs(scaled_obs - ref_intensity)
                denominator += scaled_obs

        if denominator > 0:
            return numerator / denominator
        else:
            return float("inf")

    def _extract_refined_parameters(self) -> Dict:
        """Extract refined parameters from components."""
        refined_params = {}

        for still_id in self.per_still_component.still_ids:
            refined_params[still_id] = {
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
