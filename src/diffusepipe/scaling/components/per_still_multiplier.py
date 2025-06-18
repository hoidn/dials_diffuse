"""
PerStillMultiplierComponent for diffuse scaling model.

Implements simple multiplicative scale factor with one parameter per still.
"""

import logging
from typing import Dict, List, Optional

from dials.algorithms.scaling.model.components.scale_components import (
    ScaleComponentBase,
)
from dials.array_family import flex

logger = logging.getLogger(__name__)


class PerStillMultiplierComponent(ScaleComponentBase):
    """
    Simple multiplicative scale factor component with one parameter per still.

    This is the core component of the v1 diffuse scaling model, providing
    exactly one refineable parameter (multiplicative scale) per still image.
    """

    def __init__(
        self,
        still_ids: List[int],
        initial_values: Optional[Dict[int, float]] = None,
    ):
        """
        Initialize per-still multiplier component.

        Args:
            still_ids: List of unique still identifiers
            initial_values: Optional dict mapping still_id -> initial scale
        """
        self.still_ids = sorted(still_ids)  # Ensure consistent ordering
        self.n_stills = len(self.still_ids)

        # Create mapping from still_id to parameter index
        self.still_id_to_param_idx = {
            still_id: i for i, still_id in enumerate(self.still_ids)
        }

        # Set initial parameter values
        if initial_values is None:
            initial_values = {still_id: 1.0 for still_id in self.still_ids}

        initial_params = flex.double(
            [initial_values.get(still_id, 1.0) for still_id in self.still_ids]
        )

        # Call base class constructor with initial parameters
        super().__init__(initial_params)

        # Store parameters directly
        self._parameters = initial_params

        logger.info(
            f"PerStillMultiplierComponent initialized for {self.n_stills} stills"
        )

    @property
    def n_params(self) -> int:
        """Number of parameters in this component."""
        return self.n_stills

    @property
    def parameters(self) -> flex.double:
        """Current parameter values from the parameter manager."""
        # Use the parent class parameters directly
        return self._parameters

    def calculate_scales_and_derivatives(self, reflection_table, block_id=None):
        """
        Calculate scale factors and derivatives for reflections.

        Args:
            reflection_table: DIALS reflection table or diffuse data structure
            block_id: Unused for this component

        Returns:
            tuple: (scales, derivatives) as flex arrays
        """
        # Handle different input types
        if isinstance(reflection_table, dict) and "still_ids" in reflection_table:
            # Diffuse data dictionary format
            still_ids = reflection_table["still_ids"]
            n_reflections = len(still_ids)
        elif (
            hasattr(reflection_table, "get")
            and reflection_table.get("still_id") is not None
        ):
            # DIALS flex.reflection_table
            still_ids = reflection_table.get("still_id")
            n_reflections = len(still_ids)
        else:
            raise ValueError("Invalid reflection_table format")

        if n_reflections == 0:
            return flex.double(), flex.double()

        # Get current parameter values
        current_params = self.parameters

        # Calculate scales for each reflection
        scales = flex.double(n_reflections)
        derivatives = flex.double(n_reflections)

        for i in range(n_reflections):
            still_id = still_ids[i]

            if still_id in self.still_id_to_param_idx:
                param_idx = self.still_id_to_param_idx[still_id]
                scale = current_params[param_idx]
                derivative = 1.0  # d(scale)/d(parameter) = 1.0
            else:
                # Unknown still - use unity scale
                scale = 1.0
                derivative = 0.0
                logger.warning(f"Unknown still_id {still_id}, using unity scale")

            scales[i] = scale
            derivatives[i] = derivative

        return scales, derivatives

    def get_scale_for_still(self, still_id: int) -> float:
        """
        Get current scale factor for a specific still.

        Args:
            still_id: Still identifier

        Returns:
            Current multiplicative scale factor
        """
        if still_id not in self.still_id_to_param_idx:
            logger.warning(f"Unknown still_id {still_id}")
            return 1.0

        param_idx = self.still_id_to_param_idx[still_id]
        current_params = self.parameters
        return current_params[param_idx]

    def set_scale_for_still(self, still_id: int, scale: float):
        """
        Set scale factor for a specific still.

        Args:
            still_id: Still identifier
            scale: New scale value (should be positive)
        """
        if still_id not in self.still_id_to_param_idx:
            raise ValueError(f"Unknown still_id {still_id}")

        if scale <= 0:
            raise ValueError("Scale factor must be positive")

        param_idx = self.still_id_to_param_idx[still_id]

        # Update parameter directly in the component
        self._parameters[param_idx] = scale

    def get_component_info(self) -> Dict:
        """
        Get information about this component.

        Returns:
            Dictionary with component details
        """
        current_params = self.parameters

        return {
            "component_type": "PerStillMultiplier",
            "n_parameters": self.n_params,
            "n_stills": self.n_stills,
            "still_ids": self.still_ids,
            "current_scales": {
                self.still_ids[i]: float(current_params[i])
                for i in range(self.n_stills)
            },
            "scale_statistics": {
                "mean": float(flex.mean(current_params)),
                "std": float(flex.mean_and_variance(current_params).gsl_stats_wsd()),
                "min": float(flex.min(current_params)),
                "max": float(flex.max(current_params)),
            },
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
        return (
            f"PerStillMultiplierComponent({self.n_stills} stills, "
            f"scales: {stats['mean']:.3f}±{stats['std']:.3f})"
        )

    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"PerStillMultiplierComponent(n_stills={self.n_stills}, "
            f"still_ids={self.still_ids})"
        )
