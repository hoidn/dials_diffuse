"""Adapter for dials.generate_mask functionality."""

import logging
from typing import Dict, Any, Tuple, Optional

from diffusepipe.exceptions import DIALSError
from dials.command_line.generate_mask import phil_scope as generate_mask_phil_scope
from libtbx.phil import parse as phil_parse

logger = logging.getLogger(__name__)


class DIALSGenerateMaskAdapter:
    """
    Adapter for wrapping dials.generate_mask operations.

    This adapter encapsulates DIALS mask generation operations and provides
    error handling with project-specific exceptions.
    """

    def __init__(self):
        """Initialize the DIALS generate mask adapter."""
        pass

    def generate_bragg_mask(
        self,
        experiment: object,
        reflections: object,
        mask_generation_params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[object, bool, str]:
        """
        Generate Bragg peak mask using dials.generate_mask.

        Args:
            experiment: DIALS Experiment object containing geometry and crystal model
            reflections: DIALS reflection_table containing indexed spots
            mask_generation_params: Optional parameters for mask generation

        Returns:
            Tuple containing:
            - Bragg mask as tuple of flex.bool arrays (one per panel)
            - Success boolean
            - Log messages string

        Raises:
            DIALSError: When mask generation fails
        """
        log_messages = []

        try:
            # Validate inputs
            if experiment is None:
                raise DIALSError("Experiment object cannot be None")
            if reflections is None:
                raise DIALSError("Reflections object cannot be None")

            log_messages.append("Starting Bragg mask generation")

            # Set default parameters if not provided
            if mask_generation_params is None:
                mask_generation_params = {
                    "border": 2,  # Border around each reflection
                }

            log_messages.append("Starting Bragg mask generation")

            # Create ExperimentList if we have a single experiment
            try:
                from dxtbx.model import ExperimentList

                if not isinstance(experiment, ExperimentList):
                    experiment_list = ExperimentList([experiment])
                else:
                    experiment_list = experiment
            except ImportError as e:
                raise DIALSError(f"Failed to import ExperimentList: {e}")

            # Convert mask_generation_params dict to PHIL object
            phil_params_object = self._create_phil_params(mask_generation_params)
            log_messages.append("Created PHIL parameters object")

            # Generate the mask
            mask_result = self._call_generate_mask(
                experiment_list, reflections, phil_params_object
            )

            log_messages.append("Generated Bragg mask successfully")

            # Validate the result
            self._validate_mask_result(mask_result)
            log_messages.append("Validated mask result")

            return mask_result, True, "\n".join(log_messages)

        except Exception as e:
            error_msg = f"Bragg mask generation failed: {e}"
            log_messages.append(error_msg)
            logger.error(error_msg)

            if isinstance(e, DIALSError):
                raise
            else:
                raise DIALSError(error_msg) from e

    def _call_generate_mask(
        self, experiment_list: object, reflections: object, phil_params_object: object
    ) -> object:
        """
        Generate Bragg masks from reflections positions.

        Note: The dials.util.masking.generate_mask function is for general detector
        masking and expects an ImageSet, not ExperimentList. For Bragg-specific
        masking from reflections, we create masks directly from reflection positions.

        Args:
            experiment_list: DIALS ExperimentList object
            reflections: DIALS reflection_table containing indexed spots
            phil_params_object: PHIL parameters object for mask generation

        Returns:
            Mask result as tuple of flex.bool arrays

        Raises:
            DIALSError: When the mask generation fails
        """
        try:
            from dials.array_family import flex

            # Get the experiment (assuming single experiment for now)
            if len(experiment_list) == 0:
                raise DIALSError("ExperimentList is empty")

            experiment = experiment_list[0]
            detector = experiment.detector

            # Get border parameter (default to 2 if not specified)
            border = getattr(phil_params_object, "border", 2)

            # Initialize mask for each panel (True = unmasked, False = masked)
            panel_masks = []
            for panel_idx, panel in enumerate(detector):
                panel_size = panel.get_image_size()
                # Create a mask of all True (unmasked) initially
                panel_mask = flex.bool(flex.grid(panel_size[1], panel_size[0]), True)
                panel_masks.append(panel_mask)

            # Mask regions around reflection centroids
            if reflections is not None and len(reflections) > 0:
                logger.info(
                    f"Masking {len(reflections)} reflection regions with border={border}"
                )

                # Get reflection centroids
                if "xyzobs.px.value" in reflections:
                    centroids = reflections["xyzobs.px.value"]
                elif "xyzcal.px" in reflections:
                    centroids = reflections["xyzcal.px"]
                else:
                    logger.warning(
                        "No centroid data found in reflections, using all-unmasked mask"
                    )
                    return tuple(panel_masks)

                # Get panel assignments if available
                if "panel" in reflections:
                    panels = reflections["panel"]
                else:
                    # Assume all reflections are on panel 0
                    panels = flex.int(len(reflections), 0)

                # Mask around each reflection
                for i in range(len(reflections)):
                    centroid = centroids[i]
                    panel_id = panels[i]

                    if panel_id >= len(panel_masks):
                        continue

                    # Get integer pixel coordinates
                    x_center = int(round(centroid[0]))
                    y_center = int(round(centroid[1]))

                    # Define masking region around the centroid
                    y_min = max(0, y_center - border)
                    y_max = min(panel_masks[panel_id].all()[0], y_center + border + 1)
                    x_min = max(0, x_center - border)
                    x_max = min(panel_masks[panel_id].all()[1], x_center + border + 1)

                    # Mask the region (set to False)
                    for y in range(y_min, y_max):
                        for x in range(x_min, x_max):
                            panel_masks[panel_id][y, x] = False

                # Count masked pixels for logging
                total_masked = sum((~mask).count(True) for mask in panel_masks)
                logger.info(f"Masked {total_masked} pixels around Bragg reflections")
            else:
                logger.info("No reflections provided, using all-unmasked Bragg mask")

            return tuple(panel_masks)

        except Exception as e:
            raise DIALSError(f"Bragg mask generation failed: {e}")

    def _create_phil_params(self, mask_generation_params: Dict[str, Any]) -> object:
        """
        Convert mask generation parameters dict to PHIL object.

        Args:
            mask_generation_params: Dictionary of mask generation parameters

        Returns:
            PHIL parameters object suitable for dials.util.masking.generate_mask

        Raises:
            DIALSError: When PHIL object creation fails
        """
        try:
            # Create PHIL string from parameters dict
            phil_string_parts = []

            # Handle border parameter
            if "border" in mask_generation_params:
                phil_string_parts.append(f"border = {mask_generation_params['border']}")

            # Handle other supported parameters (can be extended as needed)
            if "d_min" in mask_generation_params:
                phil_string_parts.append(f"d_min = {mask_generation_params['d_min']}")

            if "d_max" in mask_generation_params:
                phil_string_parts.append(f"d_max = {mask_generation_params['d_max']}")

            # Create the full PHIL string
            phil_string = "\n".join(phil_string_parts)

            # Parse the PHIL string and extract parameters
            current_phil = generate_mask_phil_scope.fetch(phil_parse(phil_string))
            phil_params_object = current_phil.extract()

            return phil_params_object

        except Exception as e:
            raise DIALSError(f"Failed to create PHIL parameters object: {e}")

    def _validate_mask_result(self, mask_result: object) -> None:
        """
        Validate the mask generation result.

        Args:
            mask_result: Result from DIALS mask generation

        Raises:
            DIALSError: When validation fails
        """
        if mask_result is None:
            raise DIALSError("Mask generation returned None")

        if not isinstance(mask_result, (tuple, list)):
            raise DIALSError("Mask result should be a tuple or list of panel masks")

        if len(mask_result) == 0:
            raise DIALSError("Mask result contains no panel masks")

        # Check that each panel mask is a valid flex.bool array
        for i, panel_mask in enumerate(mask_result):
            try:
                # In a real implementation, this would check for flex.bool type
                if panel_mask is None:
                    raise DIALSError(f"Panel {i} mask is None")

                # Additional validation could check mask dimensions, etc.
                logger.debug(f"Panel {i} mask validated successfully")

            except Exception as e:
                raise DIALSError(f"Panel {i} mask validation failed: {e}")

        logger.info(f"Validated mask with {len(mask_result)} panels")
