"""
Bragg mask generation for per-still Bragg peak masking.

This module implements Module 1.S.3 from the plan, providing functions to generate
Bragg peak masks for individual stills and combine them with global pixel masks.
"""

import logging
from typing import Dict, Any, Tuple, Optional, List

from diffusepipe.adapters.dials_generate_mask_adapter import DIALSGenerateMaskAdapter
from diffusepipe.exceptions import BraggMaskError

# Imports needed for patching in tests
try:
    from dials.array_family import flex
    from dials.algorithms.shoebox import MaskCode
except ImportError:
    # These imports might fail in testing environments without DIALS
    flex = None
    MaskCode = None

logger = logging.getLogger(__name__)


class BraggMaskGenerator:
    """
    Generator for Bragg peak masks and combined total masks.

    This class implements the Bragg masking logic described in Module 1.S.3,
    providing both dials.generate_mask and shoebox-based mask generation options.
    """

    def __init__(self):
        """Initialize the Bragg mask generator."""
        self.dials_adapter = DIALSGenerateMaskAdapter()

    def generate_bragg_mask_from_spots(
        self,
        experiment: object,
        reflections: object,
        config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[object, ...]:
        """
        Generate Bragg mask using dials.generate_mask (Option A).

        Args:
            experiment: DIALS Experiment object containing geometry and crystal model
            reflections: DIALS reflection_table containing indexed spots
            config: Optional configuration parameters for mask generation

        Returns:
            Tuple of flex.bool arrays, one per detector panel

        Raises:
            BraggMaskError: When mask generation fails
        """
        try:
            logger.debug("Generating Bragg mask using dials.generate_mask")

            # Use the DIALS adapter to generate the mask
            bragg_mask, success, log_messages = self.dials_adapter.generate_bragg_mask(
                experiment=experiment,
                reflections=reflections,
                mask_generation_params=config,
            )

            if not success:
                raise BraggMaskError(f"DIALS mask generation failed: {log_messages}")

            logger.debug(f"Successfully generated Bragg mask: {log_messages}")
            return bragg_mask

        except Exception as e:
            if isinstance(e, BraggMaskError):
                raise
            else:
                raise BraggMaskError(f"Failed to generate Bragg mask from spots: {e}")

    def generate_bragg_mask_from_shoeboxes(
        self, reflections: object, detector: object
    ) -> Tuple[object, ...]:
        """
        Generate Bragg mask using shoebox data (Option B).

        Args:
            reflections: DIALS reflection_table containing shoeboxes
            detector: DIALS Detector object for panel information

        Returns:
            Tuple of flex.bool arrays, one per detector panel

        Raises:
            BraggMaskError: When mask generation fails
        """
        try:
            logger.debug("Generating Bragg mask using shoebox data")

            # Import DIALS components (delayed import)
            from dials.array_family import flex
            from dials.algorithms.shoebox import MaskCode

            # Initialize per-panel masks to False (no Bragg regions initially)
            panel_masks = []
            for panel in detector:
                panel_size = panel.get_image_size()
                height, width = (
                    panel_size[1],
                    panel_size[0],
                )  # (fast, slow) -> (slow, fast)
                panel_mask = flex.bool(flex.grid(height, width), False)
                panel_masks.append(panel_mask)

            # Check if reflections have shoeboxes
            if not reflections or len(reflections) == 0:
                logger.warning("No reflections provided for shoebox-based masking")
                return tuple(panel_masks)

            if not reflections.has_key("shoebox"):
                raise BraggMaskError(
                    "Reflection table missing required 'shoebox' column"
                )

            logger.debug(f"Processing {len(reflections)} reflections with shoeboxes")

            # Iterate through reflections and extract shoebox masks
            total_masked_pixels = 0
            for ref_idx, reflection in enumerate(reflections):
                try:
                    self._process_reflection_shoebox(
                        reflection, ref_idx, panel_masks, total_masked_pixels
                    )
                except Exception as e:
                    logger.warning(f"Failed to process reflection {ref_idx}: {e}")
                    continue

            logger.info(
                f"Generated Bragg mask from shoeboxes, "
                f"masked {total_masked_pixels} total pixels"
            )

            return tuple(panel_masks)

        except Exception as e:
            if isinstance(e, BraggMaskError):
                raise
            else:
                raise BraggMaskError(
                    f"Failed to generate Bragg mask from shoeboxes: {e}"
                )

    def get_total_mask_for_still(
        self, bragg_mask: Tuple[object, ...], global_pixel_mask: Tuple[object, ...]
    ) -> Tuple[object, ...]:
        """
        Combine Bragg mask with global pixel mask to create total mask.

        Args:
            bragg_mask: Tuple of Bragg masks (one per panel)
            global_pixel_mask: Tuple of global pixel masks (one per panel)

        Returns:
            Tuple of combined total masks (one per panel)

        Behavior:
            Mask_total_2D_i(px,py) = Mask_pixel(px,py) AND (NOT BraggMask_2D_raw_i(px,py))

        Raises:
            BraggMaskError: When mask combination fails
        """
        try:
            logger.debug("Combining Bragg mask with global pixel mask")

            if len(bragg_mask) != len(global_pixel_mask):
                raise BraggMaskError(
                    f"Bragg mask and global pixel mask have different panel counts: "
                    f"{len(bragg_mask)} vs {len(global_pixel_mask)}"
                )

            total_masks = []
            for panel_idx, (bragg_panel, pixel_panel) in enumerate(
                zip(bragg_mask, global_pixel_mask)
            ):
                # Apply the combination logic: pixel_mask AND (NOT bragg_mask)
                inverted_bragg = ~bragg_panel
                total_panel_mask = pixel_panel & inverted_bragg
                total_masks.append(total_panel_mask)

                # Log statistics
                pixel_good = pixel_panel.count(True)
                bragg_masked = bragg_panel.count(True)
                total_good = total_panel_mask.count(True)
                total_pixels = len(total_panel_mask)

                logger.debug(
                    f"Panel {panel_idx}: Pixel={pixel_good}/{total_pixels}, "
                    f"Bragg_masked={bragg_masked}/{total_pixels}, "
                    f"Total_good={total_good}/{total_pixels}"
                )

            logger.info(f"Generated total masks for {len(total_masks)} panels")
            return tuple(total_masks)

        except Exception as e:
            if isinstance(e, BraggMaskError):
                raise
            else:
                raise BraggMaskError(f"Failed to combine masks: {e}")

    def _process_reflection_shoebox(
        self,
        reflection: object,
        ref_idx: int,
        panel_masks: List[object],
        total_masked_pixels: int,
    ) -> int:
        """
        Process a single reflection's shoebox and update panel masks.

        Args:
            reflection: Individual reflection from the reflection table
            ref_idx: Index of the reflection for logging
            panel_masks: List of panel masks to update
            total_masked_pixels: Running count of masked pixels

        Returns:
            Updated count of total masked pixels

        Raises:
            Exception: When shoebox processing fails
        """
        from dials.algorithms.shoebox import MaskCode

        shoebox = reflection["shoebox"]
        panel_id = reflection.get("panel", 0)  # Default to panel 0 if not specified

        if panel_id >= len(panel_masks):
            raise Exception(f"Panel ID {panel_id} exceeds available panels")

        # Get shoebox mask and data
        shoebox_mask = shoebox.mask
        shoebox_bbox = shoebox.bbox

        # Extract 3D coordinates (panels, slow, fast)
        z1, y1, x1, z2, y2, x2 = shoebox_bbox

        # Iterate through shoebox voxels
        masked_pixels_this_ref = 0
        for z in range(z1, z2):
            for y in range(y1, y2):
                for x in range(x1, x2):
                    # Calculate index in the flattened shoebox mask
                    mask_idx = (
                        (z - z1) * (y2 - y1) * (x2 - x1)
                        + (y - y1) * (x2 - x1)
                        + (x - x1)
                    )

                    if mask_idx >= len(shoebox_mask):
                        continue

                    # Check if this voxel is marked as foreground or strong
                    mask_code = shoebox_mask[mask_idx]
                    if mask_code & MaskCode.Foreground or mask_code & MaskCode.Strong:

                        # Project to 2D panel coordinates
                        # For stills, z should correspond to panel_id
                        if (
                            z == panel_id
                            and 0 <= y < len(panel_masks[panel_id])
                            and 0 <= x < len(panel_masks[panel_id][0])
                        ):
                            # Set this pixel as masked (True in Bragg mask means masked)
                            panel_masks[panel_id][y, x] = True
                            masked_pixels_this_ref += 1

        total_masked_pixels += masked_pixels_this_ref
        logger.debug(f"Reflection {ref_idx}: masked {masked_pixels_this_ref} pixels")

        return total_masked_pixels


# Convenience functions for mask configuration


def create_default_bragg_mask_config() -> Dict[str, Any]:
    """
    Create default configuration for dials.generate_mask.

    Returns:
        Dictionary with default mask generation parameters
    """
    return {
        "border": 2,  # Border around each reflection
        "algorithm": "simple",  # Simple mask algorithm
    }


def create_expanded_bragg_mask_config(border: int = 3) -> Dict[str, Any]:
    """
    Create expanded configuration for dials.generate_mask with larger borders.

    Args:
        border: Size of border around each reflection

    Returns:
        Dictionary with expanded mask generation parameters
    """
    return {"border": border, "algorithm": "simple"}


def validate_mask_compatibility(
    bragg_mask: Tuple[object, ...], pixel_mask: Tuple[object, ...]
) -> bool:
    """
    Validate that Bragg and pixel masks are compatible for combination.

    Args:
        bragg_mask: Tuple of Bragg masks
        pixel_mask: Tuple of pixel masks

    Returns:
        True if masks are compatible, False otherwise
    """
    try:
        # Check panel count
        if len(bragg_mask) != len(pixel_mask):
            logger.error(
                f"Panel count mismatch: {len(bragg_mask)} vs {len(pixel_mask)}"
            )
            return False

        # Check dimensions for each panel
        for panel_idx, (bragg_panel, pixel_panel) in enumerate(
            zip(bragg_mask, pixel_mask)
        ):
            if len(bragg_panel) != len(pixel_panel):
                logger.error(
                    f"Panel {panel_idx} size mismatch: "
                    f"{len(bragg_panel)} vs {len(pixel_panel)}"
                )
                return False

        return True

    except Exception as e:
        logger.error(f"Error validating mask compatibility: {e}")
        return False
