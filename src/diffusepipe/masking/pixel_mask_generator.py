"""
Pixel mask generation for static and dynamic bad pixel identification.

This module implements Module 1.S.2 from the plan, providing functions to generate
global pixel masks based on detector properties and dynamic features observed
across representative still images.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass

import numpy as np

from diffusepipe.exceptions import MaskGenerationError

logger = logging.getLogger(__name__)


@dataclass
class Circle:
    """Circular region definition."""
    center_x: float
    center_y: float
    radius: float


@dataclass
class Rectangle:
    """Rectangular region definition."""
    min_x: float
    max_x: float
    min_y: float
    max_y: float


@dataclass
class StaticMaskParams:
    """Parameters for static mask generation."""
    beamstop: Optional[Union[Circle, Rectangle]] = None
    untrusted_rects: Optional[List[Rectangle]] = None
    untrusted_panels: Optional[List[int]] = None


@dataclass
class DynamicMaskParams:
    """Parameters for dynamic mask generation."""
    hot_pixel_thresh: Optional[float] = None
    negative_pixel_tolerance: float = 0.0
    max_fraction_bad_pixels: float = 0.1


class PixelMaskGenerator:
    """
    Generator for static and dynamic pixel masks.
    
    This class implements the pixel masking logic described in Module 1.S.2,
    creating global masks based on detector properties and dynamic features.
    """
    
    def __init__(self):
        """Initialize the pixel mask generator."""
        pass
    
    def generate_combined_pixel_mask(
        self,
        detector: object,
        static_params: StaticMaskParams,
        representative_images: List[object],
        dynamic_params: DynamicMaskParams
    ) -> Tuple[object, ...]:
        """
        Generate combined pixel mask from static and dynamic components.
        
        Args:
            detector: DIALS Detector object
            static_params: Parameters for static mask generation
            representative_images: List of ImageSet objects for dynamic analysis
            dynamic_params: Parameters for dynamic mask generation
            
        Returns:
            Tuple of flex.bool arrays, one per detector panel
            
        Raises:
            MaskGenerationError: When mask generation fails
        """
        try:
            logger.info("Starting combined pixel mask generation")
            
            # Generate static mask
            static_mask = self.generate_static_mask(detector, static_params)
            logger.info(f"Generated static mask for {len(static_mask)} panels")
            
            # Generate dynamic mask
            dynamic_mask = self.generate_dynamic_mask(
                detector, representative_images, dynamic_params
            )
            logger.info(f"Generated dynamic mask for {len(dynamic_mask)} panels")
            
            # Combine masks (logical AND)
            combined_mask = self._combine_masks(static_mask, dynamic_mask)
            logger.info("Combined static and dynamic masks")
            
            return combined_mask
            
        except Exception as e:
            raise MaskGenerationError(f"Failed to generate combined pixel mask: {e}")
    
    def generate_static_mask(
        self,
        detector: object,
        static_params: StaticMaskParams
    ) -> Tuple[object, ...]:
        """
        Generate static pixel mask based on detector properties.
        
        Args:
            detector: DIALS Detector object
            static_params: Parameters for static masking
            
        Returns:
            Tuple of flex.bool arrays, one per detector panel
            
        Raises:
            MaskGenerationError: When static mask generation fails
        """
        try:
            from dials.array_family import flex
            
            panel_masks = []
            
            for panel_idx, panel in enumerate(detector):
                logger.debug(f"Processing panel {panel_idx}")
                
                # Get panel dimensions
                panel_size = panel.get_image_size()
                height, width = panel_size[1], panel_size[0]  # Note: DIALS uses (fast, slow)
                
                # Initialize panel mask to True (good pixels)
                panel_mask = flex.bool(flex.grid(height, width), True)
                
                # Apply trusted range mask
                panel_mask = self._apply_trusted_range_mask(panel, panel_mask)
                
                # Apply beamstop mask
                if static_params.beamstop:
                    panel_mask = self._apply_beamstop_mask(
                        panel, panel_mask, static_params.beamstop
                    )
                
                # Apply untrusted rectangle masks
                if static_params.untrusted_rects:
                    panel_mask = self._apply_untrusted_rects_mask(
                        panel, panel_mask, static_params.untrusted_rects
                    )
                
                # Check if this panel should be entirely untrusted
                if (static_params.untrusted_panels and 
                    panel_idx in static_params.untrusted_panels):
                    panel_mask = flex.bool(flex.grid(height, width), False)
                    logger.info(f"Panel {panel_idx} marked as entirely untrusted")
                
                panel_masks.append(panel_mask)
            
            return tuple(panel_masks)
            
        except Exception as e:
            raise MaskGenerationError(f"Failed to generate static mask: {e}")
    
    def generate_dynamic_mask(
        self,
        detector: object,
        representative_images: List[object],
        dynamic_params: DynamicMaskParams
    ) -> Tuple[object, ...]:
        """
        Generate dynamic pixel mask based on analysis of representative images.
        
        Args:
            detector: DIALS Detector object
            representative_images: List of ImageSet objects to analyze
            dynamic_params: Parameters for dynamic masking
            
        Returns:
            Tuple of flex.bool arrays, one per detector panel
            
        Raises:
            MaskGenerationError: When dynamic mask generation fails
        """
        try:
            from dials.array_family import flex
            
            if not representative_images:
                logger.warning("No representative images provided for dynamic masking")
                # Return all-True masks
                panel_masks = []
                for panel in detector:
                    panel_size = panel.get_image_size()
                    height, width = panel_size[1], panel_size[0]
                    panel_mask = flex.bool(flex.grid(height, width), True)
                    panel_masks.append(panel_mask)
                return tuple(panel_masks)
            
            logger.info(f"Analyzing {len(representative_images)} representative images")
            
            # Initialize accumulators for each panel
            panel_stats = []
            for panel in detector:
                panel_size = panel.get_image_size()
                height, width = panel_size[1], panel_size[0]
                
                stats = {
                    'sum': np.zeros((height, width), dtype=np.float64),
                    'count': np.zeros((height, width), dtype=np.int32),
                    'min_val': np.full((height, width), np.inf, dtype=np.float64),
                    'max_val': np.full((height, width), -np.inf, dtype=np.float64)
                }
                panel_stats.append(stats)
            
            # Accumulate statistics across all representative images
            for img_idx, image_set in enumerate(representative_images):
                logger.debug(f"Processing representative image {img_idx + 1}")
                self._accumulate_image_stats(image_set, panel_stats)
            
            # Generate dynamic masks based on accumulated statistics
            panel_masks = []
            for panel_idx, stats in enumerate(panel_stats):
                panel_mask = self._generate_panel_dynamic_mask(
                    stats, dynamic_params, panel_idx
                )
                panel_masks.append(panel_mask)
            
            return tuple(panel_masks)
            
        except Exception as e:
            raise MaskGenerationError(f"Failed to generate dynamic mask: {e}")
    
    def _apply_trusted_range_mask(
        self,
        panel: object,
        panel_mask: object
    ) -> object:
        """
        Apply trusted range mask using panel.get_trusted_range().
        
        Args:
            panel: DIALS Panel object
            panel_mask: Current panel mask (flex.bool)
            
        Returns:
            Updated panel mask
        """
        try:
            # Get trusted range for this panel
            trusted_range = panel.get_trusted_range()
            logger.debug(f"Panel trusted range: {trusted_range}")
            
            # For static masking, we use a representative raw data array
            # In practice, this would be applied per-image during processing
            # For now, just log the trusted range
            logger.debug("Trusted range mask will be applied per-image during processing")
            
            return panel_mask
            
        except Exception as e:
            logger.warning(f"Failed to apply trusted range mask: {e}")
            return panel_mask
    
    def _apply_beamstop_mask(
        self,
        panel: object,
        panel_mask: object,
        beamstop: Union[Circle, Rectangle]
    ) -> object:
        """
        Apply beamstop mask to panel.
        
        Args:
            panel: DIALS Panel object
            panel_mask: Current panel mask (flex.bool)
            beamstop: Beamstop geometry
            
        Returns:
            Updated panel mask with beamstop region masked
        """
        try:
            from dials.array_family import flex
            
            # Get panel dimensions
            panel_size = panel.get_image_size()
            height, width = panel_size[1], panel_size[0]
            
            # Create coordinate grids
            y_coords, x_coords = np.mgrid[0:height, 0:width]
            
            if isinstance(beamstop, Circle):
                # Circular beamstop
                distances_sq = ((x_coords - beamstop.center_x) ** 2 + 
                               (y_coords - beamstop.center_y) ** 2)
                beamstop_region = distances_sq <= (beamstop.radius ** 2)
                
            elif isinstance(beamstop, Rectangle):
                # Rectangular beamstop
                beamstop_region = ((x_coords >= beamstop.min_x) & 
                                 (x_coords <= beamstop.max_x) &
                                 (y_coords >= beamstop.min_y) & 
                                 (y_coords <= beamstop.max_y))
            else:
                logger.warning(f"Unknown beamstop type: {type(beamstop)}")
                return panel_mask
            
            # Convert to flex.bool and apply
            beamstop_mask = flex.bool(beamstop_region.flatten())
            beamstop_mask.reshape(flex.grid(height, width))
            
            # Mask beamstop region (set to False)
            panel_mask = panel_mask & (~beamstop_mask)
            
            logger.debug(f"Applied beamstop mask, masked {beamstop_region.sum()} pixels")
            
            return panel_mask
            
        except Exception as e:
            logger.warning(f"Failed to apply beamstop mask: {e}")
            return panel_mask
    
    def _apply_untrusted_rects_mask(
        self,
        panel: object,
        panel_mask: object,
        untrusted_rects: List[Rectangle]
    ) -> object:
        """
        Apply untrusted rectangle masks to panel.
        
        Args:
            panel: DIALS Panel object
            panel_mask: Current panel mask (flex.bool)
            untrusted_rects: List of rectangular regions to mask
            
        Returns:
            Updated panel mask with untrusted regions masked
        """
        try:
            from dials.array_family import flex
            
            # Get panel dimensions
            panel_size = panel.get_image_size()
            height, width = panel_size[1], panel_size[0]
            
            # Create coordinate grids
            y_coords, x_coords = np.mgrid[0:height, 0:width]
            
            total_masked = 0
            for rect in untrusted_rects:
                # Create mask for this rectangle
                rect_region = ((x_coords >= rect.min_x) & 
                              (x_coords <= rect.max_x) &
                              (y_coords >= rect.min_y) & 
                              (y_coords <= rect.max_y))
                
                # Convert to flex.bool and apply
                rect_mask = flex.bool(rect_region.flatten())
                rect_mask.reshape(flex.grid(height, width))
                
                # Mask this region (set to False)
                panel_mask = panel_mask & (~rect_mask)
                
                total_masked += rect_region.sum()
            
            logger.debug(f"Applied {len(untrusted_rects)} untrusted rectangles, "
                        f"masked {total_masked} pixels")
            
            return panel_mask
            
        except Exception as e:
            logger.warning(f"Failed to apply untrusted rectangles mask: {e}")
            return panel_mask
    
    def _accumulate_image_stats(
        self,
        image_set: object,
        panel_stats: List[Dict[str, np.ndarray]]
    ) -> None:
        """
        Accumulate statistics from a single image across all panels.
        
        Args:
            image_set: DIALS ImageSet object
            panel_stats: List of statistics dictionaries, one per panel
        """
        try:
            # Get raw data for this image
            raw_data = image_set.get_raw_data(0)  # First (and only) image in the set
            
            for panel_idx, panel_data in enumerate(raw_data):
                if panel_idx >= len(panel_stats):
                    logger.warning(f"Panel {panel_idx} exceeds expected number of panels")
                    continue
                
                # Convert flex array to numpy for vectorized operations
                panel_array = np.array(panel_data).astype(np.float64)
                
                stats = panel_stats[panel_idx]
                
                # Update statistics
                stats['sum'] += panel_array
                stats['count'] += 1
                stats['min_val'] = np.minimum(stats['min_val'], panel_array)
                stats['max_val'] = np.maximum(stats['max_val'], panel_array)
                
        except Exception as e:
            logger.warning(f"Failed to accumulate statistics from image: {e}")
    
    def _generate_panel_dynamic_mask(
        self,
        stats: Dict[str, np.ndarray],
        dynamic_params: DynamicMaskParams,
        panel_idx: int
    ) -> object:
        """
        Generate dynamic mask for a single panel based on accumulated statistics.
        
        Args:
            stats: Accumulated statistics for this panel
            dynamic_params: Parameters for dynamic masking
            panel_idx: Panel index for logging
            
        Returns:
            Dynamic mask for this panel (flex.bool)
        """
        try:
            from dials.array_family import flex
            
            height, width = stats['sum'].shape
            
            # Initialize dynamic mask to True (good pixels)
            dynamic_mask = np.ones((height, width), dtype=bool)
            
            # Check for sufficient data
            valid_pixels = stats['count'] > 0
            if not np.any(valid_pixels):
                logger.warning(f"Panel {panel_idx}: No valid pixel data for dynamic masking")
                return flex.bool(dynamic_mask.flatten())
            
            # Calculate mean intensity where we have data
            mean_intensity = np.zeros_like(stats['sum'])
            mean_intensity[valid_pixels] = (stats['sum'][valid_pixels] / 
                                          stats['count'][valid_pixels])
            
            # Flag hot pixels
            if dynamic_params.hot_pixel_thresh is not None:
                hot_pixels = (valid_pixels & 
                             (stats['max_val'] > dynamic_params.hot_pixel_thresh))
                dynamic_mask &= ~hot_pixels
                hot_count = hot_pixels.sum()
                logger.debug(f"Panel {panel_idx}: Flagged {hot_count} hot pixels")
            
            # Flag negative pixels (accounting for tolerance)
            negative_pixels = (valid_pixels & 
                             (stats['min_val'] < -abs(dynamic_params.negative_pixel_tolerance)))
            dynamic_mask &= ~negative_pixels
            negative_count = negative_pixels.sum()
            logger.debug(f"Panel {panel_idx}: Flagged {negative_count} negative pixels")
            
            # Check total fraction of bad pixels
            total_pixels = height * width
            good_pixels = dynamic_mask.sum()
            bad_fraction = 1.0 - (good_pixels / total_pixels)
            
            if bad_fraction > dynamic_params.max_fraction_bad_pixels:
                logger.warning(
                    f"Panel {panel_idx}: High fraction of bad pixels ({bad_fraction:.3f}), "
                    f"exceeds threshold {dynamic_params.max_fraction_bad_pixels}"
                )
            
            logger.info(f"Panel {panel_idx}: Dynamic mask flagged {total_pixels - good_pixels} "
                       f"bad pixels ({bad_fraction:.3f} fraction)")
            
            # Convert to flex.bool
            flex_mask = flex.bool(dynamic_mask.flatten())
            flex_mask.reshape(flex.grid(height, width))
            
            return flex_mask
            
        except Exception as e:
            logger.error(f"Failed to generate dynamic mask for panel {panel_idx}: {e}")
            # Return all-True mask as fallback
            from dials.array_family import flex
            height, width = stats['sum'].shape
            fallback_mask = flex.bool(flex.grid(height, width), True)
            return fallback_mask
    
    def _combine_masks(
        self,
        static_mask: Tuple[object, ...],
        dynamic_mask: Tuple[object, ...]
    ) -> Tuple[object, ...]:
        """
        Combine static and dynamic masks using logical AND.
        
        Args:
            static_mask: Tuple of static masks (one per panel)
            dynamic_mask: Tuple of dynamic masks (one per panel)
            
        Returns:
            Tuple of combined masks (one per panel)
            
        Raises:
            MaskGenerationError: When masks cannot be combined
        """
        try:
            if len(static_mask) != len(dynamic_mask):
                raise MaskGenerationError(
                    f"Static and dynamic masks have different panel counts: "
                    f"{len(static_mask)} vs {len(dynamic_mask)}"
                )
            
            combined_masks = []
            for panel_idx, (static_panel, dynamic_panel) in enumerate(
                zip(static_mask, dynamic_mask)
            ):
                # Combine using logical AND
                combined_panel = static_panel & dynamic_panel
                combined_masks.append(combined_panel)
                
                # Log statistics
                static_good = static_panel.count(True)
                dynamic_good = dynamic_panel.count(True)
                combined_good = combined_panel.count(True)
                total_pixels = len(combined_panel)
                
                logger.debug(
                    f"Panel {panel_idx}: Static={static_good}/{total_pixels}, "
                    f"Dynamic={dynamic_good}/{total_pixels}, "
                    f"Combined={combined_good}/{total_pixels}"
                )
            
            return tuple(combined_masks)
            
        except Exception as e:
            raise MaskGenerationError(f"Failed to combine masks: {e}")


# Convenience functions for creating mask parameters

def create_circular_beamstop(center_x: float, center_y: float, radius: float) -> Circle:
    """Create a circular beamstop definition."""
    return Circle(center_x=center_x, center_y=center_y, radius=radius)


def create_rectangular_beamstop(
    min_x: float, max_x: float, min_y: float, max_y: float
) -> Rectangle:
    """Create a rectangular beamstop definition."""
    return Rectangle(min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y)


def create_default_static_params() -> StaticMaskParams:
    """Create default static mask parameters."""
    return StaticMaskParams()


def create_default_dynamic_params() -> DynamicMaskParams:
    """Create default dynamic mask parameters."""
    return DynamicMaskParams(
        hot_pixel_thresh=1e6,  # Very high counts
        negative_pixel_tolerance=0.0,
        max_fraction_bad_pixels=0.1
    )