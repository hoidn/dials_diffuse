"""Masking utilities for Bragg peaks and detector pixels."""

from .pixel_mask_generator import (
    PixelMaskGenerator,
    StaticMaskParams,
    DynamicMaskParams,
    Circle,
    Rectangle,
    create_circular_beamstop,
    create_rectangular_beamstop,
    create_default_static_params,
    create_default_dynamic_params,
)

from .bragg_mask_generator import (
    BraggMaskGenerator,
    create_default_bragg_mask_config,
    create_expanded_bragg_mask_config,
    validate_mask_compatibility,
)

__all__ = [
    # Pixel mask generation
    "PixelMaskGenerator",
    "StaticMaskParams",
    "DynamicMaskParams",
    "Circle",
    "Rectangle",
    "create_circular_beamstop",
    "create_rectangular_beamstop",
    "create_default_static_params",
    "create_default_dynamic_params",
    # Bragg mask generation
    "BraggMaskGenerator",
    "create_default_bragg_mask_config",
    "create_expanded_bragg_mask_config",
    "validate_mask_compatibility",
]
