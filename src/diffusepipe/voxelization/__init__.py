"""
Voxelization module for Phase 3 of the diffuse scattering pipeline.

This module provides:
- GlobalVoxelGrid: Define common 3D reciprocal space grid
- VoxelAccumulator: Bin diffuse pixel observations with HDF5 backend
- GlobalVoxelGridConfig: Configuration for grid creation
- CorrectedDiffusePixelData: Data structure for Phase 2 output
"""

from .global_voxel_grid import (
    GlobalVoxelGrid,
    GlobalVoxelGridConfig,
    CorrectedDiffusePixelData,
)
from .voxel_accumulator import VoxelAccumulator

__all__ = [
    "GlobalVoxelGrid",
    "GlobalVoxelGridConfig",
    "CorrectedDiffusePixelData",
    "VoxelAccumulator",
]
