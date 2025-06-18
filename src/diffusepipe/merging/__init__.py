"""
Merging module for Phase 3 data merging.

This module provides:
- DiffuseDataMerger: Merge scaled observations into final voxel data
- VoxelDataRelative: Data structure for relatively-scaled results
"""

from .merger import DiffuseDataMerger, VoxelDataRelative

__all__ = ["DiffuseDataMerger", "VoxelDataRelative"]
