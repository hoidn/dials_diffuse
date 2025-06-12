"""
GlobalVoxelGrid implementation for Phase 3 voxelization.

Defines common 3D reciprocal space grid for merging diffuse scattering data.
"""

import numpy as np
import logging
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

from scitbx import matrix
from cctbx import uctbx
from dxtbx.model import Experiment, Crystal

logger = logging.getLogger(__name__)


@dataclass
class GlobalVoxelGridConfig:
    """Configuration for GlobalVoxelGrid creation."""
    d_min_target: float  # High resolution limit (Å)
    d_max_target: float  # Low resolution limit (Å)  
    ndiv_h: int          # H subdivisions per unit cell
    ndiv_k: int          # K subdivisions per unit cell
    ndiv_l: int          # L subdivisions per unit cell
    max_rms_delta_hkl: float = 0.1  # Warning threshold for Δhkl RMS


@dataclass
class CorrectedDiffusePixelData:
    """Corrected diffuse pixel data from Phase 2."""
    q_vectors: np.ndarray     # Shape (N, 3) lab-frame q-vectors
    intensities: np.ndarray   # Shape (N,) corrected intensities
    sigmas: np.ndarray       # Shape (N,) uncertainties
    still_ids: np.ndarray    # Shape (N,) still identifiers


class GlobalVoxelGrid:
    """
    Global 3D reciprocal space grid for merging diffuse scattering data.
    
    Handles crystal model averaging, HKL range determination, and voxel indexing.
    """
    
    def __init__(self, 
                 experiment_list: List[Experiment],
                 corrected_diffuse_pixel_data: List[CorrectedDiffusePixelData],
                 grid_config: GlobalVoxelGridConfig):
        """
        Initialize GlobalVoxelGrid with crystal averaging and grid definition.
        
        Args:
            experiment_list: List of Experiment objects with crystal models
            corrected_diffuse_pixel_data: List of corrected diffuse data from Phase 2
            grid_config: Grid configuration parameters
        """
        self.config = grid_config
        self._validate_config()
        self._validate_inputs(experiment_list, corrected_diffuse_pixel_data)
        
        # Extract crystal models
        self.crystal_models = [exp.crystal for exp in experiment_list]
        self.diffuse_data = corrected_diffuse_pixel_data
        
        # Average crystal models
        logger.info(f"Averaging {len(self.crystal_models)} crystal models")
        self._average_crystal_models()
        
        # Determine HKL range from diffuse data
        logger.info("Determining HKL range from diffuse data")
        self._determine_hkl_range()
        
        # Calculate total voxels
        self.total_voxels = ((self.hkl_max[0] - self.hkl_min[0] + 1) * 
                           (self.hkl_max[1] - self.hkl_min[1] + 1) * 
                           (self.hkl_max[2] - self.hkl_min[2] + 1))
        
        logger.info(f"Grid initialized: {self.total_voxels} voxels, "
                   f"HKL range: {self.hkl_min} to {self.hkl_max}")
    
    def _validate_config(self):
        """Validate grid configuration parameters."""
        if self.config.d_min_target <= 0:
            raise ValueError("d_min_target must be positive")
        if self.config.d_max_target <= self.config.d_min_target:
            raise ValueError("d_max_target must be greater than d_min_target")
        if any(ndiv <= 0 for ndiv in [self.config.ndiv_h, self.config.ndiv_k, self.config.ndiv_l]):
            raise ValueError("Grid subdivisions must be positive integers")
    
    def _validate_inputs(self, experiment_list, diffuse_data):
        """Validate input data."""
        if not experiment_list:
            raise ValueError("experiment_list cannot be empty")
        if not diffuse_data:
            raise ValueError("corrected_diffuse_pixel_data cannot be empty")
        
        for exp in experiment_list:
            if exp.crystal is None:
                raise ValueError("All experiments must have crystal models")
    
    def _average_crystal_models(self):
        """Average crystal models to create reference crystal."""
        # Average unit cells using CCTBX utilities
        unit_cells = [crystal.get_unit_cell() for crystal in self.crystal_models]
        logger.info(f"Unit cell parameters before averaging:")
        for i, uc in enumerate(unit_cells[:3]):  # Show first 3
            logger.info(f"  Crystal {i}: {uc.parameters()}")
        
        # Use CCTBX to average unit cells robustly
        avg_unit_cell = self._average_unit_cells(unit_cells)
        logger.info(f"Average unit cell: {avg_unit_cell.parameters()}")
        
        # Average U matrices
        u_matrices = [matrix.sqr(crystal.get_U()) for crystal in self.crystal_models]
        self._check_orientation_spread(u_matrices)
        u_avg_ref = self._average_u_matrices(u_matrices)
        
        # Calculate B matrix from averaged unit cell
        b_avg_ref = matrix.sqr(avg_unit_cell.fractionalization_matrix()).transpose()
        
        # Final setting matrix
        self.A_avg_ref = u_avg_ref * b_avg_ref
        
        # Create reference crystal model
        orth_matrix = matrix.sqr(avg_unit_cell.orthogonalization_matrix())
        self.crystal_avg_ref = Crystal(
            real_space_a=orth_matrix * matrix.col((1, 0, 0)),
            real_space_b=orth_matrix * matrix.col((0, 1, 0)),
            real_space_c=orth_matrix * matrix.col((0, 0, 1)),
            space_group=self.crystal_models[0].get_space_group()
        )
        self.crystal_avg_ref.set_U(u_avg_ref)
        
        # Calculate diagnostics
        self._calculate_averaging_diagnostics()
    
    def _average_unit_cells(self, unit_cells):
        """Average unit cell parameters using CCTBX utilities."""
        # Convert to parameters and average
        params_list = [uc.parameters() for uc in unit_cells]
        params_array = np.array(params_list)
        
        # Simple arithmetic mean for now - could use more robust method
        avg_params = np.mean(params_array, axis=0)
        
        return uctbx.unit_cell(tuple(avg_params))
    
    def _check_orientation_spread(self, u_matrices):
        """Check and warn about orientation spread between crystals."""
        if len(u_matrices) < 2:
            return
        
        u_ref = u_matrices[0]
        misorientations = []
        
        for u_i in u_matrices[1:]:
            # Calculate misorientation angle using trace of rotation matrix
            rotation_matrix = u_i * u_ref.transpose()
            trace = rotation_matrix.trace()
            # Handle numerical precision issues
            trace = max(-1.0, min(3.0, trace))
            angle_rad = np.arccos((trace - 1.0) / 2.0)
            angle_deg = np.degrees(angle_rad)
            misorientations.append(angle_deg)
        
        rms_misorientation = np.sqrt(np.mean(np.array(misorientations)**2))
        self.rms_misorientation_deg = rms_misorientation
        
        if rms_misorientation > 5.0:  # 5 degree threshold
            logger.warning(f"Large RMS misorientation: {rms_misorientation:.2f}° - "
                         "may cause smearing in merged diffuse map")
        else:
            logger.info(f"RMS misorientation: {rms_misorientation:.2f}°")
    
    def _average_u_matrices(self, u_matrices):
        """Average U matrices using quaternion-based method."""
        # For simplicity, use arithmetic mean (assumes small deviations)
        # Production code should use proper quaternion averaging
        if len(u_matrices) == 0:
            raise ValueError("Cannot average empty list of U matrices")
        
        # Start with first matrix
        u_sum = u_matrices[0]
        for u_matrix in u_matrices[1:]:
            u_sum = u_sum + u_matrix
        u_avg = u_sum * (1.0 / len(u_matrices))
        
        # Orthogonalize the result using SVD-like approach
        # This is a simplified version - proper implementation would use quaternions
        return self._orthogonalize_matrix(u_avg)
    
    def _orthogonalize_matrix(self, matrix_approx):
        """Orthogonalize a nearly-orthogonal matrix."""
        # Convert to numpy for SVD
        mat_np = np.array(matrix_approx).reshape(3, 3)
        u, s, vt = np.linalg.svd(mat_np)
        
        # Ensure proper rotation (det = +1)
        orthogonal = u @ vt
        if np.linalg.det(orthogonal) < 0:
            u[:, -1] *= -1
            orthogonal = u @ vt
        
        return matrix.sqr(orthogonal.flatten())
    
    def _determine_hkl_range(self):
        """Determine HKL range from diffuse data and resolution limits."""
        all_q_vectors = []
        for data in self.diffuse_data:
            all_q_vectors.append(data.q_vectors)
        
        if not all_q_vectors:
            raise ValueError("No diffuse data provided for HKL range determination")
        
        combined_q_vectors = np.vstack(all_q_vectors)
        logger.info(f"Processing {len(combined_q_vectors)} q-vectors for HKL range")
        
        # Transform to fractional HKL using average crystal
        A_inv = self.A_avg_ref.inverse()
        hkl_fractional = []
        
        for q_vec in combined_q_vectors:
            q_matrix = matrix.col(q_vec)
            hkl_frac = A_inv * q_matrix
            hkl_fractional.append(hkl_frac.elems)
        
        hkl_array = np.array(hkl_fractional)
        
        # Apply resolution filters
        q_magnitudes = np.linalg.norm(combined_q_vectors, axis=1)
        d_spacings = 1.0 / (q_magnitudes + 1e-10)  # Avoid division by zero
        
        valid_mask = ((d_spacings >= self.config.d_min_target) & 
                     (d_spacings <= self.config.d_max_target))
        
        if not np.any(valid_mask):
            raise ValueError("No data within specified resolution limits")
        
        filtered_hkl = hkl_array[valid_mask]
        logger.info(f"After resolution filtering: {len(filtered_hkl)} observations")
        
        # Determine integer HKL boundaries with buffer
        hkl_min_frac = np.min(filtered_hkl, axis=0)
        hkl_max_frac = np.max(filtered_hkl, axis=0)
        
        # Convert to integer boundaries with subdivision buffer
        buffer = 1  # Buffer for grid edges
        ndiv_list = [self.config.ndiv_h, self.config.ndiv_k, self.config.ndiv_l]
        self.hkl_min = tuple(int(np.floor(hkl_min_frac[i] * ndiv_list[i]) - buffer)
                           for i in range(3))
        self.hkl_max = tuple(int(np.ceil(hkl_max_frac[i] * ndiv_list[i]) + buffer) 
                           for i in range(3))
        
        logger.info(f"HKL range determined: {self.hkl_min} to {self.hkl_max}")
    
    def _calculate_averaging_diagnostics(self):
        """Calculate diagnostic metrics for crystal averaging quality."""
        # For now, just store what we have
        self.diagnostics = {
            "rms_misorientation_deg": getattr(self, 'rms_misorientation_deg', 0.0),
            "n_crystals_averaged": len(self.crystal_models),
            "total_diffuse_observations": sum(len(data.q_vectors) for data in self.diffuse_data)
        }
        
        # TODO: Calculate RMS Δhkl for Bragg reflections when available
        self.diagnostics["rms_delta_hkl"] = 0.0  # Placeholder
    
    def hkl_to_voxel_idx(self, h: float, k: float, l: float) -> int:
        """Map fractional Miller indices to linear voxel index."""
        # Convert to subdivision coordinates
        h_sub = int(np.round(h * self.config.ndiv_h))
        k_sub = int(np.round(k * self.config.ndiv_k))
        l_sub = int(np.round(l * self.config.ndiv_l))
        
        # Shift to positive indices
        h_idx = h_sub - self.hkl_min[0]
        k_idx = k_sub - self.hkl_min[1]  
        l_idx = l_sub - self.hkl_min[2]
        
        # Linear index
        h_range = self.hkl_max[0] - self.hkl_min[0] + 1
        k_range = self.hkl_max[1] - self.hkl_min[1] + 1
        
        return l_idx * (h_range * k_range) + k_idx * h_range + h_idx
    
    def voxel_idx_to_hkl_center(self, voxel_idx: int) -> Tuple[float, float, float]:
        """Map voxel index to center HKL coordinates."""
        h_range = self.hkl_max[0] - self.hkl_min[0] + 1
        k_range = self.hkl_max[1] - self.hkl_min[1] + 1
        
        # Inverse linear mapping
        l_idx = voxel_idx // (h_range * k_range)
        remainder = voxel_idx % (h_range * k_range)
        k_idx = remainder // h_range
        h_idx = remainder % h_range
        
        # Convert back to HKL coordinates
        h_sub = h_idx + self.hkl_min[0]
        k_sub = k_idx + self.hkl_min[1]
        l_sub = l_idx + self.hkl_min[2]
        
        # Convert subdivision coordinates to fractional HKL
        h = h_sub / self.config.ndiv_h
        k = k_sub / self.config.ndiv_k
        l = l_sub / self.config.ndiv_l
        
        return h, k, l
    
    def get_q_vector_for_voxel_center(self, voxel_idx: int) -> matrix.col:
        """Get lab-frame q-vector for voxel center."""
        h, k, l = self.voxel_idx_to_hkl_center(voxel_idx)
        hkl = matrix.col((h, k, l))
        q_vector = self.A_avg_ref * hkl
        return q_vector
    
    def get_crystal_averaging_diagnostics(self) -> Dict:
        """Return diagnostic metrics for crystal model averaging quality."""
        return {
            **self.diagnostics,
            "hkl_range_min": self.hkl_min,
            "hkl_range_max": self.hkl_max,
            "total_voxels": self.total_voxels,
            "grid_subdivisions": (self.config.ndiv_h, self.config.ndiv_k, self.config.ndiv_l)
        }