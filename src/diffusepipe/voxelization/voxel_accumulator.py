"""
VoxelAccumulator implementation for Phase 3 binning.

Bins corrected diffuse pixel observations into voxels with HDF5 backend.
"""

import numpy as np
import logging
import tempfile
import os
from typing import Dict, Optional, Any
from dataclasses import dataclass
from collections import defaultdict

try:
    import h5py

    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False
    h5py = None

from scitbx import matrix
from cctbx import sgtbx, miller
from dials.array_family import flex

from .global_voxel_grid import GlobalVoxelGrid

logger = logging.getLogger(__name__)


@dataclass
class ObservationData:
    """Single observation data structure."""

    intensity: float
    sigma: float
    still_id: int
    q_vector_lab: np.ndarray  # Shape (3,)


class VoxelAccumulator:
    """
    Bins corrected diffuse pixel observations into voxels.

    Supports both in-memory and HDF5 backends for memory management.
    Handles HKL transformation and ASU mapping.
    """

    def __init__(
        self,
        global_voxel_grid: GlobalVoxelGrid,
        space_group_info: sgtbx.space_group_info,
        backend: str = "memory",
        storage_path: Optional[str] = None,
    ):
        """
        Initialize VoxelAccumulator.

        Args:
            global_voxel_grid: Grid definition for voxel mapping
            space_group_info: Space group for ASU mapping
            backend: "memory" or "hdf5" for storage
            storage_path: Path for HDF5 file if backend="hdf5"
        """
        self.global_voxel_grid = global_voxel_grid
        self.space_group = space_group_info.group()
        self.backend = backend
        self.storage_path = storage_path

        # Statistics
        self.n_total_observations = 0
        self.n_unique_voxels = 0
        self._still_observation_counts = defaultdict(int)

        # Initialize storage backend
        self._initialize_storage()

        logger.info(f"VoxelAccumulator initialized with {backend} backend")

    def _initialize_storage(self):
        """Initialize the storage backend."""
        if self.backend == "memory":
            self._voxel_data = defaultdict(list)
        elif self.backend == "hdf5":
            if not HDF5_AVAILABLE:
                raise RuntimeError("h5py not available for HDF5 backend")
            self._initialize_hdf5_storage()
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _initialize_hdf5_storage(self):
        """Initialize HDF5 storage backend."""
        if self.storage_path is None:
            # Create temporary file
            temp_dir = tempfile.mkdtemp()
            self.storage_path = os.path.join(temp_dir, "voxel_accumulator.h5")

        self.h5_file = h5py.File(self.storage_path, "w")

        # Create main datasets with chunking and compression
        max_obs_estimate = 1000000  # Estimate for dataset size

        # Main observation storage
        self.h5_intensities = self.h5_file.create_dataset(
            "intensities",
            (0,),
            maxshape=(None,),
            dtype="f4",
            chunks=True,
            compression="gzip",
            shuffle=True,
        )
        self.h5_sigmas = self.h5_file.create_dataset(
            "sigmas",
            (0,),
            maxshape=(None,),
            dtype="f4",
            chunks=True,
            compression="gzip",
            shuffle=True,
        )
        self.h5_still_ids = self.h5_file.create_dataset(
            "still_ids",
            (0,),
            maxshape=(None,),
            dtype="i4",
            chunks=True,
            compression="gzip",
            shuffle=True,
        )
        self.h5_voxel_indices = self.h5_file.create_dataset(
            "voxel_indices",
            (0,),
            maxshape=(None,),
            dtype="i8",
            chunks=True,
            compression="gzip",
            shuffle=True,
        )
        self.h5_q_vectors = self.h5_file.create_dataset(
            "q_vectors",
            (0, 3),
            maxshape=(None, 3),
            dtype="f4",
            chunks=True,
            compression="gzip",
            shuffle=True,
        )

        # Voxel index mapping for efficient access
        self._voxel_ranges = {}  # voxel_idx -> (start, end) in arrays
        self._current_obs_count = 0

        logger.info(f"HDF5 storage initialized at {self.storage_path}")

    def add_observations(
        self,
        still_id: int,
        q_vectors_lab: np.ndarray,
        intensities: np.ndarray,
        sigmas: np.ndarray,
    ) -> int:
        """
        Add observations for a still to appropriate voxels.

        Args:
            still_id: Identifier for the still
            q_vectors_lab: Lab-frame q-vectors, shape (N, 3)
            intensities: Corrected intensities, shape (N,)
            sigmas: Intensity uncertainties, shape (N,)

        Returns:
            Number of observations successfully binned
        """
        if len(q_vectors_lab) != len(intensities) or len(intensities) != len(sigmas):
            raise ValueError("Array length mismatch in input data")

        n_input = len(q_vectors_lab)
        if n_input == 0:
            return 0

        logger.debug(f"Adding {n_input} observations from still {still_id}")

        # Transform q-vectors to fractional HKL using vectorized approach
        hkl_array = self._vectorized_hkl_transform(q_vectors_lab)

        # Apply ASU mapping
        hkl_asu = self._map_to_asu(hkl_array)

        # Get voxel indices
        voxel_indices = []
        valid_mask = []

        for i, (h, k, l) in enumerate(hkl_asu):
            try:
                voxel_idx = self.global_voxel_grid.hkl_to_voxel_idx(h, k, l)
                voxel_indices.append(voxel_idx)
                valid_mask.append(True)
            except (ValueError, IndexError):
                # Outside grid boundaries
                valid_mask.append(False)

        valid_mask = np.array(valid_mask)
        n_valid = np.sum(valid_mask)

        if n_valid == 0:
            logger.warning(f"No valid observations for still {still_id}")
            return 0

        # Filter to valid observations
        valid_q_vectors = q_vectors_lab[valid_mask]
        valid_intensities = intensities[valid_mask]
        valid_sigmas = sigmas[valid_mask]
        valid_voxel_indices = np.array(voxel_indices)[valid_mask]

        # Store observations
        if self.backend == "memory":
            self._store_observations_memory(
                still_id,
                valid_voxel_indices,
                valid_intensities,
                valid_sigmas,
                valid_q_vectors,
            )
        else:
            self._store_observations_hdf5(
                still_id,
                valid_voxel_indices,
                valid_intensities,
                valid_sigmas,
                valid_q_vectors,
            )

        # Update statistics
        self.n_total_observations += n_valid
        self._still_observation_counts[still_id] += n_valid

        logger.debug(f"Successfully binned {n_valid}/{n_input} observations")
        return n_valid

    def _vectorized_hkl_transform(self, q_vectors_lab: np.ndarray) -> np.ndarray:
        """
        Transform q-vectors to fractional HKL coordinates using vectorized operations.

        This method uses NumPy matrix multiplication for significant performance gains
        over iterative approaches (10x+ speedup for large arrays). No chunking is
        needed here as VoxelAccumulator processes filtered data of manageable size.

        Args:
            q_vectors_lab: Lab-frame q-vectors, shape (N, 3)

        Returns:
            HKL fractional coordinates, shape (N, 3)
        """
        A_inv = self.global_voxel_grid.A_avg_ref.inverse()

        # Convert scitbx matrix to NumPy array for vectorized operation
        A_inv_np = np.array(A_inv.elems).reshape(3, 3)

        # Vectorized HKL transformation: (3,3) @ (N,3).T -> (3,N) -> (N,3)
        hkl_array = (A_inv_np @ q_vectors_lab.T).T

        return hkl_array

    def _legacy_hkl_transform(self, q_vectors_lab: np.ndarray) -> np.ndarray:
        """
        Transform q-vectors to fractional HKL using legacy iterative approach.

        This method is kept for equivalence testing only. It uses the original
        Python loop approach that was identified as a performance bottleneck.

        Args:
            q_vectors_lab: Lab-frame q-vectors, shape (N, 3)

        Returns:
            HKL fractional coordinates, shape (N, 3)
        """
        A_inv = self.global_voxel_grid.A_avg_ref.inverse()
        hkl_fractional = []

        for q_vec in q_vectors_lab:
            q_matrix = matrix.col(q_vec)
            hkl_frac = A_inv * q_matrix
            hkl_fractional.append(hkl_frac.elems)

        return np.array(hkl_fractional)

    def _map_to_asu(self, hkl_array: np.ndarray) -> np.ndarray:
        """Map fractional HKL coordinates to the asymmetric unit using CCTBX symmetry operations."""
        import numpy as np
        from cctbx import crystal

        # Get crystal model from the grid
        crystal_model = self.global_voxel_grid.crystal_avg_ref

        # Create proper crystal symmetry object from unit cell and space group
        unit_cell = crystal_model.get_unit_cell()
        space_group = crystal_model.get_space_group()
        crystal_symmetry = crystal.symmetry(
            unit_cell=unit_cell, space_group=space_group
        )

        # Convert to numpy array for easier manipulation
        coords_array = np.array(hkl_array)
        
        # Round to nearest integers for miller.set creation
        miller_indices_int = np.round(coords_array).astype(int)
        miller_indices = flex.miller_index(miller_indices_int.tolist())
        miller_set = miller.set(
            crystal_symmetry=crystal_symmetry, indices=miller_indices
        )
        
        # Map to ASU
        asu_set = miller_set.map_to_asu()
        
        # Apply the same transformation to fractional coordinates
        asu_indices_int = asu_set.indices().as_vec3_double().as_numpy_array()
        transformation_applied = asu_indices_int - miller_indices_int
        asu_fractional = coords_array + transformation_applied
        
        return asu_fractional

    def _store_observations_memory(
        self,
        still_id: int,
        voxel_indices: np.ndarray,
        intensities: np.ndarray,
        sigmas: np.ndarray,
        q_vectors: np.ndarray,
    ):
        """Store observations in memory backend."""
        for i, voxel_idx in enumerate(voxel_indices):
            obs = ObservationData(
                intensity=intensities[i],
                sigma=sigmas[i],
                still_id=still_id,
                q_vector_lab=q_vectors[i],
            )
            self._voxel_data[voxel_idx].append(obs)

    def _store_observations_hdf5(
        self,
        still_id: int,
        voxel_indices: np.ndarray,
        intensities: np.ndarray,
        sigmas: np.ndarray,
        q_vectors: np.ndarray,
    ):
        """Store observations in HDF5 backend."""
        n_new = len(voxel_indices)
        start_idx = self._current_obs_count
        end_idx = start_idx + n_new

        # Resize datasets
        self.h5_intensities.resize((end_idx,))
        self.h5_sigmas.resize((end_idx,))
        self.h5_still_ids.resize((end_idx,))
        self.h5_voxel_indices.resize((end_idx,))
        self.h5_q_vectors.resize((end_idx, 3))

        # Store data
        self.h5_intensities[start_idx:end_idx] = intensities
        self.h5_sigmas[start_idx:end_idx] = sigmas
        self.h5_still_ids[start_idx:end_idx] = still_id
        self.h5_voxel_indices[start_idx:end_idx] = voxel_indices
        self.h5_q_vectors[start_idx:end_idx] = q_vectors

        self._current_obs_count = end_idx

    def get_observations_for_voxel(self, voxel_idx: int) -> Dict[str, np.ndarray]:
        """Get all observations for a specific voxel."""
        if self.backend == "memory":
            return self._get_voxel_observations_memory(voxel_idx)
        else:
            return self._get_voxel_observations_hdf5(voxel_idx)

    def _get_voxel_observations_memory(self, voxel_idx: int) -> Dict[str, np.ndarray]:
        """Get voxel observations from memory backend."""
        observations = self._voxel_data.get(voxel_idx, [])

        if not observations:
            return {
                "intensities": np.array([]),
                "sigmas": np.array([]),
                "still_ids": np.array([]),
                "q_vectors_lab": np.array([]).reshape(0, 3),
                "n_observations": 0,
            }

        intensities = np.array([obs.intensity for obs in observations])
        sigmas = np.array([obs.sigma for obs in observations])
        still_ids = np.array([obs.still_id for obs in observations])
        q_vectors = np.array([obs.q_vector_lab for obs in observations])

        return {
            "intensities": intensities,
            "sigmas": sigmas,
            "still_ids": still_ids,
            "q_vectors_lab": q_vectors,
            "n_observations": len(observations),
        }

    def _get_voxel_observations_hdf5(self, voxel_idx: int) -> Dict[str, np.ndarray]:
        """Get voxel observations from HDF5 backend."""
        # Find all indices for this voxel
        voxel_mask = self.h5_voxel_indices[:] == voxel_idx
        indices = np.where(voxel_mask)[0]

        if len(indices) == 0:
            return {
                "intensities": np.array([]),
                "sigmas": np.array([]),
                "still_ids": np.array([]),
                "q_vectors_lab": np.array([]).reshape(0, 3),
                "n_observations": 0,
            }

        return {
            "intensities": self.h5_intensities[indices],
            "sigmas": self.h5_sigmas[indices],
            "still_ids": self.h5_still_ids[indices],
            "q_vectors_lab": self.h5_q_vectors[indices],
            "n_observations": len(indices),
        }

    def get_all_binned_data_for_scaling(self) -> Dict[int, Dict[str, Any]]:
        """Get complete binned dataset for scaling algorithms."""
        binned_data = {}

        if self.backend == "memory":
            unique_voxels = list(self._voxel_data.keys())
        else:
            # Get unique voxel indices from HDF5
            unique_voxels = np.unique(self.h5_voxel_indices[:])

        self.n_unique_voxels = len(unique_voxels)

        for voxel_idx in unique_voxels:
            voxel_obs = self.get_observations_for_voxel(voxel_idx)

            if voxel_obs["n_observations"] > 0:
                # Get voxel center coordinates
                h, k, l = self.global_voxel_grid.voxel_idx_to_hkl_center(voxel_idx)
                q_center = self.global_voxel_grid.get_q_vector_for_voxel_center(
                    voxel_idx
                )

                # Format observations for scaling
                observations = []
                for i in range(voxel_obs["n_observations"]):
                    obs = {
                        "intensity": voxel_obs["intensities"][i],
                        "sigma": voxel_obs["sigmas"][i],
                        "still_id": voxel_obs["still_ids"][i],
                        "q_vector_lab": voxel_obs["q_vectors_lab"][i],
                    }
                    observations.append(obs)

                binned_data[voxel_idx] = {
                    "observations": observations,
                    "n_observations": voxel_obs["n_observations"],
                    "voxel_center_hkl": (h, k, l),
                    "voxel_center_q": np.array(q_center.elems),
                }

        logger.info(f"Retrieved binned data for {len(binned_data)} voxels")
        return binned_data

    def get_accumulation_statistics(self) -> Dict[str, Any]:
        """Get statistics about accumulated data."""
        if self.backend == "memory":
            voxel_obs_counts = [len(obs_list) for obs_list in self._voxel_data.values()]
            unique_voxels = len(self._voxel_data)
        else:
            unique_voxels = len(np.unique(self.h5_voxel_indices[:]))
            # Calculate observations per voxel
            voxel_indices, counts = np.unique(
                self.h5_voxel_indices[:], return_counts=True
            )
            voxel_obs_counts = counts.tolist()

        if voxel_obs_counts:
            obs_stats = {
                "mean": np.mean(voxel_obs_counts),
                "std": np.std(voxel_obs_counts),
                "min": int(np.min(voxel_obs_counts)),
                "max": int(np.max(voxel_obs_counts)),
            }
        else:
            obs_stats = {"mean": 0, "std": 0, "min": 0, "max": 0}

        return {
            "total_observations": self.n_total_observations,
            "unique_voxels": unique_voxels,
            "observations_per_voxel_stats": obs_stats,
            "still_distribution": dict(self._still_observation_counts),
            "backend": self.backend,
        }

    def finalize(self):
        """Finalize storage and prepare for access."""
        if self.backend == "hdf5" and hasattr(self, "h5_file"):
            self.h5_file.flush()
            logger.info("HDF5 storage finalized")

        stats = self.get_accumulation_statistics()
        logger.info(
            f"VoxelAccumulator finalized: {stats['total_observations']} observations "
            f"in {stats['unique_voxels']} voxels"
        )

    def __del__(self):
        """Cleanup HDF5 resources."""
        if hasattr(self, "h5_file") and self.h5_file:
            try:
                self.h5_file.close()
            except:
                pass
