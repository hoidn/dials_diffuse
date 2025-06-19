# VoxelAccumulator IDL

**Module Path:** `src.diffusepipe.voxelization.voxel_accumulator`

**Dependencies:**
- `@depends_on(h5py)` - HDF5 file handling for large-scale data storage
- `@depends_on(numpy)` - Array operations and data structures
- `@depends_on(cctbx.sgtbx)` - Space group operations for ASU mapping
- `@depends_on(GlobalVoxelGrid)` - Grid definitions and HKL transformations

## Interface: VoxelAccumulator

**Purpose:** Bin corrected diffuse pixel observations into voxels with HDF5 backend for memory-efficient storage. Handles HKL transformation, ASU mapping, and incremental accumulation of observations for scaling and merging.

### Constructor

```python
def __init__(self, 
            global_voxel_grid: GlobalVoxelGrid,
            space_group_info: cctbx.sgtbx.space_group_info,
            backend: str = "hdf5",
            storage_path: Optional[str] = None) -> None
```

**Preconditions:**
- `global_voxel_grid` is valid and initialized
- `space_group_info` contains valid space group for ASU mapping
- `backend` is either "memory" or "hdf5"
- `storage_path` is writable directory if backend="hdf5"

**Postconditions:**
- Storage backend initialized (HDF5 file created if needed)
- Ready to accumulate voxel observations
- ASU mapping operations configured

**Behavior:**
1. Initialize storage backend (in-memory dict or HDF5 with zstd compression)
2. Configure space group symmetry operations for ASU mapping
3. Pre-allocate HDF5 datasets if using file backend
4. Set up efficient indexing for voxel data retrieval

### Methods

```python
def add_observations(self, 
                    still_id: int,
                    q_vectors_lab: numpy.ndarray,
                    intensities: numpy.ndarray, 
                    sigmas: numpy.ndarray,
                    frame_indices: Optional[numpy.ndarray] = None) -> int
```

**Preconditions:**
- All input arrays have same length
- `q_vectors_lab` shape is (N, 3) - lab frame q-vectors
- `intensities` and `sigmas` are positive
- `still_id` is valid identifier
- `frame_indices` shape is (N,) if provided - 0-based scan frame indices

**Postconditions:**
- Observations binned to appropriate voxels using correct frame-specific transformations
- ASU symmetry applied correctly
- Returns number of observations successfully binned

**Behavior:**
For each observation, or for batches of observations grouped by unique `frame_index` from `frame_indices`:
1. **Frame-Specific Transformation:** Retrieves the frame-specific `A(φ)⁻¹` via `global_voxel_grid.get_A_inverse_for_frame(frame_idx)` for sequential data, or uses static crystal model for still data
2. **HKL Transformation:** Transforms `q_lab` to `hkl_frac` using the frame-specific `A(φ)⁻¹`: `hkl_frac = A(φ)⁻¹ * q_lab`
3. **ASU Mapping:** Maps fractional HKL coordinates to the asymmetric unit using `cctbx.sgtbx.space_group_info.map_to_asu`
4. **Voxel Assignment:** Determines voxel indices using `global_voxel_grid.hkl_to_voxel_idx()`
5. **Data Storage:** Stores `(intensity, sigma, still_id, q_lab)` for each voxel
6. **Statistics Update:** Updates accumulation statistics if using Welford's algorithm

**Expected Data Format:**
```python
# Input arrays
q_vectors_lab: numpy.ndarray     # Shape (N, 3) - lab frame q-vectors
intensities: numpy.ndarray       # Shape (N,) - corrected intensities
sigmas: numpy.ndarray           # Shape (N,) - intensity uncertainties
```

```python
def get_observations_for_voxel(self, voxel_idx: int) -> dict
```

**Preconditions:** `voxel_idx` is valid voxel index
**Postconditions:** Returns all observations for specified voxel
**Behavior:** Retrieves stored observations from backend for given voxel

**Expected Data Format:**
```python
voxel_observations = {
    "intensities": numpy.ndarray,     # All intensities for this voxel
    "sigmas": numpy.ndarray,         # Corresponding uncertainties  
    "still_ids": numpy.ndarray,      # Still identifiers
    "q_vectors_lab": numpy.ndarray,  # Original lab-frame q-vectors
    "n_observations": int            # Number of observations
}
```

```python
def get_all_binned_data_for_scaling(self) -> dict
```

**Preconditions:** Accumulation is complete
**Postconditions:** Returns complete binned dataset for scaling
**Behavior:** Assembles all voxel data in format required by scaling algorithms

**Expected Data Format:**
```python
binned_data_global = {
    voxel_idx: {
        "intensities": numpy.ndarray,     # All intensities for this voxel
        "sigmas": numpy.ndarray,         # Corresponding uncertainties  
        "still_ids": numpy.ndarray,      # Still identifiers
        "q_vectors_lab": numpy.ndarray,  # Original lab-frame q-vectors
        "n_observations": int            # Number of observations
    }
    # ... for all voxels with data
}
```

```python
def get_accumulation_statistics(self) -> dict
```

**Preconditions:** Some observations have been added
**Postconditions:** Returns statistics about accumulated data
**Behavior:** Provides summary statistics for data quality assessment

**Expected Data Format:**
```python
statistics = {
    "total_observations": int,
    "unique_voxels": int,
    "observations_per_voxel_stats": {
        "mean": float,
        "std": float,
        "min": int,
        "max": int
    },
    "still_distribution": dict[int, int],  # still_id -> n_observations
    "resolution_range": {
        "d_min": float,  # Minimum d-spacing observed
        "d_max": float   # Maximum d-spacing observed  
    }
}
```

```python
def finalize(self) -> None
```

**Preconditions:** All observations have been added
**Postconditions:** Storage backend optimized and ready for access
**Behavior:** Closes HDF5 file handles, optimizes storage, prepares for scaling phase

### Attributes

- `global_voxel_grid`: GlobalVoxelGrid - Grid definition and transformations
- `space_group`: cctbx.sgtbx.space_group - Space group for ASU mapping
- `backend`: str - Storage backend type ("memory" or "hdf5")
- `storage_path`: str - Path to HDF5 file if using file backend
- `n_total_observations`: int - Total observations accumulated
- `n_unique_voxels`: int - Number of voxels with data

### Error Conditions

**@raises_error(condition="InvalidVoxelIndex", message="Voxel index out of bounds")**
- Voxel index outside grid boundaries
- Negative voxel indices

**@raises_error(condition="StorageBackendError", message="Backend storage operation failed")**
- HDF5 file creation/write errors
- Disk space issues for large datasets

**@raises_error(condition="DataConsistencyError", message="Input array dimensions mismatch")**
- Array length mismatches between q_vectors, intensities, sigmas
- Invalid array shapes

**@raises_error(condition="SymmetryError", message="ASU mapping failed")**  
- Space group operations failed
- Invalid HKL coordinates for symmetry operations

## Implementation Notes

- Use HDF5 with zstd compression for large datasets
- Implement chunked storage for efficient random access
- Pre-allocate datasets when possible for performance
- Use memory backend for small test datasets
- Store q_vectors in original lab frame for traceability
- **ASU mapping uses correct CCTBX API pattern**: `cctbx.sgtbx.space_group_info.map_to_asu()` for direct fractional coordinate processing
- Voxel indices calculated using GlobalVoxelGrid methods
- Support both incremental addition and batch processing