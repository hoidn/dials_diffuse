# GlobalVoxelGrid IDL

**Module Path:** `src.diffusepipe.voxelization.global_voxel_grid`

**Dependencies:** 
- `@depends_on(dxtbx.model.Crystal)` - DIALS crystal models
- `@depends_on(scitbx.matrix)` - Matrix operations for HKL transformations
- `@depends_on(cctbx.uctbx)` - Unit cell averaging utilities
- `@depends_on(numpy)` - Array operations for q-vector analysis

## Interface: GlobalVoxelGrid

**Purpose:** Define a common 3D reciprocal space grid for merging diffuse scattering data from all processed still images. Handles crystal model averaging, HKL range determination, and voxel indexing operations.

### Constructor

```python
def __init__(self, 
            experiment_list: list[Experiment], 
            corrected_diffuse_pixel_data: list[CorrectedDiffusePixelData],
            grid_config: GlobalVoxelGridConfig) -> None
```

**Preconditions:**
- `experiment_list` contains valid `Experiment_dials_i` objects with crystal models
- `corrected_diffuse_pixel_data` contains q-vectors from Phase 2 processing
- `grid_config.d_min_target > 0` and `grid_config.d_max_target > grid_config.d_min_target`
- `grid_config.ndiv_h,k,l` are positive integers

**Postconditions:**
- `crystal_avg_ref` is computed as robust average of all input crystal models
- HKL range covers all diffuse data within resolution limits
- Grid subdivision parameters are stored
- Diagnostic metrics computed for crystal model averaging quality

**Behavior:**
1. Robustly average unit cell parameters using CCTBX utilities
2. Average U matrices using quaternion-based method for rotation matrices  
3. Compute `A_avg_ref = U_avg_ref * B_avg_ref` setting matrix
4. Calculate RMS Δhkl diagnostic for Bragg reflections
5. Calculate RMS misorientation diagnostic between individual U matrices
6. Transform all q-vectors to fractional HKL to determine grid boundaries
7. Store grid parameters and conversion methods

**Expected Data Format:**
```python
GlobalVoxelGridConfig = {
    "d_min_target": float,      # High resolution limit (Å)
    "d_max_target": float,      # Low resolution limit (Å) 
    "ndiv_h": int,              # H subdivisions per unit cell
    "ndiv_k": int,              # K subdivisions per unit cell
    "ndiv_l": int,              # L subdivisions per unit cell
    "max_rms_delta_hkl": float  # Warning threshold for Δhkl RMS (default 0.1)
}

CorrectedDiffusePixelData = {
    "q_vectors": numpy.ndarray,     # Shape (N, 3) lab-frame q-vectors
    "intensities": numpy.ndarray,   # Shape (N,) corrected intensities
    "sigmas": numpy.ndarray,        # Shape (N,) uncertainties
    "still_ids": numpy.ndarray      # Shape (N,) still identifiers
}
```

### Methods

```python
def hkl_to_voxel_idx(self, h: float, k: float, l: float) -> int
```

**Preconditions:** HKL values within grid boundaries
**Postconditions:** Returns unique voxel index for given HKL position
**Behavior:** Maps fractional Miller indices to linear voxel index using grid subdivisions

```python
def voxel_idx_to_hkl_center(self, voxel_idx: int) -> tuple[float, float, float]
```

**Preconditions:** `voxel_idx` is valid index within grid
**Postconditions:** Returns center HKL coordinates of voxel
**Behavior:** Inverse mapping from voxel index to fractional Miller indices

```python  
def get_q_vector_for_voxel_center(self, voxel_idx: int) -> scitbx.matrix.col
```

**Preconditions:** `voxel_idx` is valid index within grid
**Postconditions:** Returns lab-frame q-vector for voxel center
**Behavior:** Transforms voxel center HKL to lab frame using `A_avg_ref`

```python
def get_crystal_averaging_diagnostics(self) -> dict
```

**Preconditions:** Grid has been initialized
**Postconditions:** Returns diagnostic metrics for crystal model averaging quality
**Behavior:** Provides RMS Δhkl, RMS misorientation, and other quality metrics

**Expected Data Format:**
```python
diagnostics = {
    "rms_delta_hkl": float,           # RMS Δhkl for Bragg reflections
    "rms_misorientation_deg": float,   # RMS misorientation in degrees
    "n_crystals_averaged": int,        # Number of input crystal models
    "hkl_range_min": tuple,           # (h_min, k_min, l_min)
    "hkl_range_max": tuple,           # (h_max, k_max, l_max)
    "total_voxels": int               # Total number of voxels in grid
}
```

### Attributes

- `crystal_avg_ref`: dxtbx.model.Crystal - Average reference crystal model
- `A_avg_ref`: scitbx.matrix.sqr - Setting matrix for grid transformations
- `hkl_min`: tuple[int, int, int] - Minimum HKL boundaries  
- `hkl_max`: tuple[int, int, int] - Maximum HKL boundaries
- `ndiv_h, ndiv_k, ndiv_l`: int - Grid subdivisions per unit cell
- `total_voxels`: int - Total number of voxels in grid

### Error Conditions

**@raises_error(condition="InvalidGridConfig", message="Grid configuration validation failed")**
- Grid subdivisions ≤ 0
- Resolution limits invalid (d_min ≥ d_max)

**@raises_error(condition="InsufficientCrystalData", message="Cannot average crystal models")**  
- Empty experiment list
- Crystal models missing or invalid

**@raises_error(condition="ExcessiveCrystalVariation", message="Crystal model averaging quality poor")**
- RMS Δhkl exceeds warning threshold
- RMS misorientation exceeds warning threshold (logged warning, not exception)

## Implementation Notes

- Use CCTBX utilities for robust unit cell averaging
- Implement quaternion-based U matrix averaging for rotation matrices
- All HKL transformations use `A_avg_ref.inverse()` matrix
- Grid boundaries include buffer for resolution limits
- Voxel indexing uses linear mapping for memory efficiency