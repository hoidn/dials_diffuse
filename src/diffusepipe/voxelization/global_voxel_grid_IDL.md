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
- Stores the provided `ExperimentList`/scan-varying model for frame-specific transformations
- `crystal_avg_ref` is computed as robust average of all input crystal models  
- **Critical:** The average model (`A_avg_ref`) is computed **solely** for defining grid boundaries and resolution limits. This average model **must not** be used for transforming observation data.
- HKL range covers all diffuse data within resolution limits
- Grid subdivision parameters are stored
- Diagnostic metrics computed for crystal model averaging quality

**Behavior:**
1. Store the full `experiment_list` for scan-varying crystal model access
2. Robustly average unit cell parameters using CCTBX utilities
3. Average U matrices using quaternion-based method for rotation matrices  
4. Compute `A_avg_ref = U_avg_ref * B_avg_ref` setting matrix (for grid bounds only)
5. Calculate RMS Δhkl diagnostic for Bragg reflections
6. Calculate RMS misorientation diagnostic between individual U matrices
7. Transform all q-vectors to fractional HKL to determine grid boundaries
8. Store grid parameters and conversion methods

**@raises_error(condition="ExcessiveMisorientation", description="Raised if the RMS misorientation between input crystal models exceeds the safety threshold (e.g., 2.0°), indicating the stills are not suitable for merging.")**

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
def get_A_inverse_for_frame(self, frame_index: int) -> numpy.ndarray
```

**Preconditions:** `frame_index` is a valid 0-based scan point index
**Postconditions:** Returns 3x3 numpy array representing A^(-1) for the frame
**Behavior:** Retrieves scan-varying crystal orientation matrix for specific frame and computes its inverse. Caches results for performance.

**@raises_error(condition="FrameIndexOutOfBounds", description="Raised if frame_index exceeds the scan range")**
**@raises_error(condition="NoScanVaryingModel", description="Raised if experiment does not contain scan-varying crystal model")**

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

**@raises_error(condition="ExcessiveMisorientation", message="Input stills are not suitable for merging")**
- RMS misorientation between crystal models exceeds 2.0° threshold
- Prevents memory overload from creating excessively large voxel grids

## Implementation Notes

- Use CCTBX utilities for robust unit cell averaging
- Implement quaternion-based U matrix averaging for rotation matrices
- All HKL transformations use `A_avg_ref.inverse()` matrix
- Grid boundaries include buffer for resolution limits
- Voxel indexing uses linear mapping for memory efficiency