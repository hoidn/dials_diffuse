# DiffuseDataMerger IDL

**Module Path:** `src.diffusepipe.merging.merger`

**Dependencies:**
- `@depends_on(numpy)` - Array operations and structured arrays
- `@depends_on(GlobalVoxelGrid)` - Voxel grid definitions
- `@depends_on(DiffuseScalingModel)` - Refined scaling parameters
- `@depends_on(VoxelAccumulator)` - Binned pixel data

## Interface: DiffuseDataMerger

**Purpose:** Apply refined scaling model parameters to all observations and merge them within each voxel of the GlobalVoxelGrid. Produces final relatively-scaled 3D diffuse scattering map with proper error propagation.

### Constructor

```python
def __init__(self, 
            global_voxel_grid: GlobalVoxelGrid) -> None
```

**Preconditions:**
- `global_voxel_grid` is valid and initialized

**Postconditions:**
- Merger configured with grid information
- Ready to process binned data and scaling models

**Behavior:**
- Store reference to voxel grid for coordinate calculations
- Initialize data structures for merged results

### Methods

```python
def merge_scaled_data(self, 
                     binned_pixel_data: dict,
                     scaling_model: DiffuseScalingModel,
                     merge_config: dict) -> VoxelDataRelative
```

**Preconditions:**
- `binned_pixel_data` contains observations organized by voxel
- `scaling_model` has refined parameters from Module 3.S.3
- All observations have valid intensity and uncertainty values

**Postconditions:**
- Returns merged data structure with relatively-scaled intensities
- Error propagation applied correctly for v1 model constraints
- Voxel centers calculated using grid transformations

**Behavior:**
1. For each observation, retrieve refined scaling parameters from model
2. Apply scaling: `I_final_relative = (I_corr - C_i) / M_i`
3. Propagate uncertainties: `Sigma_final_relative = Sigma_corr / abs(M_i)`
4. Verify additive offset C_i is effectively zero for v1 model
5. Perform weighted merge within each voxel using inverse variance weighting
6. Calculate voxel center coordinates and q-vector attributes

**Expected Data Format:**
```python
merge_config = {
    "outlier_rejection": {
        "enabled": bool,              # Apply outlier filtering
        "sigma_threshold": float      # Reject observations >N sigma from voxel mean
    },
    "minimum_observations": int,      # Minimum observations required per voxel
    "weight_method": str              # "inverse_variance" or "uniform"
}

# Returns:
VoxelDataRelative = {
    "voxel_indices": numpy.ndarray,     # Shape (N_voxels,) - voxel indices
    "H_center": numpy.ndarray,          # Shape (N_voxels,) - H coordinates of centers  
    "K_center": numpy.ndarray,          # Shape (N_voxels,) - K coordinates of centers
    "L_center": numpy.ndarray,          # Shape (N_voxels,) - L coordinates of centers
    "q_center_x": numpy.ndarray,        # Shape (N_voxels,) - qx lab frame
    "q_center_y": numpy.ndarray,        # Shape (N_voxels,) - qy lab frame  
    "q_center_z": numpy.ndarray,        # Shape (N_voxels,) - qz lab frame
    "q_magnitude_center": numpy.ndarray, # Shape (N_voxels,) - |q| of centers
    "I_merged_relative": numpy.ndarray,  # Shape (N_voxels,) - merged intensities
    "Sigma_merged_relative": numpy.ndarray, # Shape (N_voxels,) - merged uncertainties
    "num_observations": numpy.ndarray    # Shape (N_voxels,) - observation counts
}
```

```python
def apply_scaling_to_observation(self,
                                observation: dict,
                                scaling_model: DiffuseScalingModel) -> tuple[float, float]
```

**Preconditions:**
- `observation` contains intensity, sigma, still_id, q_vector_lab
- `scaling_model` has refined parameters

**Postconditions:**
- Returns scaled intensity and propagated uncertainty
- Scaling applied according to v1 model formulation

**Behavior:**
1. Extract observation properties (still_id, q_magnitude)
2. Get multiplicative scale M_i and additive offset C_i from model
3. Apply v1 scaling formula: `I_scaled = (I_obs - C_i) / M_i`
4. Propagate uncertainty: `Sigma_scaled = Sigma_obs / abs(M_i)`
5. Verify C_i ≈ 0 for v1 model consistency

**Expected Data Format:**
```python
observation = {
    "intensity": float,
    "sigma": float,
    "still_id": int,
    "q_vector_lab": numpy.ndarray  # Shape (3,)
}

# Returns: (scaled_intensity, scaled_sigma)
(123.45, 12.34)
```

```python
def weighted_merge_voxel(self,
                        scaled_observations: list[tuple],
                        weight_method: str = "inverse_variance") -> tuple[float, float, int]
```

**Preconditions:**
- `scaled_observations` is list of (intensity, sigma) tuples
- All sigma values are positive
- At least one observation provided

**Postconditions:**
- Returns merged intensity, merged uncertainty, and observation count
- Weighting applied according to specified method

**Behavior:**
1. Calculate weights based on method (inverse variance or uniform)
2. Compute weighted average intensity
3. Propagate uncertainties using standard error propagation
4. Handle edge cases (single observation, zero weights)

**Expected Data Format:**
```python
# Input: list of (intensity, sigma) pairs
scaled_observations = [(100.0, 10.0), (120.0, 15.0), (90.0, 8.0)]

# Returns: (merged_intensity, merged_sigma, n_observations)
(105.23, 6.12, 3)
```

```python
def calculate_voxel_coordinates(self,
                               voxel_indices: list[int]) -> dict[str, numpy.ndarray]
```

**Preconditions:**
- `voxel_indices` contains valid indices within grid bounds
- Global voxel grid is properly initialized

**Postconditions:**
- Returns coordinate arrays for voxel centers
- All arrays have same length as input voxel_indices

**Behavior:**
1. For each voxel index, get center HKL coordinates
2. Transform HKL to lab-frame q-vectors using grid A_avg_ref
3. Calculate q-magnitude for each center
4. Return structured coordinate data

**Expected Data Format:**
```python
# Returns:
coordinates = {
    "H_center": numpy.ndarray,          # Fractional H coordinates
    "K_center": numpy.ndarray,          # Fractional K coordinates  
    "L_center": numpy.ndarray,          # Fractional L coordinates
    "q_center_x": numpy.ndarray,        # Lab frame qx
    "q_center_y": numpy.ndarray,        # Lab frame qy
    "q_center_z": numpy.ndarray,        # Lab frame qz
    "q_magnitude_center": numpy.ndarray  # |q| magnitudes
}
```

```python
def get_merge_statistics(self, voxel_data_relative: VoxelDataRelative) -> dict
```

**Preconditions:**
- `voxel_data_relative` is valid merged dataset
- Contains required arrays with consistent lengths

**Postconditions:**
- Returns comprehensive statistics about merged data
- Useful for quality assessment and diagnostics

**Behavior:**
- Calculate intensity and observation distribution statistics  
- Analyze resolution coverage and redundancy
- Compute data completeness metrics

**Expected Data Format:**
```python
statistics = {
    "total_voxels": int,
    "intensity_statistics": {
        "mean": float,
        "std": float,
        "min": float,
        "max": float,
        "median": float
    },
    "observation_statistics": {
        "mean_per_voxel": float,
        "total_observations": int,
        "voxels_with_single_obs": int,
        "max_observations_per_voxel": int
    },
    "resolution_coverage": {
        "q_min": float,
        "q_max": float,
        "mean_q": float
    },
    "data_quality": {
        "mean_sigma_over_intensity": float,
        "high_intensity_voxels": int,  # Above mean + 2*std
        "low_sigma_voxels": int        # Good precision voxels
    }
}
```

### Error Conditions

**@raises_error(condition="InvalidScalingModel", message="Scaling model validation failed")**
- Scaling model missing required components
- Parameters not properly refined

**@raises_error(condition="InconsistentDataFormat", message="Data format validation failed")**
- Missing required fields in binned data
- Array length mismatches

**@raises_error(condition="V1ModelViolation", message="v1 model constraints violated")**
- Additive offset C_i is not effectively zero
- Unexpected scaling model structure

**@raises_error(condition="MergeNumericalError", message="Numerical error in merging")**
- All weights are zero or negative
- Invalid uncertainty propagation

## Implementation Notes

- Use inverse variance weighting as default merging method
- Implement numerical safeguards for edge cases (single observations, zero weights)
- Verify v1 model constraints (C_i ≈ 0) with explicit checks
- Use vectorized operations for efficient coordinate calculations
- Handle memory efficiently for large voxel datasets
- Provide comprehensive statistics for data quality assessment
- Support structured array or dictionary output formats