# DiffuseScalingModel IDL

**Module Path:** `src.diffusepipe.scaling.diffuse_scaling_model`

**Dependencies:**
- `@depends_on(dials.algorithms.scaling.model.ScalingModelBase)` - Base class for scaling models
- `@depends_on(dials.algorithms.scaling.model.components.scale_components.ScaleComponentBase)` - Component framework
- `@depends_on(dials.algorithms.scaling.active_parameter_managers)` - Parameter management
- `@depends_on(dials.array_family.flex)` - Reflection tables and arrays
- `@depends_on(numpy)` - Array operations for diffuse data

## Interface: DiffuseScalingModel(ScalingModelBase)

**Purpose:** Custom scaling model for relative scaling of diffuse scattering data. Implements v1 parameter-guarded model with per-still multiplicative scales and optional resolution smoothing. Enforces hard limits on model complexity.

### Constructor

```python
def __init__(self, 
            configdict: dict,
            is_scaled: bool = False) -> None
```

**Preconditions:**
- `configdict` contains valid configuration for scaling components
- Total free parameters ≤ MAX_FREE_PARAMS (5 + N_stills)
- Advanced components disabled in v1 unless experimental flag set

**Postconditions:**
- Scaling components initialized according to v1 constraints
- Parameter manager configured with refineable parameters
- Error raised if configuration violates v1 parameter limits

**Behavior:**
1. Validate configuration against v1 parameter limits
2. Initialize enabled components (per-still multiplier, optional resolution smoother)
3. Set up active parameter manager for refinement
4. Configure restraints for smoother parameters if enabled
5. Initialize reference generation capabilities

**Expected Data Format:**
```python
configdict = {
    "per_still_scale": {
        "enabled": bool,           # Should be True for v1
        "initial_values": dict     # still_id -> initial scale (default 1.0)
    },
    "resolution_smoother": {
        "enabled": bool,           # False by default in v1
        "n_control_points": int,   # ≤5 for v1, raises error if exceeded
        "resolution_range": tuple  # (q_min, q_max) for smoother domain
    },
    "experimental_components": {
        "panel_scale": {"enabled": False},      # Hard-disabled in v1
        "spatial_scale": {"enabled": False},    # Hard-disabled in v1
        "additive_offset": {"enabled": False}   # Hard-disabled in v1
    },
    "partiality_threshold": float  # P_min_thresh for Bragg reference (default 0.1)
}
```

### Methods

```python
def refine_parameters(self, 
                     binned_pixel_data: dict,
                     bragg_reflections: dict,
                     refinement_config: dict) -> tuple[dict, dict]
```

**Preconditions:**
- `binned_pixel_data` contains voxel-organized diffuse observations
- `bragg_reflections` contains reflection tables with partiality column
- Model components are properly initialized

**Postconditions:**
- Model parameters refined via iterative minimization
- Convergence achieved or maximum iterations reached
- Returns refined parameters and refinement statistics

**Behavior:**
1. Set up iterative refinement loop with DIALS minimizer
2. Generate Bragg and diffuse references from current parameters
3. Calculate residuals using custom target function
4. Update parameters via Levenberg-Marquardt or similar
5. Check convergence and repeat until criteria met

**Expected Data Format:**
```python
refinement_config = {
    "max_iterations": int,         # Maximum refinement cycles
    "convergence_tolerance": float, # R-factor change threshold
    "minimizer_type": str          # "lm" or "gauss_newton"
}

# Returns:
refined_parameters = {
    still_id: {
        "multiplicative_scale": float,
        "additive_offset": float       # Always 0.0 in v1
    }
}

refinement_statistics = {
    "n_iterations": int,
    "final_r_factor": float,
    "convergence_achieved": bool,
    "parameter_shifts": dict       # Component -> max parameter change
}
```

```python
def get_scales_for_observation(self, 
                              still_id: int,
                              q_magnitude: float,
                              **kwargs) -> tuple[float, float]
```

**Preconditions:**
- `still_id` is valid and has refined parameters
- `q_magnitude` is positive
- Model parameters have been refined

**Postconditions:**
- Returns multiplicative scale and additive offset for observation
- Values computed from current model state

**Behavior:**
1. Retrieve per-still multiplicative scale from component
2. Apply resolution-dependent correction if smoother enabled
3. Calculate additive offset (always 0.0 in v1)
4. Return combined scale factors

**Expected Data Format:**
```python
# Returns: (multiplicative_scale, additive_offset)
(1.23, 0.0)  # additive_offset always 0.0 in v1
```

```python
def generate_references(self, 
                       binned_pixel_data: dict,
                       bragg_reflections: dict) -> tuple[dict, dict]
```

**Preconditions:**
- Current model parameters available
- Input data properly formatted with all required fields

**Postconditions:**
- Returns Bragg and diffuse reference intensities
- References computed using current scale estimates

**Behavior:**
1. Filter Bragg reflections by partiality threshold (P_spot ≥ P_min_thresh)
2. Calculate Bragg reference as weighted average of scaled intensities
3. Calculate diffuse reference for each voxel from scaled observations
4. Apply quality filters and outlier rejection

**Expected Data Format:**
```python
# Returns: (bragg_references, diffuse_references)
bragg_references = {
    hkl_asu: {
        "intensity": float,
        "sigma": float,
        "n_observations": int
    }
}

diffuse_references = {
    voxel_idx: {
        "intensity": float,
        "sigma": float, 
        "n_observations": int
    }
}
```

### Components

**PerStillMultiplierComponent:**
- Exactly one free parameter per still: `b_i`
- Initialized to 1.0 for all stills
- Enforced to be positive during refinement

**ResolutionSmootherComponent (Optional):**
- 1D Gaussian smoother over |q| with ≤5 control points
- Shared by all stills (multiplicative factor)
- Disabled by default in v1 configuration

### Error Conditions

**@raises_error(condition="ParameterLimitExceeded", message="v1 model parameter limit exceeded")**
- Total free parameters > MAX_FREE_PARAMS (5 + N_stills)
- Resolution smoother control points > 5

**@raises_error(condition="DisabledComponentAccess", message="Advanced component disabled in v1")**
- Attempt to enable panel, spatial, or additive components
- Configuration requests forbidden functionality

**@raises_error(condition="RefinementFailure", message="Parameter refinement failed")**
- Minimizer convergence failure
- Invalid parameter values during refinement

## Interface: PerStillMultiplierComponent(ScaleComponentBase)

**Purpose:** Simple multiplicative scale factor component with one parameter per still.

### Constructor

```python
def __init__(self,
            active_parameter_manager: ActiveParameterManager,
            still_ids: list[int],
            initial_values: dict = None) -> None
```

**Preconditions:**
- `still_ids` contains unique identifiers for all stills
- `initial_values` if provided maps still_id to positive float

**Postconditions:**
- One parameter registered per still in parameter manager
- Initial values set (default 1.0)

**Behavior:**
- Add one parameter per still to active parameter manager
- Initialize with unity scale or provided values
- Set up parameter indexing for efficient access

### Methods

```python
def calculate_scales_and_derivatives(self, 
                                   reflection_table: flex.reflection_table,
                                   block_id: int = None) -> tuple[flex.double, flex.double]
```

**Preconditions:**
- Reflection table contains 'still_id' column
- Parameters have been set in parameter manager

**Postconditions:**
- Returns scale factors and derivatives for all reflections
- Arrays match reflection table length

**Behavior:**
- Look up scale parameter for each reflection's still_id
- Return scales and unit derivatives (d(scale)/d(parameter) = 1.0)

## Interface: ResolutionSmootherComponent(ScaleComponentBase)

**Purpose:** 1D Gaussian smoother for resolution-dependent multiplicative scaling.

### Constructor  

```python
def __init__(self,
            active_parameter_manager: ActiveParameterManager,
            n_control_points: int,
            resolution_range: tuple[float, float]) -> None
```

**Preconditions:**
- `n_control_points` ≤ 5 for v1 compliance
- `resolution_range` is (q_min, q_max) with q_min < q_max

**Postconditions:**
- Control points distributed over resolution range
- Smoother initialized with unity values
- Restraints set up for smoothness

**Behavior:**
- Create 1D Gaussian smoother with specified control points
- Distribute control points evenly over |q| range
- Initialize control point values to 1.0 (unity scaling)
- Configure smoothness restraints

### Methods

```python
def calculate_scales_and_derivatives(self,
                                   reflection_table: flex.reflection_table,
                                   block_id: int = None) -> tuple[flex.double, flex.double]
```

**Preconditions:**
- Reflection table contains q-vector information or d-spacing
- Smoother parameters have been set

**Postconditions:**
- Returns resolution-dependent scale factors
- Derivatives computed for parameter refinement

**Behavior:**
- Calculate |q| for each reflection
- Evaluate Gaussian smoother at q-values
- Return scales and derivatives w.r.t. control point parameters

## Implementation Notes

- MAX_FREE_PARAMS = 5 + N_stills enforced at initialization
- All additive components return 0.0 in v1 implementation
- Parameter manager handles constraints and bounds
- Uses DIALS Levenberg-Marquardt minimizer for refinement
- Reference generation filters by P_spot ≥ P_min_thresh
- Convergence based on R-factor change between iterations