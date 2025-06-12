# Flex Array Manipulations

This documentation covers `dials.array_family.flex` array operations essential for crystallographic data processing and analysis, plus best practices for DIALS integration.

**Version Information:** Compatible with DIALS 3.x series. Some methods may differ in DIALS 2.x.

**Key Dependencies:**
- `dials.array_family.flex`: Core flex array types and operations
- `dxtbx.flumpy`: NumPy integration utilities
- `scitbx`: Scientific computing utilities
- `cctbx`: Crystallographic computing toolkit

## D.1. Converting Flex Arrays to NumPy Arrays

**1. Purpose:**
Convert DIALS flex arrays to NumPy arrays for compatibility with scientific Python ecosystem and integration with analysis libraries like scipy, matplotlib, and pandas.

**2. Primary Python Call(s):**
```python
from dials.array_family import flex
from dxtbx import flumpy
import numpy as np

# Method 1: Using as_numpy_array() (basic conversion)
flex_array = flex.double([1.0, 2.0, 3.0, 4.0])
numpy_array = flex_array.as_numpy_array()

# Method 2: Using flumpy (recommended for advanced types)
vec3_data = flex.vec3_double([(1,2,3), (4,5,6)])
numpy_array = flumpy.to_numpy(vec3_data)

# For vec3_double, extract components
x, y, z = vec3_data.parts()
x_np = x.as_numpy_array()
y_np = y.as_numpy_array()
z_np = z.as_numpy_array()
```

**3. Key Arguments:**
- No arguments for `as_numpy_array()`
- `flumpy.to_numpy()` handles complex flex types automatically

**4. Return Type:**
- `numpy.ndarray` with appropriate dtype (float64, int32, bool, etc.)

**5. Example Usage Snippet:**
```python
# Convert reflection table columns to NumPy
reflections = flex.reflection_table.from_file("reflections.refl")

# Simple columns
intensities = reflections["intensity.sum.value"].as_numpy_array()
variances = reflections["intensity.sum.variance"].as_numpy_array()

# Vector columns
xyz_cal = reflections["xyzcal.px"]
x, y, z = xyz_cal.parts()
positions = np.column_stack([x.as_numpy_array(), y.as_numpy_array(), z.as_numpy_array()])

print(f"Converted {len(intensities)} intensities to NumPy")
print(f"Position array shape: {positions.shape}")
```

**6. Notes/Caveats:**
- `as_numpy_array()` creates a view when possible, copy when necessary
- Complex types (vec3_double, miller_index) need component extraction
- Use `flumpy` for bidirectional conversion with proper type handling

---

## D.2. Converting NumPy Arrays to Flex Arrays

**1. Purpose:**
Convert NumPy arrays back to DIALS flex arrays for use with DIALS algorithms and maintaining compatibility with DIALS processing workflows.

**2. Primary Python Call(s):**
```python
import numpy as np
from dials.array_family import flex
from dxtbx import flumpy

# Method 1: Using flumpy (recommended)
numpy_array = np.array([1.0, 2.0, 3.0])
flex_array = flumpy.from_numpy(numpy_array)

# Method 2: Direct flex constructors
flex_double = flex.double(numpy_array)
flex_int = flex.int(numpy_array.astype(int))
flex_bool = flex.bool(numpy_array.astype(bool))

# For vector types
positions_np = np.array([[1,2,3], [4,5,6], [7,8,9]], dtype=float)
vec3_flex = flumpy.vec_from_numpy(positions_np)

# Create Miller indices from NumPy
hkl_np = np.array([[1,0,0], [2,0,0], [0,0,1]], dtype=int)
miller_flex = flex.miller_index(hkl_np)
```

**3. Key Arguments:**
- `numpy_array`: Input NumPy array with compatible dtype
- For `flex.miller_index()`: requires integer array with shape (N, 3)

**4. Return Type:**
- Appropriate flex array type (flex.double, flex.int, flex.vec3_double, etc.)

**5. Example Usage Snippet:**
```python
# Create computed data in NumPy and convert to flex
computed_intensities = np.random.exponential(1000, size=10000)
computed_positions = np.random.rand(10000, 3) * 100

# Convert to flex for DIALS
flex_intensities = flumpy.from_numpy(computed_intensities)
flex_positions = flumpy.vec_from_numpy(computed_positions)

# Add to reflection table
new_reflections = flex.reflection_table()
new_reflections["intensity.computed"] = flex_intensities
new_reflections["xyz.computed"] = flex_positions

print(f"Created reflection table with {len(new_reflections)} computed reflections")
```

**6. Notes/Caveats:**
- NumPy array dtype must be compatible with target flex type
- `flumpy` handles type conversion automatically
- Vector types require shape (N, 3) for vec3_double

---

## D.3. Accessing Elements and Properties of flex.reflection_table

**1. Purpose:**
Navigate and manipulate DIALS reflection table data for diffuse scattering analysis, including filtering, statistical analysis, and data export operations.

**2. Primary Python Call(s):**
```python
from dials.array_family import flex

# Create or load reflection table
reflections = flex.reflection_table.from_file("reflections.refl")

# Basic table properties
num_reflections = len(reflections)
column_names = list(reflections.keys())
has_column = "intensity.sum.value" in reflections

# Access columns
miller_indices = reflections["miller_index"]
intensities = reflections["intensity.sum.value"] 
pixel_positions = reflections["xyzcal.px"]

# Column statistics
mean_intensity = flex.mean(intensities)
intensity_range = (flex.min(intensities), flex.max(intensities))

# Table operations
reflections.extend(other_reflections)  # Concatenate tables
subset = reflections[:1000]            # Slice first 1000 rows
reflections.del_selected(bad_mask)     # Delete rows
```

**3. Key Methods:**
- `len(table)` → int: Number of reflections
- `table.keys()` → list: Column names
- `table[column]` → flex array: Access column data
- `table.extend(other)`: Concatenate reflection tables
- `table.select(selection)` → table: Select subset based on boolean mask

**4. Return Types:**
- Columns return appropriate flex array types
- Operations return modified reflection tables

**5. Example Usage Snippet:**
```python
reflections = flex.reflection_table.from_file("integrated.refl")

print(f"Loaded {len(reflections)} reflections")
print(f"Columns: {list(reflections.keys())}")

# Access key data for diffuse scattering
if "intensity.sum.value" in reflections:
    intensities = reflections["intensity.sum.value"]
    strong_mask = intensities > flex.mean(intensities)
    strong_reflections = reflections.select(strong_mask)
    print(f"Found {len(strong_reflections)} strong reflections")

# Work with positions
if "xyzcal.px" in reflections:
    positions = reflections["xyzcal.px"]
    x, y, z = positions.parts()
    print(f"Position range: X({flex.min(x):.1f}-{flex.max(x):.1f}), "
          f"Y({flex.min(y):.1f}-{flex.max(y):.1f})")
```

**6. Notes/Caveats:**
- Column availability depends on processing stage (find_spots, index, integrate, scale)
- Some operations modify the table in-place
- Use boolean masks with `select()` for conditional operations

---

## D.4. Common Operations on Flex Arrays

**1. Purpose:**
Perform mathematical and logical operations on flex arrays for data analysis, statistical calculations, and array manipulation in diffuse scattering processing pipelines.

**2. Primary Python Call(s):**
```python
from dials.array_family import flex
import math

# Mathematical operations
a = flex.double([1, 2, 3, 4, 5])
b = flex.double([2, 3, 4, 5, 6])

# Element-wise arithmetic
result = a + b           # Addition
result = a * 2.0         # Scalar multiplication  
result = a / b           # Element-wise division
result = flex.pow(a, 2)  # Power function

# Mathematical functions
result = flex.sqrt(a)
result = flex.exp(a)
result = flex.log(a)
result = flex.sin(a)

# Statistical operations
mean_val = flex.mean(a)
std_dev = flex.mean_and_variance(a).unweighted_sample_standard_deviation()
min_val = flex.min(a)
max_val = flex.max(a)

# Boolean operations and selection
mask = a > 3.0
selected = a.select(mask)
count = flex.sum(mask.as_1d())
```

**3. Key Methods:**
- Arithmetic: `+`, `-`, `*`, `/` (element-wise)
- Functions: `flex.sqrt()`, `flex.exp()`, `flex.log()`, `flex.sin()`, `flex.cos()`
- Statistics: `flex.mean()`, `flex.min()`, `flex.max()`, `flex.sum()`
- Selection: `array.select(mask)` where mask is flex.bool

**4. Return Types:**
- Mathematical operations return flex arrays of appropriate type
- Statistics return Python scalars
- Selection returns flex array subset

**5. Example Usage Snippet:**
```python
# Analyze intensity distribution
intensities = reflections["intensity.sum.value"]
log_intensities = flex.log(intensities)

# Calculate signal-to-noise ratio
variances = reflections["intensity.sum.variance"]
signal_to_noise = intensities / flex.sqrt(variances)

# Find outliers
mean_snr = flex.mean(signal_to_noise)
std_snr = flex.mean_and_variance(signal_to_noise).unweighted_sample_standard_deviation()
outlier_mask = flex.abs(signal_to_noise - mean_snr) > 3 * std_snr

print(f"Mean S/N: {mean_snr:.2f} ± {std_snr:.2f}")
print(f"Found {flex.sum(outlier_mask.as_1d())} outliers")

# Apply corrections
corrected_intensities = intensities * correction_factors
reflections["intensity.corrected"] = corrected_intensities
```

**6. Notes/Caveats:**
- All arrays in operations must have compatible sizes
- Mathematical functions expect appropriate input ranges (e.g., log(x) requires x > 0)
- Use `as_1d()` to convert boolean masks to size_t for counting

---

## Key Tips and Best Practices

1. **Always check experiment validity** before accessing models:
   ```python
   if experiment.detector is not None and experiment.beam is not None:
       # Safe to proceed with calculations
   ```

2. **Use flumpy for NumPy integration** rather than manual conversion methods:
   ```python
   from dxtbx import flumpy
   numpy_array = flumpy.to_numpy(flex_array)
   flex_array = flumpy.from_numpy(numpy_array)
   ```

3. **Handle multi-panel detectors correctly**:
   ```python
   raw_data = imageset.get_raw_data(0)
   if isinstance(raw_data, tuple):
       for panel_id, panel_data in enumerate(raw_data):
           # Process each panel separately
   ```

4. **Check column existence before accessing**:
   ```python
   if "intensity.sum.value" in reflections:
       intensities = reflections["intensity.sum.value"]
   ```

5. **Use matrix operations for coordinate transformations**:
   ```python
   from scitbx import matrix
   A_matrix = matrix.sqr(crystal.get_A())
   q_vector = A_matrix * matrix.col(hkl)
   ```

6. **Apply geometric corrections properly**:
   ```python
   # Corrections are typically applied as divisors
   corrected_intensity = raw_intensity / lp_correction
   ```

---

## Summary: Integration with Diffuse Scattering Pipeline

This API reference maps directly to the modules outlined in your `plan.md`:

**Module 1.1 (Data Loading):** Sections A.1-A.4 provide all file I/O operations
**Module 2.1.D (Q-space Mapping):** Sections C.1-C.3 for coordinate transformations
**Module 2.2.D (Geometric Corrections):** Section C.4 for all intensity corrections (Solid Angle, Lorentz-Polarization, Detector Efficiency, Air Attenuation)
**Module 3.0.D (Crystal Frame Analysis):** Sections B.3, C.2 for crystal coordinate systems
**Module 3.1.D (Lattice Analysis):** Sections C.2-C.3, D.3 for Miller index operations
**Module 4.1.D (Theoretical Comparison):** Section C.5 for scattering factor calculations

**Common Workflow Pattern:**
```python
# 1. Load experimental data
experiments = ExperimentListFactory.from_json_file("experiments.expt")
reflections = flex.reflection_table.from_file("reflections.refl")
imageset = experiments[0].imageset

# 2. Extract geometry models
detector = experiments[0].detector
beam = experiments[0].beam
crystal = experiments[0].crystal

# 3. Process pixel data with geometric corrections
for panel_id, panel in enumerate(detector):
    for frame_idx in range(len(imageset)):
        raw_data = imageset.get_raw_data(frame_idx)
        panel_data = raw_data[panel_id] if isinstance(raw_data, tuple) else raw_data
        
        # Apply diffuse scattering analysis using APIs from sections C.1-C.4
```

**Official Documentation Links:**
- [DIALS Documentation](https://dials.github.io/documentation/)
- [dxtbx Documentation](https://dxtbx.readthedocs.io/)
- [cctbx Documentation](https://cctbx.sourceforge.net/)

**Error Handling Best Practices:**
- Always check model existence: `if experiment.goniometer is not None:`
- Use try/except blocks for file operations with `dials.util.Sorry` exceptions
- Validate array sizes before mathematical operations
- Handle edge cases in coordinate transformations (beam center, panel edges)

---

## See Also

- **File I/O Operations**: [dials_file_io.md](dials_file_io.md)
- **Detector Models**: [dxtbx_models.md](dxtbx_models.md)
- **Crystallographic Calculations**: [crystallographic_calculations.md](crystallographic_calculations.md)
- **DIALS Scaling**: [dials_scaling.md](dials_scaling.md)