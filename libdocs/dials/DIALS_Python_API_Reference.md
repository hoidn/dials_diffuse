# DIALS Python API Reference for Diffuse Scattering Pipeline

This documentation provides precise API usage for DIALS/`dxtbx`/`cctbx` Python libraries needed for implementing diffuse scattering data processing modules. It covers file I/O, model access, crystallographic calculations, and data manipulation operations.

**Version Information:** Compatible with DIALS 3.x series. Some methods may differ in DIALS 2.x.

**Key Dependencies:**
- `dxtbx`: Detector models, beam models, image handling
- `dials.array_family.flex`: Reflection tables and array operations  
- `cctbx`: Unit cells, space groups, scattering factors
- `scitbx`: Matrix operations and mathematical utilities

---

## A. File I/O and Model Loading

### A.1. Loading Experiment Lists (`.expt` files)

**1. Purpose:**
Load DIALS experiment JSON files containing detector geometry, beam parameters, crystal orientation, and scan information for diffuse scattering pipeline modules that require geometric models and experimental metadata.

**2. Primary Python Call(s):**
```python
from dxtbx.model.experiment_list import ExperimentListFactory
from dxtbx.serialize import load

# Method 1: Using ExperimentListFactory (recommended)
experiments = ExperimentListFactory.from_json_file(
    filename="indexed_refined_detector.expt",
    check_format=True
)

# Method 2: Using generic load function
experiments = load.experiment_list("indexed_refined_detector.expt")

# Method 3: Direct from ExperimentList class
from dxtbx.model import ExperimentList
experiments = ExperimentList.from_file("indexed_refined_detector.expt")
```

**3. Key Arguments:**
- `filename` (str): Path to the `.expt` file
- `check_format` (bool, optional, default=True): Whether to perform format validation

**4. Return Type:**
- `dxtbx.model.experiment_list.ExperimentList` - A list-like container for `Experiment` objects with detector, beam, crystal, goniometer, scan, and imageset models.

**5. Example Usage Snippet:**
```python
experiments = ExperimentListFactory.from_json_file("models.expt")
if experiments:
    first_experiment = experiments[0]
    detector = first_experiment.detector
    beam = first_experiment.beam
    crystal = first_experiment.crystal
    print(f"Loaded {len(experiments)} experiments. First detector has {len(detector)} panels.")
```

**6. Notes/Caveats:**
- Requires DIALS environment to be properly sourced
- The file must contain valid JSON experiment data in DIALS format
- Multi-experiment files (e.g., from multi-lattice indexing) return multiple experiments; select appropriate one with `experiments[index]`
- Use error handling for robust file loading

**7. Error Handling Example:**
```python
from dials.util import Sorry

try:
    experiments = ExperimentListFactory.from_json_file(filename)
    if not experiments:
        raise Sorry(f"No experiments found in {filename}")
except (FileNotFoundError, IOError) as e:
    raise Sorry(f"Could not load experiments from {filename}: {e}")
except Exception as e:
    raise Sorry(f"Invalid experiment file format: {e}")
```

**8. See Also:**
- Section B: Accessing dxtbx.model Objects for working with loaded experiments
- Section A.2: Loading reflection tables to pair with experiments

---

### A.2. Loading Reflection Tables (`.refl` files)

**1. Purpose:**
Load DIALS reflection table files containing spot positions, intensities, Miller indices, and other crystallographic data for diffuse scattering analysis and intensity processing.

**2. Primary Python Call(s):**
```python
from dials.array_family import flex
from dials.serialize import load

# Method 1: Direct from reflection_table class
reflections = flex.reflection_table.from_file("indexed_refined_detector.refl")

# Method 2: Using generic load function
reflections = load.reflections("indexed_refined_detector.refl")
```

**3. Key Arguments:**
- `filename` (str): Path to the `.refl` file

**4. Return Type:**
- `dials.array_family.flex.reflection_table` - A dictionary-like container with flex arrays for each column (miller_index, intensity, xyzcal.px, etc.)

**5. Example Usage Snippet:**
```python
reflections = flex.reflection_table.from_file("reflections.refl")
print(f"Loaded {len(reflections)} reflections")
print(f"Available columns: {list(reflections.keys())}")
miller_indices = reflections['miller_index']
intensities = reflections['intensity.sum.value']
```

**6. Notes/Caveats:**
- Returns empty table if file doesn't exist rather than throwing an error
- Column names depend on processing stage (e.g., indexed, integrated, scaled)
- Can also load `.json` reflection files, though `.refl` (pickle) format is preferred for performance
- Common flag selections: `reflections.get_flags(reflections.flags.indexed)` for indexed reflections

**7. Error Handling and Flag Selection Example:**
```python
from dials.util import Sorry

try:
    reflections = flex.reflection_table.from_file(filename)
    if len(reflections) == 0:
        raise Sorry(f"No reflections found in {filename}")
    
    # Select only indexed reflections
    indexed_sel = reflections.get_flags(reflections.flags.indexed)
    indexed_reflections = reflections.select(indexed_sel)
    print(f"Loaded {len(indexed_reflections)} indexed reflections")
    
except (FileNotFoundError, IOError) as e:
    raise Sorry(f"Could not load reflections from {filename}: {e}")
```

**8. See Also:**
- Section D.3: Accessing reflection table properties and operations
- Section A.1: Loading experiments to pair with reflections

---

### A.3. Loading Image Data via ImageSet

**1. Purpose:**
Create ImageSet objects from image file paths to access raw pixel data for diffuse scattering analysis, background subtraction, and pixel-level intensity extraction.

**2. Primary Python Call(s):**
```python
from dxtbx.imageset import ImageSetFactory
from dxtbx import load

# Method 1: Create from file list
imageset = ImageSetFactory.make_imageset(
    filenames=["image_001.cbf", "image_002.cbf"],
    format_kwargs=None
)

# Method 2: Access from experiment
imageset = experiment.imageset

# Method 3: Load single image format
format_instance = load("image_001.cbf")
```

**3. Key Arguments:**
- `filenames` (list of str): List of image file paths
- `format_kwargs` (dict, optional): Format-specific parameters

**4. Return Type:**
- `dxtbx.imageset.ImageSet` - Container providing access to raw and corrected image data

**5. Example Usage Snippet:**
```python
imageset = experiment.imageset
for i in range(len(imageset)):
    raw_data = imageset.get_raw_data(i)  # Returns tuple for multi-panel
    if isinstance(raw_data, tuple):
        for panel_id, panel_data in enumerate(raw_data):
            print(f"Frame {i}, Panel {panel_id}: {panel_data.all()}")
```

**6. Notes/Caveats:**
- Multi-panel detectors return data as tuple of flex arrays
- Image data is returned as `flex.int` or `flex.double` arrays
- Use `get_corrected_data()` for pedestal/gain-corrected data
- Modern DIALS uses `ImageSetFactory.new()` and `make_imageset()` methods
- Access exposure times via: `imageset.get_scan().get_exposure_times()[frame_index]` if scan available
- Get format object: `format_class = imageset.get_format_class()`

**7. Extended Usage Examples:**
```python
# Access exposure times and format information
imageset = experiment.imageset
if hasattr(imageset, 'get_scan') and imageset.get_scan():
    scan = imageset.get_scan()
    exposure_times = scan.get_exposure_times()
    print(f"Frame 0 exposure time: {exposure_times[0]:.3f} s")

# Get format-specific metadata
format_class = imageset.get_format_class()
format_instance = format_class.get_instance(imageset.get_path(0))
print(f"Format: {format_class.__name__}")

# Handle multi-panel data correctly
for frame_idx in range(len(imageset)):
    raw_data = imageset.get_raw_data(frame_idx)
    if isinstance(raw_data, tuple):
        # Multi-panel detector
        for panel_id, panel_data in enumerate(raw_data):
            print(f"Frame {frame_idx}, Panel {panel_id}: {panel_data.all()}")
    else:
        # Single panel detector
        print(f"Frame {frame_idx}: {raw_data.all()}")
```

**8. See Also:**
- Section B.1: Detector model for coordinate transformations
- Section A.4: Loading masks to apply to image data

---

### A.4. Loading Mask Files (`.pickle` files)

**1. Purpose:**
Load detector masks that define valid/invalid pixel regions for diffuse scattering analysis.

**2. Primary Python Call(s):**
```python
import pickle
from dials.array_family import flex

# Load mask file
with open("mask.pickle", "rb") as f:
    mask_data = pickle.load(f)
```

**3. Key Arguments:**
- Standard Python pickle.load() arguments

**4. Return Type:**
- Tuple of `dials.array_family.flex.bool` arrays, one per detector panel
- `True` indicates valid pixels, `False` indicates masked pixels

**5. Example Usage Snippet:**
```python
with open("mask.pickle", "rb") as f:
    mask_tuple = pickle.load(f)
    
if isinstance(mask_tuple, tuple):
    for panel_id, panel_mask in enumerate(mask_tuple):
        valid_pixels = flex.sum(panel_mask.as_1d())
        print(f"Panel {panel_id}: {valid_pixels} valid pixels")
```

**6. Notes/Caveats:**
- Mask structure depends on how it was created (single panel vs multi-panel)
- Some masks may be stored as single flex.bool arrays for single-panel detectors
- Handle both cases: `isinstance(mask_data, tuple)` for multi-panel, single array for single-panel

**7. Comprehensive Mask Handling Example:**
```python
import pickle
from dials.array_family import flex

try:
    with open("mask.pickle", "rb") as f:
        mask_data = pickle.load(f)
    
    if isinstance(mask_data, tuple):
        # Multi-panel detector mask
        print(f"Multi-panel mask with {len(mask_data)} panels")
        for panel_id, panel_mask in enumerate(mask_data):
            valid_pixels = flex.sum(panel_mask.as_1d())
            total_pixels = len(panel_mask)
            print(f"Panel {panel_id}: {valid_pixels}/{total_pixels} valid pixels")
    else:
        # Single panel detector mask
        if isinstance(mask_data, flex.bool):
            valid_pixels = flex.sum(mask_data.as_1d())
            print(f"Single panel: {valid_pixels} valid pixels")
        else:
            raise ValueError(f"Unexpected mask data type: {type(mask_data)}")
            
except (FileNotFoundError, IOError) as e:
    print(f"Could not load mask from file: {e}")
```

**8. See Also:**
- Section A.3: Loading image data to apply masks to
- Section B.1: Detector model for understanding panel geometry

---

## B. Accessing and Using dxtbx.model Objects

### B.1. Detector Model (experiment.detector)

**1. Purpose:**
Access detector geometry information for coordinate transformations, pixel-to-lab coordinate mapping, and panel identification needed for diffuse scattering analysis across multi-panel detectors.

**2. Primary Python Call(s):**
```python
detector = experiment.detector

# Panel-level operations
for panel_id, panel in enumerate(detector):
    # Basic properties
    image_size = panel.get_image_size()  # (width, height) in pixels
    pixel_size = panel.get_pixel_size()  # (fast, slow) in mm
    
    # Coordinate transformations
    lab_coord = panel.get_pixel_lab_coord((fast_px, slow_px))
    mm_coord = panel.pixel_to_millimeter((fast_px, slow_px))
    px_coord = panel.millimeter_to_pixel((mm_x, mm_y))
    
    # Panel orientation
    fast_axis = panel.get_fast_axis()  # Fast axis direction vector
    slow_axis = panel.get_slow_axis()  # Slow axis direction vector
    origin = panel.get_origin()        # Panel origin in lab frame
```

**3. Key Methods:**
- `get_image_size()` → (int, int): Panel dimensions in pixels (fast, slow)
- `get_pixel_size()` → (float, float): Pixel size in mm (fast, slow)
- `get_pixel_lab_coord(px_coord)` → (float, float, float): Lab coordinates for pixel
- `get_trusted_range()` → (float, float): Valid pixel value range (min, max)
- `get_fast_axis()` → (float, float, float): Fast axis direction vector
- `get_slow_axis()` → (float, float, float): Slow axis direction vector
- `get_origin()` → (float, float, float): Panel origin in lab coordinates
- `detector.get_ray_intersection(s1)` → (int, (float, float)): Panel ID and intersection point
- `detector.get_panel_intersection(s1)` → int: Panel ID for ray intersection (-1 if none)

**4. Return Types:**
- Image coordinates: tuples of (fast, slow) integers or floats
- Lab coordinates: tuples of (x, y, z) floats in mm
- Axis vectors: tuples of (x, y, z) unit direction vectors

**5. Example Usage Snippet:**
```python
detector = experiment.detector
beam_centre_panel, beam_centre_mm = detector.get_ray_intersection(beam.get_s0())

if beam_centre_panel >= 0:
    panel = detector[beam_centre_panel]
    beam_centre_px = panel.millimeter_to_pixel(beam_centre_mm)
    print(f"Beam centre: Panel {beam_centre_panel}, pixel {beam_centre_px}")
```

**6. Notes/Caveats:**
- Multi-panel detectors are accessed by index: `detector[panel_id]`
- Coordinate transformations assume orthogonal detector geometry
- Lab coordinates are in detector coordinate system (not necessarily aligned with beam)
- **Panel identification from pixel coordinates:** No direct `get_panel_for_pixel()` method exists; use ray intersection methods instead
- The A matrix (A = U × B) transforms Miller indices directly to reciprocal lattice vectors in lab frame

**7. Panel Identification Examples:**
```python
# Find which panel a given s1 vector intersects
detector = experiment.detector
beam = experiment.beam

# Calculate s1 for a specific pixel on panel 0
panel = detector[0]
lab_coord = panel.get_pixel_lab_coord((100, 200))
s1 = matrix.col(lab_coord).normalize() / beam.get_wavelength()

# Find which panel this s1 intersects
panel_id = detector.get_panel_intersection(s1)
if panel_id >= 0:
    print(f"Ray intersects panel {panel_id}")
else:
    print("Ray does not intersect any panel")

# Alternative: get panel ID and intersection coordinates
panel_id, intersection_xy = detector.get_ray_intersection(s1)
if panel_id >= 0:
    panel = detector[panel_id]
    pixel_coord = panel.millimeter_to_pixel(intersection_xy)
    print(f"Ray intersects panel {panel_id} at pixel {pixel_coord}")

# Get detector coordinate system vectors
for panel_id, panel in enumerate(detector):
    fast_axis = panel.get_fast_axis()
    slow_axis = panel.get_slow_axis()
    normal = matrix.col(fast_axis).cross(matrix.col(slow_axis))
    print(f"Panel {panel_id} normal vector: {normal}")
```

**8. See Also:**
- Section C.1: Calculating q-vectors using detector geometry
- Section C.4: Geometric corrections requiring detector parameters

---

### B.2. Beam Model (experiment.beam)

**1. Purpose:**
Access incident beam properties for calculating scattering vectors and momentum transfer.

**2. Primary Python Call(s):**
```python
beam = experiment.beam

# Basic properties
wavelength = beam.get_wavelength()      # Wavelength in Angstroms
direction = beam.get_direction()        # Unit direction vector
s0 = beam.get_s0()                     # Incident beam vector (1/λ * direction)
unit_s0 = beam.get_unit_s0()           # Unit incident beam vector

# Polarization properties
polarization_normal = beam.get_polarization_normal()
polarization_fraction = beam.get_polarization_fraction()
```

**3. Key Methods:**
- `get_wavelength()` → float: Wavelength in Angstroms
- `get_s0()` → (float, float, float): Incident beam vector in 1/Angstroms
- `get_direction()` → (float, float, float): Unit direction vector
- `get_polarization_fraction()` → float: Fraction of polarized light

**4. Return Types:**
- Wavelength: float in Angstroms
- Vectors: tuples of (x, y, z) floats
- s0 magnitude: 1/wavelength in 1/Angstroms

**5. Example Usage Snippet:**
```python
beam = experiment.beam
wavelength = beam.get_wavelength()
s0 = beam.get_s0()
print(f"Wavelength: {wavelength:.4f} Å")
print(f"Beam vector s0: {s0}")
print(f"|s0| = {(s0[0]**2 + s0[1]**2 + s0[2]**2)**0.5:.6f} = 1/λ")
```

**6. Notes/Caveats:**
- s0 points in direction of incident beam with magnitude 1/λ
- For time-of-flight experiments, use `get_wavelength_range()` for wavelength distribution
- Polarization methods are available: `get_polarization_normal()` and `get_polarization_fraction()`

**7. Complete Beam Properties Example:**
```python
beam = experiment.beam

# Basic properties
wavelength = beam.get_wavelength()
direction = beam.get_direction() 
s0 = beam.get_s0()

# Polarization properties (from cctbx integration)
polarization_normal = beam.get_polarization_normal()  # Returns (x,y,z) tuple
polarization_fraction = beam.get_polarization_fraction()  # Returns float 0-1

print(f"Wavelength: {wavelength:.4f} Å")
print(f"Incident beam s0: {s0}")
print(f"Polarization: {polarization_fraction:.2f} fraction along {polarization_normal}")

# For Lorentz-polarization corrections
print(f"Polarization normal: {polarization_normal}")
print(f"Polarization fraction: {polarization_fraction}")
```

**8. See Also:**
- Section C.4: Using beam polarization in geometric corrections
- Section C.1: Using s0 vector in q-vector calculations

---

### B.3. Crystal Model (experiment.crystal)

**1. Purpose:**
Access crystal lattice parameters and orientation matrices (from cctbx.uctbx and cctbx.sgtbx) for Miller index calculations, reciprocal space transformations, and diffuse scattering analysis in crystal coordinates.

**2. Primary Python Call(s):**
```python
crystal = experiment.crystal

# Unit cell and symmetry
unit_cell = crystal.get_unit_cell()
space_group = crystal.get_space_group()
a, b, c, alpha, beta, gamma = unit_cell.parameters()

# Orientation matrices
A_matrix = crystal.get_A()  # A = U * B (orientation × metric)
U_matrix = crystal.get_U()  # Orientation matrix
B_matrix = crystal.get_B()  # Metric matrix (reciprocal lattice)

# Real space vectors
real_space_vectors = crystal.get_real_space_vectors()
```

**3. Key Methods:**
- `get_unit_cell()` → cctbx.uctbx.unit_cell: Unit cell object
- `get_A()` → matrix.sqr: Combined orientation and metric matrix
- `get_U()` → matrix.sqr: Orientation matrix
- `get_B()` → matrix.sqr: Reciprocal metric matrix
- `get_space_group()` → cctbx.sgtbx.space_group: Space group object

**4. Return Types:**
- Unit cell: cctbx.uctbx.unit_cell with .parameters() method
- Matrices: scitbx.matrix.sqr objects (3×3)
- Space group: cctbx.sgtbx.space_group with symmetry operations

**5. Example Usage Snippet:**
```python
crystal = experiment.crystal
unit_cell = crystal.get_unit_cell()
space_group = crystal.get_space_group()
A_matrix = crystal.get_A()

print(f"Unit cell: {unit_cell.parameters()}")
print(f"Space group: {space_group.info().symbol_and_number()}")

# Convert Miller index to reciprocal space vector
from scitbx import matrix
hkl = (1, 2, 3)
q_vector = matrix.col(A_matrix) * matrix.col(hkl)
print(f"Miller index {hkl} → q-vector {q_vector}")
```

**6. Notes/Caveats:**
- **A matrix interpretation:** A = U × B transforms Miller indices directly to reciprocal lattice vectors in lab frame
- **Setting consistency:** The crystal model is always in the correct setting for calculations; no direct/reciprocal setting checks needed
- For scan-varying crystals, use `get_A_at_scan_point(i)` for frame-specific matrices
- Real space vectors are in Angstroms, reciprocal vectors in 1/Angstroms
- Space group operations available through `crystal.get_space_group().all_ops()`

**7. Extended Crystal Model Usage:**
```python
crystal = experiment.crystal

# Complete unit cell information
unit_cell = crystal.get_unit_cell()
a, b, c, alpha, beta, gamma = unit_cell.parameters()
volume = unit_cell.volume()
print(f"Unit cell: a={a:.3f}, b={b:.3f}, c={c:.3f} Å")
print(f"Angles: α={alpha:.1f}, β={beta:.1f}, γ={gamma:.1f}°")
print(f"Volume: {volume:.1f} Å³")

# Space group information
space_group = crystal.get_space_group()
symbol = space_group.info().symbol_and_number()
symmetry_ops = space_group.all_ops()
print(f"Space group: {symbol}")
print(f"Number of symmetry operations: {len(symmetry_ops)}")

# Orientation matrices and transformations
U_matrix = crystal.get_U()  # Orientation matrix
B_matrix = crystal.get_B()  # Metric matrix (reciprocal lattice)
A_matrix = crystal.get_A()  # Combined A = U × B

# Real space lattice vectors
real_space_vectors = crystal.get_real_space_vectors()
a_vec, b_vec, c_vec = real_space_vectors
print(f"Real space vectors:")
print(f"a: {a_vec}")
print(f"b: {b_vec}")
print(f"c: {c_vec}")

# Calculate reciprocal lattice vectors for Miller indices
hkl_list = [(1,0,0), (0,1,0), (0,0,1), (1,1,1)]
for hkl in hkl_list:
    q_vector = matrix.col(A_matrix) * matrix.col(hkl)
    d_spacing = unit_cell.d(hkl)
    print(f"Miller index {hkl}: q={q_vector}, d={d_spacing:.3f} Å")
```

**8. See Also:**
- Section C.2: Converting q-vectors to Miller indices using A matrix
- Section C.3: Calculating d-spacings from unit cell parameters

---

### B.4. Goniometer Model (experiment.goniometer)

**1. Purpose:**
Access goniometer geometry for rotation data processing and scan-dependent coordinate transformations.

**2. Primary Python Call(s):**
```python
goniometer = experiment.goniometer

# Rotation properties
rotation_axis = goniometer.get_rotation_axis()
fixed_rotation = goniometer.get_fixed_rotation()
setting_rotation = goniometer.get_setting_rotation()

# Scan-dependent rotations
if scan is not None:
    rotation_matrix = goniometer.get_rotation_matrix_at_scan_point(scan_point)
    angle = goniometer.get_angle_from_rotation_matrix(rotation_matrix)
```

**3. Key Methods:**
- `get_rotation_axis()` → (float, float, float): Rotation axis unit vector
- `get_fixed_rotation()` → matrix.sqr: Fixed rotation matrix
- `get_setting_rotation()` → matrix.sqr: Setting rotation matrix

**4. Return Types:**
- Axis: tuple of (x, y, z) unit vector components
- Matrices: scitbx.matrix.sqr objects (3×3 rotation matrices)

**5. Example Usage Snippet:**
```python
if experiment.goniometer is not None:
    goniometer = experiment.goniometer
    rotation_axis = goniometer.get_rotation_axis()
    print(f"Rotation axis: {rotation_axis}")
    
    # For rotation experiments
    if experiment.scan is not None:
        for i in range(experiment.scan.get_num_images()):
            angle = experiment.scan.get_angle_from_image_index(i)
            print(f"Image {i}: rotation angle {angle:.2f}°")
```

**6. Notes/Caveats:**
- **Existence check required:** Only applicable for rotation experiments (None for still experiments)
- Always check `if experiment.goniometer is not None:` before accessing
- Combined with scan information to calculate frame-specific orientations
- Used in Lorentz correction calculations for rotation experiments

**7. Safe Goniometer Access Pattern:**
```python
# Always check for existence first
if experiment.goniometer is not None:
    goniometer = experiment.goniometer
    rotation_axis = goniometer.get_rotation_axis()
    print(f"Rotation experiment with axis: {rotation_axis}")
    
    # Get rotation matrices for scan points
    if experiment.scan is not None:
        scan = experiment.scan
        for i in range(scan.get_num_images()):
            # Method 1: Use scan to get angle, then goniometer for matrix
            angle = scan.get_angle_from_image_index(i)
            # Method 2: Direct from goniometer if scan points are set
            if hasattr(goniometer, 'get_rotation_matrix_at_scan_point'):
                rotation_matrix = goniometer.get_rotation_matrix_at_scan_point(i)
                print(f"Image {i}: angle={angle:.2f}°")
else:
    print("Still experiment - no goniometer model")
```

**8. See Also:**
- Section C.4: Using goniometer in Lorentz corrections
- Section B.5: Scan model for rotation angle information

---

### B.5. Scan Model (experiment.scan)

**1. Purpose:**
Access scan parameters for rotation experiments including oscillation range and frame timing.

**2. Primary Python Call(s):**
```python
scan = experiment.scan

# Basic scan properties
oscillation = scan.get_oscillation()        # (start_angle, oscillation_width)
image_range = scan.get_image_range()        # (start_image, end_image)
num_images = scan.get_num_images()          

# Frame-specific information
for i in range(num_images):
    angle = scan.get_angle_from_image_index(i)
    exposure_time = scan.get_exposure_times()[i]
```

**3. Key Methods:**
- `get_oscillation()` → (float, float): Start angle and oscillation width in degrees
- `get_image_range()` → (int, int): First and last image numbers
- `get_angle_from_image_index(i)` → float: Rotation angle for image i

**4. Return Types:**
- Angles: floats in degrees
- Image indices: integers
- Times: floats in seconds

**5. Example Usage Snippet:**
```python
if experiment.scan is not None:
    scan = experiment.scan
    start_angle, osc_width = scan.get_oscillation()
    image_range = scan.get_image_range()
    
    print(f"Scan: {osc_width}° oscillation starting at {start_angle}°")
    print(f"Images: {image_range[0]} to {image_range[1]}")
```

**6. Notes/Caveats:**
- **Existence check required:** Only applicable for rotation experiments (None for still experiments)
- Always check `if experiment.scan is not None:` before accessing
- Image indices in DIALS start from 1, not 0
- Exposure times available through `get_exposure_times()` method

**7. Complete Scan Information Access:**
```python
# Always check for existence first
if experiment.scan is not None:
    scan = experiment.scan
    
    # Basic scan properties
    start_angle, osc_width = scan.get_oscillation()
    start_image, end_image = scan.get_image_range()
    num_images = scan.get_num_images()
    
    # Exposure times
    exposure_times = scan.get_exposure_times()
    
    print(f"Scan parameters:")
    print(f"  Oscillation: {osc_width}° starting at {start_angle}°")
    print(f"  Images: {start_image} to {end_image} ({num_images} total)")
    print(f"  Exposure times: {exposure_times[0]:.3f} to {exposure_times[-1]:.3f} s")
    
    # Frame-specific information
    for i in range(min(5, num_images)):  # Show first 5 frames
        angle = scan.get_angle_from_image_index(i)
        exposure = exposure_times[i]
        image_number = start_image + i
        print(f"  Frame {i}: image #{image_number}, angle={angle:.2f}°, exposure={exposure:.3f}s")
else:
    print("Still experiment - no scan model")
```

**8. See Also:**
- Section A.3: Getting exposure times from ImageSet
- Section B.4: Combining with goniometer for rotation calculations

---

## C. Crystallographic Calculations

### C.1. Calculating Q-vector for a Pixel

**1. Purpose:**
Calculate momentum transfer vector q = k_scattered - k_incident for diffuse scattering analysis. This is fundamental for transforming pixel positions to reciprocal space coordinates.

**2. Primary Python Call(s):**
```python
from scitbx import matrix

def calculate_q_vector(detector, beam, panel_id, pixel_coord):
    """Calculate q-vector for a pixel position"""
    panel = detector[panel_id]
    
    # Convert pixel to lab coordinates
    lab_coord = panel.get_pixel_lab_coord(pixel_coord)
    
    # Calculate scattered beam vector s1 (magnitude = 1/λ)
    s1_direction = matrix.col(lab_coord).normalize()
    s1 = s1_direction * (1.0 / beam.get_wavelength())
    
    # Get incident beam vector s0
    s0 = matrix.col(beam.get_s0())
    
    # Calculate q-vector (momentum transfer)
    q = s1 - s0
    
    return q, s1
```

**3. Key Arguments:**
- `detector`: dxtbx.model.Detector object
- `beam`: dxtbx.model.Beam object
- `panel_id` (int): Panel index
- `pixel_coord` (tuple): (fast, slow) pixel coordinates

**4. Return Type:**
- `q`: scitbx.matrix.col object (momentum transfer vector)
- `s1`: scitbx.matrix.col object (scattered beam vector)

**5. Example Usage Snippet:**
```python
detector = experiment.detector
beam = experiment.beam
panel_id = 0
pixel_coord = (1000, 1000)

q_vec, s1_vec = calculate_q_vector(detector, beam, panel_id, pixel_coord)
q_magnitude = q_vec.length()
d_spacing = 1.0 / q_magnitude if q_magnitude > 0 else float('inf')

print(f"Q-vector: {q_vec}")
print(f"|q| = {q_magnitude:.6f} Å⁻¹")
print(f"d-spacing: {d_spacing:.3f} Å")
```

**6. Notes/Caveats:**
- Assumes elastic scattering (|s1| = |s0| = 1/λ)
- Lab coordinates must be normalized to unit vector before scaling by 1/λ
- Q-vector is in units of Å⁻¹
- **Numerical stability:** Handle case where pixel is very close to beam center (lab_coord ≈ 0)
- Vector normalization may fail for pixels directly on the beam path

---

### C.2. Transforming Q-vector to Fractional Miller Indices

**1. Purpose:**
Convert momentum transfer vectors to fractional Miller indices for lattice analysis and identifying diffuse scattering relative to Bragg peak positions. Handles both static crystal orientations and scan-varying crystal models.

**2. Primary Python Call(s):**
```python
from scitbx import matrix

def q_to_miller_indices_static(crystal, q_vector):
    """Transform q-vector to Miller indices for static crystal orientation"""
    # For static crystals or reference orientations
    # q_vector should be in same lab frame as crystal.get_A() is defined
    A_matrix = matrix.sqr(crystal.get_A())
    A_inverse = A_matrix.inverse()
    hkl_fractional = A_inverse * q_vector
    return hkl_fractional

def q_to_miller_indices_scan_varying(crystal, q_vector, scan_point_index):
    """Transform q-vector to Miller indices for scan-varying crystals"""
    # For scan-varying crystal orientations
    A_matrix = matrix.sqr(crystal.get_A_at_scan_point(scan_point_index))
    A_inverse = A_matrix.inverse()
    hkl_fractional = A_inverse * q_vector
    return hkl_fractional

def q_to_miller_indices_with_goniometer(crystal, q_vector_lab, goniometer, scan_point_index):
    """Transform lab-frame q-vector to Miller indices accounting for goniometer rotation"""
    # This method is for cases where q_vector is in a fixed lab frame
    # but crystal orientation changes due to rotation during scan
    
    # Get goniometer rotation for this scan point
    if hasattr(goniometer, 'get_rotation_matrix_at_scan_point'):
        rotation = goniometer.get_rotation_matrix_at_scan_point(scan_point_index)
    else:
        # Fallback: use scan to get angle, then calculate rotation
        rotation = matrix.identity(3)  # Simplified - implement based on specific goniometer
    
    # Transform q from lab frame to crystal setting frame
    # Rotation matrix transforms crystal coords to lab coords, so use transpose
    q_crystal_setting = rotation.transpose() * q_vector_lab
    
    # Now use crystal A matrix (defined in crystal setting frame)
    A_matrix = matrix.sqr(crystal.get_A())
    A_inverse = A_matrix.inverse()
    hkl_fractional = A_inverse * q_crystal_setting
    
    return hkl_fractional
```

**3. Key Arguments:**
- `crystal`: dxtbx.model.Crystal object
- `q_vector`: scitbx.matrix.col (q-vector - ensure frame consistency with method)
- `q_vector_lab`: scitbx.matrix.col (q-vector explicitly in lab frame)
- `goniometer`: dxtbx.model.Goniometer object (for rotation experiments)
- `scan_point_index`: int (scan point index for scan-varying data)

**4. Return Type:**
- `hkl_fractional`: scitbx.matrix.col with fractional Miller indices

**5. Example Usage Snippet:**
```python
crystal = experiment.crystal
goniometer = experiment.goniometer
q_vec = matrix.col((0.1, 0.2, 0.3))  # Example q-vector

# Method 1: Static crystal (stills or reference orientation)
hkl_frac_static = q_to_miller_indices_static(crystal, q_vec)
print(f"Static orientation: {q_vec} → {hkl_frac_static}")

# Method 2: Scan-varying crystal (if crystal model varies with scan)
if hasattr(crystal, 'get_A_at_scan_point'):
    scan_point = 0  # First scan point
    hkl_frac_scan = q_to_miller_indices_scan_varying(crystal, q_vec, scan_point)
    print(f"Scan point {scan_point}: {q_vec} → {hkl_frac_scan}")

# Method 3: With explicit goniometer handling (advanced case)
if goniometer is not None:
    scan_point = 0
    hkl_frac_gonio = q_to_miller_indices_with_goniometer(crystal, q_vec, goniometer, scan_point)
    print(f"With goniometer rotation: {q_vec} → {hkl_frac_gonio}")

# Convert to integer indices
hkl_int = tuple(round(x) for x in hkl_frac_static)
print(f"Nearest integer indices: {hkl_int}")

# Check if close to integer (potential Bragg reflection)
tolerance = 0.1
is_bragg_like = all(abs(x - round(x)) < tolerance for x in hkl_frac_static)
print(f"Close to Bragg reflection: {is_bragg_like}")
```

**6. Notes/Caveats:**
- Returns fractional Miller indices; round to nearest integers for identifying Bragg reflections
- **Coordinate frame consistency:** Ensure q-vector and crystal A matrix are in compatible frames:
  - For `crystal.get_A()`: q-vector should be in the lab frame where A is defined
  - For scan-varying: use `crystal.get_A_at_scan_point(i)` with q-vectors from that scan point
  - For goniometer rotations: account for rotation between lab frame and crystal setting
- **Scan-varying crystals:** Use appropriate method based on whether crystal orientation changes
- **Fractional indices interpretation:** Values close to integers indicate proximity to Bragg reflections
- **Diffuse scattering analysis:** Non-integer values represent diffuse scattering in fractional hkl space

**7. Coordinate Frame Decision Tree:**
```python
# Decision logic for choosing the right method:
if experiment.scan is None:
    # Still experiment - use static method
    hkl = q_to_miller_indices_static(crystal, q_vector)
elif hasattr(crystal, 'num_scan_points') and crystal.num_scan_points > 1:
    # Scan-varying crystal - use scan-specific A matrix
    hkl = q_to_miller_indices_scan_varying(crystal, q_vector, scan_point)
elif goniometer is not None and "need_gonio_correction":
    # Fixed crystal with goniometer rotation (advanced case)
    hkl = q_to_miller_indices_with_goniometer(crystal, q_vector, goniometer, scan_point)
else:
    # Standard rotation with fixed crystal orientation
    hkl = q_to_miller_indices_static(crystal, q_vector)
```

**8. See Also:**
- Section C.1: Calculating q-vectors using detector geometry
- Section C.3: Calculating d-spacings from Miller indices
- Section B.3: Crystal model orientation matrices
- Section B.4: Goniometer model for rotation handling

---

### C.3. Calculating D-spacing from Q-vector or Miller Indices

**1. Purpose:**
Calculate resolution (d-spacing) from momentum transfer or crystallographic indices. Essential for resolution-dependent analysis and filtering in diffuse scattering studies.

**2. Primary Python Call(s):**
```python
import math
from scitbx import matrix

def d_spacing_from_q(q_vector):
    """Calculate d-spacing from q-vector magnitude"""
    q_magnitude = q_vector.length()
    if q_magnitude > 0:
        return 2 * math.pi / q_magnitude
    else:
        return float('inf')

def d_spacing_from_miller_indices(unit_cell, hkl):
    """Calculate d-spacing from Miller indices using unit cell"""
    # Use cctbx unit cell d-spacing calculation
    from cctbx import miller
    d_spacing = unit_cell.d(hkl)
    return d_spacing
```

**3. Key Arguments:**
- `q_vector`: scitbx.matrix.col (momentum transfer vector)
- `unit_cell`: cctbx.uctbx.unit_cell object
- `hkl`: tuple of (h, k, l) Miller indices

**4. Return Type:**
- `d_spacing`: float in Angstroms

**5. Example Usage Snippet:**
```python
# From q-vector
q_vec = matrix.col((0.1, 0.2, 0.3))
d_from_q = d_spacing_from_q(q_vec)

# From Miller indices
unit_cell = experiment.crystal.get_unit_cell()
hkl = (1, 2, 3)
d_from_hkl = d_spacing_from_miller_indices(unit_cell, hkl)

print(f"d-spacing from |q|: {d_from_q:.3f} Å")
print(f"d-spacing from Miller indices {hkl}: {d_from_hkl:.3f} Å")
```

**6. Notes/Caveats:**
- d = 2π/|q| for X-ray crystallography convention
- cctbx unit_cell.d() handles complex unit cell geometries correctly
- Both methods should give identical results for Bragg reflections

---

### C.4. Geometric Corrections

**1. Purpose:**
Apply Lorentz-polarization, solid angle, detector efficiency, and air attenuation corrections essential for accurate diffuse scattering intensity analysis and comparison with theoretical models.

**2. Primary Python Call(s):**
```python
from dials.algorithms.integration import CorrectionsMulti, Corrections
from dials.algorithms.integration.kapton_correction import KaptonAbsorption
import math
from scitbx import matrix

# Method 1: Using DIALS correction calculator
corrector = CorrectionsMulti()
for exp in experiments:
    corrector.append(Corrections(exp.beam, exp.goniometer, exp.detector))

# Calculate Lorentz-polarization correction
lp_correction = corrector.lp(reflections["id"], reflections["s1"])

# Calculate quantum efficiency correction
qe_correction = corrector.qe(reflections["id"], reflections["s1"], reflections["panel"])

# Method 2: Manual geometric corrections for custom analysis
def calculate_solid_angle_correction(detector, panel_id, pixel_coord, beam):
    """Calculate solid angle subtended by a pixel"""
    panel = detector[panel_id]
    
    # Get pixel position in lab coordinates
    lab_coord = matrix.col(panel.get_pixel_lab_coord(pixel_coord))
    
    # Distance from sample to pixel
    distance = lab_coord.length()
    
    # Pixel area in mm^2
    pixel_size = panel.get_pixel_size()
    pixel_area = pixel_size[0] * pixel_size[1]
    
    # Angle between pixel normal and beam direction
    panel_normal = matrix.col(panel.get_fast_axis()).cross(matrix.col(panel.get_slow_axis()))
    beam_to_pixel = lab_coord.normalize()
    cos_angle = abs(panel_normal.dot(beam_to_pixel))
    
    # Solid angle = projected_area / distance^2
    solid_angle = (pixel_area * cos_angle) / (distance * distance)
    
    return solid_angle

def calculate_polarization_correction(beam, s1_vector):
    """Calculate polarization correction factor"""
    s0 = matrix.col(beam.get_s0())
    s1 = matrix.col(s1_vector)
    
    # Polarization properties
    pol_normal = matrix.col(beam.get_polarization_normal())
    pol_fraction = beam.get_polarization_fraction()
    
    # Scattering angle
    cos_2theta = s1.dot(s0) / (s1.length() * s0.length())
    
    # Angle between scattering vector and polarization normal
    s1_unit = s1.normalize()
    cos_phi = abs(pol_normal.dot(s1_unit))
    sin_phi_sq = 1 - cos_phi * cos_phi
    
    # Polarization factor (Kahn et al. 1982 formulation)
    pol_factor = (1 - pol_fraction) * (1 + cos_2theta * cos_2theta) + pol_fraction * sin_phi_sq
    
    return pol_factor

def calculate_lorentz_correction(beam, s1_vector, goniometer=None):
    """Calculate Lorentz correction factor"""
    s0 = matrix.col(beam.get_s0())
    s1 = matrix.col(s1_vector)
    
    if goniometer is not None:
        # Rotation experiment - Lorentz factor from rotation geometry
        rotation_axis = matrix.col(goniometer.get_rotation_axis())
        # L = 1/|sin(ψ)| where ψ is angle between rotation axis and scattering plane
        scattering_plane_normal = s1.cross(s0).normalize()
        sin_psi = abs(rotation_axis.dot(scattering_plane_normal))
        lorentz_factor = 1.0 / sin_psi if sin_psi > 1e-6 else 1.0
    else:
        # Still experiment - standard Lorentz factor
        cos_2theta = s1.dot(s0) / (s1.length() * s0.length())
        sin_2theta = math.sqrt(1 - cos_2theta * cos_2theta)
        lorentz_factor = 1.0 / sin_2theta if sin_2theta > 1e-6 else 1.0
    
    return lorentz_factor

def calculate_detector_efficiency_correction(detector, panel_id, s1_vector, 
                                           attenuation_coeff=0.1, thickness_mm=0.45):
    """Calculate detector quantum efficiency correction"""
    panel = detector[panel_id]
    
    # Panel normal vector
    fast_axis = matrix.col(panel.get_fast_axis())
    slow_axis = matrix.col(panel.get_slow_axis())
    panel_normal = fast_axis.cross(slow_axis).normalize()
    
    # Angle between incoming ray and panel normal
    s1_unit = matrix.col(s1_vector).normalize()
    cos_theta = abs(panel_normal.dot(s1_unit))
    
    # Path length through detector material
    if cos_theta > 1e-6:
        path_length = thickness_mm / cos_theta
        # Quantum efficiency: QE = 1 - exp(-μt/cos(θ))
        qe_factor = 1 - math.exp(-attenuation_coeff * path_length)
    else:
        qe_factor = 1.0  # Grazing incidence approximation
    
    return qe_factor

def calculate_air_attenuation_correction(lab_coord, wavelength_angstrom, 
                                       air_path_mm=None, pressure_atm=1.0, temp_kelvin=293.15):
    """Calculate X-ray attenuation through air"""
    if air_path_mm is None:
        # Use distance to detector as air path
        air_path_mm = matrix.col(lab_coord).length()
    
    # X-ray attenuation coefficient for air at standard conditions
    # Approximate values for common wavelengths
    if wavelength_angstrom < 1.0:  # Hard X-rays
        mu_air_per_cm = 0.001  # Very approximate
    else:  # Softer X-rays
        mu_air_per_cm = 0.01 * (wavelength_angstrom ** 3)  # Rough scaling
    
    # Adjust for pressure and temperature
    density_correction = (pressure_atm * 273.15) / (1.0 * temp_kelvin)
    mu_air_corrected = mu_air_per_cm * density_correction
    
    # Attenuation factor
    path_cm = air_path_mm / 10.0
    attenuation_factor = math.exp(-mu_air_corrected * path_cm)
    
    return attenuation_factor
```

**3. Key Arguments:**
- `experiments`: ExperimentList object
- `reflections`: flex.reflection_table with s1 vectors and panel IDs
- For manual calculations: detector, beam, goniometer models and geometric parameters
- `attenuation_coeff`: Detector material attenuation coefficient (mm⁻¹)
- `thickness_mm`: Detector active layer thickness

**4. Return Type:**
- `correction_factor`: float or flex.double array of correction factors
- **Application convention:** See "Physical Interpretation and Application" below for how to apply each correction type

**5. Comprehensive Example Usage:**
```python
from dials.algorithms.integration import CorrectionsMulti, Corrections

# Method A: Using DIALS built-in corrections
corrector = CorrectionsMulti()
for exp in experiments:
    corrector.append(Corrections(exp.beam, exp.goniometer, exp.detector))

# Apply standard corrections
lp_corrections = corrector.lp(reflections["id"], reflections["s1"])  # Returns divisors
qe_corrections = corrector.qe(reflections["id"], reflections["s1"], reflections["panel"])  # Returns multipliers

# Apply corrections with proper convention
# LP and QE corrections from DIALS have different application conventions
corrected_intensities = (reflections["intensity.sum.value"] / lp_corrections) * qe_corrections

# Alternative: Convert all to multiplicative form for consistency
lp_mult = 1.0 / lp_corrections
total_multiplicative_corrections = lp_mult * qe_corrections
corrected_intensities_alt = reflections["intensity.sum.value"] * total_multiplicative_corrections

print(f"LP corrections (divisors): {flex.min(lp_corrections):.3f} to {flex.max(lp_corrections):.3f}")
print(f"QE corrections (multipliers): {flex.min(qe_corrections):.3f} to {flex.max(qe_corrections):.3f}")

# Method B: Manual corrections for diffuse scattering analysis
detector = experiments[0].detector
beam = experiments[0].beam
goniometer = experiments[0].goniometer

# Calculate corrections for specific pixels
for panel_id in range(len(detector)):
    panel = detector[panel_id]
    
    # Example pixel coordinates
    for px, py in [(100, 100), (500, 500), (1000, 1000)]:
        # Calculate s1 vector
        lab_coord = panel.get_pixel_lab_coord((px, py))
        s1 = matrix.col(lab_coord).normalize() / beam.get_wavelength()
        
        # Individual correction factors
        solid_angle = calculate_solid_angle_correction(detector, panel_id, (px, py), beam)
        pol_correction = calculate_polarization_correction(beam, s1)
        lorentz_correction = calculate_lorentz_correction(beam, s1, goniometer)
        qe_correction = calculate_detector_efficiency_correction(detector, panel_id, s1)
        air_correction = calculate_air_attenuation_correction(lab_coord, beam.get_wavelength())
        
        # Apply corrections using proper conventions
        # I_corrected = I_raw / (LP * Ω * QE * A) * P
        # Or in multiplicative form: I_corrected = I_raw * (1/LP) * (1/Ω) * (1/QE) * (1/A) * P
        
        lp_mult = 1.0 / lorentz_correction
        solid_angle_mult = 1.0 / solid_angle
        qe_mult = 1.0 / qe_correction  
        air_mult = 1.0 / air_correction
        pol_mult = pol_correction
        
        total_multiplicative_correction = lp_mult * solid_angle_mult * qe_mult * air_mult * pol_mult
        
        print(f"Panel {panel_id}, Pixel ({px},{py}):")
        print(f"  Solid angle: {solid_angle:.2e} sr (apply as divisor)")
        print(f"  Polarization: {pol_correction:.3f} (apply as multiplier)")
        print(f"  Lorentz: {lorentz_correction:.3f} (apply as divisor)")
        print(f"  QE: {qe_correction:.3f} (apply as divisor)")
        print(f"  Air attenuation: {air_correction:.4f} (apply as divisor)")
        print(f"  Total multiplicative correction: {total_multiplicative_correction:.2e}")
        print(f"  Usage: I_corrected = I_raw * {total_multiplicative_correction:.2e}")

# Method C: Using Kapton tape absorption correction (for beamlines with tape)
from dials.algorithms.integration.kapton_correction import KaptonAbsorption

kapton = KaptonAbsorption(
    height_mm=0.02,
    thickness_mm=0.05,
    half_width_mm=1.5875,
    rotation_angle_deg=1.15,
    wavelength_ang=beam.get_wavelength()
)

# Apply kapton correction to s1 vectors
for s1_vec in reflections["s1"][:10]:  # First 10 reflections
    kapton_correction = kapton.abs_correction(s1_vec)
    print(f"Kapton correction factor: {kapton_correction:.4f}")
```

**6. Physical Interpretation and Application:**

**Correction Factor Definitions:**
- **Lorentz-Polarization (LP):** Corrects for geometric and polarization effects → **Apply as divisor**
- **Solid Angle (Ω):** Normalizes intensity per unit solid angle → **Apply as divisor** 
- **Polarization (P):** Additional polarization correction → **Apply as multiplier**
- **Detector Efficiency (QE):** Corrects for incomplete absorption → **Apply as divisor**
- **Air Attenuation (A):** Corrects for X-ray absorption in air → **Apply as divisor**

**Combined Correction Formula:**
```python
# Standard diffuse scattering intensity correction:
I_corrected = I_raw / (LP_factor * solid_angle * QE_factor * air_attenuation_factor) * polarization_factor

# Or equivalently, using multiplicative forms:
LP_mult = 1.0 / LP_factor
solid_angle_mult = 1.0 / solid_angle  
QE_mult = 1.0 / QE_factor
air_mult = 1.0 / air_attenuation_factor
pol_mult = polarization_factor

I_corrected = I_raw * LP_mult * solid_angle_mult * QE_mult * air_mult * pol_mult
```

**7. Notes/Caveats:**
- **Coordinate systems:** Ensure s1 vectors are in lab frame and properly normalized
- **Detector efficiency:** Values depend on detector type (silicon, CdTe, etc.) and X-ray energy
- **Air attenuation:** Significant for low-energy X-rays and long air paths
- **Solid angle:** Critical for accurate diffuse scattering intensity comparisons
- **DIALS integration:** `CorrectionsMulti.lp()` returns factors to divide by; `CorrectionsMulti.qe()` returns factors to multiply by
- **Sign conventions:** Always verify factor definitions in your specific analysis context

**8. Physical Formulas and Application:**
- **Solid Angle:** Ω = (A_pixel × cos θ) / r² → I_corrected = I_raw / Ω
- **Lorentz (rotation):** L = 1/|sin ψ| → I_corrected = I_raw / L  
- **Lorentz (still):** L = 1/sin(2θ) → I_corrected = I_raw / L
- **Polarization:** P = (1-f)(1+cos²2θ) + f×sin²φ → I_corrected = I_raw × P
- **Quantum Efficiency:** QE = 1 - exp(-μt/cos θ) → I_corrected = I_raw / QE
- **Air Attenuation:** A = exp(-μ_air × path_length) → I_corrected = I_raw / A

**Combined:** I_corrected = (I_raw × P) / (L × Ω × QE × A)

**9. See Also:**
- Section B.1: Detector model for panel geometry
- Section B.2: Beam model for polarization properties  
- Section C.1: Calculating s1 vectors for correction inputs
- DIALS integration documentation for standard correction workflows
- `dials.algorithms.integration.corrections.h` for C++ implementation details

---

### C.5. Theoretical Scattering Factor Calculation

**1. Purpose:**
Calculate atomic form factors and incoherent scattering intensities (from cctbx.eltbx) for comparison with experimental diffuse scattering and theoretical model validation.

**2. Primary Python Call(s):**
```python
from cctbx.eltbx import sasaki, henke, henke_cdb
from cctbx.eltbx.xray_scattering import wk1995

def get_atomic_form_factor(element, q_magnitude, table="wk1995"):
    """Get atomic form factor for an element at given q"""
    
    # Convert q to sin(theta)/lambda (s parameter)
    s = q_magnitude / (4 * math.pi)
    
    if table == "wk1995":
        # Waasmaier-Kirfel 1995 form factors
        scattering_factor_info = wk1995(element, True)
        f0 = scattering_factor_info.at_stol(s)
    elif table == "sasaki":
        # Sasaki 1989 form factors
        fp_fdp = sasaki.table(element).at_angstrom(wavelength)
        f0 = fp_fdp.fp()  # Real part
    
    return f0

def get_incoherent_scattering(element, q_magnitude):
    """Get incoherent scattering intensity for an element"""
    s = q_magnitude / (4 * math.pi)
    
    # Use Hubbell et al. tables for incoherent scattering
    scattering_factor_info = wk1995(element, True)
    # Incoherent scattering approximation
    atomic_number = scattering_factor_info.atomic_number()
    s_incoh = atomic_number * (1 - math.exp(-2 * s**2))
    
    return s_incoh
```

**3. Key Arguments:**
- `element`: str (element symbol, e.g., "C", "N", "O")
- `q_magnitude`: float (magnitude of q-vector in Å⁻¹)
- `wavelength`: float (X-ray wavelength in Å)
- `table`: str (form factor table to use)

**4. Return Type:**
- `f0`: float (atomic form factor in electrons)
- `s_incoh`: float (incoherent scattering intensity)

**5. Example Usage Snippet:**
```python
# Calculate form factors for carbon at different q values
import math
import numpy as np

element = "C"
q_values = np.linspace(0.1, 2.0, 20)

for q in q_values:
    f0 = get_atomic_form_factor(element, q)
    s_incoh = get_incoherent_scattering(element, q)
    d_spacing = 2 * math.pi / q
    
    print(f"q={q:.3f} Å⁻¹, d={d_spacing:.3f} Å: f0={f0:.3f}, I_incoh={s_incoh:.3f}")
```

**6. Notes/Caveats:**
- Form factors depend on scattering parameter s = |q|/(4π) = sin(θ)/λ
- Different tables available: wk1995 (Waasmaier-Kirfel), sasaki, henke
- Anomalous scattering corrections available through sasaki and henke tables

---

## D. Flex Array Manipulations

### D.1. Converting Flex Arrays to NumPy Arrays

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

### D.2. Converting NumPy Arrays to Flex Arrays

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

### D.3. Accessing Elements and Properties of flex.reflection_table

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

### D.4. Common Operations on Flex Arrays

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

This comprehensive API reference provides the exact function calls, arguments, and usage patterns needed for implementing your diffuse scattering pipeline modules using DIALS/dxtbx/cctbx Python libraries.