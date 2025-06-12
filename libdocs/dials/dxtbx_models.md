# DXTBX Model Objects

This documentation covers accessing and using dxtbx.model objects for detector geometry, beam properties, crystal orientation, and experimental parameters needed for diffuse scattering analysis.

**Version Information:** Compatible with DIALS 3.x series. Some methods may differ in DIALS 2.x.

**Key Dependencies:**
- `dxtbx`: Detector models, beam models, image handling
- `scitbx`: Matrix operations and mathematical utilities

---

## B.0. DXTBX Masking Utilities

**1. Purpose:**
Access detector-level masking utilities from DXTBX for creating trusted range masks and basic geometric masks needed for diffuse scattering preprocessing.

**2. Primary Python Call(s):**
```python
from dxtbx.model import Panel
from dials.array_family import flex

# Panel trusted range masking
def create_trusted_range_mask(panel, image_data):
    """Generate mask based on panel's trusted pixel value range"""
    trusted_range = panel.get_trusted_range()
    min_trusted, max_trusted = trusted_range
    
    # Create mask where True = valid pixels
    mask = (image_data >= min_trusted) & (image_data <= max_trusted)
    return mask

# Multi-panel trusted range masking
def create_detector_trusted_masks(detector, raw_data):
    """Create trusted range masks for all detector panels"""
    panel_masks = []
    
    if isinstance(raw_data, tuple):
        # Multi-panel detector
        for panel_id, (panel, panel_data) in enumerate(zip(detector, raw_data)):
            panel_mask = create_trusted_range_mask(panel, panel_data)
            panel_masks.append(panel_mask)
    else:
        # Single panel detector
        panel = detector[0]
        panel_mask = create_trusted_range_mask(panel, raw_data)
        panel_masks.append(panel_mask)
    
    return panel_masks
```

**3. Key Panel Methods for Masking:**
- `panel.get_trusted_range()` → (float, float): Valid pixel value range (min, max)
- `panel.get_image_size()` → (int, int): Panel dimensions (fast, slow) for mask creation
- `panel.get_pixel_size()` → (float, float): Physical pixel size for geometric masking

**4. Return Types:**
- `trusted_range`: tuple of (min_value, max_value) floats
- `mask`: flex.bool array with same shape as input image data
- Panel dimensions: tuple of (fast_pixels, slow_pixels) integers

**5. Example Usage Snippet:**
```python
from dxtbx.model.experiment_list import ExperimentListFactory
from dials.array_family import flex

# Load experiment and get detector model
experiments = ExperimentListFactory.from_json_file("experiments.expt")
detector = experiments[0].detector
imageset = experiments[0].imageset

# Get raw image data
raw_data = imageset.get_raw_data(0)  # First frame

# Create trusted range masks
trusted_masks = create_detector_trusted_masks(detector, raw_data)

# Apply masks to data
if isinstance(raw_data, tuple):
    # Multi-panel case
    for panel_id, (panel_data, panel_mask) in enumerate(zip(raw_data, trusted_masks)):
        valid_data = panel_data.select(panel_mask.as_1d())
        n_valid = len(valid_data)
        n_total = len(panel_data)
        print(f"Panel {panel_id}: {n_valid}/{n_total} pixels within trusted range")
        
        # Get trusted range for this panel
        panel = detector[panel_id]
        trusted_min, trusted_max = panel.get_trusted_range()
        print(f"  Trusted range: {trusted_min} to {trusted_max}")
else:
    # Single panel case
    panel_mask = trusted_masks[0]
    valid_data = raw_data.select(panel_mask.as_1d())
    print(f"Single panel: {len(valid_data)}/{len(raw_data)} pixels within trusted range")
```

**6. Integration with Image Processing:**
```python
def apply_trusted_range_filtering(experiments, frame_index=0):
    """
    Complete workflow for applying trusted range masks to image data
    """
    detector = experiments[0].detector
    imageset = experiments[0].imageset
    
    # Get image data
    raw_data = imageset.get_raw_data(frame_index)
    
    # Create trusted masks
    trusted_masks = create_detector_trusted_masks(detector, raw_data)
    
    # Apply masks and collect statistics
    filtered_data = []
    mask_statistics = {}
    
    if isinstance(raw_data, tuple):
        for panel_id, (panel_data, panel_mask) in enumerate(zip(raw_data, trusted_masks)):
            # Apply mask
            masked_data = panel_data.deep_copy()
            masked_data.set_selected(~panel_mask, 0)  # Set invalid pixels to 0
            filtered_data.append(masked_data)
            
            # Collect statistics
            n_valid = flex.sum(panel_mask.as_1d())
            n_total = len(panel_mask)
            panel = detector[panel_id]
            trusted_range = panel.get_trusted_range()
            
            mask_statistics[panel_id] = {
                'n_valid': n_valid,
                'n_total': n_total,
                'fraction_valid': n_valid / n_total,
                'trusted_range': trusted_range
            }
    else:
        # Single panel
        panel_mask = trusted_masks[0]
        masked_data = raw_data.deep_copy()
        masked_data.set_selected(~panel_mask, 0)
        filtered_data.append(masked_data)
        
        mask_statistics[0] = {
            'n_valid': flex.sum(panel_mask.as_1d()),
            'n_total': len(panel_mask),
            'fraction_valid': flex.sum(panel_mask.as_1d()) / len(panel_mask),
            'trusted_range': detector[0].get_trusted_range()
        }
    
    return filtered_data, trusted_masks, mask_statistics

# Usage
filtered_data, masks, stats = apply_trusted_range_filtering(experiments)
for panel_id, panel_stats in stats.items():
    print(f"Panel {panel_id}: {panel_stats['fraction_valid']:.3f} valid fraction")
    print(f"  Trusted range: {panel_stats['trusted_range']}")
```

**7. Notes/Caveats:**
- **Data Types:** Trusted range comparison requires compatible data types (typically int or float)
- **Overload Handling:** Pixels above trusted max are typically overloaded; pixels below trusted min may be noisy
- **Panel Indexing:** Always verify panel indexing consistency when working with multi-panel detectors
- **Memory Efficiency:** Create masks as needed rather than storing all panel masks simultaneously for large detectors

**8. Integration with Diffuse Scattering Pipeline:**
Trusted range masks form the foundation for all subsequent masking operations, ensuring only reliable pixel data is used in geometric corrections and diffuse scattering analysis.

**9. See Also:**
- Section A.6: dials.generate_mask for advanced masking capabilities
- Section A.3: Loading image data for mask application
- Section B.1: Detector model for geometric properties

---

## B.1. Detector Model (experiment.detector)

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

## B.2. Beam Model (experiment.beam)

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

## B.3. Crystal Model (experiment.crystal)

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

## B.4. Goniometer Model (experiment.goniometer)

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

## B.5. Scan Model (experiment.scan)

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