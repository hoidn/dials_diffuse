# Crystallographic Calculations

This documentation covers crystallographic calculations including Q-vector computations, Miller index operations, geometric corrections, and scattering factor calculations for diffuse scattering analysis.

**Version Information:** Compatible with DIALS 3.x series. Some methods may differ in DIALS 2.x.

**Key Dependencies:**
- `cctbx`: Unit cells, space groups, scattering factors
- `scitbx`: Matrix operations and mathematical utilities
- `dials.array_family.flex`: Reflection tables and array operations
- `dials.algorithms.integration`: Corrections framework

## C.1. Calculating Q-vector for a Pixel

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

## C.2. Transforming Q-vector to Fractional Miller Indices

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

## C.3. Calculating D-spacing from Q-vector or Miller Indices

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

## C.4. Geometric Corrections

**1. Purpose:**
Apply Lorentz-polarization, solid angle, detector efficiency, and air attenuation corrections essential for accurate diffuse scattering intensity analysis. **CRITICAL: For the diffuse scattering pipeline, LP and QE corrections should be obtained from the robust DIALS Corrections API to avoid reimplementation errors. Only Solid Angle and Air Attenuation require custom implementations for diffuse pixels.**

**2. Primary Python Call(s):**
```python
from dials.algorithms.integration import CorrectionsMulti, Corrections
from dials.algorithms.integration.kapton_correction import KaptonAbsorption
import math
from scitbx import matrix

# Method 1: Using DIALS correction calculator (RECOMMENDED for LP and QE)
corrector = CorrectionsMulti()
for exp in experiments:
    corrector.append(Corrections(exp.beam, exp.goniometer, exp.detector))

# Calculate Lorentz-polarization correction (returns divisors)
lp_correction = corrector.lp(reflections["id"], reflections["s1"])

# Calculate quantum efficiency correction (returns multipliers)
qe_correction = corrector.qe(reflections["id"], reflections["s1"], reflections["panel"])

# For diffuse scattering: Use single Corrections object per still
corrections_obj = Corrections(experiment.beam, experiment.goniometer, experiment.detector)
# Apply to arrays of s1 vectors for diffuse pixels

# Method 2: Manual geometric corrections - ONLY for Solid Angle & Air Attenuation
# (LP and QE should use DIALS API above to avoid errors)
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

# Method A: Hybrid approach for diffuse scattering (RECOMMENDED)
# Use DIALS API for LP & QE, custom calculations for Solid Angle & Air Attenuation

corrections_obj = Corrections(experiment.beam, experiment.goniometer, experiment.detector)

# For arrays of diffuse pixel s1 vectors and panel IDs
lp_corrections = corrections_obj.lp(s1_array)  # Returns divisors
qe_corrections = corrections_obj.qe(s1_array, panel_ids)  # Returns multipliers

# Custom calculations for remaining factors
solid_angle_factors = []  # Custom calculation needed
air_attenuation_factors = []  # Custom calculation needed

# Convert all to multiplicative form for consistency
lp_mult = 1.0 / lp_corrections
total_multiplicative_corrections = lp_mult * qe_corrections * solid_angle_mult * air_mult
corrected_intensities = raw_intensities * total_multiplicative_corrections

print(f"LP corrections (divisors): {flex.min(lp_corrections):.3f} to {flex.max(lp_corrections):.3f}")
print(f"QE corrections (multipliers): {flex.min(qe_corrections):.3f} to {flex.max(qe_corrections):.3f}")
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

---

## C.5. Theoretical Scattering Factor Calculation

**1. Purpose:**
Calculate atomic form factors and incoherent scattering intensities (from cctbx.eltbx) for comparison with experimental diffuse scattering and theoretical model validation.

**2. Primary Python Call(s):**
```python
from cctbx.eltbx import sasaki, henke, henke_cdb
from cctbx.eltbx.xray_scattering import wk1995
import math

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

## C.6. CCTBX Crystal Model Averaging

**1. Purpose:**
Average multiple crystal unit cells and orientation matrices from stills diffraction for obtaining consensus crystal parameters and assessing crystal parameter distributions in diffuse scattering analysis.

**2. Primary Python Call(s):**
```python
from cctbx import uctbx, sgtbx
from scitbx import matrix
import math

def average_unit_cells(unit_cells, weights=None):
    """
    Average multiple unit cell parameters with optional weighting
    
    Args:
        unit_cells: List of cctbx.uctbx.unit_cell objects
        weights: Optional list of weights for each unit cell
    
    Returns:
        cctbx.uctbx.unit_cell: Averaged unit cell
    """
    if not unit_cells:
        return None
    
    if weights is None:
        weights = [1.0] * len(unit_cells)
    
    # Extract parameters from all unit cells
    all_params = [uc.parameters() for uc in unit_cells]
    
    # Calculate weighted averages
    total_weight = sum(weights)
    avg_params = []
    
    for param_idx in range(6):  # a, b, c, alpha, beta, gamma
        weighted_sum = sum(params[param_idx] * weight 
                          for params, weight in zip(all_params, weights))
        avg_params.append(weighted_sum / total_weight)
    
    # Create averaged unit cell
    averaged_unit_cell = uctbx.unit_cell(avg_params)
    return averaged_unit_cell

def create_average_crystal_model(crystal_models, weights=None):
    """
    Create averaged crystal model from multiple crystal models
    
    Args:
        crystal_models: List of dxtbx.model.Crystal objects
        weights: Optional list of weights for each crystal
    
    Returns:
        Averaged crystal parameters as dictionary
    """
    if not crystal_models:
        return None
    
    # Extract unit cells and orientation matrices
    unit_cells = [crystal.get_unit_cell() for crystal in crystal_models]
    U_matrices = [crystal.get_U() for crystal in crystal_models]
    space_groups = [crystal.get_space_group() for crystal in crystal_models]
    
    # Check space group consistency
    reference_sg = space_groups[0]
    if not all(sg.info().symbol_and_number() == reference_sg.info().symbol_and_number() 
               for sg in space_groups):
        print("Warning: Space groups are not consistent across crystal models")
    
    # Average unit cells and orientations
    avg_unit_cell = average_unit_cells(unit_cells, weights)
    
    # Calculate average B matrix from averaged unit cell
    avg_B_matrix = matrix.sqr(avg_unit_cell.fractionalization_matrix()).transpose()
    
    return {
        'unit_cell': avg_unit_cell,
        'space_group': reference_sg,
        'B_matrix': avg_B_matrix,
        'unit_cell_parameters': avg_unit_cell.parameters(),
        'unit_cell_volume': avg_unit_cell.volume()
    }
```

**3. Example Usage Snippet:**
```python
# Load multiple crystal models from stills processing
crystal_models = []
for expt_file in ["still_001_integrated.expt", "still_002_integrated.expt", "still_003_integrated.expt"]:
    experiments = ExperimentListFactory.from_json_file(expt_file)
    if len(experiments) > 0:
        crystal_models.append(experiments[0].crystal)

print(f"Loaded {len(crystal_models)} crystal models for averaging")

# Create averaged crystal model
avg_crystal_params = create_average_crystal_model(crystal_models)

if avg_crystal_params:
    print("Averaged Crystal Parameters:")
    print(f"Unit cell: {avg_crystal_params['unit_cell_parameters']}")
    print(f"Volume: {avg_crystal_params['unit_cell_volume']:.1f} Å³")
    print(f"Space group: {avg_crystal_params['space_group'].info().symbol_and_number()}")
```

---

## C.7. CCTBX Asymmetric Unit (ASU) Mapping

**1. Purpose:**
Map Miller indices and fractional coordinates to crystallographic asymmetric unit for proper handling of symmetry-equivalent reflections in diffuse scattering analysis and comparison with theoretical models.

**2. Primary Python Call(s):**
```python
from cctbx import miller, sgtbx, crystal
from scitbx import matrix

def map_miller_indices_to_asu(miller_indices, space_group):
    """
    Map Miller indices to asymmetric unit
    
    Args:
        miller_indices: List of (h,k,l) tuples or flex.miller_index
        space_group: cctbx.sgtbx.space_group object
    
    Returns:
        List of ASU-mapped Miller indices
    """
    # Create miller set for batch ASU mapping
    miller_set = miller.set(
        crystal_symmetry=crystal.symmetry(
            space_group=space_group
        ),
        indices=miller_indices
    )
    
    # Map to ASU
    asu_miller_set = miller_set.map_to_asu()
    return asu_miller_set.indices()

def map_fractional_coords_to_asu(fractional_coords, space_group, epsilon=1e-6):
    """
    Map fractional coordinates to asymmetric unit
    
    Args:
        fractional_coords: List of (x,y,z) fractional coordinate tuples
        space_group: cctbx.sgtbx.space_group object
        epsilon: Tolerance for boundary conditions
    
    Returns:
        List of ASU-mapped fractional coordinates
    """
    asu_coords = []
    space_group_info = space_group.info()
    
    for coord in fractional_coords:
        # Map individual coordinate to ASU
        asu_coord = space_group_info.map_to_asu(coord, epsilon=epsilon)
        asu_coords.append(asu_coord)
    
    return asu_coords
```

**3. Example Usage Snippet:**
```python
from cctbx import crystal
from dxtbx.model.experiment_list import ExperimentListFactory

# Load crystal model with space group
experiments = ExperimentListFactory.from_json_file("experiments.expt")
crystal_model = experiments[0].crystal
space_group = crystal_model.get_space_group()
unit_cell = crystal_model.get_unit_cell()

# Create crystal symmetry object
crystal_symmetry = crystal.symmetry(
    unit_cell=unit_cell,
    space_group=space_group
)

print(f"Space group: {space_group.info().symbol_and_number()}")
print(f"ASU conditions: {space_group.info().asu()}")

# Example Miller indices from diffuse scattering analysis
test_indices = [(1, 2, 3), (2, 4, 6), (-1, -2, -3), (3, 2, 1)]

# Map to ASU
asu_indices = map_miller_indices_to_asu(test_indices, space_group)

print("Miller index ASU mapping:")
for orig, asu in zip(test_indices, asu_indices):
    print(f"  {orig} → {asu}")
```

---

## C.8. CCTBX Tabulated Compton Scattering Factors

**1. Purpose:**
Access tabulated incoherent (Compton) scattering factors for elements as a function of momentum transfer, essential for calculating theoretical diffuse scattering backgrounds and separating elastic from inelastic contributions.

**2. Primary Python Call(s):**
```python
from cctbx.eltbx import sasaki  # Primary CCTBX scattering factor access
from cctbx.eltbx.xray_scattering import wk1995
import math

def get_compton_scattering_factor(element, q_magnitude_or_stol, energy_kev=12.4, table="it1992"):
    """
    Get incoherent (Compton) scattering factor for an element using CCTBX tabulated data
    
    Args:
        element: Element symbol (e.g., "C", "N", "O")
        q_magnitude_or_stol: |q| in Å⁻¹ or sin(θ)/λ value
        energy_kev: X-ray energy in keV (for future compatibility)
        table: Scattering factor table ("it1992" ONLY - others have limitations)
    
    Returns:
        Incoherent scattering factor (electrons)
        
    Note:
        Based on CCTBX codebase investigation:
        - IT1992 is the ONLY table with direct incoherent() method access
        - Chantler tables are NOT available in current CCTBX distribution
        - Henke tables only provide fp()/fdp() anomalous factors, not incoherent
        - Use table="it1992" exclusively for reliable incoherent factor access
    """
    # Convert q to sin(theta)/lambda if needed
    if q_magnitude_or_stol > 2.0:  # Assume it's |q| in Å⁻¹
        stol = q_magnitude_or_stol / (4 * math.pi)
    else:  # Assume it's already sin(theta)/lambda
        stol = q_magnitude_or_stol
    
    if table == "it1992":
        # International Tables for Crystallography 1992 - RECOMMENDED
        # This is the ONLY table with reliable incoherent() method access in CCTBX
        try:
            scattering_factor_table = sasaki.table(element)
            s_incoh = scattering_factor_table.at_stol(stol).incoherent()
            return s_incoh
            
        except (ImportError, AttributeError, RuntimeError) as e:
            print(f"Warning: IT1992 incoherent factors not accessible for {element}: {e}")
            # Fallback to approximation using atomic number
            atomic_number = sasaki.table(element).atomic_number()
            return atomic_number * (1 - math.exp(-2 * stol**2))
    
    else:
        raise ValueError(f"Unknown table type: {table}. Use 'it1992' for reliable incoherent access")

def calculate_theoretical_compton_background(elements, concentrations, q_values, 
                                           wavelength_angstrom=1.0):
    """
    Calculate theoretical Compton scattering background for a compound
    
    Args:
        elements: List of element symbols
        concentrations: List of atomic fractions for each element
        q_values: Array of q-values (Å⁻¹)
        wavelength_angstrom: X-ray wavelength in Angstroms
    
    Returns:
        Array of Compton scattering intensities
    """
    compton_intensities = []
    
    for q_mag in q_values:
        total_compton = 0.0
        
        for element, concentration in zip(elements, concentrations):
            # Get incoherent scattering factor
            s_incoh = get_compton_scattering_factor(element, q_mag, table="it1992")
            
            # Weight by concentration
            total_compton += concentration * s_incoh
        
        compton_intensities.append(total_compton)
    
    return compton_intensities
```

**3. Example Usage Snippet:**
```python
import numpy as np

# Define compound composition (e.g., protein: C, N, O, S)
elements = ["C", "N", "O", "S"]
concentrations = [0.50, 0.16, 0.23, 0.03]  # Approximate protein composition
wavelength = 1.0  # Å

# Define q-range for analysis
q_values = np.linspace(0.1, 2.0, 100)

# Calculate Compton background
compton_background = calculate_theoretical_compton_background(
    elements, concentrations, q_values, wavelength
)

print("Compton scattering factors for different elements at |q| = 0.5 Å⁻¹:")
for element in ["C", "N", "O", "P", "S"]:
    s_incoh = get_compton_scattering_factor(element, 0.5)
    print(f"{element}: Compton = {s_incoh:.3f}")
```

**4. Notes/Caveats:**
- **Table Accuracy:** IT1992 is the only table with reliable incoherent factor access in CCTBX
- **Energy Dependence:** Incoherent scattering is weakly energy-dependent; coherent scattering shows strong energy dependence near absorption edges
- **Q-dependence:** Compton scattering factors increase with q; coherent factors decrease
- **Units:** Ensure consistent units (Å⁻¹ for q, keV for energy, electrons for scattering factors)

**5. Integration with Diffuse Scattering Pipeline:**
Compton scattering calculations provide theoretical baselines for separating elastic diffuse scattering from inelastic backgrounds, essential for accurate diffuse scattering analysis.

---

## See Also

- **File I/O Operations**: [dials_file_io.md](dials_file_io.md)
- **Detector Models**: [dxtbx_models.md](dxtbx_models.md)
- **DIALS Scaling**: [dials_scaling.md](dials_scaling.md)
- **Array Operations**: [flex_arrays.md](flex_arrays.md)