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

### A.0. dials.stills_process Python API

**1. Purpose:**
Provides a complete pipeline for processing still diffraction images through spot finding, indexing, refinement, and integration. Essential for generating per-still crystal models, reflection data with partiality, and shoebox data for Bragg mask generation in stills diffuse scattering pipelines.

**2. Primary Python Call(s):**
```python
from dials.command_line.stills_process import Processor, phil_scope, do_import
from libtbx.phil import parse

# Initialize processor with parameters
params = phil_scope.fetch(parse("")).extract()  # Default parameters
processor = Processor(params, composite_tag="pipeline_run", rank=0)

# Import image to experiment
experiments = do_import("image.cbf", load_models=True)

# Process through full pipeline
processor.process_experiments(tag="image_001", experiments=experiments)

# Access results
integrated_experiments = processor.all_integrated_experiments
integrated_reflections = processor.all_integrated_reflections
```

**3. Key Classes and Methods:**

**Processor Class:**
- `__init__(self, params, composite_tag=None, rank=0)`: Initialize processor
  - `params`: PHIL parameter object controlling processing behavior
  - `composite_tag`: Tag for aggregated output files
  - `rank`: Process rank for parallel processing
- `process_experiments(self, tag, experiments)`: Main processing pipeline
  - `tag`: String identifier for this processing run
  - `experiments`: ExperimentList object from do_import()
- `find_spots(self, experiments)` → flex.reflection_table: Spot finding step
- `index(self, experiments, reflections)` → (ExperimentList, flex.reflection_table): Indexing step
- `refine(self, experiments, reflections)` → (ExperimentList, flex.reflection_table): Refinement step  
- `integrate(self, experiments, reflections)` → flex.reflection_table: Integration step
- `finalize(self)`: Write final outputs to files

**do_import Function:**
- `do_import(filename, load_models=True)` → ExperimentList: Convert image file to experiments
  - `filename`: Path to image file (CBF, HDF5, etc.)
  - `load_models`: Whether to load detector/beam models from headers

**4. Key Output Attributes:**
- `processor.all_integrated_experiments`: ExperimentList with refined crystal models
- `processor.all_integrated_reflections`: Reflection table with partiality data
- `processor.all_strong_reflections`: Reflection table from spot finding
- `processor.all_indexed_reflections`: Reflection table with Miller indices

**5. Example Usage Snippet:**
```python
from dials.command_line.stills_process import Processor, phil_scope, do_import
from libtbx.phil import parse

# Configure parameters for stills processing
custom_phil = """
dispatch {
  find_spots = True
  index = True
  refine = True
  integrate = True
  squash_errors = False
}
output {
  composite_output = True
  shoeboxes = True
}
integration {
  debug {
    output = True
    separate_files = False
    delete_shoeboxes = True
  }
}
"""

# Initialize processor
params = phil_scope.fetch(parse(custom_phil)).extract()
processor = Processor(params, composite_tag="diffuse_pipeline")

# Process a still image
image_path = "still_001.cbf"
experiments = do_import(image_path)
processor.process_experiments(tag="still_001", experiments=experiments)

# Extract results for pipeline
for i, expt in enumerate(processor.all_integrated_experiments):
    # Per-still crystal model (Experiment_dials_i)
    crystal_model = expt.crystal
    unit_cell = crystal_model.get_unit_cell().parameters()
    space_group = crystal_model.get_space_group().info().symbol_and_number()
    A_matrix = crystal_model.get_A()  # Orientation + metric matrix
    U_matrix = crystal_model.get_U()  # Orientation matrix only
    
    print(f"Still {i} crystal model:")
    print(f"  Unit cell: a={unit_cell[0]:.3f}, b={unit_cell[1]:.3f}, c={unit_cell[2]:.3f} Å")
    print(f"  Space group: {space_group}")
    print(f"  A matrix determinant: {A_matrix.determinant():.6f}")
    
    # Per-still reflections with partiality (Reflections_dials_i)
    expt_reflections = processor.all_integrated_reflections.select(
        processor.all_integrated_reflections["id"] == i
    )
    
    # Access partiality data (P_spot)
    if "partiality" in expt_reflections:
        partialities = expt_reflections["partiality"]
        print(f"  {len(expt_reflections)} reflections, "
              f"partiality range: {flex.min(partialities):.3f}-{flex.max(partialities):.3f}")
    
    # Access shoeboxes for mask generation (BraggMask_2D_raw_i)
    if "shoebox" in expt_reflections:
        shoeboxes = expt_reflections["shoebox"]
        print(f"  {len(shoeboxes)} shoeboxes with 3D mask data")

processor.finalize()
```

**6. Parameter Configuration:**
The PHIL parameter system controls all processing aspects:
```python
# Essential parameters for diffuse scattering pipeline
key_params = """
# Algorithm control
dispatch.find_spots = True      # Enable spot finding
dispatch.index = True           # Enable indexing  
dispatch.refine = True          # Enable refinement (recommended)
dispatch.integrate = True       # Enable integration

# Hit finder (quality control)
dispatch.hit_finder.minimum_number_of_reflections = 10
dispatch.hit_finder.maximum_number_of_reflections = None

# Output control
output.composite_output = True  # Aggregate results efficiently
output.shoeboxes = True         # Save shoebox data for masking

# Integration settings for diffuse scattering
integration.integrator = stills  # Use stills-specific algorithms
integration.debug.output = True  # Enable shoebox saving
integration.debug.delete_shoeboxes = True  # Manage memory usage

# Refinement constraints for stills
refinement.parameterisation.beam.fix = all      # Fix beam parameters
refinement.parameterisation.detector.fix = all  # Fix detector parameters
"""
```

**7. Notes/Caveats:**
- **Environment Dependency:** Requires full DIALS/cctbx environment to be properly initialized
- **Memory Management:** Enable `integration.debug.delete_shoeboxes=True` for large datasets
- **Error Handling:** Set `dispatch.squash_errors=False` for debugging; `True` for production batch processing
- **Partiality Quality:** Partiality calculations are production-quality, based on 3-sigma Gaussian profile modeling
- **Per-Still Processing:** Each input image becomes a separate experiment with its own crystal model
- **Composite Output Behavior:** When `composite_output=True`, all processed stills are aggregated into single output files (processor.all_integrated_experiments, processor.all_integrated_reflections). When `False`, each still generates separate files with tag-based naming (e.g., `tag_integrated.refl`)

**8. Integration with Diffuse Scattering Pipeline:**
```python
# Module 1.S.1 Implementation Pattern
def process_still_for_diffuse_pipeline(image_path, base_params):
    """Process single still for diffuse scattering pipeline"""
    # Initialize processor
    processor = Processor(base_params, composite_tag=f"still_{image_id}")
    
    # Import and process
    experiments = do_import(image_path)
    processor.process_experiments(tag=f"image_{image_id}", experiments=experiments)
    
    # Extract pipeline outputs
    if len(processor.all_integrated_experiments) > 0:
        # Experiment_dials_i: Per-still crystal model
        crystal_model_i = processor.all_integrated_experiments[0]
        
        # Reflections_dials_i: Per-still reflections with P_spot  
        reflections_i = processor.all_integrated_reflections.select(
            processor.all_integrated_reflections["id"] == 0
        )
        
        # Extract P_spot partiality data
        partiality_i = reflections_i["partiality"] if "partiality" in reflections_i else None
        
        # BraggMask_2D_raw_i: From shoebox mask projection
        shoeboxes_i = reflections_i["shoebox"] if "shoebox" in reflections_i else None
        
        return crystal_model_i, reflections_i, partiality_i, shoeboxes_i
    else:
        return None, None, None, None
```

**9. See Also:**
- Section B.3: Crystal model access for per-still crystal parameters
- Section D.3: Reflection table operations for partiality data
- Section A.1: Loading experiment lists from stills_process outputs
- Section A.2: Loading reflection tables from stills_process outputs

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

---

### A.5. Shoebox Data and MaskCode (for Bragg Mask Generation)

**1. Purpose:**
Access and manipulate reflection shoebox data structures containing 3D pixel data and mask codes for generating 2D Bragg masks (BraggMask_2D_raw_i) from integrated reflections in diffuse scattering analysis.

**2. Primary Python Call(s):**
```python
from dials.array_family import flex
from dials.model.data import Shoebox
from dials.algorithms.shoebox import MaskCode

# Access shoeboxes from reflection table
reflections = flex.reflection_table.from_file("integrated.refl")
if "shoebox" in reflections:
    shoeboxes = reflections["shoebox"]
    
    for i, shoebox in enumerate(shoeboxes):
        # Shoebox properties
        data = shoebox.data           # flex.double - 3D pixel intensities
        mask = shoebox.mask           # flex.int - 3D mask codes
        bbox = shoebox.bbox           # (x1, x2, y1, y2, z1, z2) bounding box
        
        # Check mask consistency
        is_consistent = shoebox.is_consistent()
        
        # Count pixels by mask code
        n_foreground = shoebox.count_mask_values(MaskCode.Foreground)
        n_background = shoebox.count_mask_values(MaskCode.Background)
        n_valid = shoebox.count_mask_values(MaskCode.Valid)
```

**3. Key Shoebox Attributes and Methods:**

**Shoebox Object:**
- `data`: flex.double - 3D array of pixel intensities (z, y, x)
- `mask`: flex.int - 3D array of mask codes (same dimensions as data)
- `bbox`: tuple - Bounding box (x1, x2, y1, y2, z1, z2) in panel coordinates
- `panel`: int - Panel ID for multi-panel detectors
- `is_consistent()` → bool: Check data/mask size consistency
- `count_mask_values(mask_code)` → int: Count pixels with specific mask code
- `flatten()`: Project 3D shoebox to 2D (if available)

**MaskCode Enumeration:**
- `MaskCode.Valid`: Pixel is within trusted detector region
- `MaskCode.Foreground`: Pixel classified as signal/reflection
- `MaskCode.Background`: Pixel classified as background
- `MaskCode.Strong`: Pixel above strong threshold
- `MaskCode.BackgroundUsed`: Pixel used in background calculation
- `MaskCode.Overlapped`: Pixel overlaps with another reflection

**4. Return Types:**
- `data`: flex.double array with shape (nz, ny, nx)
- `mask`: flex.int array with same shape, containing bitwise mask codes
- `bbox`: tuple of 6 integers defining bounding box
- Count methods: integer pixel counts

**5. Example Usage Snippet:**
```python
from dials.array_family import flex
from dials.algorithms.shoebox import MaskCode

# Load reflections with shoebox data
reflections = flex.reflection_table.from_file("integrated.refl")

if "shoebox" in reflections:
    shoeboxes = reflections["shoebox"]
    print(f"Found {len(shoeboxes)} shoeboxes")
    
    # Analyze first few shoeboxes
    for i in range(min(5, len(shoeboxes))):
        shoebox = shoeboxes[i]
        
        # Basic shoebox information
        bbox = shoebox.bbox
        panel_id = shoebox.panel
        data_shape = shoebox.data.accessor().all()
        
        # Count pixels by classification
        n_valid = shoebox.count_mask_values(MaskCode.Valid)
        n_foreground = shoebox.count_mask_values(MaskCode.Foreground)
        n_background = shoebox.count_mask_values(MaskCode.Background)
        n_strong = shoebox.count_mask_values(MaskCode.Strong)
        
        print(f"Shoebox {i}:")
        print(f"  Panel: {panel_id}, BBox: {bbox}")
        print(f"  Shape: {data_shape} (z, y, x)")
        print(f"  Valid: {n_valid}, Foreground: {n_foreground}, Background: {n_background}")
        print(f"  Strong pixels: {n_strong}")
        print(f"  Consistent: {shoebox.is_consistent()}")
```

**6. Generating 2D Bragg Masks from 3D Shoeboxes:**
```python
def create_2d_bragg_mask_from_shoeboxes(reflections, detector, mask_type="foreground"):
    """
    Generate 2D Bragg mask for each detector panel from 3D shoebox data.
    
    Args:
        reflections: flex.reflection_table with shoebox data
        detector: dxtbx.model.Detector object
        mask_type: "foreground", "strong", or "all_signal"
    
    Returns:
        List of flex.bool arrays (one per panel) for 2D Bragg masks
    """
    from dials.algorithms.shoebox import MaskCode
    
    # Initialize 2D masks for each panel
    panel_masks = []
    for panel_id, panel in enumerate(detector):
        panel_shape = panel.get_image_size()  # (fast, slow)
        panel_mask = flex.bool(flex.grid(panel_shape[1], panel_shape[0]), False)
        panel_masks.append(panel_mask)
    
    # Process each shoebox
    if "shoebox" in reflections:
        shoeboxes = reflections["shoebox"]
        
        for shoebox in shoeboxes:
            panel_id = shoebox.panel
            bbox = shoebox.bbox  # (x1, x2, y1, y2, z1, z2)
            mask_3d = shoebox.mask
            
            # Select mask criteria based on type
            if mask_type == "foreground":
                target_mask = MaskCode.Foreground | MaskCode.Valid
            elif mask_type == "strong": 
                target_mask = MaskCode.Strong | MaskCode.Valid
            elif mask_type == "all_signal":
                target_mask = (MaskCode.Foreground | MaskCode.Strong) & MaskCode.Valid
            
            # Project 3D mask to 2D by OR-ing across z-slices
            mask_shape = mask_3d.accessor().all()  # (nz, ny, nx)
            
            for y in range(mask_shape[1]):  # slow direction
                for x in range(mask_shape[0]):  # fast direction
                    # Check if any z-slice has the target mask
                    has_signal = False
                    for z in range(mask_shape[2]):
                        mask_value = mask_3d[z, y, x]
                        if (mask_value & target_mask) == target_mask:
                            has_signal = True
                            break
                    
                    # Set 2D mask pixel
                    if has_signal:
                        panel_y = bbox[2] + y  # Global panel coordinates
                        panel_x = bbox[0] + x
                        if (0 <= panel_x < panel_masks[panel_id].accessor().all()[1] and
                            0 <= panel_y < panel_masks[panel_id].accessor().all()[0]):
                            panel_masks[panel_id][panel_y, panel_x] = True
    
    return panel_masks

# Usage example
bragg_masks = create_2d_bragg_mask_from_shoeboxes(
    reflections, detector, mask_type="foreground"
)

for panel_id, mask in enumerate(bragg_masks):
    n_bragg_pixels = flex.sum(mask.as_1d())
    total_pixels = len(mask)
    print(f"Panel {panel_id}: {n_bragg_pixels}/{total_pixels} Bragg pixels")
```

**7. Advanced Shoebox Analysis:**
```python
def analyze_shoebox_properties(reflections):
    """Detailed analysis of shoebox data for quality assessment"""
    if "shoebox" not in reflections:
        return
    
    shoeboxes = reflections["shoebox"]
    
    # Collect shoebox statistics
    volumes = []
    foreground_fractions = []
    signal_to_noise = []
    
    for i, shoebox in enumerate(shoeboxes):
        if not shoebox.is_consistent():
            continue
            
        # Volume analysis
        data_shape = shoebox.data.accessor().all()
        volume = data_shape[0] * data_shape[1] * data_shape[2]
        volumes.append(volume)
        
        # Mask analysis
        n_valid = shoebox.count_mask_values(MaskCode.Valid)
        n_foreground = shoebox.count_mask_values(MaskCode.Foreground)
        n_background = shoebox.count_mask_values(MaskCode.Background)
        
        if n_valid > 0:
            fg_fraction = n_foreground / n_valid
            foreground_fractions.append(fg_fraction)
        
        # Signal analysis (if intensity data available)
        if hasattr(reflections, 'select') and i < len(reflections):
            refl_subset = reflections.select(flex.size_t([i]))
            if "intensity.sum.value" in refl_subset and "intensity.sum.variance" in refl_subset:
                intensity = refl_subset["intensity.sum.value"][0]
                variance = refl_subset["intensity.sum.variance"][0]
                if variance > 0:
                    snr = intensity / (variance ** 0.5)
                    signal_to_noise.append(snr)
    
    # Print statistics
    if volumes:
        print(f"Shoebox volumes: {min(volumes)} - {max(volumes)} pixels")
        print(f"Mean volume: {sum(volumes)/len(volumes):.1f} pixels")
    
    if foreground_fractions:
        print(f"Foreground fraction: {min(foreground_fractions):.3f} - {max(foreground_fractions):.3f}")
        print(f"Mean foreground fraction: {sum(foreground_fractions)/len(foreground_fractions):.3f}")
    
    if signal_to_noise:
        print(f"Signal-to-noise: {min(signal_to_noise):.1f} - {max(signal_to_noise):.1f}")
        print(f"Mean S/N: {sum(signal_to_noise)/len(signal_to_noise):.1f}")

# Usage
analyze_shoebox_properties(reflections)
```

**8. Notes/Caveats:**
- **Memory Usage:** Shoeboxes can consume significant memory; use `integration.debug.delete_shoeboxes=True` for production
- **3D Structure:** Shoebox data is organized as (z, y, x) with z=frame, y=slow, x=fast detector coordinates
- **Mask Codes:** Use bitwise operations for combining mask criteria: `(mask & MaskCode.Foreground) != 0`
- **Bounding Box Coordinates:** bbox coordinates are in panel pixel units, not global detector coordinates
- **Panel Consistency:** Always check `shoebox.panel` when working with multi-panel detectors
- **Consistency Check:** Use `shoebox.is_consistent()` to verify data integrity before processing

**9. Integration with Diffuse Scattering Pipeline:**
This shoebox data provides the foundation for generating `BraggMask_2D_raw_i` in Module 2.1.D by projecting 3D reflection regions to 2D detector masks, enabling accurate separation of Bragg and diffuse scattering components.

**10. See Also:**
- Section A.0: dials.stills_process for generating shoebox data
- Section A.4: Loading mask files for comparison with generated Bragg masks
- Section D.3: Reflection table operations for accessing shoebox columns

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

### A.6. Python API for dials.generate_mask

**1. Purpose:**
Generate detector pixel masks programmatically using DIALS masking utilities, including resolution limits, geometric exclusions, and powder ring masking for diffuse scattering data preprocessing.

**2. Primary Python Call(s):**
```python
from dials.util.masking import generate_mask
from dials.command_line.generate_mask import Script as GenerateMaskScript
from libtbx.phil import parse
from dxtbx.model import ExperimentList

# Method 1: Using generate_mask utility function directly
def create_detector_mask(experiments, phil_params):
    """Generate mask using DIALS masking utilities"""
    mask = generate_mask(experiments, phil_params)
    return mask

# Method 2: Using generate_mask command-line script programmatically  
def create_mask_from_script(experiments, mask_params):
    """Use dials.generate_mask script logic programmatically"""
    script = GenerateMaskScript()
    
    # Combine experiments with mask parameters
    mask_data = script.generate_mask_from_experiments(
        experiments, mask_params
    )
    return mask_data
```

**3. Key Functions and Parameters:**

**generate_mask Function:**
- `generate_mask(experiments, params)` → tuple of flex.bool arrays
  - `experiments`: ExperimentList object with detector/beam models
  - `params`: PHIL parameter object with masking criteria
  - Returns: tuple of flex.bool masks (one per detector panel)

**Key PHIL Parameters for Masking:**
```python
mask_phil_str = """
border = 0
  .type = int
  .help = "Number of pixels to mask around detector edge"

d_min = None
  .type = float  
  .help = "Minimum resolution limit (Angstroms)"

d_max = None
  .type = float
  .help = "Maximum resolution limit (Angstroms)"

resolution_range = None
  .type = floats(size=2)
  .help = "Resolution range to mask [d_min, d_max]"
  .multiple = True

untrusted {
  circle = None
    .type = floats(size=3)
    .help = "Circular exclusion: x_center y_center radius"
    .multiple = True
  
  rectangle = None
    .type = floats(size=4) 
    .help = "Rectangular exclusion: x1 y1 x2 y2"
    .multiple = True
    
  polygon = None
    .type = floats
    .help = "Polygon vertices: x1 y1 x2 y2 ..."
    .multiple = True
    
  pixel = None
    .type = ints(size=2)
    .help = "Individual pixel: x y"
    .multiple = True
}

ice_rings {
  filter = False
    .type = bool
    .help = "Apply ice ring masking"
    
  d_min = 10.0
    .type = float
    .help = "Minimum d-spacing for ice ring detection"
}
"""
```

**4. Return Type:**
- Tuple of `flex.bool` arrays, one per detector panel
- `True` indicates valid pixels, `False` indicates masked pixels
- Compatible with DIALS masking conventions

**5. Example Usage Snippet:**
```python
from dials.util.masking import generate_mask
from dxtbx.model.experiment_list import ExperimentListFactory
from libtbx.phil import parse

# Load experiments
experiments = ExperimentListFactory.from_json_file("experiments.expt")

# Define masking parameters
mask_phil = """
border = 5
d_min = 1.5
d_max = 50.0
resolution_range = 2.0 3.0
resolution_range = 4.0 5.0
untrusted {
  circle = 1000 1000 50
  circle = 500 500 30
  rectangle = 0 0 100 100
}
ice_rings {
  filter = True
  d_min = 8.0  
}
"""

# Parse parameters
mask_params = parse(mask_phil).extract()

# Generate mask
detector_mask = generate_mask(experiments, mask_params)

# Apply mask to image data
if isinstance(detector_mask, tuple):
    for panel_id, panel_mask in enumerate(detector_mask):
        n_masked = flex.sum((~panel_mask).as_1d())
        n_total = len(panel_mask)
        print(f"Panel {panel_id}: {n_masked}/{n_total} pixels masked")
```

**6. Diffuse Scattering Mask Strategy:**
```python
def create_diffuse_scattering_mask(experiments, bragg_masks=None):
    """
    Create comprehensive mask for diffuse scattering analysis
    combining resolution limits, geometric exclusions, and Bragg masking
    
    Args:
        experiments: ExperimentList with detector/beam models
        bragg_masks: Optional tuple of flex.bool masks for Bragg reflection exclusion
    
    Returns:
        Tuple of flex.bool masks optimized for diffuse scattering analysis
    """
    from dials.util.masking import generate_mask
    from libtbx.phil import parse
    
    # Define comprehensive masking strategy for diffuse scattering
    phil_string = """
    # Edge masking - exclude detector borders
    border = 10
    
    # Resolution limits for diffuse scattering range
    d_min = 1.0    # High resolution cutoff (avoid noise)
    d_max = 100.0  # Low resolution cutoff (avoid beamstop shadow)
    
    # Exclude problematic resolution ranges
    resolution_range = 3.9 3.7   # Ice ring ~3.8 Å
    resolution_range = 2.7 2.6   # Ice ring ~2.65 Å
    resolution_range = 2.25 2.20 # Ice ring ~2.22 Å
    
    # Geometric exclusions for common artifacts
    untrusted {
      circle = 1024 1024 80     # Beamstop shadow (adjust coordinates)
      rectangle = 1020 0 1030 2048  # Module gap (adjust for detector)
      rectangle = 0 1020 2048 1030  # Module gap (adjust for detector)
    }
    
    # Automatic ice ring detection and masking
    ice_rings {
      filter = True
      d_min = 10.0
    }
    """
    
    params = parse(phil_string).extract()
    detector_mask = generate_mask(experiments, params)
    
    # Combine with Bragg masks if provided
    if bragg_masks is not None:
        combined_masks = []
        for panel_id, (detector_panel_mask, bragg_panel_mask) in enumerate(zip(detector_mask, bragg_masks)):
            # Exclude pixels that are geometrically excluded OR contain Bragg reflections
            combined_mask = detector_panel_mask & (~bragg_panel_mask)
            combined_masks.append(combined_mask)
        return tuple(combined_masks)
    
    return detector_mask

# Example usage in diffuse scattering pipeline
detector_mask = create_diffuse_scattering_mask(experiments)
print(f"Created base diffuse scattering mask for {len(detector_mask)} panels")

# Optional: combine with Bragg masks from Section A.5
if have_bragg_reflections:
    bragg_masks = create_2d_bragg_mask_from_shoeboxes(reflections, experiments[0].detector)
    combined_mask = create_diffuse_scattering_mask(experiments, bragg_masks)
    print("Combined detector mask with Bragg exclusions")
```

**7. Advanced Custom Masking Functions:**
```python
def create_custom_resolution_mask(experiments, d_min=None, d_max=None, exclude_ranges=None):
    """
    Create resolution-based mask with custom d-spacing exclusions
    
    Args:
        experiments: ExperimentList
        d_min: Minimum d-spacing (high resolution limit)
        d_max: Maximum d-spacing (low resolution limit)  
        exclude_ranges: List of (d_min, d_max) tuples to exclude
    """
    detector = experiments[0].detector
    beam = experiments[0].beam
    wavelength = beam.get_wavelength()
    
    panel_masks = []
    
    for panel_id, panel in enumerate(detector):
        panel_shape = panel.get_image_size()  # (fast, slow)
        panel_mask = flex.bool(flex.grid(panel_shape[1], panel_shape[0]), True)
        
        # Check each pixel's resolution
        for slow in range(panel_shape[1]):
            for fast in range(panel_shape[0]):
                # Calculate resolution for this pixel
                lab_coord = panel.get_pixel_lab_coord((fast, slow))
                distance = (lab_coord[0]**2 + lab_coord[1]**2 + lab_coord[2]**2)**0.5
                
                # Scattering angle
                two_theta = math.atan2(
                    (lab_coord[0]**2 + lab_coord[1]**2)**0.5, 
                    lab_coord[2]
                )
                
                # d-spacing calculation: d = λ / (2 * sin(θ))
                if two_theta > 0:
                    d_spacing = wavelength / (2 * math.sin(two_theta / 2))
                else:
                    d_spacing = float('inf')
                
                # Apply resolution limits
                mask_pixel = True
                
                if d_min is not None and d_spacing < d_min:
                    mask_pixel = False
                if d_max is not None and d_spacing > d_max:
                    mask_pixel = False
                    
                # Apply exclusion ranges
                if exclude_ranges:
                    for range_min, range_max in exclude_ranges:
                        if range_min <= d_spacing <= range_max:
                            mask_pixel = False
                            break
                
                panel_mask[slow, fast] = mask_pixel
        
        panel_masks.append(panel_mask)
    
    return tuple(panel_masks)

# Usage example
resolution_mask = create_custom_resolution_mask(
    experiments, 
    d_min=1.2, 
    d_max=20.0,
    exclude_ranges=[(3.9, 3.7), (2.7, 2.6), (2.25, 2.20)]  # Ice rings
)
```

**8. Mask Persistence and Reuse:**
```python
import pickle

def save_detector_mask(mask_tuple, filename):
    """Save detector mask to pickle file"""
    with open(filename, 'wb') as f:
        pickle.dump(mask_tuple, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved mask to {filename}")

def load_detector_mask(filename):
    """Load detector mask from pickle file"""
    with open(filename, 'rb') as f:
        mask_tuple = pickle.load(f)
    return mask_tuple

# Usage
save_detector_mask(detector_mask, "diffuse_scattering_mask.pickle")
loaded_mask = load_detector_mask("diffuse_scattering_mask.pickle")
```

**9. Notes/Caveats:**
- **Coordinate Systems:** Masking coordinates are in panel pixel coordinates (fast, slow)
- **Resolution Calculations:** Require accurate detector geometry and beam models
- **Ice Ring Detection:** Built-in ice ring positions may need adjustment for specific wavelengths
- **Memory Efficiency:** Large detector masks can consume significant memory; save/load as needed
- **Panel Indexing:** Always verify panel indexing consistency across mask operations
- **Boolean Convention:** DIALS uses `True` for valid pixels, `False` for masked pixels
- **Strategy Recommendation:** For diffuse scattering, use the comprehensive masking approach in example 6 rather than individual custom functions

**10. Integration with Pipeline:**
Generated masks are essential for Module 2.1.D geometric corrections and Module 2.2.D diffuse scattering extraction, providing precise control over which detector regions are included in analysis.

**11. See Also:**
- Section A.5: Shoebox data for Bragg mask generation to combine with detector masks
- Section B.1: Detector model for coordinate transformations in masking
- Section C.1: Q-vector calculations for resolution-based masking
- Section A.4: Loading existing mask files

---

## B. Accessing and Using dxtbx.model Objects

### B.0. DXTBX Masking Utilities

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

### C.6. CCTBX Crystal Model Averaging

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

def average_orientation_matrices(U_matrices, weights=None):
    """
    Average orientation matrices using quaternion interpolation
    
    Args:
        U_matrices: List of scitbx.matrix.sqr orientation matrices
        weights: Optional list of weights for each matrix
    
    Returns:
        scitbx.matrix.sqr: Averaged orientation matrix
    """
    from scitbx.math import r3_rotation_axis_and_angle_from_matrix
    
    if not U_matrices:
        return None
    
    if weights is None:
        weights = [1.0] * len(U_matrices)
    
    # Convert matrices to axis-angle representation for averaging
    axis_angles = []
    for U in U_matrices:
        try:
            axis, angle = r3_rotation_axis_and_angle_from_matrix(U)
            axis_angles.append((matrix.col(axis), angle))
        except RuntimeError:
            # Handle identity or near-identity matrices
            axis_angles.append((matrix.col((0, 0, 1)), 0.0))
    
    # Average axis-angle representations (simplified approach)
    total_weight = sum(weights)
    
    # For small angle approximation, average the rotation vectors
    weighted_rotation_vectors = []
    for (axis, angle), weight in zip(axis_angles, weights):
        rotation_vector = axis * angle * weight
        weighted_rotation_vectors.append(rotation_vector)
    
    # Sum weighted rotation vectors
    avg_rotation_vector = sum(weighted_rotation_vectors, matrix.col((0, 0, 0))) / total_weight
    
    # Convert back to matrix
    avg_angle = avg_rotation_vector.length()
    if avg_angle > 1e-6:
        avg_axis = avg_rotation_vector.normalize()
        # Create rotation matrix from axis and angle
        cos_angle = math.cos(avg_angle)
        sin_angle = math.sin(avg_angle)
        one_minus_cos = 1 - cos_angle
        
        x, y, z = avg_axis
        
        avg_U = matrix.sqr((
            cos_angle + x*x*one_minus_cos,     x*y*one_minus_cos - z*sin_angle, x*z*one_minus_cos + y*sin_angle,
            y*x*one_minus_cos + z*sin_angle,  cos_angle + y*y*one_minus_cos,    y*z*one_minus_cos - x*sin_angle,
            z*x*one_minus_cos - y*sin_angle,  z*y*one_minus_cos + x*sin_angle, cos_angle + z*z*one_minus_cos
        ))
    else:
        avg_U = matrix.identity(3)
    
    return avg_U

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
    avg_U_matrix = average_orientation_matrices(U_matrices, weights)
    
    # Calculate average B matrix from averaged unit cell
    avg_B_matrix = matrix.sqr(avg_unit_cell.fractionalization_matrix()).transpose()
    
    # Calculate average A matrix
    avg_A_matrix = avg_U_matrix * avg_B_matrix
    
    return {
        'unit_cell': avg_unit_cell,
        'space_group': reference_sg,
        'U_matrix': avg_U_matrix,
        'B_matrix': avg_B_matrix,
        'A_matrix': avg_A_matrix,
        'unit_cell_parameters': avg_unit_cell.parameters(),
        'unit_cell_volume': avg_unit_cell.volume()
    }
```

**3. Key Arguments:**
- `unit_cells`: List of cctbx.uctbx.unit_cell objects
- `U_matrices`: List of scitbx.matrix.sqr (3×3) orientation matrices
- `crystal_models`: List of dxtbx.model.Crystal objects
- `weights`: Optional list of float weights for each input

**4. Return Types:**
- `averaged_unit_cell`: cctbx.uctbx.unit_cell object
- `averaged_U_matrix`: scitbx.matrix.sqr (3×3) matrix
- `average_crystal_dict`: Dictionary with averaged crystal parameters

**5. Example Usage Snippet:**
```python
# Load multiple crystal models from stills processing
crystal_models = []
for expt_file in ["still_001_integrated.expt", "still_002_integrated.expt", "still_003_integrated.expt"]:
    experiments = ExperimentListFactory.from_json_file(expt_file)
    if len(experiments) > 0:
        crystal_models.append(experiments[0].crystal)

print(f"Loaded {len(crystal_models)} crystal models for averaging")

# Calculate weighted average based on integration quality (example)
weights = []
for i, crystal in enumerate(crystal_models):
    # Weight by unit cell volume consistency (example metric)
    volume = crystal.get_unit_cell().volume()
    weight = 1.0 / (1.0 + abs(volume - 1000.0) / 1000.0)  # Example weighting
    weights.append(weight)

# Create averaged crystal model
avg_crystal_params = create_average_crystal_model(crystal_models, weights)

if avg_crystal_params:
    print("Averaged Crystal Parameters:")
    print(f"Unit cell: {avg_crystal_params['unit_cell_parameters']}")
    print(f"Volume: {avg_crystal_params['unit_cell_volume']:.1f} Å³")
    print(f"Space group: {avg_crystal_params['space_group'].info().symbol_and_number()}")
    
    # Analyze parameter distributions
    unit_cells = [crystal.get_unit_cell() for crystal in crystal_models]
    volumes = [uc.volume() for uc in unit_cells]
    
    print(f"\nUnit cell volume statistics:")
    print(f"Mean: {sum(volumes)/len(volumes):.1f} Å³")
    print(f"Range: {min(volumes):.1f} - {max(volumes):.1f} Å³")
    print(f"Std dev: {(sum((v - sum(volumes)/len(volumes))**2 for v in volumes)/(len(volumes)-1))**0.5:.1f} Å³")
```

**6. Statistical Analysis of Crystal Parameters:**
```python
def analyze_crystal_parameter_distributions(crystal_models):
    """
    Analyze distributions of crystal parameters for quality assessment
    """
    unit_cells = [crystal.get_unit_cell() for crystal in crystal_models]
    
    # Extract all parameters
    all_params = [uc.parameters() for uc in unit_cells]
    param_names = ['a', 'b', 'c', 'alpha', 'beta', 'gamma']
    
    statistics = {}
    
    for i, param_name in enumerate(param_names):
        values = [params[i] for params in all_params]
        mean_val = sum(values) / len(values)
        var_val = sum((v - mean_val)**2 for v in values) / (len(values) - 1) if len(values) > 1 else 0
        std_val = var_val**0.5
        
        statistics[param_name] = {
            'mean': mean_val,
            'std': std_val,
            'min': min(values),
            'max': max(values),
            'cv': std_val / mean_val if mean_val > 0 else 0  # Coefficient of variation
        }
    
    # Volume statistics
    volumes = [uc.volume() for uc in unit_cells]
    mean_vol = sum(volumes) / len(volumes)
    std_vol = (sum((v - mean_vol)**2 for v in volumes) / (len(volumes) - 1))**0.5 if len(volumes) > 1 else 0
    
    statistics['volume'] = {
        'mean': mean_vol,
        'std': std_vol,
        'min': min(volumes),
        'max': max(volumes),
        'cv': std_vol / mean_vol if mean_vol > 0 else 0
    }
    
    return statistics

# Usage
param_stats = analyze_crystal_parameter_distributions(crystal_models)
for param, stats in param_stats.items():
    print(f"{param}: {stats['mean']:.3f} ± {stats['std']:.3f} (CV: {stats['cv']:.3f})")
```

**7. Notes/Caveats:**
- **Space Group Consistency:** Ensure all crystal models have the same space group before averaging
- **Orientation Matrix Averaging:** Uses simplified axis-angle averaging; more sophisticated quaternion SLERP may be needed for large rotations
- **Weight Selection:** Choose weights based on integration quality, resolution, or other relevant metrics
- **Statistical Validation:** Large parameter variations may indicate processing issues or genuine crystal variation
- **Coordinate System:** Orientation matrices must be in the same coordinate system (lab frame)

**8. Integration with Diffuse Scattering Pipeline:**
Averaged crystal parameters provide a reference model for diffuse scattering analysis and help assess the quality and consistency of per-still crystal determination.

---

### C.7. CCTBX Asymmetric Unit (ASU) Mapping

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

def create_asu_mapping_for_diffuse_analysis(crystal_symmetry, q_vectors):
    """
    Create ASU mapping for diffuse scattering q-vectors in fractional coordinates
    
    Args:
        crystal_symmetry: cctbx.crystal.symmetry object
        q_vectors: List of q-vectors in reciprocal space
    
    Returns:
        Dictionary with ASU mappings and multiplicities
    """
    space_group = crystal_symmetry.space_group()
    unit_cell = crystal_symmetry.unit_cell()
    
    # Convert q-vectors to fractional Miller indices
    A_matrix = matrix.sqr(unit_cell.fractionalization_matrix()).transpose()
    A_inverse = A_matrix.inverse()
    
    fractional_indices = []
    for q_vec in q_vectors:
        hkl_frac = A_inverse * matrix.col(q_vec)
        fractional_indices.append(tuple(hkl_frac))
    
    # Map to ASU
    asu_coords = map_fractional_coords_to_asu(fractional_indices, space_group)
    
    # Calculate multiplicities (number of symmetry equivalents)
    multiplicities = []
    for coord in fractional_indices:
        # Generate all symmetry equivalents
        equiv_coords = []
        for sym_op in space_group.all_ops():
            equiv_coord = sym_op * coord
            # Reduce to unit cell
            equiv_coord = tuple(x - int(x) if x >= 0 else x - int(x) + 1 for x in equiv_coord)
            equiv_coords.append(equiv_coord)
        
        # Count unique equivalents (within tolerance)
        unique_coords = []
        tolerance = 1e-4
        for equiv in equiv_coords:
            is_unique = True
            for unique in unique_coords:
                if all(abs(equiv[i] - unique[i]) < tolerance for i in range(3)):
                    is_unique = False
                    break
            if is_unique:
                unique_coords.append(equiv)
        
        multiplicities.append(len(unique_coords))
    
    return {
        'original_coords': fractional_indices,
        'asu_coords': asu_coords,
        'multiplicities': multiplicities,
        'space_group': space_group,
        'space_group_symbol': space_group.info().symbol_and_number()
    }
```

**3. Key Arguments:**
- `miller_indices`: List of (h,k,l) tuples or flex.miller_index array
- `fractional_coords`: List of (x,y,z) fractional coordinate tuples
- `space_group`: cctbx.sgtbx.space_group object
- `crystal_symmetry`: cctbx.crystal.symmetry object
- `epsilon`: Float tolerance for ASU boundary handling

**4. Return Types:**
- `asu_miller_indices`: flex.miller_index array mapped to ASU
- `asu_coords`: List of (x,y,z) tuples in ASU
- `mapping_dict`: Dictionary with original coords, ASU coords, and multiplicities

**5. Example Usage Snippet:**
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

# Example fractional coordinates
fractional_coords = [(0.1, 0.2, 0.3), (0.7, 0.8, 0.9), (-0.1, 1.2, 0.5)]
asu_coords = map_fractional_coords_to_asu(fractional_coords, space_group)

print("\nFractional coordinate ASU mapping:")
for orig, asu in zip(fractional_coords, asu_coords):
    print(f"  {orig} → {asu}")
```

**6. Advanced ASU Analysis for Diffuse Scattering:**
```python
def analyze_diffuse_scattering_asu_coverage(reflections, crystal_symmetry):
    """
    Analyze ASU coverage of diffuse scattering data
    """
    from dials.array_family import flex
    
    space_group = crystal_symmetry.space_group()
    unit_cell = crystal_symmetry.unit_cell()
    
    # Extract fractional Miller indices from reflections
    if "miller_index" in reflections:
        miller_indices = reflections["miller_index"]
    else:
        raise ValueError("Reflections table must contain Miller indices")
    
    # Create miller set and map to ASU
    miller_set = miller.set(
        crystal_symmetry=crystal_symmetry,
        indices=miller_indices
    )
    
    asu_miller_set = miller_set.map_to_asu()
    asu_indices = asu_miller_set.indices()
    
    # Analyze coverage
    unique_asu_indices = set(tuple(hkl) for hkl in asu_indices)
    
    # Calculate multiplicities for unique indices
    multiplicity_analysis = {}
    for unique_hkl in unique_asu_indices:
        # Count occurrences in original data
        orig_count = 0
        for orig_hkl in miller_indices:
            # Map original to ASU and compare
            test_set = miller.set(
                crystal_symmetry=crystal_symmetry,
                indices=flex.miller_index([orig_hkl])
            )
            test_asu = test_set.map_to_asu().indices()[0]
            if tuple(test_asu) == unique_hkl:
                orig_count += 1
        
        # Theoretical multiplicity
        theoretical_mult = space_group.multiplicity(unique_hkl)
        
        multiplicity_analysis[unique_hkl] = {
            'observed_count': orig_count,
            'theoretical_multiplicity': theoretical_mult,
            'coverage_fraction': orig_count / theoretical_mult if theoretical_mult > 0 else 0
        }
    
    return {
        'total_reflections': len(miller_indices),
        'unique_asu_reflections': len(unique_asu_indices),
        'redundancy': len(miller_indices) / len(unique_asu_indices) if unique_asu_indices else 0,
        'multiplicity_analysis': multiplicity_analysis,
        'space_group_info': space_group.info().symbol_and_number()
    }

# Usage example
if "miller_index" in reflections:
    asu_analysis = analyze_diffuse_scattering_asu_coverage(reflections, crystal_symmetry)
    print(f"ASU Analysis Results:")
    print(f"Total reflections: {asu_analysis['total_reflections']}")
    print(f"Unique ASU reflections: {asu_analysis['unique_asu_reflections']}")
    print(f"Average redundancy: {asu_analysis['redundancy']:.2f}")
```

**7. Notes/Caveats:**
- **Space Group Accuracy:** ASU mapping requires correct space group determination
- **Boundary Conditions:** Use appropriate epsilon values for fractional coordinates near ASU boundaries
- **Systematic Absences:** ASU mapping doesn't handle systematic absences; use separate validation
- **Coordinate Precision:** Numerical precision affects ASU boundary assignment
- **Multiplicity Calculations:** Account for special positions with lower multiplicity

**8. Integration with Diffuse Scattering Pipeline:**
ASU mapping ensures proper symmetry handling in diffuse scattering analysis, enabling accurate comparison with theoretical models and proper statistical analysis of symmetry-equivalent regions.

**9. See Also:**
- Section B.3: Crystal model for space group access
- Section C.2: Miller index transformations
- Section C.3: d-spacing calculations for ASU reflections

---

### C.8. CCTBX Tabulated Compton Scattering Factors

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
            
    elif table == "chantler":
        # WARNING: Chantler tables are NOT available in standard CCTBX distribution
        print(f"Warning: Chantler tables not available in CCTBX. Using IT1992 instead.")
        return get_compton_scattering_factor(element, q_magnitude_or_stol, energy_kev, table="it1992")
        
    elif table == "henke":
        # WARNING: Henke tables only provide fp/fdp, not incoherent factors
        print(f"Warning: Henke tables do not provide incoherent factors. Using IT1992 instead.")
        return get_compton_scattering_factor(element, q_magnitude_or_stol, energy_kev, table="it1992")
            
    elif table == "wk1995":
        # Waasmaier-Kirfel 1995 form factors (coherent only, so approximate incoherent)
        sf_info = wk1995(element, True)
        atomic_number = sf_info.atomic_number()
        
        # Approximation for incoherent scattering based on atomic physics
        # S_incoh ≈ Z * [1 - exp(-2*(sin(θ)/λ)²)] * correction_factor
        correction = 1 + 0.2 * stol  # Slight momentum transfer dependence
        s_incoh = atomic_number * (1 - math.exp(-2 * stol**2)) * correction
        return s_incoh
    
    else:
        raise ValueError(f"Unknown table type: {table}. Use 'it1992' for reliable incoherent access, or 'wk1995' for approximation")

def get_tabulated_incoherent_factors_direct(element, stol_values, table="it1992"):
    """
    Direct access to tabulated incoherent scattering factors from CCTBX tables
    
    Args:
        element: Element symbol
        stol_values: Array of sin(θ)/λ values
        table: Table type ("it1992" ONLY table with reliable incoherent access)
    
    Returns:
        Array of incoherent scattering factors
        
    Note:
        This is the most direct way to get tabulated incoherent factors from CCTBX.
        IT1992 is the only table with direct incoherent() method access.
    """
    try:
        if table == "it1992":
            # Use sasaki interface which provides access to IT1992 data
            scattering_factor_table = sasaki.table(element)
            
            incoherent_factors = []
            for stol in stol_values:
                try:
                    factor_data = scattering_factor_table.at_stol(stol)
                    s_incoh = factor_data.incoherent()
                    incoherent_factors.append(s_incoh)
                except Exception as e:
                    # Fallback for problematic stol values
                    atomic_number = scattering_factor_table.atomic_number()
                    s_incoh = atomic_number * (1 - math.exp(-2 * stol**2))
                    incoherent_factors.append(s_incoh)
                    print(f"Warning: Using approximation for {element} at stol={stol}: {e}")
            
            return incoherent_factors
            
    except ImportError as e:
        print(f"Warning: CCTBX sasaki module not available: {e}")
        
    # Fallback: use WK1995 with approximation
    sf_info = wk1995(element, True)
    atomic_number = sf_info.atomic_number()
    
    return [atomic_number * (1 - math.exp(-2 * stol**2)) for stol in stol_values]

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

def get_coherent_scattering_factor(element, q_magnitude, wavelength_angstrom=1.0):
    """
    Get coherent (elastic) scattering factor for comparison with Compton
    """
    stol = q_magnitude / (4 * math.pi)
    
    # Use Waasmaier-Kirfel form factors
    sf_info = wk1995(element, True)
    f0 = sf_info.at_stol(stol)
    
    return f0
```

**3. Key Arguments:**
- `element`: String element symbol ("C", "N", "O", etc.)
- `q_magnitude`: Float q-vector magnitude in Å⁻¹
- `stol`: Float sin(θ)/λ value
- `wavelength_angstrom`: Float X-ray wavelength in Angstroms
- `table`: String table identifier ("chantler", "henke", "wk1995")

**4. Return Types:**
- `s_incoh`: Float incoherent scattering factor in electrons
- `f0`: Float coherent scattering factor in electrons
- `compton_background`: List of float intensity values

**5. Example Usage Snippet:**
```python
import numpy as np
import matplotlib.pyplot as plt

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

# Calculate coherent scattering for comparison
coherent_total = []
for q_mag in q_values:
    total_coherent = 0.0
    for element, concentration in zip(elements, concentrations):
        f0 = get_coherent_scattering_factor(element, q_mag, wavelength)
        total_coherent += concentration * f0**2  # Intensity proportional to |f|²
    coherent_total.append(total_coherent)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(q_values, compton_background, label='Compton (incoherent)', linewidth=2)
plt.plot(q_values, coherent_total, label='Coherent scattering', linewidth=2)
plt.xlabel('|q| (Å⁻¹)')
plt.ylabel('Scattering factor')
plt.title('Theoretical Scattering Components')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("Compton scattering factors for different elements at |q| = 0.5 Å⁻¹:")
for element in ["C", "N", "O", "P", "S"]:
    s_incoh = get_compton_scattering_factor(element, 0.5)
    f0 = get_coherent_scattering_factor(element, 0.5)
    print(f"{element}: Compton = {s_incoh:.3f}, Coherent = {f0:.3f}")
```

**6. Energy-Dependent Scattering (Advanced):**
```python
def get_energy_dependent_scattering(element, energy_kev, q_magnitude):
    """
    Get energy-dependent scattering factors using Chantler tables
    
    Args:
        element: Element symbol
        energy_kev: X-ray energy in keV
        q_magnitude: Momentum transfer in Å⁻¹
    
    Returns:
        Dictionary with coherent and incoherent components
    """
    # Convert energy to wavelength
    wavelength_angstrom = 12.398 / energy_kev  # keV to Å conversion
    
    # Use Chantler tables for energy-dependent factors
    try:
        chantler_table = chantler.table(element)
        scattering_data = chantler_table.at_kev(energy_kev)
        
        # Get photoabsorption data
        f1 = scattering_data.fp()  # Real part of anomalous scattering
        f2 = scattering_data.fpp()  # Imaginary part of anomalous scattering
        
        # Calculate form factor at given q
        stol = q_magnitude / (4 * math.pi)
        sf_info = wk1995(element, True)
        f0 = sf_info.at_stol(stol)
        
        # Total coherent factor: f = f0 + f' + if''
        f_total_real = f0 + f1
        f_total_imag = f2
        f_total_magnitude = (f_total_real**2 + f_total_imag**2)**0.5
        
        # Approximate incoherent factor (energy-independent at this level)
        atomic_number = sf_info.atomic_number()
        s_incoh = atomic_number * (1 - math.exp(-2 * stol**2))
        
        return {
            'f0': f0,
            'f_prime': f1,
            'f_double_prime': f2,
            'f_total_magnitude': f_total_magnitude,
            's_incoherent': s_incoh,
            'wavelength_angstrom': wavelength_angstrom,
            'energy_kev': energy_kev
        }
        
    except Exception as e:
        print(f"Error accessing Chantler data for {element} at {energy_kev} keV: {e}")
        return None

# Usage example
energy_kev = 12.4  # keV (1 Å wavelength)
q_mag = 0.5  # Å⁻¹

for element in ["C", "N", "O"]:
    scattering_data = get_energy_dependent_scattering(element, energy_kev, q_mag)
    if scattering_data:
        print(f"\n{element} at {energy_kev} keV, |q| = {q_mag} Å⁻¹:")
        print(f"  f0 = {scattering_data['f0']:.3f}")
        print(f"  f' = {scattering_data['f_prime']:.3f}")
        print(f"  f'' = {scattering_data['f_double_prime']:.3f}")
        print(f"  |f_total| = {scattering_data['f_total_magnitude']:.3f}")
        print(f"  S_incoh = {scattering_data['s_incoherent']:.3f}")
```

**7. Notes/Caveats:**
- **Table Accuracy:** Different tables have varying accuracy ranges; Chantler is generally most accurate
- **Energy Dependence:** Incoherent scattering is weakly energy-dependent; coherent scattering shows strong energy dependence near absorption edges
- **Q-dependence:** Compton scattering factors increase with q; coherent factors decrease
- **Approximations:** Simple exponential approximation may not be accurate for all q-ranges
- **Units:** Ensure consistent units (Å⁻¹ for q, keV for energy, electrons for scattering factors)

**8. Integration with Diffuse Scattering Pipeline:**
Compton scattering calculations provide theoretical baselines for separating elastic diffuse scattering from inelastic backgrounds, essential for accurate diffuse scattering analysis in Module 4.1.D.

**9. See Also:**
- Section C.5: Theoretical scattering factor calculation
- Section C.1: Q-vector calculations for scattering factor evaluation
- Section C.3: d-spacing calculations for momentum transfer

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

## D. DIALS Algorithms Scaling Framework

### D.0. dials.algorithms.scaling Python Framework (Major)

**1. Purpose:**
Comprehensive framework for implementing custom scaling models, components, and refinement procedures for relative scaling of diffraction data in Module 3.S.3. Provides the infrastructure for building sophisticated scaling algorithms that can handle multi-crystal datasets and custom parameterizations.

**2. Primary Python Call(s):**
```python
from dials.algorithms.scaling.model import ScalingModelBase
from dials.algorithms.scaling.model.components.scale_components import ScaleComponentBase
from dials.algorithms.scaling.model.components.smooth_scale_components import (
    GaussianSmoother1D, GaussianSmoother2D, GaussianSmoother3D
)
from dials.algorithms.scaling.active_parameter_managers import (
    multi_active_parameter_manager,
    active_parameter_manager
)
from dials.algorithms.scaling.scaling_refiner import scaling_refinery
from dials.algorithms.scaling.target_function import ScalingTargetFunction

# Base classes for custom scaling models
class CustomScalingModel(ScalingModelBase):
    def __init__(self, configdict, is_scaled=False):
        super().__init__(configdict, is_scaled)
        # Initialize custom components
        
class CustomScaleComponent(ScaleComponentBase):
    def __init__(self, active_parameter_manager, params):
        super().__init__(active_parameter_manager, params)
        # Initialize component-specific parameters
```

**3. Key Framework Classes:**

**ScalingModelBase:**
- `__init__(self, configdict, is_scaled=False)`: Initialize scaling model
  - `configdict`: Dictionary with model configuration parameters
  - `is_scaled`: Boolean indicating if model has been applied to data
- `configure_components(self, reflection_table, experiment, params)`: Setup model components
- `_components`: Dictionary storing individual scale components
- `components`: Property returning list of active components
- `get_scales(self, reflection_table)` → flex.double: Calculate scale factors
- `get_inverse_scales(self, reflection_table)` → flex.double: Calculate inverse scales

**ScaleComponentBase:**
- `__init__(self, active_parameter_manager, params)`: Initialize component
- `calculate_scales_and_derivatives(self, reflection_table, block_id=None)`: Core scaling calculation
- `parameters`: Property accessing current parameter values
- `n_params`: Property returning number of parameters
- `update_reflection_data(self, reflection_table)`: Update component state

**GaussianSmoother Classes:**
- `GaussianSmoother1D(n_parameters, value_range)`: 1D smoothing component
- `GaussianSmoother2D(n_x_params, n_y_params, x_range, y_range)`: 2D smoothing
- `GaussianSmoother3D(n_x_params, n_y_params, n_z_params, x_range, y_range, z_range)`: 3D smoothing
- `value_error_for_location(locations)` → (values, errors): Evaluate smoother at coordinates

**4. Return Types:**
- `get_scales()`: flex.double array of scale factors
- `calculate_scales_and_derivatives()`: tuple of (scales, derivatives) flex arrays
- `value_error_for_location()`: tuple of (flex.double values, flex.double errors)

**5. Example Usage - Custom Scaling Model:**
```python
from dials.algorithms.scaling.model import ScalingModelBase
from dials.algorithms.scaling.model.components.scale_components import ScaleComponentBase
from dials.algorithms.scaling.active_parameter_managers import active_parameter_manager
from dials.array_family import flex
from scitbx import matrix

class SimpleMultiplicativeComponent(ScaleComponentBase):
    """Simple multiplicative scale factor component"""
    
    def __init__(self, active_parameter_manager, initial_scale=1.0):
        # Initialize with single parameter
        super().__init__(active_parameter_manager)
        self._initial_scale = initial_scale
        
        # Add parameter to manager
        self.active_parameter_manager.add_parameters(
            flex.double([initial_scale])  # Single scale parameter
        )
    
    @property
    def n_params(self):
        return 1
    
    @property  
    def parameters(self):
        return self.active_parameter_manager.get_parameters()[:self.n_params]
    
    def calculate_scales_and_derivatives(self, reflection_table, block_id=None):
        """
        Calculate scales and derivatives for all reflections
        
        Returns:
            tuple: (scales, derivatives) where derivatives[i,j] = ∂scale_i/∂param_j
        """
        n_refl = len(reflection_table)
        scale_param = self.parameters[0]
        
        # All reflections get the same scale factor
        scales = flex.double(n_refl, scale_param)
        
        # Derivatives: ∂scale_i/∂param_0 = 1 for all i
        derivatives = flex.double(flex.grid(n_refl, 1), 1.0)
        
        return scales, derivatives
    
    def update_reflection_data(self, reflection_table):
        """Update component with new reflection data if needed"""
        pass

class ResolutionDependentComponent(ScaleComponentBase):
    """Scale component that varies with resolution"""
    
    def __init__(self, active_parameter_manager, n_resolution_bins=10, d_min=1.0, d_max=50.0):
        super().__init__(active_parameter_manager)
        self.n_resolution_bins = n_resolution_bins
        self.d_min = d_min
        self.d_max = d_max
        
        # Initialize parameters (one per resolution bin)
        initial_params = flex.double([1.0] * n_resolution_bins)
        self.active_parameter_manager.add_parameters(initial_params)
        
        # Store bin edges
        import math
        log_d_min = math.log(d_min)
        log_d_max = math.log(d_max)
        self.log_d_edges = [log_d_min + i * (log_d_max - log_d_min) / n_resolution_bins 
                           for i in range(n_resolution_bins + 1)]
    
    @property
    def n_params(self):
        return self.n_resolution_bins
    
    @property
    def parameters(self):
        return self.active_parameter_manager.get_parameters()[:self.n_params]
    
    def calculate_scales_and_derivatives(self, reflection_table, block_id=None):
        """Calculate resolution-dependent scales"""
        import math
        
        # Get d-spacings
        if "d" not in reflection_table:
            raise ValueError("Reflection table must contain 'd' column for resolution-dependent scaling")
        
        d_spacings = reflection_table["d"]
        n_refl = len(d_spacings)
        
        scales = flex.double(n_refl)
        derivatives = flex.double(flex.grid(n_refl, self.n_params), 0.0)
        
        # Assign scales based on resolution bins
        for i, d_val in enumerate(d_spacings):
            if d_val <= 0:
                scales[i] = 1.0
                continue
                
            log_d = math.log(d_val)
            
            # Find appropriate bin
            bin_idx = 0
            for j in range(len(self.log_d_edges) - 1):
                if self.log_d_edges[j] <= log_d < self.log_d_edges[j + 1]:
                    bin_idx = j
                    break
            
            # Ensure bin_idx is valid
            bin_idx = max(0, min(bin_idx, self.n_params - 1))
            
            # Set scale and derivative
            scales[i] = self.parameters[bin_idx]
            derivatives[i, bin_idx] = 1.0
        
        return scales, derivatives

class CustomRelativeScalingModel(ScalingModelBase):
    """Custom scaling model for Module 3.S.3 relative scaling"""
    
    def __init__(self, configdict, is_scaled=False):
        super().__init__(configdict, is_scaled)
        self._components = {}
        
    def configure_components(self, reflection_table, experiment, params):
        """Configure scaling components based on data and parameters"""
        
        # Set up parameter manager
        self.active_parameter_manager = active_parameter_manager()
        
        # Add multiplicative component
        self._components["multiplicative"] = SimpleMultiplicativeComponent(
            self.active_parameter_manager, initial_scale=1.0
        )
        
        # Add resolution-dependent component if requested
        if hasattr(params, 'resolution_dependent') and params.resolution_dependent:
            self._components["resolution"] = ResolutionDependentComponent(
                self.active_parameter_manager, 
                n_resolution_bins=params.n_resolution_bins,
                d_min=params.d_min, 
                d_max=params.d_max
            )
        
        # Calculate d-spacings if needed
        if "resolution" in self._components and "d" not in reflection_table:
            # Add d-spacing calculation
            from cctbx import uctbx
            unit_cell = experiment.crystal.get_unit_cell()
            miller_indices = reflection_table["miller_index"]
            d_spacings = flex.double([unit_cell.d(hkl) for hkl in miller_indices])
            reflection_table["d"] = d_spacings
    
    @property
    def components(self):
        return list(self._components.values())
    
    def get_scales(self, reflection_table):
        """Calculate combined scale factors from all components"""
        if not self._components:
            return flex.double(len(reflection_table), 1.0)
        
        # Start with unit scales
        total_scales = flex.double(len(reflection_table), 1.0)
        
        # Multiply scales from all components
        for component in self.components:
            component_scales, _ = component.calculate_scales_and_derivatives(reflection_table)
            total_scales *= component_scales
        
        return total_scales
    
    def get_inverse_scales(self, reflection_table):
        """Calculate inverse scale factors"""
        scales = self.get_scales(reflection_table)
        return 1.0 / scales
```

**6. Advanced Framework Usage - Parameter Management and Refinement:**
```python
from dials.algorithms.scaling.active_parameter_managers import multi_active_parameter_manager
from dials.algorithms.scaling.scaling_refiner import scaling_refinery
from dials.algorithms.scaling.target_function import ScalingTargetFunction
from scitbx.lstbx import normal_eqns

class CustomScalingTargetFunction:
    """Custom target function for scaling refinement"""
    
    def __init__(self, scaling_models, reflection_tables):
        self.scaling_models = scaling_models
        self.reflection_tables = reflection_tables
    
    def compute_residuals_and_gradients(self):
        """
        Compute residuals and gradients for least squares refinement
        
        Returns:
            tuple: (residuals, jacobian) for least squares optimization
        """
        from dials.array_family import flex
        import math
        
        all_residuals = flex.double()
        all_gradients = []
        
        for model, refl_table in zip(self.scaling_models, self.reflection_tables):
            if len(refl_table) == 0:
                continue
                
            # Get observed intensities and calculate expected from model
            obs_intensities = refl_table["intensity.sum.value"]
            obs_variances = refl_table["intensity.sum.variance"]
            
            # Get scales from model
            scales = model.get_scales(refl_table)
            
            # Example residual: (I_obs - scale * I_ref) / sigma
            # For relative scaling, we might compare intensities between datasets
            
            # Simple example: residual based on deviation from mean intensity
            mean_intensity = flex.mean(obs_intensities)
            expected_intensities = scales * mean_intensity
            
            # Calculate residuals
            residuals = flex.double()
            gradients_matrix = []
            
            for i, (obs, exp, var) in enumerate(zip(obs_intensities, expected_intensities, obs_variances)):
                if var > 0:
                    sigma = math.sqrt(var)
                    residual = (obs - exp) / sigma
                    residuals.append(residual)
                    
                    # Calculate gradients with respect to model parameters
                    # ∂residual/∂param = -(∂exp/∂param) / sigma = -(∂(scale*I_ref)/∂param) / sigma
                    
                    # Get parameter gradients from components
                    component_gradients = flex.double()
                    for component in model.components:
                        _, derivatives = component.calculate_scales_and_derivatives(refl_table)
                        # ∂exp/∂param = I_ref * ∂scale/∂param
                        param_gradient = -mean_intensity * derivatives[i, :] / sigma
                        component_gradients.extend(param_gradient)
                    
                    gradients_matrix.append(component_gradients)
            
            all_residuals.extend(residuals)
            all_gradients.extend(gradients_matrix)
        
        # Convert gradients to proper format for least squares
        if all_gradients:
            n_residuals = len(all_residuals)
            n_params = len(all_gradients[0]) if all_gradients else 0
            jacobian = flex.double(flex.grid(n_residuals, n_params))
            
            for i, grad_row in enumerate(all_gradients):
                for j, grad_val in enumerate(grad_row):
                    jacobian[i, j] = grad_val
        else:
            jacobian = flex.double(flex.grid(0, 0))
        
        return all_residuals, jacobian

def run_scaling_refinement(scaling_models, reflection_tables, max_iterations=10):
    """
    Run scaling parameter refinement using DIALS scaling framework
    """
    from scitbx.lstbx import normal_eqns_solving
    
    # Create target function
    target_function = CustomScalingTargetFunction(scaling_models, reflection_tables)
    
    # Set up parameter manager for all models
    param_manager = multi_active_parameter_manager()
    for model in scaling_models:
        if hasattr(model, 'active_parameter_manager'):
            param_manager.add_parameter_manager(model.active_parameter_manager)
    
    # Refinement loop
    for iteration in range(max_iterations):
        print(f"Refinement iteration {iteration + 1}")
        
        # Calculate residuals and gradients
        residuals, jacobian = target_function.compute_residuals_and_gradients()
        
        if len(residuals) == 0:
            print("No data for refinement")
            break
        
        # Set up normal equations
        normal_eqns = normal_eqns_solving.levenberg_marquardt_iterations(
            residuals=residuals,
            jacobian=jacobian,
            step_threshold=1e-6,
            gradient_threshold=1e-6
        )
        
        # Solve for parameter shifts
        try:
            normal_eqns.build_and_solve()
            parameter_shifts = normal_eqns.solution()
            
            # Apply parameter shifts
            current_params = param_manager.get_parameters()
            new_params = current_params + parameter_shifts
            param_manager.set_parameters(new_params)
            
            # Check convergence
            shift_magnitude = flex.max(flex.abs(parameter_shifts))
            residual_rms = math.sqrt(flex.mean(residuals * residuals))
            
            print(f"  RMS residual: {residual_rms:.6f}")
            print(f"  Max parameter shift: {shift_magnitude:.6f}")
            
            if shift_magnitude < 1e-6:
                print("Refinement converged")
                break
                
        except Exception as e:
            print(f"Refinement failed at iteration {iteration + 1}: {e}")
            break
    
    return scaling_models

# Usage example
def create_and_refine_scaling_model(experiments, reflection_tables):
    """Complete example of creating and refining a custom scaling model"""
    
    # Create scaling models for each dataset
    scaling_models = []
    for i, (expt, refl_table) in enumerate(zip(experiments, reflection_tables)):
        
        # Configuration for this model
        configdict = {
            'model_type': 'custom_relative',
            'resolution_dependent': True,
            'n_resolution_bins': 10,
            'd_min': 1.5,
            'd_max': 50.0
        }
        
        # Create model
        model = CustomRelativeScalingModel(configdict, is_scaled=False)
        
        # Configure components based on data
        from types import SimpleNamespace
        params = SimpleNamespace()
        params.resolution_dependent = True
        params.n_resolution_bins = 10
        params.d_min = 1.5
        params.d_max = 50.0
        
        model.configure_components(refl_table, expt, params)
        scaling_models.append(model)
        
        print(f"Model {i}: {len(model.components)} components, "
              f"{sum(c.n_params for c in model.components)} parameters")
    
    # Run refinement
    refined_models = run_scaling_refinement(scaling_models, reflection_tables)
    
    # Apply scaling to data
    for model, refl_table in zip(refined_models, reflection_tables):
        scales = model.get_scales(refl_table)
        
        # Apply scales to intensities
        refl_table["intensity.scaled.value"] = refl_table["intensity.sum.value"] * scales
        refl_table["intensity.scaled.variance"] = refl_table["intensity.sum.variance"] * (scales * scales)
        
        print(f"Applied scaling: scale range {flex.min(scales):.3f} - {flex.max(scales):.3f}")
    
    return refined_models, reflection_tables
```

**7. Gaussian Smoothing Components Usage:**
```python
from dials.algorithms.scaling.model.components.smooth_scale_components import GaussianSmoother1D

class SmoothScaleComponent(ScaleComponentBase):
    """Component using Gaussian smoothing for parameter regularization"""
    
    def __init__(self, active_parameter_manager, coordinate_values, n_control_points=20):
        super().__init__(active_parameter_manager)
        
        # Set up coordinate range
        self.coord_min = flex.min(coordinate_values) 
        self.coord_max = flex.max(coordinate_values)
        self.coord_range = (self.coord_min, self.coord_max)
        
        # Create Gaussian smoother
        self.smoother = GaussianSmoother1D(
            n_parameters=n_control_points,
            value_range=self.coord_range
        )
        
        # Initialize control point values
        initial_values = flex.double([1.0] * n_control_points)
        self.active_parameter_manager.add_parameters(initial_values)
        
        # Store coordinate values for evaluation
        self.coordinate_values = coordinate_values
    
    @property
    def n_params(self):
        return self.smoother.n_parameters
    
    def calculate_scales_and_derivatives(self, reflection_table, block_id=None):
        """Calculate smoothed scales using Gaussian interpolation"""
        
        # Get current parameter values
        control_values = self.parameters
        
        # Set control point values in smoother
        self.smoother.set_parameters(control_values)
        
        # Evaluate smoother at coordinate positions
        scales, errors = self.smoother.value_error_for_location(self.coordinate_values)
        
        # Calculate derivatives: ∂scale_i/∂control_j
        derivatives = flex.double(flex.grid(len(scales), self.n_params), 0.0)
        
        # Use analytical derivatives from smoother weight matrices
        # DIALS GaussianSmoother classes provide analytical derivatives through weight matrices
        # This is more accurate and efficient than finite differences
        for i, coord in enumerate(self.coordinate_values):
            # Get analytical weights and values at coordinate location
            weight_result = self.smoother.value_weight(coord)
            weights = weight_result.get_weight()  # Weight matrix for this coordinate
            
            # Weights provide direct derivatives: ∂scale_i/∂control_j = weight[i,j]
            for j in range(self.n_params):
                derivatives[i, j] = weights[j]  # Direct analytical derivative
        
        return scales, derivatives
```

**8. Notes/Caveats:**
- **Framework Complexity:** The scaling framework is sophisticated; start with simple components before building complex models
- **Parameter Management:** Always use `active_parameter_manager` for proper parameter handling and refinement
- **Data Requirements:** Ensure reflection tables contain required columns (intensity, variance, Miller indices)
- **Memory Usage:** Large datasets with many parameters can consume significant memory during refinement
- **Convergence:** Monitor refinement convergence carefully; poor initial guesses can lead to non-convergence
- **Component Ordering:** The order of components in `_components` affects the final scaling calculation
- **Analytical Derivatives:** GaussianSmoother classes provide analytical derivatives through weight matrices from `value_weight()` methods, eliminating the need for finite difference calculations

**9. Integration with Diffuse Scattering Pipeline:**
This framework enables implementation of Module 3.S.3 relative scaling by providing:
- Custom scaling models tailored to diffuse scattering requirements  
- Flexible parameter management for multi-dataset scaling
- Robust refinement algorithms for optimizing scale factors
- Integration with DIALS reflection table infrastructure

**10. See Also:**
- Section A.0: dials.stills_process for generating input reflection data
- Section D.3: Reflection table operations for data manipulation
- Section B.3: Crystal models for unit cell information in scaling
- DIALS scaling documentation for additional built-in scaling models

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