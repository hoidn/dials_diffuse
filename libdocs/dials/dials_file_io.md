# DIALS File I/O and Model Loading

This documentation covers file I/O operations and model loading for DIALS/`dxtbx`/`cctbx` Python libraries needed for implementing diffuse scattering data processing modules.

**Version Information:** Compatible with DIALS 3.x series. Some methods may differ in DIALS 2.x.

**Key Dependencies:**
- `dxtbx`: Detector models, beam models, image handling
- `dials.array_family.flex`: Reflection tables and array operations  
- `cctbx`: Unit cells, space groups, scattering factors
- `scitbx`: Matrix operations and mathematical utilities

---

## A.0. dials.stills_process Python API

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

## A.1. Loading Experiment Lists (`.expt` files)

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

## A.2. Loading Reflection Tables (`.refl` files)

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

## A.3. Loading Image Data via ImageSet

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

## A.4. Loading Mask Files (`.pickle` files)

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

## A.5. Shoebox Data and MaskCode (for Bragg Mask Generation)

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

---

## A.6. Python API for dials.generate_mask

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