# DIALS Programs Documentation

This document contains compiled documentation for all major DIALS (Diffraction Integration for Advanced Light Sources) programs.

## Table of Contents

1. [dials.import](#dialsimport)
2. [dials.find_spots](#dialsfind_spots)
3. [dials.index](#dialsindex)
4. [dials.refine_bravais_settings](#dialsrefine_bravais_settings)
5. [dials.reindex](#dialsreindex)
6. [dials.refine](#dialsrefine)
7. [dials.integrate](#dialsintegrate)
8. [dials.two_theta_refine](#dialstwo_theta_refine)
9. [dials.cosym](#dialscosym)
10. [dials.symmetry](#dialssymmetry)
11. [dials.scale](#dialsscale)
12. [dials.export](#dialsexport)
13. [xia2.multiplex](#xia2multiplex)
14. [dials.show](#dialsshow)
15. [dials.image_viewer](#dialsimage_viewer)
16. [dials.generate_mask](#dialsgenerate_mask)
17. [dials.check_indexing_symmetry](#dialscheck_indexing_symmetry)
18. [dials.search_beam_position](#dialssearch_beam_position)
19. [dials.report](#dialsreport)
20. [dials.plot_scan_varying_model](#dialsplot_scan_varying_model)
21. [dials.find_spots_server](#dialsfind_spots_server)
22. [dials.apply_mask](#dialsapply_mask)
23. [dials.create_profile_model](#dialscreate_profile_model)
24. [dials.estimate_gain](#dialsestimate_gain)
25. [dials.estimate_resolution](#dialsestimate_resolution)
26. [dials.predict](#dialspredict)
27. [dials.merge_cbf](#dialsmerge_cbf)
28. [dials.export_bitmaps](#dialsexport_bitmaps)
29. [dials.slice_sequence](#dialsslice_sequence)
30. [dials.compare_orientation_matrices](#dialscompare_orientation_matrices)
31. [dials.spot_counts_per_image](#dialsspot_counts_per_image)
32. [dials.stereographic_projection](#dialsstereographic_projection)
33. [dials.combine_experiments](#dialscombine_experiments)
34. [dials.align_crystal](#dialsalign_crystal)
35. [dials.anvil_correction](#dialsanvil_correction)
36. [dials.missing_reflections](#dialsmissing_reflections)
37. [dials.filter_reflections](#dialsfilter_reflections)
38. [dials.import_xds](#dialsimport_xds)

---

## dials.import

### Introduction

The `dials.import` program is used to import image data files into a format compatible with DIALS. Key features include:

- Analyzing metadata and filenames to determine image set relationships
- Creating an experiments object specifying file relationships
- Supporting multiple input methods:
  - Command line arguments
  - Stdin input
  - File template specification

### Basic Usage Examples

```
dials.import /data/directory-containing-images/
dials.import image_*.cbf
dials.import template=image_1_####.cbf
find . -name "image_*.cbf" | dials.import
```

### Key Parameters

#### Output Parameters
- `experiments`: Output experiment file (default: `imported.expt`)
- `log`: Log filename (default: `dials.import.log`)
- `compact`: Use compact JSON representation (default: `False`)

#### Input Parameters
- `ignore_unhandled`: Ignore unhandled input files (default: `True`)
- `template`: Image sequence template
- `directory`: Directory containing images
- `reference_geometry`: Override geometry from reference experiment file

#### Geometry Overrides
Allows manual setting of:
- Beam properties
- Detector characteristics
- Goniometer settings
- Scan parameters

### Advanced Features
- Support for multi-panel detectors
- Geometry reference override options
- Beam centre adjustment
- Scan sequence manipulation

---

## dials.find_spots

### Introduction

The `dials.find_spots` program is designed to identify strong spots on a sequence of images. Key features include:

- Can process images via "models.expt" file or direct image files
- Performs spot finding on logically grouped image sets
- Identifies strong pixels and forms spots using connected component labelling
- Calculates spot centroids and intensities
- Filters spots based on user preferences
- Outputs a "strong.refl" file for use in subsequent indexing

### Basic Usage Examples

```
dials.find_spots image1.cbf
dials.find_spots imager_00*.cbf
dials.find_spots models.expt
dials.find_spots models.expt output.reflections=strong.refl
```

### Key Parameters

#### Output Options
- `reflections`: Output filename (default: 'strong.refl')
- `shoeboxes`: Save raw pixel values (default: True)
- `log`: Log filename (default: 'dials.find_spots.log')

#### Spotfinder Configuration
- Supports various threshold algorithms (dispersion, radial profile)
- Configurable filtering options:
  - Spot size limits
  - Resolution range
  - Border settings
  - Untrusted region masking

#### Threshold Algorithms
1. Dispersion
2. Dispersion Extended
3. Radial Profile

### Detailed Configuration

Users can extensively customize spot finding through parameters like:
- Scan ranges
- Region of interest
- Background computation
- Spot filtering criteria
- Multiprocessing settings

For full parameter details, refer to the comprehensive configuration section in the documentation.

---

## dials.index

### Overview

The `dials.index` program is a critical component of the DIALS crystallography software suite designed to perform autoindexing on strong spots detected in diffraction data.

### Key Features

- Attempts to index strong spots output by `dials.find_spots`
- Uses input files: "imported.expt" and "strong.refl"
- Provides multiple indexing methods:
  - One-dimensional FFT
  - Three-dimensional FFT
  - Real space grid search
  - Other specialized methods

### Basic Usage

```bash
dials.index imported.expt strong.refl
```

Optional parameters include:
- Specifying unit cell
- Specifying space group
- Choosing indexing method (e.g., `indexing.method=fft1d`)

### Indexing Process

1. Searches for a primitive lattice
2. Refines crystal orientation and experimental geometry
3. Minimizes differences between observed and predicted spot centroids
4. Outputs:
   - "indexed.expt" file with crystal model
   - "indexed.refl" file with Miller indices and predicted centroids

### Advanced Configuration

The program offers extensive configuration options for:
- Indexing methods
- Refinement protocols
- Reflection management
- Outlier rejection
- Experimental geometry parameterization

### Recommended Use

Ideal for processing crystallographic diffraction data, particularly when dealing with:
- Rotation data
- Still image collections
- Complex lattice scenarios

---

## dials.refine_bravais_settings

### Overview
The `dials.refine_bravais_settings` program is a tool for refining Bravais settings consistent with a primitive unit cell. It takes indexed experimental data and performs full refinement of crystal and experimental geometry parameters across potential Bravais settings.

### Key Features
- Refinement of crystal and experimental geometry parameters
- Generation of `.expt` files for each Bravais setting
- Detailed output including:
  - Metric fit
  - Root-mean-square deviations (RMSD)
  - Refined unit cell parameters
  - Change of basis operators

### Basic Usage
```
dials.refine_bravais_settings indexed.expt indexed.refl
dials.refine_bravais_settings indexed.expt indexed.refl nproc=4
```

### Main Parameters
- `lepage_max_delta`: Default 5
- `nproc`: Number of processors (default Auto)
- `best_monoclinic_beta`: Prefer less oblique monoclinic cells (default True)

### Detailed Refinement Options
The program offers extensive configuration for:
- Beam parameters
- Crystal parameters
- Detector settings
- Goniometer configuration
- Reflection management
- Outlier rejection strategies

### Output
- Generates Bravais setting `.expt` files
- Produces a log file with refinement details
- Provides a table of potential Bravais settings

---

## dials.reindex

### Introduction

The `dials.reindex` program allows users to re-index experimental and reflection files from one setting to another. Key features include:

- Change of basis operator in multiple conventions (h,k,l, a,b,c, x,y,z)
- Optional space group modification
- Ability to reindex datasets using a reference dataset

### Basic Usage Examples

```
dials.reindex indexed.expt change_of_basis_op=b+c,a+c,a+b
dials.reindex indexed.refl change_of_basis_op=-b,a+b+2*c,-a
dials.reindex indexed.expt indexed.refl change_of_basis_op=l,h,k
```

### Key Parameters

- `change_of_basis_op`: Specifies the change of basis operator
- `hkl_offset`: Optional integer offset
- `space_group`: Space group to apply after change of basis
- `reference`: Optional reference experiments/reflections for reindexing

### Output Options

- `experiments`: Reindexed experimental models (default: `reindexed.expt`)
- `reflections`: Reindexed reflections (default: `reindexed.refl`)
- `log`: Logging file (default: `dials.reindex.log`)

---

## dials.refine

### Overview

`dials.refine` is a tool for refining the diffraction geometry of experimental models against indexed reflections. Key features include:

#### Basic Functionality
- Refines experimental models for rotation scan data
- Supports static or scan-varying model parameterization
- Allows fixing specific model parameters

#### Key Parameters
- Input: Indexed experiments and reflections
- Output options for experiments, reflections, and logs
- Configurable refinement strategies

#### Refinement Modes
1. Static refinement (same parameters for all reflections)
2. Scan-varying refinement (parameters dependent on image number)

#### Example Usage
```
dials.refine indexed.expt indexed.refl
dials.refine indexed.expt indexed.refl scan_varying=(False/True/Auto)
```

#### Configurable Components
- Beam parameters
- Crystal parameters
- Detector parameters
- Goniometer settings

#### Advanced Features
- Outlier rejection algorithms
- Weighting strategies
- Multiprocessing support
- Detailed logging and tracking options

The documentation provides extensive configuration possibilities for precise geometric refinement in crystallographic data processing.

---

## dials.integrate

### Overview
`dials.integrate` is a program used to integrate reflections on diffraction images. It processes experiment lists and strong spots to generate integrated reflections and experiment data.

### Basic Usage Examples
```
dials.integrate models.expt refined.refl
dials.integrate models.expt refined.refl output.reflections=integrated.refl
dials.integrate models.expt refined.refl profile.fitting=False
```

### Key Features
- Integrates reflections from indexed and refined experiment data
- Creates profile models
- Supports multiple integration and background algorithms
- Configurable output options

### Main Parameters
- `output`: Controls output file names and formats
- `integration`: Configures integration process
  - Background algorithms
  - Profile fitting
  - Multiprocessing options
- `profile`: Defines profile modeling approach
- `prediction`: Sets resolution and prediction parameters

### Profile Modeling Algorithms
1. Ellipsoid
2. Gaussian Reciprocal Space (default)

### Background Subtraction Options
- Auto (default)
- GLM
- Global model
- Null
- Simple

### Multiprocessing Methods
- Multiprocessing (default)
- DRMAA
- SGE
- LSF
- PBS

The documentation provides extensive configuration options for customizing the integration process across different experimental setups.

---

## dials.two_theta_refine

### Introduction

A DIALS tool to "Refine the unit cell(s) of input experiments against the input indexed reflections using a 2θ angle target."

### Basic Usage

```
dials.two_theta_refine integrated.expt integrated.refl
```

Optional additional parameters:
```
dials.two_theta_refine integrated.expt integrated.refl \
    correlation_plot.filename=corrplot.png cif=refined_cell.cif
```

### Key Parameters

#### Output Options
- `experiments`: Filename for refined experimental models
- `log`: Log file name
- `cif`: Optional Crystallographic Information File output
- `correlation_plot`: Optional parameter correlation visualization

#### Refinement Options
- `filter_integrated_centroids`: Filter centroids (default: True)
- `partiality_threshold`: Minimum reflection partiality (default: 0.4)
- `combine_crystal_models`: Combine multiple experiment models (default: True)
- `triclinic`: Remove symmetry constraints (default: False)

### Typical Workflow

1. Provide indexed experimental and reflection files
2. Optionally specify output and refinement parameters
3. Generate refined unit cell parameters

---

## dials.cosym

### Overview
`dials.cosym` is a program for determining Patterson group symmetry from multi-crystal datasets, implementing methods from "Gildea, R. J. & Winter, G. (2018)".

### Key Features
- Analyzes symmetry elements in multiple crystal datasets
- Handles indexing ambiguities
- Performs unit cell clustering
- Normalizes intensities
- Determines resolution limits
- Identifies symmetry operations and Patterson groups

### Basic Usage
```
dials.cosym models.expt observations.refl
```

### Main Analysis Steps
1. Unit cell metric symmetry analysis
2. Hierarchical unit cell clustering
3. Intensity normalization
4. Resolution limit determination
5. Symmetry element scoring
6. Laue group identification

### Key Parameters
- `partiality_threshold`: Reflection inclusion threshold (default 0.4)
- `min_reflections`: Minimum merged reflections per experiment (default 10)
- `d_min`: High-resolution cutoff
- `min_cc_half`: Minimum correlation coefficient threshold (default 0.6)

### Output
- Reindexed experiments and reflections
- HTML and JSON reports
- Log file with detailed analysis

### Typical Workflow
The program automatically:
- Clusters unit cells
- Normalizes intensities
- Determines optimal analysis dimensions
- Scores symmetry elements
- Identifies best Patterson group

---

## dials.symmetry

### Introduction

The `dials.symmetry` program implements symmetry determination methods from POINTLESS, referencing two key publications:
- Evans, P. (2006). Acta Cryst. D62, 72-82
- Evans, P. R. (2011). Acta Cryst. D67, 282-292

### Basic Usage

```
dials.symmetry models.expt observations.refl
```

### Key Parameters

#### Basic Configuration
- `d_min`: Resolution limit (default: Auto)
- `min_i_mean_over_sigma_mean`: 4
- `min_cc_half`: 0.6
- `normalisation`: Options include kernel quasi, ml_iso, ml_aniso
- `lattice_group`: None by default
- `laue_group`: Auto detection

#### Systematic Absences
- Configurable checking method
- Direct or Fourier analysis
- Significance level: 0.95

### Output Files
- Log file: `dials.symmetry.log`
- Experiments: `symmetrized.expt`
- Reflections: `symmetrized.refl`
- JSON report: `dials.symmetry.json`
- HTML report: `dials.symmetry.html`

### Advanced Options
- Image exclusion
- Partiality threshold
- Lattice symmetry tolerances
- Monoclinic cell optimization

---

## dials.scale

### Overview

`dials.scale` is a program for scaling integrated X-ray diffraction datasets, designed to improve the internal consistency of reflection intensities by correcting for various experimental effects.

### Key Features

- Performs scaling on integrated datasets
- Corrects for experimental effects like scale, decay, and absorption
- Supports multiple datasets scaled against a common target
- Outputs scaled reflection and experiment files
- Generates an HTML report with interactive plots

### Basic Usage Examples

```bash
# Regular single-sequence scaling without absorption correction
dials.scale integrated.refl integrated.expt physical.absorption_correction=False

# Scaling multiple datasets with resolution limit
dials.scale 1_integrated.refl 1_integrated.expt 2_integrated.refl 2_integrated.expt d_min=1.4

# Incremental scaling with different options
dials.scale integrated.refl integrated.expt physical.scale_interval=10.0
dials.scale integrated_2.refl integrated_2.expt scaled.refl scaled.expt physical.scale_interval=15.0
```

### Scaling Models

The program supports several scaling models:
- KB (default)
- Array
- Dose decay
- Physical

### Key Parameters

- `model`: Choose scaling model (KB, array, dose_decay, physical)
- `output`: Configure output files
- `reflection_selection`: Control reflection subset selection
- `weighting`: Define error model and weighting scheme
- `scaling_options`: Set refinement and outlier rejection parameters

### Additional Resources

More detailed documentation is available in the [DIALS scale user guide](https://dials.github.io/dials_scale_user_guide.html).

---

## dials.export

### Overview
`dials.export` is a program used to export crystallographic processing results in various file formats.

### Supported Output Formats
- MTZ: Unmerged MTZ file for downstream processing
- NXS: NXmx file
- MMCIF: mmcif file
- XDS_ASCII: Intensity data in XDS CORRECT step format
- SADABS: Intensity data in ersatz-SADABS format
- MOSFLM: Matrix and instruction files
- XDS: XDS.INP and XPARM.XDS files
- SHELX: Intensity data in HKLF 4 format
- PETS: CIF format for dynamic diffraction refinement
- JSON: Reciprocal lattice point data

### Basic Usage Examples
```
# Export to MTZ
dials.export integrated.expt integrated.refl
dials.export scaled.expt scaled.refl intensity=scale

# Export to Nexus
dials.export integrated.expt integrated.refl format=nxs

# Export to XDS
dials.export indexed.expt indexed.refl format=xds
```

### Key Parameters
- `format`: Choose output file format (default: MTZ)
- `intensity`: Select intensity type (default: auto)
- `debug`: Output additional debugging information

### Detailed Configuration
Each format has specific configuration options, such as:
- MTZ: Partiality thresholds, resolution cutoffs
- MMCIF: Compression, PDB version
- SHELX: Composition, scaling
- PETS: Virtual frame settings

### Notes
Supports exporting both integrated and scaled crystallographic data with flexible configuration options.

---

## xia2.multiplex

### Overview

xia2.multiplex is a DIALS program for processing multi-crystal X-ray diffraction datasets. It performs several key functions:

#### Key Features
- Symmetry analysis
- Scaling and merging of multi-crystal datasets
- Analysis of dataset pathologies like:
  - Non-isomorphism
  - Radiation damage
  - Preferred orientation

#### Internal Programs Used
- dials.cosym
- dials.two_theta_refine
- dials.scale
- dials.symmetry
- dials.estimate_resolution

### Example Usage

Basic command syntax:
```
xia2.multiplex integrated.expt integrated.refl
```

Advanced examples:
```
# Multiple input files
xia2.multiplex integrated_1.expt integrated_1.refl \
  integrated_2.expt integrated_2.refl

# Override space group and resolution
xia2.multiplex space_group=C2 resolution.d_min=2.5 \
  integrated_1.expt integrated_1.refl

# Filter datasets using ΔCC½ method
xia2.multiplex filtering.method=deltacchalf \
  integrated.expt integrated.refl
```

### Citation

For academic use, cite: "Gildea, R. J. et al. (2022) Acta Cryst. D78, 752-769"

### Key Parameters

The documentation provides extensive configuration options for:
- Unit cell clustering
- Scaling
- Symmetry determination
- Resolution limits
- Filtering
- Multi-crystal analysis

### Detailed Documentation

The full documentation includes comprehensive parameter definitions with help text, types, and expert-level settings for advanced users.

---

## dials.show

### Overview
`dials.show` is a command-line tool for displaying information about experimental data, reflections, and images in crystallography.

### Basic Usage Examples
- `dials.show models.expt`
- `dials.show image_*.cbf`
- `dials.show observations.refl`

### Parameters

#### Display Options
- `show_scan_varying` (default: False)
  - "Whether or not to show the crystal at each scan point."

- `show_shared_models` (default: False)
  - "Show which models are linked to which experiments"

- `show_all_reflection_data` (default: False)
  - "Whether or not to print individual reflections"

- Additional toggles:
  - `show_intensities`
  - `show_centroids`
  - `show_profile_fit`
  - `show_flags`
  - `show_identifiers`

#### Image Statistics
```
image_statistics {
  show_corrected = False  # "Show statistics on the distribution of values in each corrected image"
  show_raw = False        # "Show statistics on the distribution of values in each raw image"
}
```

#### Additional Parameter
- `max_reflections` (default: None)
  - "Limit the number of reflections in the output."

### Supported by
- Diamond Light Source
- CCP4
- STFC
- Lawrence Berkeley National Laboratory
- SSRL/SMB

---

## dials.image_viewer

### Overview
The `dials.image_viewer` is a program for viewing diffraction images with optional overlays of analysis results.

### Usage Examples
```
dials.image_viewer image.cbf
dials.image_viewer models.expt
dials.image_viewer models.expt strong.refl
dials.image_viewer models.expt integrated.refl
```

### Key Features
- View diffraction images
- Optional overlays:
  - Spot finding results
  - Indexing results
  - Integration results

### Basic Parameters
#### Display Options
- `brightness`: Image brightness level
- `color_scheme`: Options include grayscale, rainbow, heatmap, invert
- `projection`: Lab or image view
- Toggleable display elements:
  - Beam center
  - Resolution rings
  - Ice rings
  - Center of mass
  - Maximum pixels
  - Predictions
  - Miller indices
  - Indexed/integrated reflections

#### Image Processing
- Background calculation
- Thresholding
- Masking
- Powder arc analysis
- Calibration options

### Advanced Features
- Profile modeling
- Reflection prediction
- Detailed masking controls
- Extensive configuration options for image analysis

### Recommended Use
Primarily used by crystallographers for detailed examination and analysis of diffraction image data.

---

## dials.generate_mask

### Introduction

The `dials.generate_mask` program is used to "mask images to remove unwanted pixels" during crystallography data processing. It allows users to:

- Create masks using detector trusted range
- Define masks with simple shapes
- Set resolution range filters
- Combine multiple mask files

### Basic Usage Examples

```
dials.generate_mask models.expt border=5

dials.generate_mask models.expt \
  untrusted.rectangle=50,100,50,100 \
  untrusted.circle=200,200,100

dials.generate_mask models.expt d_max=2.00

dials.generate_mask backstop.mask shadow.mask
```

### Key Parameters

- `output.mask`: Name of output mask file
- `border`: Border size around image edge
- `d_min`: High resolution limit (Angstrom)
- `d_max`: Low resolution limit (Angstrom)
- `untrusted`: Options to mask specific regions
  - Panels
  - Circles
  - Rectangles
  - Polygons
  - Individual pixels

### Additional Features

- Can generate masks based on resolution ranges
- Optional ice ring filtering
- Supports parallax correction toggle

---

## dials.check_indexing_symmetry

### Introduction

This DIALS program analyzes correlation coefficients between reflections related by symmetry operators in a space group. It can help detect potential misindexing of a diffraction pattern, possibly due to an incorrect beam centre.

### Example Usage

```
dials.check_indexing_symmetry indexed.expt indexed.refl \
  grid=1 symop_threshold=0.7

dials.check_indexing_symmetry indexed.expt indexed.refl \
  grid_l=3 symop_threshold=0.7
```

### Basic Parameters

- `d_min`: High resolution limit (default: 0)
- `d_max`: Low resolution limit (default: 0)
- `symop_threshold`: Threshold for symmetry operator confidence (default: 0)
- `grid`: Search scope for testing misindexing on h, k, l
- `asu`: Perform search comparing within ASU
- `normalise`: Normalize intensities before calculating correlation coefficients
- `reference`: Correctly indexed reference set for comparison

### Output

- Default log file: `dials.check_indexing_symmetry.log`

The program is part of the DIALS software suite, developed collaboratively by Diamond Light Source, Lawrence Berkeley National Laboratory, and STFC.

---

## dials.search_beam_position

### Overview

A DIALS function to find beam center from diffraction images. The default method is based on the work of Sauter et al. (J. Appl. Cryst. 37, 399-409 (2004)) and uses spot finding results.

### Basic Usage

#### Default Method
```
dials.search_beam_position imported.expt strong.refl
```

#### Projection Method
```
dials.search_beam_position method=midpoint imported.exp
```

### Key Methods

Available beam position search methods:
- `default`
- `midpoint`
- `maximum`
- `inversion`

### Main Parameters

#### Default Parameters
- `nproc`: Number of processors (default: Auto)
- `max_reflections`: Maximum reflections to use (default: 10,000)
- `mm_search_scope`: Global radius of origin offset search (default: 4.0)
- `n_macro_cycles`: Iterative beam centre search cycles (default: 1)

#### Projection Parameters
- `method_x`: Projection method for x-axis
- `method_y`: Projection method for y-axis
- Options for plotting and image processing

### Output

- Optimized experiment file: `optimised.expt`
- Log file: `dials.search_beam_position.log`
- JSON beam positions: `beam_positions.json`

More details available at: https://autoed.readthedocs.io/en/latest/pages/beam_position_methods.html

---

## dials.report

### Overview
`dials.report` is a DIALS program that "Generates a html report given the output of various DIALS programs (observations.refl and/or models.expt)".

### Example Usage
```
dials.report strong.refl
dials.report indexed.refl
dials.report refined.refl
dials.report integrated.refl
dials.report refined.expt
dials.report integrated.refl integrated.expt
```

### Basic Parameters
- `output.html`: Default filename is `dials.report.html`
- `output.json`: Optional JSON file for plot data
- `output.external_dependencies`: Options are remote, local, or embed
- `grid_size`: Defaults to Auto
- `pixels_per_bin`: Default is 40

### Advanced Parameters
- `centroid_diff_max`: Optional parameter for heatmap color mapping
- `orientation_decomposition`: Configures orientation matrix decomposition
  - Allows setting rotation axes
  - Option to decompose relative to static orientation

### Key Features
- Generates HTML reports from DIALS processing output
- Flexible input file types (.refl and .expt)
- Configurable visualization and reporting options

---

## dials.plot_scan_varying_model

### Overview
A DIALS tool to "Generate plots of scan-varying models, including crystal orientation, unit cell and beam centre"

### Basic Usage
```
dials.plot_scan_varying_model refined.expt
```

### Parameters

#### Output Options
- `directory`: Directory to store results (default: current directory)
- `format`: Output file format (png, pdf)
- `debug`: Print tables of plotted values (expert level)

#### Orientation Decomposition
- `e1`, `e2`, `e3`: Rotation axes (default: standard coordinate axes)
- `relative_to_static_orientation`: Whether rotations are relative to reference crystal model

### Key Features
- Visualizes changes in:
  - Crystal orientation
  - Unit cell
  - Beam centre

### Supported Outputs
- PNG images
- PDF files

### Notes
- Requires a refined experiment file (`refined.expt`) as input
- Provides detailed visualization of scan-varying crystallographic parameters

---

## dials.find_spots_server

### Overview
A client/server version of `dials.find_spots` designed for quick feedback on image quality during grid scans and data collections.

### Server Setup
```
dials.find_spots_server [nproc=8] [port=1234]
```

### Client Usage
```
dials.find_spots_client [host=hostname] [port=1234] [nproc=8] /path/to/image.cbf
```

### XML Response Example
```xml
<response>
<image>/path/to/image_0001.cbf</image>
<spot_count>352</spot_count>
<spot_count_no_ice>263</spot_count_no_ice>
<d_min>1.46</d_min>
<d_min_method_1>1.92</d_min_method_1>
<d_min_method_2>1.68</d_min_method_2>
<total_intensity>56215</total_intensity>
</response>
```

#### Response Fields
- `spot_count`: Total spots found
- `spot_count_no_ice`: Spots excluding ice ring regions
- `d_min_method_1/2`: Resolution estimates
- `total_intensity`: Total intensity of strong spots

### Additional Client Options
- Can pass standard `dials.find_spots` parameters
- Example: `dials.find_spots_client /path/to/image.cbf min_spot_size=2 d_min=2`

### Stopping the Server
```
dials.find_spots_client stop [host=hostname] [port=1234]
```

### Default Parameters
- `nproc`: Auto
- `port`: 1701

---

## dials.apply_mask

### Introduction

This program allows users to augment an experiments JSON file with masks. "Its only function is to input the mask file paths to the experiments JSON file"

### Key Requirements

- Mask files must be provided in the same order as their corresponding imagesets in the experiments JSON file

### Usage Examples

```
dials.apply_mask models.expt input.mask=pixels.mask
dials.apply_mask models.expt input.mask=pixels1.mask input.mask=pixels2.mask
```

### Parameters

#### Input
- `mask`: Mask filename(s)
  - Type: String
  - Multiple masks allowed
  - Default: None

#### Output
- `experiments`: Output experiments file
  - Default: `masked.expt`

### Full Parameter Definitions

```
input {
  mask = None
    .help = "The mask filenames, one mask per imageset"
    .type = str
    .multiple = True
}
output {
  experiments = masked.expt
    .help = "Name of output experiments file"
    .type = str
}
```

---

## dials.create_profile_model

### Overview

This DIALS program computes a profile model from input reflections and saves a modified experiments file with profile model information. It can be run independently of integration, though it's typically performed during that process.

### Basic Usage

```
dials.create_profile_model models.expt observations.refl
```

### Key Parameters

#### Profile Modeling Algorithms
- Two primary algorithms:
  1. Ellipsoid
  2. Gaussian RS (recommended)

#### Profile Configuration Options
- Scan-varying model
- Minimum spot requirements
- Mosaicity computation
- Centroid definition
- Filtering parameters

#### Example Configuration
```
profile {
  algorithm = gaussian_rs
  gaussian_rs {
    scan_varying = False
    min_spots {
      overall = 50
      per_degree = 20
    }
  }
}
```

### Important Settings
- Can subtract background before profile computation
- Configurable resolution limits
- Refinement parameters for profile model
- Grid methods for profile fitting

### Recommended Use
Typically used during crystal diffraction data processing to characterize reflection profiles and improve integration accuracy.

---

## dials.estimate_gain

### Introduction

This program estimates the detector's gain. For pixel array detectors, the gain is typically 1.00, representing Poisson statistics. For older CCD detectors, the gain may vary, which can impact spot finding algorithms.

### Example Usage

```
dials.estimate_gain models.expt
```

### Parameters

#### Basic Parameters
- `kernel_size`: Default is 10,10
- `max_images`: Default is 1
- `output.gain_map`: Default is None

#### Full Parameter Definitions

##### kernel_size
- Type: 2 integers
- Minimum value: 1

##### max_images
- Type: Integer (can be None)
- Help: "For multi-file images (NeXus for example), report a gain for each image, up to max_images, and then report an average gain"

##### output.gain_map
- Type: String
- Help: "Name of output gain map file"

### Key Points
- Useful for understanding detector pixel behavior
- Particularly important for older CCD detectors
- Can help improve spot finding accuracy by understanding noise characteristics

---

## dials.estimate_resolution

### Overview
A DIALS command-line tool for estimating resolution limits in crystallographic data processing based on various statistical metrics.

### Supported Resolution Metrics
- cc_half (default)
- isigma (unmerged <I/sigI>)
- misigma (merged <I/sigI>)
- i_mean_over_sigma_mean
- cc_ref
- completeness
- rmerge

### Basic Usage Examples
```
dials.estimate_resolution scaled.expt scaled.refl
dials.estimate_resolution scaled_unmerged.mtz
dials.estimate_resolution scaled.expt scaled.refl cc_half=0.1
```

### Key Features
- Estimates resolution by fitting curves to merging statistics in resolution bins
- Chooses resolution limit based on specified criteria
- Supports multiple resolution estimation methods
- Generates log, HTML, and optional JSON output

### Resolution Estimation Methods
Different metrics use specific fitting approaches:
- cc_half: Fits tanh function
- Other metrics: Polynomial fits to log-transformed data

### Parameters
Important configurable parameters include:
- `cc_half`: Minimum CC½ threshold (default 0.3)
- `reflections_per_bin`: Minimum reflections per resolution bin
- `nbins`: Maximum number of resolution bins
- `reference`: Optional reference dataset

### Output
Generates resolution estimates with optional visualization of curve fits.

---

## dials.predict

### Overview
`dials.predict` is a program that "takes a set of experiments and predicts the reflections" which are then saved to a file.

### Basic Usage Examples
```
dials.predict models.expt
dials.predict models.expt force_static=True
dials.predict models.expt d_min=2.0
```

### Key Parameters

#### Basic Parameters
- `output`: Filename for predicted reflections (default: `predicted.refl`)
- `force_static`: Force static prediction for scan-varying model (default: `False`)
- `ignore_shadows`: Ignore dynamic shadowing (default: `True`)
- `buffer_size`: Prediction buffer zone around scan images (default: `0`)
- `d_min`: Minimum d-spacing for predicted reflections (default: `None`)

#### Profile Modeling
Two primary algorithms:
1. Ellipsoid
2. Gaussian Reciprocal Space (default)

##### Gaussian RS Options
- `scan_varying`: Calculate scan-varying model (default: `False`)
- Minimum spot requirements
- Mosaicity computation algorithm
- Centroid definition
- Filtering and fitting parameters

### Detailed Configuration
The program offers extensive configuration options for reflection prediction, including:
- Mosaicity models
- Wavelength spread
- Unit cell and orientation refinement
- Indexing and prediction tolerances

### Recommended Use
Ideal for crystallography experiments requiring precise reflection prediction and modeling.

---

## dials.merge_cbf

### Introduction

The `dials.merge_cbf` program merges consecutive CBF image files into fewer images. For example:

- Merging 100 images with `merge_n_images=2` will produce 50 summed images
- Currently supports only CBF format images

### Usage Examples

```
dials.merge_cbf image_*.cbf
dials.merge_cbf image_*.cbf merge_n_images=10
```

### Basic Parameters

- `merge_n_images = 2`: Number of input images to average into a single output image
- `output.image_prefix = sum_`: Prefix for output image files

### Full Parameter Definitions

#### merge_n_images
- Type: Integer
- Minimum value: 1
- Allows None: Yes
- Help: "Number of input images to average into a single output image"

#### get_raw_data_from_imageset
- Type: Boolean
- Default: True
- Expert level: 2
- Help: "By default the raw data is read via the imageset. This limits use to single panel detectors"

### Output

Images will be prefixed with `sum_` by default.

---

## dials.export_bitmaps

### Overview
A DIALS utility to export raw diffraction image files as bitmap images, with options for:
- Exporting images from intermediate spot-finding steps
- Adjusting image appearance
- Controlling image output parameters

### Basic Usage Examples
```
dials.export_bitmaps image.cbf
dials.export_bitmaps models.expt
dials.export_bitmaps image.cbf display=variance colour_scheme=inverse_greyscale
```

### Key Parameters
- `binning`: Pixel binning (default: 1)
- `brightness`: Image brightness (default: 100)
- `colour_scheme`: Options include greyscale, rainbow, heatmap
- `display`: Image display mode (default: image)
- `output format`: PNG (default), JPEG, TIFF

### Advanced Features
- Resolution ring display
- Ice ring visualization
- Threshold and sigma-based image processing
- Customizable image compression/quality settings

### Supported Input Types
- Raw image files (.cbf)
- Experiment files (.expt)

### Typical Use Cases
- Visualizing diffraction images
- Analyzing spot-finding intermediate steps
- Generating diagnostic image outputs

---

## dials.slice_sequence

### Overview
A DIALS command-line tool for slicing experimental sequences and reflections within specified image ranges.

### Purpose
"Slice a sequence to produce a smaller sequence within the bounds of the original"

### Usage Examples
```
dials.slice_sequence models.expt observations.refl "image_range=1 20"
dials.slice_sequence models.expt "image_range=1 20"
dials.slice_sequence models.expt observations.refl "image_range=1 20" "image_range=5 30"
```

### Key Parameters
- `output.reflections_filename`: Output filename for sliced reflections
- `output.experiments_filename`: Output filename for sliced experiments
- `image_range`: Specify image ranges to slice
- `block_size`: Optional parameter to split sequences into equal blocks
- `exclude_images_multiple`: Advanced option for splitting scans at specific intervals

### Behavior
- Modifies scan objects in experiments
- Removes reflections outside specified image ranges
- Supports multiple experiments with different image ranges

### Advanced Features
Includes an expert-level option for handling interrupted scans, particularly useful for cRED (continuous rotation electron diffraction) data.

---

## dials.compare_orientation_matrices

### Overview
A DIALS tool that "Computes the change of basis operator that minimises the difference between two orientation matrices" and calculates related transformation details.

### Usage Examples
```
dials.compare_orientation_matrices models.expt
dials.compare_orientation_matrices models_1.expt models_2.expt
dials.compare_orientation_matrices models_1.expt models_2.expt hkl=1,0,0
```

### Parameters
- `hkl`: Miller indices for comparison (default: None)
  - Type: Integer array of size 3
  - Can specify multiple indices
- `comparison`: Comparison method
  - Default: Pairwise sequential
- `space_group`: Optional space group specification

### Key Features
- Calculates rotation matrix between orientation matrices
- Computes Euler angles
- Optional Miller index angle calculation

### Supported Inputs
- Experimental model files (.expt)
- Multiple model file comparisons

Note: This documentation is derived from the DIALS project documentation, with attribution to Diamond Light Source, Lawrence Berkeley National Laboratory, and STFC.

---

## dials.spot_counts_per_image

### Introduction

A DIALS tool that "Reports the number of strong spots and computes an estimate of the resolution limit for each image" after running dials.find_spots.

### Example Usage

```
dials.spot_counts_per_image imported.expt strong.refl
dials.spot_counts_per_image imported.expt strong.refl plot=per_image.png
```

### Parameters

#### Basic Parameters
- `resolution_analysis`: Boolean (default: True)
- `plot`: Path (default: None)
- `json`: Path (default: None)
- `split_json`: Boolean (default: False)
- `joint_json`: Boolean (default: True)
- `id`: Integer (default: None)

### Full Parameter Definitions

All parameters match the basic parameter descriptions, with additional type specifications:
- `resolution_analysis`: Boolean
- `plot`: Path
- `json`: Path
- `split_json`: Boolean
- `joint_json`: Boolean
- `id`: Integer (minimum 0, can be None)

---

## dials.stereographic_projection

### Overview
A DIALS tool for calculating stereographic projection images of crystal models and Miller indices.

### Key Features
- Generates stereographic projections for crystal models
- Supports projection in crystal or laboratory frame
- Can expand to symmetry equivalents
- Eliminates systematically absent reflections optionally

### Basic Usage Examples
```
dials.stereographic_projection indexed.expt hkl=1,0,0 hkl=0,1,0
dials.stereographic_projection indexed.expt hkl_limit=2
dials.stereographic_projection indexed_1.expt indexed_2.expt hkl=1,0,0 expand_to_p1=True
```

### Key Parameters
- `hkl`: Specific Miller indices to project
- `hkl_limit`: Maximum Miller index to include
- `expand_to_p1`: Expand to symmetry equivalents (default: True)
- `eliminate_sys_absent`: Remove systematically absent reflections (default: False)
- `frame`: Projection frame (laboratory or crystal)

### Plot Configuration
Supports customizing:
- Filename
- Marker size
- Font size
- Color mapping
- Grid size
- Labels

### Output Options
- PNG image generation
- Optional JSON coordinate export

---

## dials.combine_experiments

### Overview

A DIALS utility script for combining multiple reflections and experiments files into a single multi-experiment reflections and experiments file.

### Key Features

- Matches experiments to reflections in the order they are provided
- Allows selection of reference models from input experiment files
- Can replace models like beam, crystal, detector, etc.
- Supports complex experiment combinations through multiple runs

### Basic Usage Example

```
dials.combine_experiments experiments_0.expt experiments_1.expt \
  reflections_0.refl reflections_1.refl \
  reference_from_experiment.beam=0 \
  reference_from_experiment.detector=0
```

### Main Parameters

#### Output Options
- Log file
- Experiments filename
- Reflections filename
- Subset selection methods

#### Reference Model Selection
- Choose reference models from specific experiments
- Options to average or compare models
- Configurable tolerances for model comparisons

#### Clustering Options
- Optional experiment clustering
- Dendrogram generation
- Cluster threshold and size controls

### Advanced Features

- Significance filtering
- Resolution cutoff
- Reflection count filtering
- Batch size management

### Typical Use Cases

- Combining datasets from multiple experiments
- Standardizing experimental models
- Preparing data for further analysis

---

## dials.align_crystal

### Introduction

The `dials.align_crystal` program calculates possible goniometer settings to re-align crystal axes. By default, it attempts to align primary crystal axes with the principle goniometer axis.

### Key Features

- Can align vectors in two modes:
  1. `mode=main` (default): First vector aligned along principle goniometer axis
  2. `mode=cusp`: First vector aligned perpendicular to beam and goniometer axis

### Example Commands

```
dials.align_crystal models.expt
dials.align_crystal models.expt vector=0,0,1 vector=0,1,0
dials.align_crystal models.expt frame=direct
```

### Parameters

#### Basic Parameters
- `space_group`: Default is None
- `align.mode`: Choose between `main` or `cusp`
- `align.crystal.vector`: Specify crystal vectors
- `align.crystal.frame`: Choose `reciprocal` or `direct`
- `output.json`: Default output file is `align_crystal.json`

### Modes of Operation

- **Main Mode**: Aligns first vector along principle goniometer axis
- **Cusp Mode**: Aligns first vector perpendicular to beam and goniometer axis

---

## dials.anvil_correction

### Overview
A DIALS utility to correct integrated reflection intensities for attenuation caused by diamond anvil cells in high-pressure X-ray diffraction experiments.

### Purpose
"Correct integrated intensities to account for attenuation by a diamond anvil cell."

### Key Features
- Calculates path lengths of incident and diffracted beams through diamond anvils
- Corrects reflection intensities before scaling
- Uses absorption and density calculations to estimate intensity attenuation

### Usage Examples
```
dials.anvil_correction integrated.expt integrated.refl
dials.anvil_correction integrated.expt integrated.refl thickness=1.2 normal=1,0,0
```

### Parameters
#### Anvil Properties
- `density`: Diamond density (default 3510 kg/m³)
- `thickness`: Anvil thickness in mm (default 1.5925 mm)
- `normal`: 3-vector orthogonal to anvil surfaces

#### Output Options
- `experiments`: Output experiment list file
- `reflections`: Output reflection table file (default: corrected.refl)
- `log`: Log file (default: dials.anvil_correction.log)

### Theoretical Basis
The correction calculates beam path lengths through diamond anvils and applies an exponential attenuation factor based on:
- Linear absorption coefficient
- Beam vector orientation
- Anvil material properties

### Reference
Hubbell & Seltzer (2004), NIST X-Ray Mass Attenuation Coefficients

---

## dials.missing_reflections

### Overview
A DIALS program designed to "Identify connected regions of missing reflections in the asymmetric unit."

### Method
The program works by:
1. Generating a complete set of possible Miller indices
2. Performing connected components analysis on missing reflections

### Usage Examples
```
dials.missing_reflections integrated.expt integrated.refl
dials.missing_reflections scaled.expt scaled.refl min_component_size=10
```

### Parameters
- `min_component_size` (int, default=0): "Only show connected regions larger than or equal to this."
- `d_min` (float, optional): Minimum resolution limit
- `d_max` (float, optional): Maximum resolution limit

### Key Characteristics
- Helps identify gaps in reflection data
- Provides analysis of missing reflection regions
- Configurable through resolution and component size parameters

### Licensing
Part of the DIALS software suite, with copyright held by Diamond Light Source, Lawrence Berkeley National Laboratory, and STFC.

---

## dials.filter_reflections

### Overview
A DIALS program that filters reflection files based on user-specified criteria, allowing selective output of reflection data.

### Key Features
- Filter reflections using boolean flag expressions
- Filter by resolution (d_min, d_max)
- Select reflections by:
  - Experiment IDs
  - Panels
  - Partiality
  - Intensity quality
  - Dead time
  - Ice rings

### Basic Usage Examples
```
dials.filter_reflections refined.refl flag_expression=used_in_refinement
dials.filter_reflections integrated.refl flag_expression="indexed & (failed_during_summation | failed_during_profile_fitting)"
dials.filter_reflections indexed.refl indexed.expt d_max=20 d_min=2.5
```

### Key Parameters
- `output.reflections`: Output filename for filtered reflections
- `flag_expression`: Boolean expression to select reflections
- `d_min`/`d_max`: Resolution limits
- `partiality`: Min/max reflection partiality
- `select_good_intensities`: Filter for trustworthy intensities
- `ice_rings.filter`: Option to filter ice ring reflections

### Filtering Logic
1. Evaluate optional boolean flag expression
2. Apply additional filters on reflection table values
3. If no parameters set, print available flag values

---

## dials.import_xds

### Introduction

The `dials.import_xds` program imports XDS processed data for use in DIALS. It requires up to three components:

1. An XDS.INP file to specify geometry
2. One of:
   - "INTEGRATE.HKL"
   - "XPARM.XDS"
   - Alternatively, "XDS_ASCII.HKL" or "GXPARM.XDS"
3. INTEGRATE.HKL or SPOT.XDS file to create a reflection table

### Example Usage

```bash
# Extract files from a directory
dials.import_xds /path/to/folder/containing/xds/inp/

# Specify INTEGRATE.HKL path
dials.import_xds /path/to/folder/containing/xds/inp/INTEGRATE.HKL

# Be explicit about reflection file
dials.import_xds /path/to/folder/containing/xds/inp/ SPOT.XDS

# Specify experiment metadata file
dials.import_xds /path/to/folder/containing/xds/inp/ xds_file=XPARM.XDS
```

### Basic Parameters

- `input.xds_file`: Specify XDS file (default: None)
- `output.reflections`: Output reflections filename
- `output.xds_experiments`: Output experiment list filename (default: "xds_models.expt")
- `remove_invalid`: Remove non-index reflections
- `add_standard_columns`: Add empty standard columns
- `read_varying_crystal`: Create scan-varying crystal model

### Additional Details

The program is flexible in handling XDS input files and can extract necessary metadata and reflection information from various XDS-generated files.