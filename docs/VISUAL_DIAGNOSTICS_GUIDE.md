# Visual Diagnostics Guide for DiffusePipe

This guide documents the visual diagnostic tools for verifying the correctness of diffuse scattering data extraction and processing in the DiffusePipe pipeline.

## Table of Contents

1. [Overview](#overview)
2. [Phase 2 Visual Diagnostics](#phase-2-visual-diagnostics)
   - [check_diffuse_extraction.py](#check_diffuse_extractionpy)
   - [run_phase2_e2e_visual_check.py](#run_phase2_e2e_visual_checkpy)
3. [Phase 3 Visual Diagnostics](#phase-3-visual-diagnostics)
   - [check_phase3_outputs.py](#check_phase3_outputspy)
   - [run_phase3_e2e_visual_check.py](#run_phase3_e2e_visual_checkpy)
4. [Integration Workflow](#integration-workflow)
5. [Troubleshooting](#troubleshooting)

## Overview

The visual diagnostics system provides comprehensive tools for validating DiffusePipe processing at different pipeline stages:

### Phase 2 Diagnostics
1. **`check_diffuse_extraction.py`** - Generates visual diagnostics from Phase 2 processing outputs
2. **`run_phase2_e2e_visual_check.py`** - Orchestrates Phase 1-2 pipeline with automatic diagnostics

### Phase 3 Diagnostics  
3. **`check_phase3_outputs.py`** - Generates visual diagnostics from Phase 3 processing outputs
4. **`run_phase3_e2e_visual_check.py`** - Orchestrates Phase 1-3 pipeline with automatic diagnostics

These tools are essential for:
- Validating implementation correctness at each processing stage
- Debugging voxelization, scaling, and merging pipeline issues
- Generating reference outputs for testing
- Quality control of diffuse scattering analysis
- Visual verification of 3D reciprocal space maps

---

## Phase 2 Visual Diagnostics

## check_diffuse_extraction.py

### Purpose

Generates visual diagnostics to verify diffuse scattering extraction and correction processes. Takes outputs from Phase 1 (DIALS processing) and Phase 2 (DataExtractor) to create comprehensive diagnostic plots.

### Location

`scripts/visual_diagnostics/check_diffuse_extraction.py`

### Key Features

- **8 Diagnostic Plot Types**: Comprehensive visual verification suite
- **Pixel Coordinate Support**: Enhanced visualizations when coordinates are available
- **Flexible Input**: Works with various mask types and background maps
- **Summary Reports**: Automated generation of diagnostic summaries

### Usage

#### Basic Command

```bash
python check_diffuse_extraction.py \
  --raw-image path/to/image.cbf \
  --expt path/to/experiment.expt \
  --total-mask path/to/total_mask.pickle \
  --npz-file path/to/extracted_data.npz
```

#### Full Command with Options

```bash
python check_diffuse_extraction.py \
  --raw-image path/to/image.cbf \
  --expt path/to/experiment.expt \
  --total-mask path/to/total_mask.pickle \
  --npz-file path/to/extracted_data.npz \
  --bragg-mask path/to/bragg_mask.pickle \
  --pixel-mask path/to/pixel_mask.pickle \
  --bg-map path/to/background.npy \
  --output-dir my_diagnostics \
  --verbose
```

### Command-Line Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--raw-image` | Yes | Path to raw CBF image file |
| `--expt` | Yes | Path to DIALS experiment .expt file |
| `--total-mask` | Yes | Path to total diffuse mask .pickle file |
| `--npz-file` | Yes | Path to DataExtractor output .npz file |
| `--bragg-mask` | No | Path to Bragg mask .pickle file (for overlay plots) |
| `--pixel-mask` | No | Path to global pixel mask .pickle file |
| `--bg-map` | No | Path to background map .npy/.pickle file |
| `--output-dir` | No | Output directory (default: extraction_visual_check) |
| `--verbose` | No | Enable verbose logging |

### Generated Diagnostics

#### 1. Diffuse Pixel Overlay (`diffuse_pixel_overlay.png`)
- **Purpose**: Verify mask application and pixel selection
- **Content**: Raw detector image with extracted diffuse pixels highlighted
- **Requirements**: Pixel coordinates in NPZ file
- **Key Checks**: 
  - Diffuse pixels avoid Bragg regions
  - Proper mask boundaries
  - No obvious Bragg peaks in diffuse selection

#### 2. Intensity Correction Summary (`intensity_correction_summary.txt`)
- **Purpose**: Document correction effects
- **Content**: Sample of corrected intensity values with Q-vectors
- **Format**: Tabular text with I, σ, I/σ values
- **Note**: Full transformation tracking requires enhanced DataExtractor logging

#### 3. Q-Space Projections (`q_projection_*.png`)
- **Purpose**: Verify Q-space coverage and distribution
- **Files**: 
  - `q_projection_qx_qy.png`
  - `q_projection_qx_qz.png`
  - `q_projection_qy_qz.png`
- **Content**: Scatter plots colored by intensity
- **Key Checks**: Even Q-space sampling, reasonable coverage

#### 4. Radial Q Distribution (`radial_q_distribution.png`)
- **Purpose**: Analyze intensity vs resolution
- **Content**: Intensity vs |Q| scatter plot
- **Key Checks**: Expected intensity falloff with resolution

#### 5. Intensity Histogram (`intensity_histogram.png`)
- **Purpose**: Validate intensity distribution
- **Content**: Dual histograms (linear and log scale)
- **Key Checks**: 
  - Positive intensities
  - Reasonable distribution shape
  - No unexpected bimodality

#### 6. Intensity Heatmap (`intensity_heatmap_panel_0.png`)
- **Purpose**: Spatial intensity distribution on detector
- **Content**: 2D heatmap of intensities at pixel positions
- **Requirements**: Pixel coordinates in NPZ file
- **Key Checks**: Smooth spatial variations, no artifacts

#### 7. Sigma vs Intensity (`sigma_vs_intensity.png`)
- **Purpose**: Validate error propagation
- **Content**: Scatter plot with Poisson noise reference
- **Key Checks**: Errors follow expected σ ≈ √I relationship

#### 8. I/σ Histogram (`isigi_histogram.png`)
- **Purpose**: Data quality assessment
- **Content**: Distribution with mean/median markers
- **Key Checks**: Reasonable I/σ values (typically > 1)

### Output Structure

```
output-dir/
├── diffuse_pixel_overlay.png
├── intensity_correction_summary.txt
├── q_projection_qx_qy.png
├── q_projection_qx_qz.png
├── q_projection_qy_qz.png
├── radial_q_distribution.png
├── intensity_histogram.png
├── intensity_heatmap_panel_0.png
├── sigma_vs_intensity.png
├── isigi_histogram.png
└── extraction_diagnostics_summary.txt
```

### NPZ File Requirements

The input NPZ file must contain:
- **Required**: `q_vectors`, `intensities`, `sigmas`
- **Optional**: `original_panel_ids`, `original_fast_coords`, `original_slow_coords`

Note: Pixel coordinate arrays enable additional diagnostic plots (overlay, heatmap).

---

## run_phase2_e2e_visual_check.py

### Purpose

Orchestrates the complete Phase 1 and Phase 2 pipeline for a single CBF image, then automatically runs visual diagnostics. Provides an end-to-end verification pathway from raw data to diagnostic plots.

### Location

`scripts/dev_workflows/run_phase2_e2e_visual_check.py`

### Key Features

- **Complete Pipeline Automation**: DIALS → Masking → Extraction → Diagnostics
- **Configurable Processing**: JSON-based parameter overrides
- **Pixel Coordinate Tracking**: Automatically enables for visual diagnostics
- **Organized Output**: Structured directory with all intermediate files
- **Comprehensive Logging**: Detailed progress and error tracking

### Usage

#### Basic Command

```bash
python run_phase2_e2e_visual_check.py \
  --cbf-image path/to/image.cbf \
  --output-base-dir ./e2e_outputs
```

#### With PDB Validation

```bash
python run_phase2_e2e_visual_check.py \
  --cbf-image path/to/image.cbf \
  --output-base-dir ./e2e_outputs \
  --pdb-path path/to/reference.pdb
```

#### Full Configuration Example

```bash
python run_phase2_e2e_visual_check.py \
  --cbf-image path/to/image.cbf \
  --output-base-dir ./e2e_outputs \
  --dials-phil-path custom_params.phil \
  --pdb-path reference.pdb \
  --static-mask-config '{"beamstop": {"type": "circle", "center_x": 1250, "center_y": 1250, "radius": 50}}' \
  --bragg-mask-config '{"border": 3}' \
  --extraction-config-json '{"pixel_step": 2, "min_intensity": 10.0}' \
  --use-bragg-mask-option-b \
  --verbose
```

### Command-Line Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--cbf-image` | Yes | Path to input CBF image file |
| `--output-base-dir` | Yes | Base output directory |
| `--dials-phil-path` | No | Custom DIALS PHIL configuration |
| `--pdb-path` | No | External PDB for validation |
| `--static-mask-config` | No | JSON for static mask parameters |
| `--bragg-mask-config` | No | JSON for Bragg mask parameters |
| `--use-bragg-mask-option-b` | No | Use shoebox-based Bragg masking |
| `--extraction-config-json` | No | JSON for extraction overrides |
| `--pixel-step` | No | Pixel sampling step size |
| `--save-pixel-coords` | No | Save pixel coordinates (default: True) |
| `--verbose` | No | Enable verbose logging |

### Pipeline Stages

#### Phase 1: DIALS Processing
1. **Import**: Load CBF file into DIALS format
2. **Find Spots**: Detect diffraction spots
3. **Index**: Determine crystal orientation
4. **Refine**: Optimize detector geometry

#### Phase 1: Mask Generation
1. **Pixel Masks**: Create static (beamstop, untrusted) and dynamic (hot pixels) masks
2. **Bragg Masks**: Create masks around Bragg peaks (spot or shoebox-based)
3. **Total Mask**: Combine all masks for diffuse region selection

#### Phase 2: Data Extraction
1. **Configure**: Apply extraction parameters
2. **Process**: Extract diffuse scattering with corrections
3. **Save**: Generate NPZ file with pixel coordinates

#### Phase 3: Visual Diagnostics
1. **Invoke**: Run check_diffuse_extraction.py
2. **Generate**: Create all diagnostic plots
3. **Report**: Save summaries and logs

### Configuration Examples

#### Static Mask Configuration

```json
{
  "beamstop": {
    "type": "circle",
    "center_x": 1250,
    "center_y": 1250,
    "radius": 50
  },
  "untrusted_rects": [
    {"min_x": 0, "max_x": 100, "min_y": 0, "max_y": 100}
  ],
  "untrusted_panels": [1, 3]
}
```

#### Bragg Mask Configuration

```json
{
  "border": 3,
  "algorithm": "simple",
  "resolution_range": [50.0, 2.0]
}
```

#### Extraction Configuration

```json
{
  "pixel_step": 2,
  "min_intensity": 10.0,
  "max_intensity": 100000.0,
  "min_res": 50.0,
  "max_res": 2.0,
  "gain": 1.0,
  "lp_correction_enabled": true,
  "air_temperature_k": 293.15,
  "air_pressure_atm": 1.0
}
```

### Output Structure

For input `image_001.cbf`:

```
output-base-dir/
└── image_001/
    ├── e2e_visual_check.log                # Complete pipeline log
    ├── imported.expt                       # DIALS import
    ├── strong.refl                         # Found spots
    ├── indexed_initial.expt                # Initial indexing
    ├── indexed_initial.refl                
    ├── indexed_refined_detector.expt       # Final refined
    ├── indexed_refined_detector.refl       
    ├── global_pixel_mask.pickle            # Pixel masks
    ├── bragg_mask.pickle                   # Bragg mask
    ├── total_diffuse_mask.pickle           # Combined mask
    ├── diffuse_data.npz                    # Extracted data
    └── extraction_diagnostics/             # Visual plots
        ├── diffuse_pixel_overlay.png
        ├── q_projection_*.png
        ├── intensity_histogram.png
        ├── [... all diagnostic plots ...]
        └── extraction_diagnostics_summary.txt
```

### Error Handling

The script provides detailed error reporting for each phase:

- **DIALS Failures**: Missing files, indexing failures, insufficient spots
- **Mask Failures**: Invalid parameters, memory issues
- **Extraction Failures**: Configuration errors, no valid pixels
- **Diagnostic Failures**: Missing dependencies (non-fatal)

---

## Integration Workflow

### Recommended Workflow

1. **Initial Testing**: Run on a single representative CBF file
2. **Parameter Optimization**: Adjust configurations based on diagnostics
3. **Batch Processing**: Apply optimized parameters to dataset
4. **Quality Control**: Review diagnostic plots for anomalies

### Example Workflow

```bash
# Step 1: Test with defaults
python run_phase2_e2e_visual_check.py \
  --cbf-image test_image.cbf \
  --output-base-dir ./test_run

# Step 2: Review diagnostics
ls test_run/test_image/extraction_diagnostics/

# Step 3: Adjust parameters based on results
python run_phase2_e2e_visual_check.py \
  --cbf-image test_image.cbf \
  --output-base-dir ./optimized_run \
  --extraction-config-json '{"pixel_step": 1, "min_intensity": 5.0}' \
  --bragg-mask-config '{"border": 5}'

# Step 4: Apply to multiple images
for cbf in data/*.cbf; do
  python run_phase2_e2e_visual_check.py \
    --cbf-image "$cbf" \
    --output-base-dir ./batch_run \
    --extraction-config-json '{"pixel_step": 1, "min_intensity": 5.0}'
done
```

### Integration with CI/CD

Both scripts can be integrated into automated testing:

```yaml
# Example GitHub Actions workflow
- name: Run E2E Visual Check
  run: |
    python scripts/dev_workflows/run_phase2_e2e_visual_check.py \
      --cbf-image test_data/reference.cbf \
      --output-base-dir ./ci_output \
      --pdb-path test_data/reference.pdb
    
- name: Verify Outputs
  run: |
    test -f ci_output/reference/diffuse_data.npz
    test -f ci_output/reference/extraction_diagnostics/extraction_diagnostics_summary.txt
```

---

## Troubleshooting

### Common Issues

#### 1. "No pixel coordinates in NPZ file"
**Cause**: DataExtractor not configured to save coordinates
**Solution**: Ensure `--save-pixel-coords` is set (default: True)

#### 2. "DIALS processing failed"
**Cause**: Poor diffraction, wrong parameters, or format issues
**Solution**: 
- Check CBF file is valid
- Review DIALS logs in output directory
- Try different indexing methods

#### 3. "No pixels passed filtering criteria"
**Cause**: Overly restrictive filters or masking
**Solution**:
- Reduce `min_intensity` threshold
- Check mask coverage isn't too aggressive
- Increase `pixel_step` for faster testing

#### 4. Memory errors with large detectors
**Cause**: Processing all pixels at once
**Solution**: Increase `pixel_step` to reduce memory usage

#### 5. Visual diagnostics subprocess fails
**Cause**: Missing dependencies or path issues
**Solution**: 
- Ensure matplotlib is installed
- Check script paths are correct
- Review subprocess error logs

### Performance Optimization

#### Processing Time Factors
- **Detector Size**: 6M pixel detector takes longer than 1M
- **Pixel Step**: step=1 (all pixels) vs step=2 (1/4 pixels)
- **DIALS Parameters**: Spot finding thresholds affect speed
- **Diagnostic Plots**: Disable with `plot_diagnostics: false`

#### Recommended Settings by Use Case

**Quick Testing**:
```json
{"pixel_step": 4, "plot_diagnostics": false}
```

**Full Analysis**:
```json
{"pixel_step": 1, "plot_diagnostics": true}
```

**Memory Limited**:
```json
{"pixel_step": 2, "min_intensity": 10.0}
```

### Debug Mode

For detailed debugging, use verbose mode and check logs:

```bash
# Enable verbose logging
python run_phase2_e2e_visual_check.py \
  --cbf-image image.cbf \
  --output-base-dir ./debug_run \
  --verbose

# Check detailed logs
tail -f debug_run/image/e2e_visual_check.log

# Check DIALS processing logs
cat debug_run/image/dials.*.log
```

### File Format Requirements

#### CBF Files
- Must have valid crystallographic headers
- Oscillation or still image data
- Supported detector formats

#### Pickle Files
- DIALS-compatible mask tuples
- Proper panel indexing
- Boolean mask arrays

#### NPZ Files
- Required arrays: q_vectors, intensities, sigmas
- Optional: original_panel_ids, original_fast_coords, original_slow_coords
- Compressed format supported

---

## Phase 3 Visual Diagnostics

### Purpose

Phase 3 visual diagnostics provide comprehensive verification of the voxelization, relative scaling, and merging processes that transform corrected diffuse pixel observations into a 3D reciprocal space map. These tools are essential for validating the final stages of the DiffusePipe pipeline.

### Key Components

1. **`check_phase3_outputs.py`** - Standalone diagnostic generation from Phase 3 outputs
2. **`run_phase3_e2e_visual_check.py`** - Complete Phase 1-3 pipeline orchestration with automatic diagnostics

---

## check_phase3_outputs.py

### Purpose

Generates comprehensive visual diagnostics to verify Phase 3 voxelization, relative scaling, and merging processes. Takes outputs from the complete Phase 3 pipeline (global voxel grid, refined scaling parameters, and merged voxel data) to create diagnostic plots and summary reports.

### Location

`scripts/visual_diagnostics/check_phase3_outputs.py`

### Key Features

- **Multi-dimensional Visualization**: 3D reciprocal space slicing and projections
- **Scaling Analysis**: Per-still scale factor and resolution smoother visualization
- **Quality Metrics**: Voxel occupancy, redundancy, and intensity/sigma analysis
- **Comprehensive Reporting**: Automated generation of summary statistics
- **Performance Optimized**: Configurable point limits for large datasets

### Usage

#### Basic Command

```bash
python check_phase3_outputs.py \
  --grid-definition-file phase3_outputs/global_voxel_grid_definition.json \
  --scaling-model-params-file phase3_outputs/refined_scaling_model_params.json \
  --voxel-data-file phase3_outputs/voxel_data_relative.npz \
  --output-dir phase3_diagnostics
```

#### Full Command with Options

```bash
python check_phase3_outputs.py \
  --grid-definition-file grid_def.json \
  --scaling-model-params-file scaling_params.json \
  --voxel-data-file voxel_data.npz \
  --output-dir diagnostics \
  --experiments-list-file experiments_list.txt \
  --corrected-pixel-data-dir pixel_data_dirs.txt \
  --max-plot-points 25000 \
  --verbose
```

### Command-Line Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--grid-definition-file` | Yes | Path to GlobalVoxelGrid definition JSON file |
| `--scaling-model-params-file` | Yes | Path to refined scaling model parameters JSON file |
| `--voxel-data-file` | Yes | Path to VoxelData_relative NPZ/HDF5 file |
| `--output-dir` | Yes | Output directory for diagnostic plots and reports |
| `--experiments-list-file` | No | Path to file containing list of experiment (.expt) file paths |
| `--corrected-pixel-data-dir` | No | Path to file containing list of corrected pixel data directories |
| `--max-plot-points` | No | Maximum number of points in scatter plots (default: 50000) |
| `--verbose` | No | Enable verbose logging |

### Generated Diagnostics

#### 1. Grid Summary (`grid_summary.txt`, `grid_visualization_conceptual.png`)
- **Purpose**: Document global voxel grid parameters and visualize coverage
- **Text Content**: 
  - Average crystal parameters (unit cell, space group)
  - HKL bounds and voxel dimensions
  - Total voxels and voxel sizes
- **Plot Content**: 3D wireframe showing HKL bounds with sample grid points
- **Key Checks**: Reasonable grid coverage, appropriate voxel resolution

#### 2. Voxel Occupancy Analysis
- **Files**: 
  - `voxel_occupancy_slice_L0.png`, `voxel_occupancy_slice_K0.png`, `voxel_occupancy_slice_H0.png`
  - `voxel_occupancy_histogram.png`
- **Purpose**: Analyze data redundancy and completeness
- **Content**: 
  - 2D heatmap slices of observation counts per voxel
  - Histogram of voxel occupancy distribution
- **Key Checks**: Even data distribution, adequate redundancy, identification of gaps

#### 3. Scaling Model Parameters
- **Files**:
  - `scaling_params_b_i.png` - Per-still multiplicative scale factors
  - `scaling_resolution_smoother.png` - Resolution smoother curve (if enabled)
  - `scaling_parameters_summary.txt` - Parameter statistics
- **Purpose**: Validate relative scaling convergence and parameters
- **Content**: Scale factor trends, smoother function, refinement statistics
- **Key Checks**: Reasonable scale factor range, smooth convergence, stable parameters

#### 4. Merged Voxel Data Visualization
- **Intensity Slices**: `merged_intensity_slice_L0.png`, etc. (log scale)
- **Sigma Slices**: `merged_sigma_slice_L0.png`, etc.
- **I/Sigma Slices**: `merged_isigi_slice_L0.png`, etc.
- **Radial Analysis**: `merged_radial_average.png` - Intensity vs |q|
- **Distribution**: `merged_intensity_histogram.png` - Intensity distribution
- **Purpose**: Verify final merged diffuse scattering map quality
- **Key Checks**: Smooth intensity variations, reasonable I/σ ratios, proper resolution trends

#### 5. Comprehensive Summary (`phase3_diagnostics_summary.txt`)
- **Purpose**: Consolidated report of all Phase 3 processing statistics
- **Content**:
  - Input file paths and processing parameters
  - Grid definition summary
  - Voxel occupancy statistics
  - Scaling model convergence metrics
  - Merged intensity quality metrics
  - Resolution coverage analysis
  - Generated plot inventory

### Output Structure

```
output-dir/
├── grid_summary.txt
├── grid_visualization_conceptual.png
├── voxel_occupancy_slice_L0.png
├── voxel_occupancy_slice_K0.png
├── voxel_occupancy_slice_H0.png
├── voxel_occupancy_histogram.png
├── scaling_params_b_i.png
├── scaling_resolution_smoother.png (if enabled)
├── scaling_parameters_summary.txt
├── merged_intensity_slice_L0.png
├── merged_intensity_slice_K0.png
├── merged_intensity_slice_H0.png
├── merged_sigma_slice_L0.png
├── merged_sigma_slice_K0.png
├── merged_sigma_slice_H0.png
├── merged_isigi_slice_L0.png
├── merged_isigi_slice_K0.png
├── merged_isigi_slice_H0.png
├── merged_radial_average.png
├── merged_intensity_histogram.png
└── phase3_diagnostics_summary.txt
```

### Input File Requirements

#### Grid Definition JSON
- `crystal_avg_ref`: Average crystal parameters (unit cell, space group)
- `hkl_bounds`: H/K/L min/max values
- `ndiv_h/k/l`: Voxel divisions per unit cell
- `total_voxels`: Total number of voxels

#### Scaling Parameters JSON
- `refined_parameters`: Per-still scale factors and offsets
- `refinement_statistics`: Convergence metrics and R-factors
- `resolution_smoother`: Smoother configuration and control points

#### Voxel Data NPZ/HDF5
- **Required Arrays**: `voxel_indices`, `H_center`, `K_center`, `L_center`, `q_center_x/y/z`, `q_magnitude_center`, `I_merged_relative`, `Sigma_merged_relative`, `num_observations`
- **Format**: Compressed NPZ or HDF5 with consistent array lengths

---

## run_phase3_e2e_visual_check.py

### Purpose

Orchestrates the complete Phase 1 (True Sequence Processing), Phase 2 (diffuse extraction with shared model), and Phase 3 (voxelization, scaling, merging) pipeline for multiple CBF images, then automatically runs Phase 3 visual diagnostics. Provides end-to-end verification from raw data to final 3D diffuse maps with perfect crystal orientation consistency.

**KEY ARCHITECTURAL CHANGE**: This script now uses true sequence processing where all images are processed together as a single cohesive dataset, achieving perfect crystal orientation consistency (0.0000° RMS misorientation) and eliminating indexing ambiguity issues.

### Location

`scripts/dev_workflows/run_phase3_e2e_visual_check.py`

### Key Features

- **True Sequence Processing**: All images processed as single DIALS dataset for perfect consistency
- **Perfect Crystal Orientation**: Achieves 0.0000° RMS misorientation between images
- **Shared Model Extraction**: Phase 2 uses consistent crystal model for all images
- **Complete Pipeline**: Phases 1-3 orchestration with automatic diagnostics
- **Configurable Parameters**: JSON-based configuration for all pipeline stages
- **Intermediate Output Management**: Optional saving of all intermediate files
- **Robust Error Handling**: Comprehensive error tracking with graceful continuation
- **Comprehensive Logging**: Detailed progress tracking and debugging information

### Usage

#### Basic Command

```bash
python run_phase3_e2e_visual_check.py \
  --cbf-image-paths image1.cbf image2.cbf image3.cbf \
  --output-base-dir ./e2e_outputs_phase3
```

#### With PDB Validation and Configuration

```bash
python run_phase3_e2e_visual_check.py \
  --cbf-image-paths path/to/image1.cbf path/to/image2.cbf \
  --output-base-dir ./outputs \
  --pdb-path reference.pdb \
  --dials-phil-path custom_dials.phil \
  --extraction-config-json '{"pixel_step": 2}' \
  --relative-scaling-config-json '{"enable_res_smoother": true}' \
  --grid-config-json '{"ndiv_h": 100, "ndiv_k": 100, "ndiv_l": 50}' \
  --save-intermediate-phase-outputs \
  --verbose
```

#### Testing RMS Misorientation Validation

```bash
# Test with images from 10_6 series (expect 2.74° misorientation > 2.0° threshold)
python run_phase3_e2e_visual_check.py \
  --cbf-image-paths 747/lys_nitr_10_6_0491.cbf 747/lys_nitr_10_6_0492.cbf \
  --output-base-dir ./test_fix_output \
  --pdb-path 6o2h.pdb \
  --verbose
```

### Command-Line Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--cbf-image-paths` | Yes | Space-separated paths to input CBF image files |
| `--output-base-dir` | Yes | Base output directory (subdirectories created automatically) |
| `--pdb-path` | No | Path to external PDB file for validation |
| `--dials-phil-path` | No | Path to custom DIALS PHIL configuration file |
| `--static-mask-config` | No | JSON string for static mask configuration |
| `--bragg-mask-config` | No | JSON string for Bragg mask configuration |
| `--use-bragg-mask-option-b` | No | Use shoebox-based Bragg masking |
| `--extraction-config-json` | No | JSON string for Phase 2 extraction configuration |
| `--relative-scaling-config-json` | No | JSON string for Phase 3 relative scaling configuration |
| `--grid-config-json` | No | JSON string for voxel grid configuration |
| `--save-intermediate-phase-outputs` | No | Save manifest of all intermediate files |
| `--verbose` | No | Enable verbose logging |

### Pipeline Stages

#### Phase 1: True Sequence Processing
- **Single Dataset Processing**: All CBF files processed together using DIALS sequence workflow
- **Scan-Varying Refinement**: DIALS maintains consistent crystal orientation across all images
- **Shared Experiment Model**: Single experiment file used for all subsequent processing
- **Per-Image Mask Generation**: Individual masks created using the consistent crystal model
- **Output**: Single shared experiment/reflection files, per-image mask files

#### Phase 2: Shared Model Data Extraction
- **Consistent Model**: All images use the same crystal model from sequence processing
- **Per-Image Extraction**: DataExtractor processing with pixel corrections for each image
- **Coordinate Tracking**: Automatic saving of original pixel coordinates
- **Configuration**: Flexible extraction parameter overrides
- **Output**: NPZ files with corrected diffuse observations per image

#### Phase 3: Voxelization and Merging
- **Grid Definition**: Global voxel grid creation from all crystal models
- **Observation Binning**: Assignment of diffuse pixels to voxels with symmetry
- **Relative Scaling**: Iterative refinement of per-image scale factors
- **Data Merging**: Weighted combination of scaled observations per voxel
- **Output**: Grid definition, scaling parameters, merged voxel data

#### Phase 3 Diagnostics
- **Automatic Invocation**: `check_phase3_outputs.py` called with generated outputs
- **Comprehensive Analysis**: All Phase 3 diagnostic plots and summaries
- **Quality Assessment**: Visual verification of voxelization and scaling

### Configuration Examples

#### Basic Grid Configuration
```json
{
  "ndiv_h": 100,
  "ndiv_k": 100, 
  "ndiv_l": 50
}
```

#### Advanced Scaling Configuration
```json
{
  "enable_res_smoother": true,
  "max_iterations": 10,
  "convergence_tolerance": 0.001
}
```

#### Extraction Configuration
```json
{
  "pixel_step": 2,
  "min_intensity": 0.0,
  "save_original_pixel_coordinates": true
}
```

### Output Structure

```
output-base-dir/
├── phase1_image1/
│   ├── indexed_refined_detector.expt
│   ├── indexed_refined_detector.refl
│   ├── global_pixel_mask.pickle
│   ├── bragg_mask.pickle
│   └── total_diffuse_mask.pickle
├── phase1_image2/
│   └── ... (similar structure)
├── phase3_outputs/
│   ├── global_voxel_grid_definition.json
│   ├── refined_scaling_model_params.json
│   ├── voxel_data_relative.npz
│   └── diagnostics/
│       ├── grid_summary.txt
│       ├── voxel_occupancy_slice_L0.png
│       ├── scaling_params_b_i.png
│       ├── merged_intensity_slice_L0.png
│       └── phase3_diagnostics_summary.txt
├── intermediate_outputs_manifest.json (if --save-intermediate-phase-outputs)
└── e2e_phase3_visual_check.log
```

### Performance Considerations

- **Memory Management**: Large datasets may require disk-based voxel accumulation
- **Processing Time**: Scales with number of images and voxel grid resolution
- **Disk Space**: Intermediate files can be substantial for large datasets
- **Parallelization**: Phase 1 processing parallelized across available CPU cores

---

## Summary

These visual diagnostic tools provide comprehensive verification of the complete DiffusePipe pipeline:

### Phase 2 Diagnostics
- **`check_diffuse_extraction.py`**: Standalone diagnostic generation from Phase 2 outputs
- **`run_phase2_e2e_visual_check.py`**: Phase 1-2 pipeline orchestration with automatic diagnostics

### Phase 3 Diagnostics
- **`check_phase3_outputs.py`**: Standalone diagnostic generation from Phase 3 outputs  
- **`run_phase3_e2e_visual_check.py`**: Complete Phase 1-3 pipeline orchestration with automatic diagnostics

Together, these tools enable thorough validation of:
- **Phase 2**: Diffuse scattering extraction, pixel corrections, and data quality
- **Phase 3**: Voxelization, relative scaling, merging, and 3D reciprocal space map generation

This comprehensive diagnostic framework is essential for ensuring the correctness of the entire DiffusePipe processing pipeline, from raw crystallographic images to final diffuse scattering maps ready for scientific analysis.