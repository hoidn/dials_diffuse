# Visual Diagnostics Guide for DiffusePipe

This guide documents the visual diagnostic tools for verifying the correctness of diffuse scattering data extraction and processing in the DiffusePipe pipeline.

## Table of Contents

1. [Overview](#overview)
2. [check_diffuse_extraction.py](#check_diffuse_extractionpy)
3. [run_phase2_e2e_visual_check.py](#run_phase2_e2e_visual_checkpy)
4. [Integration Workflow](#integration-workflow)
5. [Troubleshooting](#troubleshooting)

## Overview

The visual diagnostics system provides two key scripts:

1. **`check_diffuse_extraction.py`** - Generates comprehensive visual diagnostics from existing processing outputs
2. **`run_phase2_e2e_visual_check.py`** - Orchestrates the complete pipeline from raw data to visual diagnostics

These tools are essential for:
- Validating Phase 2 implementation correctness
- Debugging extraction pipeline issues
- Generating reference outputs for testing
- Quality control of diffuse scattering analysis

---

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

## Summary

These visual diagnostic tools provide comprehensive verification of the DiffusePipe Phase 2 implementation:

- **`check_diffuse_extraction.py`**: Standalone diagnostic generation from existing outputs
- **`run_phase2_e2e_visual_check.py`**: Complete pipeline orchestration with automatic diagnostics

Together, they enable thorough validation of diffuse scattering extraction, pixel corrections, and data quality, essential for ensuring the correctness of the DiffusePipe processing pipeline.