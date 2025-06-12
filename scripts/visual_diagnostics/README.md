# Visual Diagnostics for DiffusePipe Phase 0 & 1

This directory contains scripts for visual verification of Phase 0 and Phase 1 processing outputs. These tools help validate that DIALS processing, mask generation, and data transformations are working correctly through manual inspection of generated plots.

**📖 For comprehensive documentation on the latest visual diagnostic tools (including Phase 2), see [docs/VISUAL_DIAGNOSTICS_GUIDE.md](../../docs/VISUAL_DIAGNOSTICS_GUIDE.md)**

## Overview

The visual diagnostic scripts are designed to:
- Load real crystallographic data and processing outputs
- Generate informative plots and visualizations  
- Save results for manual inspection
- Provide summary statistics and configuration details

## Scripts

### 1. `check_dials_processing.py` - Module 1.S.1 Visual Checks

**Purpose**: Verify DIALS stills processing results by visualizing spot finding, indexing, and refinement outputs.

**Key Visualizations**:
- Raw still image with logarithmic intensity scaling
- Observed spot positions overlaid on raw image
- Predicted spot positions from crystal model
- Combined view showing observed vs. predicted spots

**Usage Examples**:
```bash
# Basic DIALS processing check
python check_dials_processing.py \
  --raw-image ../../747/lys_nitr_10_6_0491.cbf \
  --expt ../../lys_nitr_10_6_0491_dials_processing/indexed_refined_detector.expt \
  --refl ../../lys_nitr_10_6_0491_dials_processing/indexed_refined_detector.refl \
  --output-dir dials_visual_check

# With verbose output
python check_dials_processing.py --raw-image image.cbf --expt processed.expt --refl processed.refl --verbose
```

**Output Files**:
- `raw_image.png` - Raw detector image
- `image_with_observed_spots.png` - Raw image with observed spots (red)
- `image_with_predicted_spots.png` - Raw image with predicted spots (blue) 
- `image_with_both_spots.png` - Combined observed and predicted spots
- `dials_processing_info.txt` - Summary statistics and crystal model details

**What to Look For**:
- Spots should be visible and well-distributed across the detector
- Observed and predicted spots should overlap well (good indexing)
- Crystal model parameters should be reasonable
- Partiality values should be in expected range (0-1)

---

### 2. `check_pixel_masks.py` - Module 1.S.2 Visual Checks

**Purpose**: Verify static and dynamic pixel mask generation by visualizing masked regions and mask combination logic.

**Key Visualizations**:
- Static mask showing beamstop, untrusted regions, panel exclusions
- Dynamic mask showing hot/negative pixels from image analysis
- Combined mask showing final pixel selection
- Panel-by-panel comparison views

**Usage Examples**:
```bash
# Basic pixel mask check with defaults
python check_pixel_masks.py \
  --expt ../../lys_nitr_10_6_0491_dials_processing/indexed_refined_detector.expt \
  --images ../../747/lys_nitr_10_6_0491.cbf \
  --output-dir pixel_mask_check

# With custom beamstop configuration
python check_pixel_masks.py \
  --expt detector.expt \
  --images image1.cbf image2.cbf \
  --static-config '{"beamstop": {"type": "circle", "center_x": 1250, "center_y": 1250, "radius": 50}}' \
  --output-dir pixel_mask_check

# Multiple representative images for dynamic analysis
python check_pixel_masks.py \
  --expt detector.expt \
  --images image1.cbf image2.cbf image3.cbf \
  --dynamic-config '{"hot_pixel_thresh": 500000, "negative_pixel_tolerance": 5.0}' \
  --output-dir pixel_mask_check
```

**Configuration Options**:

Static mask config (JSON):
```json
{
  "beamstop": {
    "type": "circle",
    "center_x": 1250,
    "center_y": 1250, 
    "radius": 50
  },
  "untrusted_rects": [
    {"min_x": 0, "max_x": 10, "min_y": 0, "max_y": 10}
  ],
  "untrusted_panels": [1, 3]
}
```

Dynamic mask config (JSON):
```json
{
  "hot_pixel_thresh": 1000000,
  "negative_pixel_tolerance": 0.0,
  "max_fraction_bad_pixels": 0.1
}
```

**Output Files**:
- `panel_X_static_mask.png` - Static mask for panel X
- `panel_X_dynamic_mask.png` - Dynamic mask for panel X  
- `panel_X_combined_mask.png` - Combined mask for panel X
- `panel_X_mask_comparison.png` - Side-by-side comparison
- `pixel_mask_info.txt` - Mask statistics and configuration summary

**What to Look For**:
- Static masks should correctly exclude beamstop and untrusted regions
- Dynamic masks should flag obvious hot/negative pixels without being overly aggressive
- Combined masks should preserve most detector area for diffuse analysis
- Mask fraction should be reasonable (typically <10-20% rejected)

---

### 3. `check_total_mask.py` - Module 1.S.3 Visual Checks

**Purpose**: Verify Bragg mask generation and combination with pixel masks to create final diffuse analysis masks.

**Key Visualizations**:
- Raw image with Bragg peak mask overlay
- Raw image with total mask showing diffuse regions
- Masked image showing only diffuse pixels
- Side-by-side mask comparison
- Individual mask components

**Usage Examples**:
```bash
# Basic total mask check (uses dials.generate_mask)
python check_total_mask.py \
  --raw-image ../../747/lys_nitr_10_6_0491.cbf \
  --expt ../../lys_nitr_10_6_0491_dials_processing/indexed_refined_detector.expt \
  --refl ../../lys_nitr_10_6_0491_dials_processing/indexed_refined_detector.refl \
  --output-dir total_mask_check

# Using shoebox-based Bragg masking (Option B)
python check_total_mask.py \
  --raw-image image.cbf --expt processed.expt --refl processed.refl \
  --use-option-b --output-dir total_mask_check

# With custom Bragg mask configuration
python check_total_mask.py \
  --raw-image image.cbf --expt processed.expt --refl processed.refl \
  --bragg-config '{"border": 3, "algorithm": "simple"}' \
  --output-dir total_mask_check
```

**Output Files**:
- `raw_image.png` - Raw detector image
- `image_with_bragg_mask.png` - Raw image with Bragg regions highlighted (red)
- `image_with_total_diffuse_mask.png` - Raw image with diffuse regions highlighted (blue)
- `image_diffuse_pixels_only.png` - Masked image showing only diffuse pixels
- `mask_comparison.png` - Side-by-side comparison of all masks
- `global_pixel_mask.png`, `bragg_mask.png`, `total_diffuse_mask.png` - Individual masks
- `total_mask_info.txt` - Mask statistics and processing summary

**What to Look For**:
- Bragg masks should cover indexed reflection positions
- Diffuse regions should exclude both Bragg peaks and bad pixels
- Final diffuse mask should preserve most detector area
- No obvious Bragg peaks should remain in diffuse regions
- Reflection statistics should match expectations

---

### 4. `check_diffuse_extraction.py` - Diffuse Extraction Verification

**Purpose**: Verify diffuse scattering extraction and correction processes by visualizing the outputs from Phase 1 (DIALS processing) and Phase 2 (DataExtractor).

**Key Visualizations**:
- Raw image with extracted diffuse pixels overlay
- Q-space coverage projections (qx vs qy, qx vs qz, qy vs qz)
- Radial Q-space distribution (intensity vs |Q|)
- Intensity distribution histograms
- Intensity heatmap on detector (if pixel coordinates available)
- Sigma vs intensity scatter plot with Poisson noise reference
- I/σ distribution histogram with statistics

**Usage Examples**:
```bash
# Basic diffuse extraction check
python check_diffuse_extraction.py \
  --raw-image ../../747/lys_nitr_10_6_0491.cbf \
  --expt ../../lys_nitr_10_6_0491_dials_processing/indexed_refined_detector.expt \
  --total-mask ../../lys_nitr_10_6_0491_dials_processing/total_diffuse_mask.pickle \
  --npz-file extraction_output.npz

# With optional masks and background map
python check_diffuse_extraction.py \
  --raw-image image.cbf --expt experiment.expt \
  --total-mask total_mask.pickle --npz-file data.npz \
  --bragg-mask bragg_mask.pickle \
  --pixel-mask pixel_mask.pickle \
  --bg-map background_map.npy \
  --output-dir custom_output \
  --verbose
```

**Output Files**:
- `diffuse_pixel_overlay.png` - Raw image with diffuse pixels highlighted (green)
- `q_projection_qx_qy.png`, `q_projection_qx_qz.png`, `q_projection_qy_qz.png` - Q-space projections
- `radial_q_distribution.png` - Intensity vs radial Q scatter plot
- `intensity_histogram.png` - Intensity distribution (linear and log scale)
- `intensity_heatmap_panel_0.png` - Intensity mapped back to detector coordinates
- `sigma_vs_intensity.png` - Error analysis with Poisson noise reference
- `isigi_histogram.png` - I/σ distribution with mean and median markers
- `intensity_correction_summary.txt` - Sample of corrected intensity values
- `extraction_diagnostics_summary.txt` - Overall summary and file listing

**What to Look For**:
- Diffuse pixels should avoid Bragg peak regions and bad pixel areas
- Q-space coverage should be reasonable and evenly distributed
- Intensity distributions should be physically reasonable (positive, not bimodal)
- I/σ values should be reasonable (typically > 1 for good data)
- Intensity heatmap should show smooth spatial distribution
- Sigma values should follow roughly Poisson statistics (σ ≈ √I)

**Important Notes**:
- Some plots (pixel overlay, intensity heatmap) require original pixel coordinates to be saved in the NPZ file
- If pixel coordinates are not available, these plots will be skipped with a warning
- The intensity correction plot is simplified due to lack of intermediate processing data
- For multi-panel detectors, only panel 0 is visualized by default

---

## General Usage Tips

### Prerequisites
- DIALS must be installed and importable
- Raw crystallographic data (CBF files) and processed outputs (EXPT/REFL files)
- Python packages: matplotlib, numpy, pathlib

### Common Workflows

1. **Full Pipeline Check**: Run all three scripts sequentially on the same dataset
2. **Troubleshooting**: Use verbose mode (`--verbose`) for detailed logging
3. **Parameter Tuning**: Modify configuration JSON to test different masking parameters
4. **Batch Analysis**: Create shell scripts to run checks on multiple datasets

### Interpreting Results

**Good Results Indicate**:
- Clear, well-distributed spot patterns
- Good agreement between observed and predicted spots
- Reasonable mask coverage (not too aggressive)
- Clean separation of Bragg and diffuse regions

**Warning Signs**:
- Very few or no indexed spots
- Large discrepancies between observed/predicted positions
- Excessive masking (>50% of detector excluded)
- Obvious Bragg peaks remaining in diffuse regions

### Output Management

All scripts create timestamped output directories and preserve input configurations in summary files. This allows for:
- Tracking parameter changes over time
- Reproducing specific analysis conditions
- Comparing results across different datasets

### Integration with CI/CD

These scripts can be integrated into automated testing pipelines by:
- Running on reference datasets with known good outputs
- Checking that output files are generated successfully
- Validating summary statistics fall within expected ranges

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure DIALS is properly installed and PYTHONPATH includes project src/
2. **File Not Found**: Check that input file paths are correct and files exist
3. **Memory Issues**: Large detector images may require sufficient RAM for visualization
4. **Display Issues**: Scripts use non-interactive matplotlib backend (Agg) for automation

### Getting Help

- Use `--help` flag with any script for detailed usage information
- Check log output for specific error messages
- Ensure input files are valid DIALS formats
- Verify detector geometry is reasonable for your experimental setup

## Future Enhancements

Potential improvements for these diagnostic tools:
- Interactive plotting modes for detailed inspection
- Automated quality metrics and pass/fail criteria  
- Integration with DIALS viewer commands
- Support for multi-panel detector visualization
- Statistical comparison against reference datasets