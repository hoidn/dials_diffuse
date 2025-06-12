# Development Workflows for DiffusePipe

This directory contains development and testing workflows that orchestrate multiple pipeline components to provide end-to-end processing and validation capabilities.

**📖 For comprehensive documentation on all visual diagnostic tools, see [docs/VISUAL_DIAGNOSTICS_GUIDE.md](../../docs/VISUAL_DIAGNOSTICS_GUIDE.md)**

## Scripts

### `run_phase2_e2e_visual_check.py` - End-to-End Phase 2 Visual Verification

**Purpose**: Complete end-to-end pipeline that processes a single CBF image through Phase 1 (DIALS processing, masking) and Phase 2 (DataExtractor), then generates comprehensive visual diagnostics to verify correctness.

**What it does:**
1. **Phase 1 - DIALS Processing**: Imports, finds spots, indexes, and refines detector geometry
2. **Phase 1 - Mask Generation**: Creates pixel masks (static + dynamic) and Bragg masks 
3. **Phase 1 - Total Mask**: Combines masks to create final diffuse analysis mask
4. **Phase 2 - Data Extraction**: Extracts diffuse scattering data with pixel corrections
5. **Phase 3 - Visual Diagnostics**: Runs comprehensive visual verification plots

**Key Features:**
- Fully automated pipeline from raw CBF to visual diagnostics
- Configurable DIALS processing parameters
- Support for both spot-based and shoebox-based Bragg masking
- Pixel coordinate tracking for detailed visual analysis
- Comprehensive error handling and logging
- Organized output directory structure

**Usage Examples:**

```bash
# Basic usage with CBF file and PDB reference
python run_phase2_e2e_visual_check.py \
  --cbf-image ../../747/lys_nitr_10_6_0491.cbf \
  --output-base-dir ./e2e_outputs \
  --pdb-path ../../6o2h.pdb

# With custom configurations
python run_phase2_e2e_visual_check.py \
  --cbf-image image.cbf \
  --output-base-dir ./outputs \
  --dials-phil-path custom_dials.phil \
  --static-mask-config '{"beamstop": {"type": "circle", "center_x": 1250, "center_y": 1250, "radius": 50}}' \
  --bragg-mask-config '{"border": 3}' \
  --extraction-config-json '{"pixel_step": 2}' \
  --verbose

# Using shoebox-based Bragg masking
python run_phase2_e2e_visual_check.py \
  --cbf-image image.cbf \
  --output-base-dir ./outputs \
  --use-bragg-mask-option-b \
  --verbose
```

**Command-Line Arguments:**

**Required:**
- `--cbf-image`: Path to input CBF image file
- `--output-base-dir`: Base output directory (unique subdirectory will be created)

**Optional Processing:**
- `--dials-phil-path`: Path to custom DIALS PHIL configuration file
- `--pdb-path`: Path to external PDB file for validation

**Masking Configuration:**
- `--static-mask-config`: JSON string for static mask configuration (beamstop, untrusted regions)
- `--bragg-mask-config`: JSON string for Bragg mask configuration (border size, algorithm)
- `--use-bragg-mask-option-b`: Use shoebox-based Bragg masking instead of spot-based

**Extraction Configuration:**
- `--extraction-config-json`: JSON string for extraction configuration overrides
- `--pixel-step`: Pixel sampling step size for extraction (default from config)
- `--save-pixel-coords`: Save original pixel coordinates in NPZ output (default: True)

**General:**
- `--verbose`: Enable verbose logging

**Output Structure:**

For a CBF file named `lys_nitr_10_6_0491.cbf`, the script creates:

```
output-base-dir/
└── lys_nitr_10_6_0491/
    ├── e2e_visual_check.log                    # Complete pipeline log
    ├── imported.expt                           # DIALS import output
    ├── indexed_initial.expt                    # Initial indexing
    ├── indexed_initial.refl                    # Initial reflections
    ├── indexed_refined_detector.expt           # Final refined experiment
    ├── indexed_refined_detector.refl           # Final refined reflections
    ├── global_pixel_mask.pickle                # Combined pixel mask
    ├── bragg_mask.pickle                       # Bragg peak mask
    ├── total_diffuse_mask.pickle               # Final diffuse analysis mask
    ├── diffuse_data.npz                        # Extracted diffuse data
    └── extraction_diagnostics/                 # Visual diagnostic plots
        ├── diffuse_pixel_overlay.png           # Raw image with diffuse pixels
        ├── q_projection_qx_qy.png              # Q-space projections
        ├── q_projection_qx_qz.png
        ├── q_projection_qy_qz.png
        ├── radial_q_distribution.png           # Intensity vs |Q|
        ├── intensity_histogram.png             # Intensity distributions
        ├── intensity_heatmap_panel_0.png       # Detector heatmap
        ├── sigma_vs_intensity.png              # Error analysis
        ├── isigi_histogram.png                 # I/σ distribution
        ├── intensity_correction_summary.txt    # Correction effects
        └── extraction_diagnostics_summary.txt  # Overall summary
```

**Configuration Examples:**

Static mask configuration (JSON):
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

Bragg mask configuration (JSON):
```json
{
  "border": 3,
  "algorithm": "simple"
}
```

Extraction configuration (JSON):
```json
{
  "pixel_step": 2,
  "min_intensity": 10.0,
  "max_intensity": 100000.0,
  "lp_correction_enabled": true,
  "plot_diagnostics": false
}
```

**Error Handling:**

The script includes comprehensive error handling for each phase:
- **Phase 1 Failures**: DIALS processing errors, mask generation failures
- **Phase 2 Failures**: Data extraction errors, file I/O issues
- **Phase 3 Failures**: Visual diagnostic script errors (non-fatal)

If any phase fails, detailed error information is logged and the script exits gracefully.

**Integration with Visual Diagnostics:**

The script automatically invokes `check_diffuse_extraction.py` with the correct file paths and arguments, providing:
- Raw image overlay with extracted diffuse pixels
- Q-space coverage analysis
- Intensity distribution validation
- Error propagation verification
- Pixel correction effect visualization

**Development Use Cases:**

1. **Phase 2 Implementation Validation**: Verify that DataExtractor correctly processes real crystallographic data
2. **Parameter Optimization**: Test different extraction and masking parameters
3. **Regression Testing**: Generate reference outputs for automated testing
4. **Debugging**: Identify issues in the complete processing pipeline
5. **Documentation**: Generate example outputs for documentation

**Dependencies:**

- DIALS must be properly installed and importable
- All DiffusePipe components must be available
- Sufficient disk space for intermediate files and plots
- CBF input files with proper crystallographic headers

**Performance Notes:**

Processing time depends on:
- CBF image size and detector geometry
- Pixel step size (lower = more pixels = longer processing)
- DIALS processing complexity (number of spots, indexing difficulty)
- Visual diagnostic generation

Typical processing times:
- Small detector (1M pixels): 2-5 minutes
- Large detector (6M pixels): 5-15 minutes
- With pixel_step=1: Longer processing but full resolution
- With pixel_step=2: Faster processing, reduced resolution

## Future Enhancements

Potential improvements for this workflow:
- Multi-panel detector support
- Batch processing of multiple CBF files
- Automated parameter optimization
- Integration with automated testing frameworks
- Performance profiling and optimization
- Support for different detector formats beyond CBF