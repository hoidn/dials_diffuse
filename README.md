# Crystal Diffuse Scattering Pipeline

This project wraps DIALS into a scripted workflow capable of handling both true still images and sequence data.

The output will be a merged 3D diffuse scattering map.


## Features

*   **Dual Processing Modes**: Automatically detects whether input CBF files contain true stills (`Angle_increment = 0.0°`) or oscillation data (`Angle_increment > 0.0°`) and routes them to the appropriate DIALS processing workflow.
*   **DIALS Integration**: Uses DIALS for spot finding, auto-indexing, geometric refinement, and integration.
*   **Masking**: Generates static (beamstop, detector gaps), dynamic (hot/bad pixels), and per-still Bragg masks to isolate the diffuse signal.
*   **corrections**: Applies a series of physical corrections to each diffuse pixel:
    *   Lorentz-Polarization (LP) and Quantum Efficiency (QE) via the robust DIALS API.
    *   Custom Solid Angle ($\Omega$) and Air Attenuation corrections.
*   **Relative Scaling and Merging**: A custom scaling model, built on the DIALS framework, places all datasets on a common relative scale before merging them into a final 3D map via inverse-variance weighting.
*   **Visual Diagnostics**: Includes scripts for visual verification of each processing step.

## Project Structure

```
├── src/diffusepipe/           # Main Python package
│   ├── adapters/              # Wrappers for DIALS/DXTBX APIs
│   ├── config/                # Default DIALS PHIL configuration files
│   ├── crystallography/       # Crystal model processing and validation
│   ├── diagnostics/           # Q-vector calculation and consistency checking
│   ├── extraction/            # Diffuse data extraction and correction
│   ├── masking/               # Pixel and Bragg mask generation
│   ├── merging/               # Voxel data merging
│   ├── orchestration/         # Pipeline coordination
│   ├── scaling/               # Relative scaling model and components
│   └── types/                 # Pydantic data models for configuration and outcomes
├── docs/                      # Comprehensive project documentation
├── scripts/                   # Development and visual diagnostic scripts
├── tests/                     # Test suite (integration-focused)
└── libdocs/                   # Internal documentation for key libraries (e.g., DIALS)
```

## Getting Started

### Prerequisites

*   Python 3.10+
*   DIALS, cctbx and dxtbx
*   Required Python packages (see `pyproject.toml` or `requirements.txt`)

### End-to-End Visual Check

The quickest way to see the full pipeline in action is to use the end-to-end visual check script. This script processes a single CBF image from raw data through all masking and extraction steps, and generates a comprehensive set of diagnostic plots. Phase 3 (voxelization) is complete but not tested. You can run phase 2 end to end:

```bash
python scripts/dev_workflows/run_phase2_e2e_visual_check.py \
  --cbf-image /path/to/your/image.cbf \
  --output-base-dir ./e2e_outputs \
  --pdb-path /path/to/reference.pdb
```

See the **[Visual Diagnostics Guide](docs/VISUAL_DIAGNOSTICS_GUIDE.md)**.

## Pipeline Workflow

The processing is divided into three main phases:

1.  **Phase 1: Per-Still Processing & Masking**
    *   **Data Type Detection:** Analyzes CBF headers to select the correct DIALS workflow (stills vs. sequence).
    *   **DIALS Processing:** Performs spot-finding, indexing, and refinement to obtain a crystal model for each image.
    *   **Geometric Validation:** Verifies the quality of the crystal model using Q-vector consistency checks.
    *   **Mask Generation:** Creates static, dynamic, and Bragg masks.

2.  **Phase 2: Diffuse Intensity Extraction & Correction**
    *   **Data Extraction:** Iterates through unmasked pixels of each image.
    *   **Q-Vector Calculation:** Computes the scattering vector $\mathbf{q}$ for each pixel.
    *   **Correction Application:** Applies LP, QE, Solid Angle, and Air Attenuation corrections to pixel intensities.

3.  **Phase 3: Voxelization, Scaling & Merging**
    *   **Grid Definition:** Creates a common 3D reciprocal space grid based on an average crystal model.
    *   **Voxel Accumulation:** Bins all corrected diffuse pixels from all stills into the 3D grid using an HDF5-backed accumulator for memory efficiency.
    *   **Relative Scaling:** Refines a multiplicative scale factor for each still to place all datasets on a consistent relative scale.
    *   **Merging:** Performs an inverse-variance weighted merge of all observations within each voxel to produce the final 3D diffuse map.

