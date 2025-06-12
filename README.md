# Crystal Diffuse Scattering Pipeline

This project wraps DIALS into a scripted workflow capable of handling both true still images and sequence data.

The output will be a merged 3D diffuse scattering map.

## Methodology 

The approach follows Meisburger's published work. We use standard libraries wherever possible. Certain things -- especially background estimation -- need to be adapted to work with stills data. 


## Features

*   **Dual Processing Modes**: Automatically detects whether input CBF files contain true stills (`Angle_increment = 0.0°`) or oscillation data (`Angle_increment > 0.0°`) and routes them to the appropriate DIALS processing workflow.
*   **DIALS Integration**: Uses DIALS for spot finding, auto-indexing, geometric refinement, and integration.
*   **Masking**: Generates static (beamstop, detector gaps), dynamic (hot/bad pixels), and per-still Bragg masks to isolate the diffuse signal.
*   **corrections**: Applies a series of physical corrections to each diffuse pixel:
    *   Lorentz-Polarization (LP) and Quantum Efficiency (QE) via DIALS.
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

### Phase 1: Per-Still Processing & Masking
The goal of this phase is to process each raw detector image to obtain a validated crystallographic model and a comprehensive mask that isolates the diffuse scattering signal.

*   **Data Type Detection:** The pipeline first inspects the CBF header to determine if the image is a true still or part of an oscillation sequence. It then automatically routes the data to the appropriate DIALS processing adapter.
*   **Crystallographic Processing:** DIALS is used to perform spot-finding, auto-indexing, and geometric refinement. This step yields a precise crystal orientation matrix ($\mathbf{U}_i$) and unit cell for each image.
*   **Geometric Validation:** The quality of the crystal model is verified using a Q-vector consistency check, which ensures that the model accurately maps between reciprocal space and detector space.
*   **Mask Generation:** A total mask is created for each still by combining a global bad-pixel mask (for detector gaps, shadows, etc.) with a per-still Bragg mask that covers the regions of intense Bragg diffraction for that specific orientation.

### Phase 2: Diffuse Intensity Extraction & Correction
This phase iterates through the valid pixels identified in Phase 1 and applies a series of physical corrections to obtain accurate intensity measurements.

*   **Data Extraction:** For every unmasked pixel in an image, its raw intensity is read, and its position is used to calculate the corresponding scattering vector ($\mathbf{q}$).
*   **Physical Corrections:** The intensity of each pixel is corrected for a series of experimental and geometric effects. The full correction is applied multiplicatively, combining four key factors:
    1.  **Lorentz-Polarization (LP):** Accounts for polarization and the geometry of diffraction.
    2.  **Quantum Efficiency (QE):** Account for detector sensitivity.
    3.  **Solid Angle ($\Omega$):** Normalizes intensity by the solid angle subtended by the pixel.
    4.  **Air Attenuation:** Corrects for X-ray absorption by air between the sample and detector.
*   **Output:** The result of this phase is a list of corrected `{q-vector, intensity, sigma}` data points for each individual still image.

### Phase 3: Voxelization, Scaling & Merging
This phase combines the processed data from all individual stills into a single, self-consistent 3D diffuse scattering map.

*   **Global Grid Definition:** A common 3D grid in reciprocal space is defined by first calculating an average crystal model from all successfully processed stills. This ensures all data can be mapped to a consistent reference frame.
*   **Voxel Accumulation:** The corrected diffuse data points from every still are binned into this global 3D grid. For memory efficiency with large datasets, this process is handled by a `VoxelAccumulator` that can use an HDF5 file as a backend.
*   **Relative Scaling:** A custom scaling model is refined to correct for inter-image variations ( changes in beam intensity or illuminated crystal volume). This model determines a multiplicative scale factor for each still, placing all datasets on a common relative scale.
*   **Merging:** The final scaled observations are merged. Within each voxel of the 3D grid, all contributing intensity measurements are combined using an inverse-variance weighted average to produce the final merged intensity and its associated error.

## Next Steps: Phase 4 (Pending)

The final phase of the pipeline will focus on placing the merged data onto an absolute scale and preparing it for scientific interpretation.

*   **Absolute Scaling:** The relatively-scaled diffuse map will be converted to absolute units ( electron units per unit cell). This will be achieved by matching the total experimental scattering (diffuse + Bragg) to the total theoretical scattering from a known unit cell composition, using the Krogh-Moe/Norman summation method.
*   **Incoherent Subtraction:** The theoretical incoherent (Compton) scattering background will be calculated from the sample composition and subtracted from the absolute-scaled map.
*   **Final Output:** The result will be the final, absolutely-scaled 3D diffuse scattering map, ready for analysis.
