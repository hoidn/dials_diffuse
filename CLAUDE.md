# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Structure
- DiffusePipe is a Python project for processing crystallography data
- Core package is `src/diffusepipe/`
- Configuration files (PHIL files) are stored in `src/diffusepipe/config/`
- Uses DIALS crystallography software through CLI tools

## Important File Locations
- Main processing script: `/src/scripts/process_pipeline.sh`
- PHIL configuration files:
  - `/src/diffusepipe/config/find_spots.phil`
  - `/src/diffusepipe/config/refine_detector.phil`
- IDL specifications are in files named `*_IDL.md` next to the implementation files

## External Dependencies
- DIALS crystallography software (primarily accessed via its Python API, e.g., `dials.stills_process`)
- Python 3.10+
- PDB file for consistency checks (6o2h.pdb in the example)

## Documentation
- Main DIALS/CCTBX/DXTBX documentation: `libdocs/dials/DIALS_Python_API_Reference.md`
- For features not covered in documentation, explore source code under `./bio/` as last resort

## Testing Approach
- Use pytest for testing
- Strongly emphasize integration tests over unit tests
- Avoid mocks whenever possible - use real components instead
- Only mock external APIs with usage limits or services requiring complex infrastructure

## Code Style and Documentation
- Format using Black
- Lint with Ruff
- Google Style docstrings
- Follow PEP 8 naming conventions
- IDL files serve as the specification/contract for each component

# Input data Format 
The diffraction images are stored in .cbf format under ./747/. Example header:
<cbf header>
###CBF: VERSION 1.5, CBFlib v0.7.8 - PILATUS detectors

data_lys_nitr_10_1_0048

_array_data.header_convention "PILATUS_1.2"
_array_data.header_contents
;
# Detector: PILATUS3 6M, S/N 60-0127
# 2017-06-26T02:54:23.347
# Pixel_size 172e-6 m x 172e-6 m
# Silicon sensor, thickness 0.001000 m
# Exposure_time 0.0990000 s
# Exposure_period 0.1000000 s
# Tau = 0 s
# Count_cutoff 1009797 counts
# Threshold_setting: 6344 eV
# Gain_setting: autog (vrf = 1.000)
# N_excluded_pixels = 852
# Excluded_pixels: badpixel_mask.tif
# Flat_field: FF_p60-0127_E12688_T6344_vrf_m0p100.tif
# Trim_file: p60-0127_E12688_T6344.bin
# Image_path: /F1a/ando/20170624/lysozyme/nitrate/ubatch4/
# Ratecorr_lut_directory: ContinuousStandard_v1.1
# Retrigger_mode: 1
# Wavelength 0.97680 A
# Detector_distance 0.22982 m
# Beam_xy (1264.48, 1242.52) pixels
# Start_angle 254.7000 deg.
# Angle_increment 0.1000 deg.
# Phi 250.0000 deg.
# Oscillation_axis PHI
# N_oscillations 1
# Shutter_time 0.1000000 s
;

_array_data.data
;
--CIF-BINARY-FORMAT-SECTION--
Content-Type: application/octet-stream;
     conversions="x-CBF_BYTE_OFFSET"
Content-Transfer-Encoding: BINARY
X-Binary-Size: 6224641
X-Binary-ID: 1
X-Binary-Element-Type: "signed 32-bit integer"
X-Binary-Element-Byte-Order: LITTLE_ENDIAN
Content-MD5: v+hI0TIRqPDQ7ABGUxprUA==
X-Binary-Number-of-Elements: 6224001
X-Binary-Size-Fastest-Dimension: 2463
X-Binary-Size-Second-Dimension: 2527
X-Binary-Size-Padding: 4095
</cbf header>

## Common Tasks
- When running the process_pipeline.sh script, use absolute paths to PHIL files
- For any changes to the pipeline, consult the relevant IDL files first
- Always check the log files in the processing directories when debugging pipeline issues
- The script creates a summary log at the root directory (`dials_processing_summary.log`)

## Running the Pipeline
```bash
# Basic usage with a single CBF file
bash src/scripts/process_pipeline.sh path/to/image.cbf --external_pdb path/to/reference.pdb

# Processing multiple CBF files
bash src/scripts/process_pipeline.sh path/to/image1.cbf path/to/image2.cbf --external_pdb path/to/reference.pdb

# With verbose Python script output
bash src/scripts/process_pipeline.sh path/to/image.cbf --external_pdb path/to/reference.pdb --verbose

# Disabling diagnostic scripts
bash src/scripts/process_pipeline.sh path/to/image.cbf --external_pdb path/to/reference.pdb --run_diagnostics false
```

## Other bash commands:
# generating a task-aware context file:
repomix --include "**/*.md,**/*.py,**/*.phil" --ignore .aider.chat.history.md
bash adaptive_substitute.sh context_package.json 10000 expanded_context.txt

## Development Process
1. Review IDL specification for the component to be modified
2. Implement changes following the IDL contract
3. Add integration tests with real components (minimize mocks)
4. Format code (Black) and run linter (Ruff)
5. Verify changes work with the pipeline script

## Debugging the Pipeline

For comprehensive debugging guidance, see `docs/06_DIALS_DEBUGGING_GUIDE.md`.

### Key Debugging Principles

**Test Suite Remediation Patterns:**
- Use `MagicMock` instead of `Mock` for DIALS objects that need magic method support (`__getitem__`, `__and__`, `__or__`)
- Replace complex mocking with real DIALS `flex` arrays for authentic integration testing
- Only mock external APIs with usage limits or complex infrastructure requirements

**Quick DIALS Troubleshooting:**
1. **Check CBF headers** for oscillation data vs stills (`Angle_increment` value)
2. **Compare with working DIALS logs** in existing processing directories
3. **Verify PHIL parameters** match working approach (see critical parameters in debugging guide)
4. **Test DIALS API imports** independently
5. **Use CLI-based adapters** as fallback for API issues

**Primary Validation Method:** Q-vector consistency checking (`|Δq|` typically < 0.01 Å⁻¹)

## Pipeline Steps and Expected Outputs
1. **DIALS Stills Processing (via `dials.stills_process` Python API Adapter)**
   - Input: CBF file, base geometry, PHIL configuration for `dials.stills_process`.
   - Process: The adapter uses `dials.stills_process.Processor` to internally perform import, spot finding, indexing, refinement, and integration (including partiality and optional shoebox generation).
   - Output: 
     - `Experiment_dials_i` (DIALS Python Experiment object containing refined crystal model).
     - `Reflections_dials_i` (DIALS Python reflection_table object with indexed spots, partialities, and optionally shoeboxes).
     - Corresponding DIALS output files (e.g., `integrated.expt`, `integrated.refl`) are saved to the working directory.
   - Log: The `dials.stills_process` tool typically produces its own comprehensive log. Individual step logs (like `dials.find_spots.log`) might also be generated by it if configured.

2. **Bragg Mask Generation (using `dials.generate_mask` or shoebox data)**
   - Input: `Experiment_dials_i` and `Reflections_dials_i` (and shoeboxes if that option is chosen).
   - Output: `BraggMask_2D_raw_i` (Python `flex.bool` mask object), and `bragg_mask.pickle` file.
   - Log: `dials.generate_mask.log` (if `dials.generate_mask` is used).

3. **Python Extraction (DataExtractor component)**
   - Input: Raw image data, `Experiment_dials_i`, `BraggMask_2D_raw_i`, static pixel mask, PDB file (optional).
   - Output: NPZ file with diffuse data (`q_vectors`, `intensities`, `variances`).
   - Log: `extract_diffuse_data.log` (or similar, managed by the Python component).
