**`plan2.md`**

**Supporting Components and Advanced Utilities Plan**

**Nomenclature:**
*This document uses nomenclature consistent with `plan.md`. Specific new terms related to Phase 3 outputs will be defined within their respective module descriptions.*

*   `VoxelData_relative`: The primary output of Phase 3, a data structure (e.g., NumPy structured array or `flex.reflection_table`) where each row represents a voxel. Contains `(voxel_idx, H_center, K_center, L_center, q_center_x, q_center_y, q_center_z, |q|_center, I_merged_relative, Sigma_merged_relative, num_observations_in_voxel)`.
*   `GlobalVoxelGrid_obj`: The Python object instance defining the 3D reciprocal space grid.
*   `ScalingModel_refined_params`: A data structure (e.g., dictionary or list of Pydantic models) holding the refined parameters of the `DiffuseScalingModel` (e.g., per-still `b_i` scales, resolution smoother `a(|q|)` parameters).

---

**0. Introduction and Purpose**

This document, `plan2.md`, outlines the design and implementation requirements for supporting components and advanced utilities that complement the core processing pipeline defined in `plan.md`.

**Precedence:** `plan2.md` has **lower precedence** than `plan.md`. In case of conflicts regarding core pipeline logic, `plan.md` is the authoritative source. `plan2.md` focuses on auxiliary systems like advanced diagnostics, specialized data analysis tools, or alternative workflow orchestrators that build upon the core pipeline outputs.

This initial version of `plan2.md` will focus on creating a **Phase 3 Visual Check System**, analogous to the visual diagnostics already implemented for earlier phases.

---

**Section P3.VC: Phase 3 Visual Check System**

**Overview:**
The Phase 3 Visual Check System aims to provide developers and users with tools to visually inspect and validate the outputs of the voxelization, relative scaling, and merging steps (Modules 3.S.1 - 3.S.4). It will consist of an orchestration script to run the full pipeline up to Phase 3 and generate inputs for a dedicated diagnostic script, which will then produce plots and summary reports.

This system will be modeled on the existing `run_phase2_e2e_visual_check.py` and `scripts/visual_diagnostics/check_diffuse_extraction.py` framework.

---

**Module P3.VC.O: Orchestration Script for Phase 3 Visual Checks**

*   **Script Name:** `scripts/dev_workflows/run_phase3_e2e_visual_check.py`
*   **Action:** Orchestrate the complete DiffusePipe pipeline from raw CBF image(s) through all of Phase 1, Phase 2, and Phase 3. It will then invoke the Phase 3 diagnostic script (`check_phase3_outputs.py`) with the generated outputs.
*   **Input (Command Line Arguments):**
    *   `--cbf-image-paths`: List of paths to input CBF image files (allowing multiple stills for realistic Phase 3 testing).
    *   `--output-base-dir`: Base output directory for all intermediate files and final diagnostics.
    *   `--pdb-path` (optional): Path to external PDB file for Phase 1 validation.
    *   `--dials-phil-path` (optional): Custom DIALS PHIL for Phase 1 processing.
    *   `--static-mask-config` (optional): JSON string/file for static mask parameters (Phase 1).
    *   `--bragg-mask-config` (optional): JSON string/file for Bragg mask parameters (Phase 1).
    *   `--use-bragg-mask-option-b` (optional): Flag to use shoebox-based Bragg masking.
    *   `--extraction-config-json` (optional): JSON string/file for `ExtractionConfig` overrides (Phase 2).
    *   `--relative-scaling-config-json` (optional): JSON string/file for `RelativeScalingConfig` overrides (Phase 3).
    *   `--grid-config-json` (optional): JSON string/file for `GlobalVoxelGrid` parameters (e.g., `d_min_target`, `ndiv_hkl`) (Phase 3).
    *   `--save-intermediate-phase-outputs`: Flag to explicitly save key outputs from Phase 1 and 2 (e.g., `Experiment_dials_i` list, `CorrectedDiffusePixelData_i` list as NPZ files) that might be needed by `check_phase3_outputs.py`.
    *   `--verbose`: Enable verbose logging.
*   **Process:**
    1.  **Setup:** Create a unique output subdirectory within `--output-base-dir` (e.g., based on the first CBF filename or a timestamp). Setup logging.
    2.  **Run Phase 1 (Modules 1.S.0 - 1.S.3):** For each CBF file:
        *   Perform DIALS processing (still/sequence as detected).
        *   Perform geometric validation.
        *   Generate pixel masks and Bragg masks.
        *   Generate total diffuse mask.
        *   Store/collect `Experiment_dials_i` objects and paths to `Mask_total_2D_i` (or the mask objects themselves).
    3.  **Run Phase 2 (Modules 2.S.1 - 2.S.2):** For each successfully processed still from Phase 1:
        *   Instantiate `DataExtractor`.
        *   Extract diffuse data (`q_vector, I_corrected, Sigma_corrected`, etc.).
        *   Save the output per-still NPZ file (`CorrectedDiffusePixelData_i.npz`) containing these arrays into the working directory. Collect paths to these files.
    4.  **Run Phase 3 (Modules 3.S.1 - 3.S.4):**
        *   **Module 3.S.1 (Grid Definition):**
            *   Collect all `Experiment_dials_i` objects.
            *   Load all `CorrectedDiffusePixelData_i.npz` files to get q-vector ranges.
            *   Instantiate and build `GlobalVoxelGrid_obj` using configurations.
            *   (Optional) Serialize `GlobalVoxelGrid_obj` definition (e.g., to JSON/pickle) for the diagnostic script. Path: `global_voxel_grid_definition.json`.
        *   **Module 3.S.2 (Binning):**
            *   Instantiate `VoxelAccumulator` (with HDF5 backend in the working directory).
            *   Iterate through `CorrectedDiffusePixelData_i.npz` files, binning observations into `VoxelAccumulator`.
            *   Generate `ScalingModel_initial_list`.
        *   **Module 3.S.3 (Relative Scaling):**
            *   Instantiate `DiffuseScalingModel`.
            *   Perform relative scaling using data from `VoxelAccumulator` and `Reflections_dials_i` (for Bragg ref).
            *   Store `ScalingModel_refined_params`. Serialize parameters (e.g., to JSON). Path: `refined_scaling_model_params.json`.
        *   **Module 3.S.4 (Merging):**
            *   Apply refined scales and merge data.
            *   Generate and save `VoxelData_relative` (e.g., as `voxel_data_relative.npz` or `voxel_data_relative.hdf5`).
    5.  **Invoke Phase 3 Diagnostic Script:**
        *   Construct command for `scripts/visual_diagnostics/check_phase3_outputs.py`.
        *   Pass paths to:
            *   `global_voxel_grid_definition.json` (or equivalent)
            *   `refined_scaling_model_params.json`
            *   `voxel_data_relative.npz` (or `.hdf5`)
            *   Optionally, paths to the collection of `Experiment_dials_i` (if needed for context) and `CorrectedDiffusePixelData_i.npz` files.
            *   Output directory for diagnostic plots (e.g., `phase3_diagnostics/` within the main output dir).
        *   Execute the script.
*   **Output:**
    *   All intermediate files from Phases 1, 2, and 3 saved in the unique output subdirectory.
    *   Log file for the entire orchestration.
    *   A dedicated subdirectory (e.g., `phase3_diagnostics/`) containing plots and reports from `check_phase3_outputs.py`.
*   **Relevant `libdocs/dials/` API:** This script will use high-level project components which internally use DIALS APIs. No direct DIALS API calls from this script.

---

**Module P3.VC.D: Phase 3 Diagnostic Script**

*   **Script Name:** `scripts/visual_diagnostics/check_phase3_outputs.py`
*   **Action:** Load outputs from Phase 3 (grid definition, scaling model, merged voxel data) and generate visualizations and summary statistics to verify correctness.
*   **Input (Command Line Arguments):**
    *   `--grid-definition-file`: Path to serialized `GlobalVoxelGrid_obj` definition.
    *   `--scaling-model-params-file`: Path to serialized `ScalingModel_refined_params`.
    *   `--voxel-data-file`: Path to the `VoxelData_relative` file (NPZ or HDF5).
    *   `--output-dir`: Directory to save diagnostic plots and reports.
    *   `--experiments-list-file` (optional): Path to a file listing paths to individual `Experiment_dials_i.expt` files (if needed for still-specific context).
    *   `--corrected-pixel-data-dir` (optional): Path to directory containing per-still `CorrectedDiffusePixelData_i.npz` files (if comparison to pre-binned data is desired).
    *   `--max-plot-points` (optional): Max points for scatter plots.
    *   `--verbose`: Enable verbose logging.
*   **Process & Generated Diagnostics (Plots & Text Summaries):**
    1.  **Load Inputs:** Deserialize/load all required input files.
    2.  **Global Voxel Grid Summary (`grid_summary.txt`, `grid_visualization_conceptual.png`):**
        *   Text: Report `Crystal_avg_ref` parameters (unit cell, space group), calculated HKL bounds of the grid, voxel dimensions (`1/ndiv_h`, etc.), total number of voxels.
        *   Plot (Conceptual): A 3D scatter plot showing a subset of `q_vector` points (from optional `--corrected-pixel-data-dir`) overlaid with the bounding box of the `GlobalVoxelGrid` in q-space to visualize coverage.
    3.  **Voxel Occupancy/Redundancy Plots (`voxel_occupancy_slice_L0.png`, `voxel_occupancy_histogram.png`):**
        *   Requires `num_observations_in_voxel` from `VoxelData_relative`.
        *   Plot: 2D heatmap slices (e.g., H-K plane at L=0, H-L at K=0, K-L at H=0) of `num_observations_in_voxel`. Colormap indicating redundancy.
        *   Plot: Histogram of `num_observations_in_voxel` values.
        *   Text: Min, max, mean, median redundancy. Percentage of voxels with redundancy < N.
    4.  **Relative Scaling Model Parameter Plots (`scaling_model_params.png`, `scaling_residuals.png`):**
        *   Requires `ScalingModel_refined_params`.
        *   Plot (if applicable): `b_i` (per-still scales) vs. still index or `p_order(i)`.
        *   Plot (if applicable): `a(|q|)` (resolution smoother parameters/curve).
        *   Text: Summary of refined parameters.
        *   (Advanced, if residuals saved by main pipeline): Plot histogram of residuals from scaling refinement.
    5.  **Merged Voxel Data Visualization (`merged_intensity_slice_L0.png`, `merged_isigi_slice_L0.png`, `merged_radial_avg.png`, `merged_intensity_histogram.png`):**
        *   Requires `VoxelData_relative`.
        *   **Reciprocal Space Slices:**
            *   Plot: 2D heatmap slices (e.g., H-K at L=0) of `I_merged_relative` (log scale).
            *   Plot: 2D heatmap slices of `Sigma_merged_relative`.
            *   Plot: 2D heatmap slices of `I_merged_relative / Sigma_merged_relative`.
        *   **Radial Average Plot:**
            *   Calculate `|q|_center` for each voxel.
            *   Plot `I_merged_relative` vs. `|q|_center` (scatter or binned average).
        *   **Intensity Histogram:**
            *   Plot histogram of `I_merged_relative` values (linear and log y-scale).
    6.  **Summary Report (`phase3_diagnostics_summary.txt`):**
        *   Text file summarizing key statistics from all plots, input file names, configurations used (if available).
*   **Output Files:** PNG plot files and TXT summary files in the specified `--output-dir`.
*   **Relevant `libdocs/dials/` API:**
    *   For loading `Experiment_dials_i` (if passed): `dials_file_io.md` (A.1).
    *   For potentially re-calculating q-vectors from HKL centers: `dxtbx_models.md` (B.3 for `crystal.get_A()`) and `crystallographic_calculations.md` (C.2 implied `q = A * hkl`).
    *   For array manipulations (if loading NPZ and working with NumPy): `flex_arrays.md` concepts might be mapped to NumPy equivalents.
    *   Primarily uses project-specific data structures and plotting libraries (matplotlib).

---

**Module P3.VC.U: Plotting Utilities (Enhancements to `scripts/visual_diagnostics/plot_utils.py`)**

*   **Action:** Extend `plot_utils.py` with functions needed for Phase 3 diagnostics.
*   **New/Enhanced Functions:**
    1.  `plot_3d_grid_slice(grid_data_3d, slice_dim, slice_idx, title, output_path, cmap, norm, xlabel, ylabel)`:
        *   Input: 3D NumPy array (`grid_data_3d`), dimension to slice along (`'H'`, `'K'`, or `'L'`), index for the slice.
        *   Behavior: Extracts a 2D slice and plots it as a heatmap using `imshow`. Handles axis labeling based on slice.
    2.  `plot_radial_average(q_magnitudes, intensities, num_bins, title, output_path)`:
        *   Input: Arrays of q-magnitudes and corresponding intensities.
        *   Behavior: Bins intensities by q-magnitude, calculates mean intensity per bin, plots mean intensity vs. q-bin center. Includes error bars if sigmas are provided.
    3.  `plot_parameter_vs_index(param_values, index_values, title, param_label, index_label, output_path)`:
        *   For plotting scaling parameters like `b_i` vs. still index.
    4.  `plot_smoother_curve(smoother_params_or_evaluator, q_range, title, output_path)`:
        *   For plotting the resolution smoother `a(|q|)`.
*   **Relevant `libdocs/dials/` API:** Not directly calling DIALS APIs, but uses matplotlib for plotting data derived from DIALS/project outputs.

---

**Module P3.VC.T: Testing the Phase 3 Visual Check System**

*   **Action:** Create tests for `run_phase3_e2e_visual_check.py` and `check_phase3_outputs.py`.
*   **Process:**
    1.  **Test `run_phase3_e2e_visual_check.py`:**
        *   Requires mock components for Phase 1, 2, and 3 that produce dummy output files in the expected formats.
        *   Verify that the orchestrator calls each phase correctly.
        *   Verify that it correctly invokes `check_phase3_outputs.py` with the paths to generated files.
    2.  **Test `check_phase3_outputs.py`:**
        *   Create synthetic input files (`global_voxel_grid_definition.json`, `refined_scaling_model_params.json`, `voxel_data_relative.npz`) with known characteristics.
        *   Run the script with these inputs.
        *   Assert that all expected plot files and summary files are generated.
        *   (Optional, advanced) For simple synthetic data, potentially load generated plots and check for key features or summary statistics.
*   **Relevant `libdocs/dials/` API:** None directly, focuses on testing script logic and file I/O.

---
