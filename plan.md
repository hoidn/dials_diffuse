# Diffuse Scattering Processing Plan (DIALS-Integrated with Integrated Testing)

**Nomenclature:**

*   `I_raw(px, py, f)`: Raw intensity at detector pixel `(px, py)` for frame `f`. Accessed via `imageset.get_raw_data(f)` (Ref: API A.3).
*   `t_exp(f)`: Exposure time for frame `f`. Accessed via `scan.get_exposure_times()[f]` (Ref: API B.5, A.3).
*   `Experiment_dials_w`: DIALS `ExperimentList` object for wedge `w` (from `.expt` file). Type: `dxtbx.model.experiment_list.ExperimentList` (Ref: API A.1).
*   `Reflections_dials_w`: DIALS `reflection_table` for wedge `w` (from `.refl` file). Type: `dials.array_family.flex.reflection_table` (Ref: API A.2).
*   `BraggMask_dials_w(px, py)`: Boolean mask from DIALS (`.pickle` file) for wedge `w`, indexed by panel and pixel. Type: Tuple of `dials.array_family.flex.bool` arrays (Ref: API A.4).
*   `IS_w`: `io.ImageSeries` object for wedge `w` (for accessing raw image data). Type: `dxtbx.imageset.ImageSet` (Ref: API A.3).
*   `Mask_pixel(px, py)`: Static bad pixel mask (e.g., beamstop, bad pixels), indexed by panel and pixel. Type: Tuple/list of `dials.array_family.flex.bool` arrays (Ref: API D.4 for creation/manipulation).
*   `v`: Voxel index in a 3D reciprocal space grid (defined in Phase 3).
*   `p`: Detector pixel index `(px,py)`.
*   `|q|`: Scattering vector magnitude (magnitude of `q_vector`). Calculated from `q_vector.length()` (Ref: API C.1). (Original `s` renamed to `|q|` for clarity).
*   `H,K,L`: Integer Miller indices. Often stored as `flex.miller_index` (Ref: API D.2, D.3).
*   `h,k,l`: Continuous (fractional) Miller indices. Result of transforming `q_vector` using `crystal.get_A().inverse()` (Ref: API C.2).
*   `w`: Wedge index.

---
**0. Testing Principles and Conventions**

*   **0.1 Granular Testing:** Each significant computational module or processing step defined in this plan will have corresponding tests to verify its correctness and expected outputs based on controlled inputs.
*   **0.2 Input Data Strategy:**
    *   Initial pipeline steps consuming raw data (e.g., CBF files) will be tested using a small, curated set of real representative files.
    *   Subsequent modules will primarily be tested using the **serialized output files** (e.g., `.expt`, `.refl`, `.pickle` from DIALS; NPZ files from Python components) generated from successfully running tests of preceding modules. This promotes end-to-end validation of data flow and component integration.
    *   For highly focused unit tests of specific algorithms or utility functions *within* a module, inputs may be programmatically constructed in test code if it enhances clarity or allows for more precise control over edge cases.
*   **0.3 Test Data Management:** All test input files (curated CBFs, serialized DIALS outputs, reference outputs) will be stored in a dedicated directory within the test suite (e.g., `tests/data/`).
*   **0.4 Verification:** Assertions will check the correctness of calculations, data transformations, and the structure/content of output data structures against pre-calculated expected values or known properties. For modules producing file outputs, tests may compare against reference output files or check key statistics.
*   **0.5 Scope:** Testing will focus on the logic implemented within this project. The internal correctness of external tools like DIALS is assumed, but their integration and invocation by this project will be tested.

---

**Phase 1: DIALS-Based Initial Processing & Masking**
*(This phase replaces original `plan.md` Modules 1.0, 1.1, 1.2)*

**Module 1.D: DIALS Processing Orchestration**
*   **Action:** Execute DIALS command-line tools for each data wedge `w` to determine geometry, index reflections, and generate Bragg masks.
*   **Input:**
    *   Raw image files for wedge `w`.
    *   DIALS PHIL files (for spot finding, refinement).
    *   Unit cell parameters, space group.
*   **Process (Typically orchestrated by a shell script like `src/scripts/process_pipeline.sh` or a Python orchestrator):**
    1.  `dials.import <image_files_w>`: Generates initial experiment geometry (e.g., `imported.expt`).
    2.  `dials.find_spots <imported.expt> <find_spots.phil>`: Identifies strong reflections (e.g., `strong.refl`).
    3.  `dials.index <imported.expt> <strong.refl> unit_cell=... space_group=...`: Determines crystal orientation and indexes reflections (e.g., `indexed_initial.expt`, `indexed_initial.refl`).
    4.  `dials.refine <indexed_initial.expt> <indexed_initial.refl> <refine_detector.phil>`: Refines experimental geometry and crystal model (e.g., `indexed_refined_detector.expt`, `indexed_refined_detector.refl`).
    5.  `dials.generate_mask <indexed_refined_detector.expt> <indexed_refined_detector.refl>`: Creates a mask to exclude Bragg peak regions (e.g., `bragg_mask.pickle`).
*   **Output (for each wedge `w`):**
    *   `Experiment_dials_w`: Refined DIALS `ExperimentList` object (from `indexed_refined_detector.expt`). Type: `dxtbx.model.experiment_list.ExperimentList` (Ref: API A.1).
    *   `Reflections_dials_w`: Refined DIALS `reflection_table` (from `indexed_refined_detector.refl`). Type: `dials.array_family.flex.reflection_table` (Ref: API A.2).
    *   `BraggMask_dials_w`: Boolean mask for Bragg peaks (from `bragg_mask.pickle`), typically a tuple of `dials.array_family.flex.bool` arrays, one per panel (Ref: API A.4).
*   **Consistency Check:** Successful completion of all DIALS steps (check logs). Reasonableness of refined geometry and indexing statistics.
*   **Testing (Module 1.D):**
    *   **Input Data:**
        *   Curated CBF image files from `tests/data/cbf_input/`.
        *   Standard DIALS PHIL files used by the orchestrator.
    *   **Execution:** Invoke the DIALS processing pipeline orchestrator (e.g., `src/scripts/process_pipeline.sh` or its Python equivalent) on these CBF files.
    *   **Verification:**
        *   Successful execution of all DIALS commands (check exit codes, review DIALS logs).
        *   Generation of expected output files (`indexed_refined_detector.expt`, `indexed_refined_detector.refl`, `bragg_mask.pickle`) in `tests/data/dials_outputs/wedge_W/` (or a temporary processing directory whose contents are checked).
        *   Basic sanity checks on output files (e.g., loadability by `dxtbx.serialize.load` - Ref: API A.1, A.2, A.4).

**Module 1.1 (Revised): Static Pixel Masking**
*   **Action:** Create a static 2D detector mask based on detector properties and known bad regions, independent of diffraction data.
*   **Input:**
    *   `Experiment_dials_w.detector`: The `dxtbx.model.Detector` object for each wedge `w` (Ref: API B.1).
    *   Optionally, a representative raw image `I_raw(px, py, first_frame)` accessed via `IS_w.get_raw_data(first_frame)` (Ref: API A.3) for dynamic masking (e.g., negative counts).
*   **Process:** For each panel in `Experiment_dials_w.detector`:
    1.  `Mask_static_panel(px,py)`: From panel properties like `panel.get_image_size()`, `panel.get_pixel_size()`, `panel.get_trusted_range()` (Ref: API B.1). Create `flex.bool` array (Ref: API D.4).
    2.  `Mask_beamstop_panel(px,py)`: User-defined or geometrically derived beamstop mask relevant to the panel, potentially using `panel.get_pixel_lab_coord()` (Ref: API B.1). Create `flex.bool` array (Ref: API D.4).
    3.  `Mask_virtual_corrected_panel(px,py)`: Mask for virtual or corrected pixels if applicable to panel type, checking properties like `panel.is_virtual_panel()` (Ref: API B.1). Create `flex.bool` array (Ref: API D.4).
    4.  `Mask_negative_counts_panel(px,py)`: If using a raw image, `(I_raw_panel(px,py,first_frame) >= 0)`. `I_raw_panel` is a `flex.int` or `flex.double` array from `imageset.get_raw_data()` (Ref: API A.3). Create `flex.bool` array using comparison (Ref: API D.4).
    5.  `Mask_pixel_panel(px,py) = Mask_static_panel & Mask_beamstop_panel & Mask_virtual_corrected_panel & Mask_negative_counts_panel`. Combine `flex.bool` arrays using logical AND (Ref: API D.4).
*   **Output:** `Mask_pixel_w`: A tuple/list of 2D `dials.array_family.flex.bool` arrays, one per panel for detector in wedge `w` (Ref: API D.4).
*   **Consistency Check:** Visual inspection of `Mask_pixel_w`. Ensure it covers known bad regions without excessively masking good data.
*   **Testing (Module 1.1):**
    *   **Input Data:**
        *   Programmatically constructed `dxtbx.model.Detector` objects (Ref: API B.1) with known features.
        *   Optionally, small `flex.int` or `flex.double` arrays (Ref: API D.2) for dynamic mask aspects per panel.
    *   **Execution:** Call the Python function for static mask generation, passing the detector model and optional raw data.
    *   **Verification:** Assert the output (list/tuple of `flex.bool` arrays for `Mask_pixel_w`) correctly identifies known bad regions based on inputs for each panel.

---

**Phase 2: Diffuse Intensity Extraction and Pixel-Level Correction**

**Module 2.1.D: Pixel-Based Diffuse Data Extraction**
*   **Action:** For each wedge `w`, read raw images, apply combined masks (`BraggMask_dials_w` and `Mask_pixel_w`), calculate q-vector for each unmasked pixel, and extract its raw intensity.
*   **Input:**
    *   `Experiment_dials_w` (for geometry, includes `dxtbx.model.Detector` and `dxtbx.model.Beam`) (Ref: API B.1, B.2).
    *   `IS_w` (`dxtbx.imageset.ImageSet` for accessing raw image data for wedge `w`) (Ref: API A.3).
    *   `BraggMask_dials_w` (tuple of `flex.bool` from Module 1.D) (Ref: API A.4, D.4).
    *   `Mask_pixel_w` (tuple/list of `flex.bool` from Module 1.1) (Ref: API D.4).
*   **Process (for each frame `f` in `IS_w`, for each panel `panel_idx` in `Experiment_dials_w.detector`, for each pixel `p=(px,py)` on that panel):**
    1.  If `NOT BraggMask_dials_w[panel_idx](py, px)` AND `Mask_pixel_w[panel_idx](py, px)`: (Note: flex.bool is often (slow,fast) so py,px). Access elements of `flex.bool` arrays and perform logical operations (Ref: API D.4).
        a.  `I_raw_pf = IS_w.get_raw_data(f)[panel_idx][py, px]`. Access raw data (`flex.int`/`flex.double` array) from `ImageSet` and index pixel (Ref: API A.3, D.4).
        b.  Calculate `q_vector(p,f)`:
            i.  Get laboratory coordinates of pixel `p` on `Experiment_dials_w.detector[panel_idx]` using `panel.get_pixel_lab_coord(px, py)` (Ref: API B.1).
            ii. Get incident beam vector `s_incident` from `Experiment_dials_w.beam` using `beam.get_s0()` (Ref: API B.2).
            iii.Calculate scattered vector `s_scattered` from sample to pixel (normalized lab_coord scaled by `1.0 / beam.get_wavelength()`, Ref: API B.1, B.2, C.1).
            iv. `q_vector(p,f) = s_scattered - s_incident`. Perform vector subtraction using `scitbx.matrix.col` (Ref: API C.1).
            *   This entire calculation is summarized by API C.1.
        c.  Store `(q_vector(p,f), I_raw_pf, frame_index=f, panel_index=panel_idx, pixel_coords=(px,py))`. `q_vector` is a `scitbx.matrix.col`.
*   **Output (for each wedge `w`):**
    *   `DiffusePixelList_w`: A list of tuples, where each tuple is `(q_vector, raw_intensity, frame_idx, panel_idx, px, py)` for all selected diffuse pixels in the wedge. `q_vector` is a `scitbx.matrix.col` object.
*   **Consistency Check:** Sanity check on the number of extracted pixels. Visual inspection of q-vector distribution if feasible.
*   **Testing (Module 2.1.D):**
    *   **Input Data (Primary - Integration with DIALS Outputs):**
        *   Load `Experiment_dials_w` from `tests/data/dials_outputs/wedge_W/indexed_refined_detector.expt` (Ref: API A.1).
        *   Load `BraggMask_dials_w` from `tests/data/dials_outputs/wedge_W/bragg_mask.pickle` (Ref: API A.4).
        *   Load/construct `Mask_pixel_w` (tuple of `flex.bool`) (Ref: API D.4).
        *   Use an `ImageSet` pointing to `tests/data/cbf_input/wedge_W_image_F.cbf` (Ref: API A.3).
    *   **Input Data (Secondary - Focused Unit Test):**
        *   Mock `ImageSet` yielding a pre-defined NumPy array (Ref: API D.2 for conversion to flex). Programmatically construct minimal `Experiment_dials_w` (Ref: API A.1, B.1, B.2), `BraggMask_dials_w` (tuple of `flex.bool`, Ref: API D.4), `Mask_pixel_w` (tuple of `flex.bool`, Ref: API D.4).
    *   **Execution:** Call the Python function(s) implementing this module.
    *   **Verification:** Assert `DiffusePixelList_w` contains correct q-vectors (`scitbx.matrix.col`) and raw intensities for expected unmasked pixels.

**Module 2.2.D: Pixel-Based Correction Factor Application**
*   **Action:** Apply geometric, experimental, and background corrections to each extracted diffuse pixel's intensity.
*   **Input:**
    *   `Experiment_dials_w` (for geometry, source, spindle information, includes `dxtbx.model.Detector`, `dxtbx.model.Beam`, `dxtbx.model.Goniometer`) (Ref: API B.1, B.2, B.4).
    *   `IS_w` (`dxtbx.imageset.ImageSet` for `scan` to get `t_exp(f)`) (Ref: API A.3, B.5).
    *   `DiffusePixelList_w` (from Module 2.1.D, contains `q_vector` as `scitbx.matrix.col`).
    *   Optionally, background image series `IS_bkg_w` (`ImageSet`) and their exposure times (Ref: API A.3, B.5).
*   **Process (for each entry `(q_vec, I_raw_val, f_idx, panel_idx, px, py)` in `DiffusePixelList_w`):**
    1.  **Calculate Per-Pixel Corrections `CF(p,f_idx)` for pixel `p=(px,py)` at frame `f_idx`:**
        *   `d(p)`: Distance from sample to pixel, derived from `panel.get_pixel_lab_coord(px, py).length()` (Ref: API B.1, C.1).
        *   `SA(p)`: Solid angle correction. Use `calculate_solid_angle_correction` or similar (Ref: API C.4). Requires `Detector` (API B.1), `Beam` (API B.2).
        *   `Pol(p,f_idx)`: Polarization correction. Use `calculate_polarization_correction` or similar (Ref: API C.4). Requires `Beam` (API B.2) and `s1` vector (derived from `panel.get_pixel_lab_coord`, API B.1).
        *   `Eff(p)`: Detector efficiency correction. Use `calculate_detector_efficiency_correction` or similar (Ref: API C.4). Requires `Detector` (API B.1) and `s1` vector.
        *   `Att(p)`: Attenuation correction (e.g., air path). Use `calculate_air_attenuation_correction` or similar (Ref: API C.4). Requires `lab_coord` (from API B.1) and `Beam.get_wavelength()` (Ref: API B.2).
        *   Lorentz correction (L(p,f_idx)): If applicable for rotation data. Use `calculate_lorentz_correction` or similar (Ref: API C.4). Requires `Beam` (API B.2), `s1` vector, and `Goniometer` (API B.4).
        *   Combine corrections: `TotalGeomCorr(p,f_idx) = SA(p) * Eff(p) * Att(p) / (L(p,f_idx) * Pol(p,f_idx))`. Note application conventions (Ref: API C.4, physical formulas).
    2.  **Background Subtraction (if applicable):**
        *   `I_bkg_pixel_rate(p)`: Pre-calculate average background rate per pixel if background images `IS_bkg_w` are provided. Access raw data (`flex.int`/`flex.double`) and exposure times (`flex.double`) from `IS_bkg_w` (Ref: API A.3, B.5). Use `flex.sum` and arithmetic (Ref: API D.4).
        *   `I_raw_per_sec = I_raw_val / t_exp(f_idx)`: Scalar division. `t_exp` from `IS_w.get_scan().get_exposure_times()` (Ref: API A.3, B.5).
        *   `I_bkg_subtracted_per_sec = I_raw_per_sec - I_bkg_pixel_rate(p)` (if background used, else `I_bkg_subtracted_per_sec = I_raw_per_sec`). Scalar subtraction.
    3.  **Apply Corrections:**
        *   `I_corrected_pf = I_bkg_subtracted_per_sec / TotalGeomCorr(p,f_idx)`: Scalar division.
    4.  **Error Propagation:**
        *   `Var_raw = I_raw_val / gain` (Poisson assumption, gain applied). Gain might be from `panel.get_gain()` (Ref: API B.1, if available) or external.
        *   `Var_bkg_rate`: Calculate using arithmetic on background data (Ref: API D.4).
        *   `Var_corrected_pf = (Var_raw / t_exp(f_idx)^2 + Var_bkg_rate) / TotalGeomCorr(p,f_idx)^2`: Arithmetic.
        *   `Sigma_corrected_pf = sqrt(Var_corrected_pf)`: `math.sqrt`.
    5.  Update the entry in `DiffusePixelList_w` to store `(q_vec, I_corrected_pf, Sigma_corrected_pf, sx, sy, sz, ix_orig=px, iy_orig=py, iz_frame=f_idx, panel_idx_orig=panel_idx)`.
        *   `sx, sy, sz` are components of `q_vec` (`scitbx.matrix.col`).
*   **Output (for each wedge `w`):**
    *   `CorrectedDiffusePixelList_w`: List of tuples `(q_vector, corrected_intensity, corrected_sigma, sx, sy, sz, ix_orig, iy_orig, iz_frame, panel_idx_orig)`. `q_vector` is `scitbx.matrix.col`.
*   **Consistency Check:** Corrected intensities should be physically reasonable. Sigma values positive.
*   **Testing (Module 2.2.D):**
    *   **Input Data:**
        *   `DiffusePixelList_w` (as output from Module 2.1.D tests, or a simplified, programmatically constructed version with known raw intensities and q-vectors - `q_vector` as `scitbx.matrix.col`).
        *   Corresponding `Experiment_dials_w` (loaded or constructed, including `Detector`, `Beam`, `Goniometer`, `Scan`) (Ref: API A.1, B.1, B.2, B.4, B.5).
        *   `IS_w` (for exposure times, can be minimal `ImageSet`) (Ref: API A.3, B.5).
    *   **Execution:** Call the Python function(s) for applying corrections.
    *   **Verification:** Assert values in `CorrectedDiffusePixelList_w` (corrected intensities, sigmas) match expected values based on known correction formulas and inputs.

---

**Phase 3: Voxelization, Scaling (Relative), and Merging**

**Module 3.0.D: Global Voxel Grid Definition**
*   **Action:** Define a common 3D reciprocal space grid for merging diffuse data from all wedges.
*   **Input:**
    *   `Experiment_dials_w` (from all wedges, for an average reference crystal model - `dxtbx.model.Crystal`) (Ref: API B.3).
    *   `CorrectedDiffusePixelList_w` (from all wedges, to determine q-space coverage and HKL range). Contains `q_vector` as `scitbx.matrix.col`.
    *   Options: `|q|_max` (maximum scattering vector magnitude), `ndiv` (number of divisions per HKL unit).
*   **Process:**
    1.  Determine an average reference crystal model from all `Experiment_dials_w.crystal`. This involves averaging `cctbx.uctbx.unit_cell` parameters and `scitbx.matrix.sqr` orientation matrices (Ref: API B.3).
    2.  Determine overall minimum and maximum `(h,k,l)` ranges from all `q_vector` components in `CorrectedDiffusePixelList_w` when transformed to fractional Miller indices using the average reference crystal model's `crystal.get_A().inverse()` (Ref: API C.2).
    3.  Create `GlobalGrid = grid.Sub3dRef()` (or equivalent grid object):
        *   `GlobalGrid.ref_crystal`: The average reference `dxtbx.model.Crystal` model (Ref: API B.3).
        *   `GlobalGrid.hkl_range`: Min/max H, K, L integer indices based on observed range and `|q|_max`.
        *   `GlobalGrid.ndiv = [n_h, n_k, n_l]` (subdivisions per unit cell, from options).
*   **Output:** `GlobalGrid` (a single grid object for the entire dataset, containing a `dxtbx.model.Crystal` object).
*   **Testing (Module 3.0.D):**
    *   **Input Data:**
        *   List of minimal `Experiment_dials_w` objects (programmatically constructed, containing `dxtbx.model.Crystal`) or `CorrectedDiffusePixelList_w` (constructed with `scitbx.matrix.col` q-vectors) to define HKL ranges.
        *   Grid definition options (`|q|_max`, `ndiv`).
    *   **Execution:** Call the grid definition function.
    *   **Verification:** Assert `GlobalGrid` object has correct reference crystal (`dxtbx.model.Crystal`), HKL range, and divisions.

**Module 3.1.D: Binning Corrected Diffuse Pixels into Global Voxel Grid**
*   **Action:** For each wedge `w`, assign each entry in `CorrectedDiffusePixelList_w` to a voxel in `GlobalGrid`.
*   **Input:**
    *   `CorrectedDiffusePixelList_w` (for all wedges, contains `q_vec` as `scitbx.matrix.col`).
    *   `GlobalGrid` (from Module 3.0.D, contains `ref_crystal` as `dxtbx.model.Crystal`).
*   **Process (for each entry `(q_vec, I_corr, Sigma_corr, sx, sy, sz, ix, iy, iz_frame, panel_idx)` in each `CorrectedDiffusePixelList_w`):**
    1.  Transform `q_vec` to fractional Miller indices `(h,k,l)` using `GlobalGrid.ref_crystal.get_A().inverse()` (Ref: API C.2, B.3). Result is `scitbx.matrix.col`.
    2.  `voxel_idx = GlobalGrid.hkl2index(h,k,l)` (get the index of the voxel in `GlobalGrid`).
    3.  Store the observation: `(voxel_idx, I_corr, Sigma_corr, sx, sy, sz, ix_orig, iy_orig, iz_frame, panel_idx_orig, wedge_idx=w)`. This data could be stored in a `dials.array_family.flex.reflection_table` (Ref: API D.3, D.2).
*   **Output:**
    *   `BinnedPixelTable_Global`: A single table or list containing all diffuse observations from all wedges, each associated with a `voxel_idx`. Likely a `dials.array_family.flex.reflection_table` (Ref: API D.3).
    *   `ScalingModel_initial_list`: A list of initial `ScalingModel` objects, one per wedge `w`.
*   **Testing (Module 3.1.D):**
    *   **Input Data:**
        *   Sample `CorrectedDiffusePixelList_w` (programmatically constructed with known `scitbx.matrix.col` q-vectors and values).
        *   A `GlobalGrid` (constructed, containing a `dxtbx.model.Crystal`).
    *   **Execution:** Call the binning function.
    *   **Verification:** Assert observations in `BinnedPixelTable_Global` (likely `flex.reflection_table`) are correctly assigned to `voxel_idx` and contain correct values.

**Module 3.2.D: Relative Scaling of Binned Observations**
*   **Action:** Iteratively refine `ScalingModel` parameters (`a,b,c,d` control points) for each wedge `w`.
*   **Input:**
    *   `BinnedPixelTable_Global` (likely `flex.reflection_table`, accessing columns like intensity, sigma, voxel_idx, wedge_idx) (Ref: API D.3).
    *   `ScalingModel_initial_list`.
*   **Process (Iterative, using a `MultiBatchScaler`-like approach):** Involves arithmetic and statistical operations on columns of `BinnedPixelTable_Global` (Ref: API D.4).
*   **Output:** `ScalingModel_refined_list`.
*   **Testing (Module 3.2.D):**
    *   **Input Data:**
        *   A representative `BinnedPixelTable_Global` (programmatically constructed as `flex.reflection_table` with known systematic variations).
        *   `ScalingModel_initial_list`.
    *   **Execution:** Run the relative scaling process.
    *   **Verification:** Assert `ScalingModel_refined_list` parameters correct known variations. Check residuals using arithmetic (Ref: API D.4).

**Module 3.3.D: Merging Relatively Scaled Data into Voxels**
*   **Action:** Apply final refined scales and merge observations into `GlobalGrid` voxels.
*   **Input:**
    *   `BinnedPixelTable_Global` (likely `flex.reflection_table`, accessing columns) (Ref: API D.3).
    *   `ScalingModel_refined_list`.
    *   `GlobalGrid` (contains `ref_crystal` as `dxtbx.model.Crystal`).
*   **Process:** Apply scales using arithmetic on intensity/sigma columns (Ref: API D.4). Merge observations (e.g., weighted average) using arithmetic and `flex.sum` (Ref: API D.4). Calculate `H_center, K_center, L_center` from `voxel_idx` and `GlobalGrid`. Calculate `|q|_center` from HKL_center using `GlobalGrid.ref_crystal.get_A()` to get `q_center` (`scitbx.matrix.col`), then `q_center.length()` (Ref: API B.3, C.1).
*   **Output:**
    *   `VoxelData_relative`: Table/array: `(voxel_idx, H_center, K_center, L_center, |q|_center, I_merged_relative, Sigma_merged_relative, num_obs_in_voxel)`. Likely a `flex.reflection_table` (Ref: API D.3).
*   **Testing (Module 3.3.D):**
    *   **Input Data:**
        *   `BinnedPixelTable_Global` (constructed as `flex.reflection_table`).
        *   `ScalingModel_refined_list` (constructed with known scaling factors).
        *   `GlobalGrid`.
    *   **Execution:** Call the merging function.
    *   **Verification:** Assert `I_merged_relative` and `Sigma_merged_relative` in `VoxelData_relative` (likely `flex.reflection_table`) are correct.

---

**Phase 4: Absolute Scaling**

**Module 4.1.D: Absolute Scaling and Incoherent Subtraction**
*   **Action:** Convert `VoxelData_relative` to absolute units and subtract theoretical incoherent scattering.
*   **Input:**
    *   `VoxelData_relative` (from Module 3.3.D, likely `flex.reflection_table`, accessing columns) (Ref: API D.3).
    *   `unitCellInventory`: Atomic composition of the unit cell (external input).
    *   `Crystal`: Average crystal model (`dxtbx.model.Crystal`, from `GlobalGrid.ref_crystal` or similar) (Ref: API B.3).
    *   `Reflections_dials_w` (all wedges): Used to create `I_bragg_relative(HKL_asu, |q|)` by applying `ScalingModel_refined_list`. Access Bragg intensities and HKLs (Ref: API A.2, D.3).
*   **Process:** Determine absolute scale factor (e.g., by comparing Bragg intensities to theoretical values). Apply scale factor to `VoxelData_relative` intensities and sigmas (Ref: API D.4). Calculate theoretical incoherent scattering for each voxel's `|q|_center` using `cctbx.eltbx` functions like `wk1995` (Ref: API C.5), summing contributions based on `unitCellInventory`. Subtract incoherent scattering from scaled diffuse intensities (Ref: API D.4).
*   **Output:**
    *   `VoxelData_absolute`: Final diffuse map `(voxel_idx, H_center, K_center, L_center, |q|_center, I_abs_diffuse, Sigma_abs_diffuse)`. Likely a `flex.reflection_table` (Ref: API D.3).
    *   `ScalingModel_final_list`: Updated scaling models.
*   **Testing (Module 4.1.D):**
    *   **Input Data:**
        *   `VoxelData_relative` (constructed as `flex.reflection_table` with known relative intensities).
        *   Known `unitCellInventory` and `Crystal` model (Ref: API B.3).
        *   Constructed list of Bragg reflections (`flex.reflection_table`) with known relative intensities (Ref: API A.2, D.3).
    *   **Execution:** Call the absolute scaling function.
    *   **Verification:** Assert `Scale_Abs` is correct. Assert `I_abs_diffuse` in `VoxelData_absolute` (likely `flex.reflection_table`) matches expected values. Use arithmetic and comparison (Ref: API D.4).

---
