Okay, here is the full, updated `plan.md` incorporating DIALS for initial processing and the revised, integrated testing strategy.

```markdown
# Diffuse Scattering Processing Plan (DIALS-Integrated with Integrated Testing)

**Nomenclature:**

*   `I_raw(px, py, f)`: Raw intensity at detector pixel `(px, py)` for frame `f`.
*   `t_exp(f)`: Exposure time for frame `f`.
*   `Experiment_dials_w`: DIALS `ExperimentList` object for wedge `w` (from `.expt` file).
*   `Reflections_dials_w`: DIALS `reflection_table` for wedge `w` (from `.refl` file).
*   `BraggMask_dials_w(px, py)`: Boolean mask from DIALS (`.pickle` file) for wedge `w`, indexed by panel and pixel.
*   `IS_w`: `io.ImageSeries` object for wedge `w` (for accessing raw image data).
*   `Mask_pixel(px, py)`: Static bad pixel mask (e.g., beamstop, bad pixels), indexed by panel and pixel.
*   `v`: Voxel index in a 3D reciprocal space grid (defined in Phase 3).
*   `p`: Detector pixel index `(px,py)`.
*   `s`: Scattering vector magnitude `|s|`.
*   `H,K,L`: Integer Miller indices.
*   `h,k,l`: Continuous (fractional) Miller indices.
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
    *   `Experiment_dials_w`: Refined DIALS `ExperimentList` object (from `indexed_refined_detector.expt`).
    *   `Reflections_dials_w`: Refined DIALS `reflection_table` (from `indexed_refined_detector.refl`).
    *   `BraggMask_dials_w`: Boolean mask for Bragg peaks (from `bragg_mask.pickle`), typically a tuple of flex.bool arrays, one per panel.
*   **Consistency Check:** Successful completion of all DIALS steps (check logs). Reasonableness of refined geometry and indexing statistics.
*   **Testing (Module 1.D):**
    *   **Input Data:**
        *   Curated CBF image files from `tests/data/cbf_input/`.
        *   Standard DIALS PHIL files used by the orchestrator.
    *   **Execution:** Invoke the DIALS processing pipeline orchestrator (e.g., `src/scripts/process_pipeline.sh` or its Python equivalent) on these CBF files.
    *   **Verification:**
        *   Successful execution of all DIALS commands (check exit codes, review DIALS logs).
        *   Generation of expected output files (`indexed_refined_detector.expt`, `indexed_refined_detector.refl`, `bragg_mask.pickle`) in `tests/data/dials_outputs/wedge_W/` (or a temporary processing directory whose contents are checked).
        *   Basic sanity checks on output files (e.g., loadability by `dxtbx`).

**Module 1.1 (Revised): Static Pixel Masking**
*   **Action:** Create a static 2D detector mask based on detector properties and known bad regions, independent of diffraction data.
*   **Input:**
    *   `Experiment_dials_w.detector` (for detector geometry for each wedge `w`).
    *   Optionally, a representative raw image `I_raw(px, py, first_frame)` for dynamic masking (e.g., negative counts).
*   **Process:** For each panel in `Experiment_dials_w.detector`:
    1.  `Mask_static_panel(px,py)`: From panel properties (e.g., panel gaps, edges).
    2.  `Mask_beamstop_panel(px,py)`: User-defined or geometrically derived beamstop mask relevant to the panel.
    3.  `Mask_virtual_corrected_panel(px,py)`: Mask for virtual or corrected pixels if applicable to panel type.
    4.  `Mask_negative_counts_panel(px,py)`: If using a raw image, `(I_raw_panel(px,py,first_frame) >= 0)`.
    5.  `Mask_pixel_panel(px,py) = Mask_static_panel & Mask_beamstop_panel & Mask_virtual_corrected_panel & Mask_negative_counts_panel`.
*   **Output:** `Mask_pixel_w`: A tuple/list of 2D boolean arrays, one per panel for detector in wedge `w`.
*   **Consistency Check:** Visual inspection of `Mask_pixel_w`. Ensure it covers known bad regions without excessively masking good data.
*   **Testing (Module 1.1):**
    *   **Input Data:**
        *   Programmatically constructed `dxtbx.model.Detector` objects (containing one or more panels) with known features.
        *   Optionally, small NumPy arrays for dynamic mask aspects per panel.
    *   **Execution:** Call the Python function for static mask generation, passing the detector model.
    *   **Verification:** Assert the output (list/tuple of NumPy arrays for `Mask_pixel_w`) correctly identifies known bad regions based on inputs for each panel.

---

**Phase 2: Diffuse Intensity Extraction and Pixel-Level Correction**

**Module 2.1.D: Pixel-Based Diffuse Data Extraction**
*   **Action:** For each wedge `w`, read raw images, apply combined masks (`BraggMask_dials_w` and `Mask_pixel_w`), calculate q-vector for each unmasked pixel, and extract its raw intensity.
*   **Input:**
    *   `Experiment_dials_w` (for geometry).
    *   `IS_w` (`io.ImageSeries` for accessing raw image data for wedge `w`).
    *   `BraggMask_dials_w` (from Module 1.D).
    *   `Mask_pixel_w` (from Module 1.1).
*   **Process (for each frame `f` in `IS_w`, for each panel `panel_idx` in `Experiment_dials_w.detector`, for each pixel `p=(px,py)` on that panel):**
    1.  If `NOT BraggMask_dials_w[panel_idx](py, px)` AND `Mask_pixel_w[panel_idx](py, px)`: (Note: flex.bool is often (slow,fast) so py,px)
        a.  `I_raw_pf = IS_w.get_raw_data(f)[panel_idx][py, px]`
        b.  Calculate `q_vector(p,f)`:
            i.  Get laboratory coordinates of pixel `p` on `Experiment_dials_w.detector[panel_idx]`.
            ii. Get incident beam vector `s_incident` from `Experiment_dials_w.beam`.
            iii.Calculate scattered vector `s_scattered` from sample to pixel.
            iv. `q_vector(p,f) = s_scattered - s_incident`.
        c.  Store `(q_vector(p,f), I_raw_pf, frame_index=f, panel_index=panel_idx, pixel_coords=(px,py))`.
*   **Output (for each wedge `w`):**
    *   `DiffusePixelList_w`: A list of tuples, where each tuple is `(q_vector, raw_intensity, frame_idx, panel_idx, px, py)` for all selected diffuse pixels in the wedge.
*   **Consistency Check:** Sanity check on the number of extracted pixels. Visual inspection of q-vector distribution if feasible.
*   **Testing (Module 2.1.D):**
    *   **Input Data (Primary - Integration with DIALS Outputs):**
        *   Load `Experiment_dials_w` from `tests/data/dials_outputs/wedge_W/indexed_refined_detector.expt`.
        *   Load `BraggMask_dials_w` from `tests/data/dials_outputs/wedge_W/bragg_mask.pickle`.
        *   Load/construct `Mask_pixel_w`.
        *   Use an `ImageSet` pointing to `tests/data/cbf_input/wedge_W_image_F.cbf`.
    *   **Input Data (Secondary - Focused Unit Test):**
        *   Mock `ImageSet` yielding a pre-defined NumPy array. Programmatically construct minimal `Experiment_dials_w`, `BraggMask_dials_w`, `Mask_pixel_w`.
    *   **Execution:** Call the Python function(s) implementing this module.
    *   **Verification:** Assert `DiffusePixelList_w` contains correct q-vectors and raw intensities for expected unmasked pixels.

**Module 2.2.D: Pixel-Based Correction Factor Application**
*   **Action:** Apply geometric, experimental, and background corrections to each extracted diffuse pixel's intensity.
*   **Input:**
    *   `Experiment_dials_w` (for geometry, source, spindle information).
    *   `IS_w` (for `t_exp(f)`).
    *   `DiffusePixelList_w` (from Module 2.1.D).
    *   Optionally, background image series `IS_bkg_w` and their exposure times.
*   **Process (for each entry `(q_vec, I_raw_val, f_idx, panel_idx, px, py)` in `DiffusePixelList_w`):**
    1.  **Calculate Per-Pixel Corrections `CF(p,f_idx)` for pixel `p=(px,py)` at frame `f_idx`:**
        *   `d(p), cosω(p)`: Distance and angle from `Experiment_dials_w.detector[panel_idx]` and pixel `(px,py)`.
        *   `SA(p) = geom.Corrections.solidAngle(Detector_panel, px, py)`
        *   `Pol(p,f_idx) = geom.Corrections.polarization(Source, Detector_panel, Spindle_at_frame_f_idx, px, py)`
        *   `Eff(p) = geom.Corrections.efficiency(Detector_panel, px, py)`
        *   `Att(p) = geom.Corrections.attenuation(Detector_panel, px, py)` (e.g., air path)
        *   `TotalGeomCorr(p,f_idx) = SA(p) * Pol(p,f_idx) * Eff(p) * Att(p)`
    2.  **Background Subtraction (if applicable):**
        *   `I_bkg_pixel_rate(p)`: Pre-calculate average background rate per pixel if background images `IS_bkg_w` are provided.
            `I_bkg_pixel_rate(p) = (Σ_{f_bkg in IS_bkg_w} I_raw_bkg(p, f_bkg)) / (Σ_{f_bkg in IS_bkg_w} t_exp_bkg(f_bkg))`
        *   `I_raw_per_sec = I_raw_val / t_exp(f_idx)`
        *   `I_bkg_subtracted_per_sec = I_raw_per_sec - I_bkg_pixel_rate(p)` (if background used, else `I_bkg_subtracted_per_sec = I_raw_per_sec`)
    3.  **Apply Corrections:**
        *   `I_corrected_pf = I_bkg_subtracted_per_sec / TotalGeomCorr(p,f_idx)`
    4.  **Error Propagation:**
        *   `Var_raw = I_raw_val / gain` (Poisson assumption, gain applied)
        *   `Var_bkg_rate = (Σ_{f_bkg} I_raw_bkg(p,f_bkg)/gain) / (Σ_{f_bkg} t_exp_bkg(f_bkg))^2` (if background used)
        *   `Var_corrected_pf = (Var_raw / t_exp(f_idx)^2 + Var_bkg_rate) / TotalGeomCorr(p,f_idx)^2`
        *   `Sigma_corrected_pf = sqrt(Var_corrected_pf)`
    5.  Update the entry in `DiffusePixelList_w` to store `(q_vec, I_corrected_pf, Sigma_corrected_pf, sx, sy, sz, ix_orig=px, iy_orig=py, iz_frame=f_idx, panel_idx_orig=panel_idx)`.
        *   `sx, sy, sz` are components of `q_vec`.
*   **Output (for each wedge `w`):**
    *   `CorrectedDiffusePixelList_w`: List of tuples `(q_vector, corrected_intensity, corrected_sigma, sx, sy, sz, ix_orig, iy_orig, iz_frame, panel_idx_orig)`.
*   **Consistency Check:** Corrected intensities should be physically reasonable. Sigma values positive.
*   **Testing (Module 2.2.D):**
    *   **Input Data:**
        *   `DiffusePixelList_w` (as output from Module 2.1.D tests, or a simplified, programmatically constructed version with known raw intensities and q-vectors).
        *   Corresponding `Experiment_dials_w` (loaded or constructed).
        *   `IS_w` (for exposure times, can be minimal).
    *   **Execution:** Call the Python function(s) for applying corrections.
    *   **Verification:** Assert values in `CorrectedDiffusePixelList_w` (corrected intensities, sigmas) match expected values based on known correction formulas and inputs.

---

**Phase 3: Voxelization, Scaling (Relative), and Merging**

**Module 3.0.D: Global Voxel Grid Definition**
*   **Action:** Define a common 3D reciprocal space grid for merging diffuse data from all wedges.
*   **Input:**
    *   `Experiment_dials_w` (from all wedges, for an average reference crystal model).
    *   `CorrectedDiffusePixelList_w` (from all wedges, to determine q-space coverage and HKL range).
    *   Options: `s_max` (maximum scattering vector magnitude), `ndiv` (number of divisions per HKL unit).
*   **Process:**
    1.  Determine an average reference crystal model from all `Experiment_dials_w.crystal`.
    2.  Determine overall minimum and maximum `(h,k,l)` ranges from all `q_vector` components in `CorrectedDiffusePixelList_w` when transformed to fractional Miller indices using the average reference crystal model.
    3.  Create `GlobalGrid = grid.Sub3dRef()` (or equivalent grid object):
        *   `GlobalGrid.ref_crystal`: The average reference crystal model.
        *   `GlobalGrid.hkl_range`: Min/max H, K, L integer indices based on observed range and `s_max`.
        *   `GlobalGrid.ndiv = [n_h, n_k, n_l]` (subdivisions per unit cell, from options).
*   **Output:** `GlobalGrid` (a single grid object for the entire dataset).
*   **Testing (Module 3.0.D):**
    *   **Input Data:**
        *   List of minimal `Experiment_dials_w` objects (programmatically constructed) or `CorrectedDiffusePixelList_w` (constructed) to define HKL ranges.
        *   Grid definition options (`s_max`, `ndiv`).
    *   **Execution:** Call the grid definition function.
    *   **Verification:** Assert `GlobalGrid` object has correct reference cell, HKL range, and divisions.

**Module 3.1.D: Binning Corrected Diffuse Pixels into Global Voxel Grid**
*   **Action:** For each wedge `w`, assign each entry in `CorrectedDiffusePixelList_w` to a voxel in `GlobalGrid`.
*   **Input:**
    *   `CorrectedDiffusePixelList_w` (for all wedges).
    *   `GlobalGrid` (from Module 3.0.D).
*   **Process (for each entry `(q_vec, I_corr, Sigma_corr, sx, sy, sz, ix, iy, iz_frame, panel_idx)` in each `CorrectedDiffusePixelList_w`):**
    1.  Transform `q_vec` to fractional Miller indices `(h,k,l)` using `GlobalGrid.ref_crystal`.
    2.  `voxel_idx = GlobalGrid.hkl2index(h,k,l)` (get the index of the voxel in `GlobalGrid`).
    3.  Store the observation: `(voxel_idx, I_corr, Sigma_corr, sx, sy, sz, ix_orig, iy_orig, iz_frame, panel_idx_orig, wedge_idx=w)`.
*   **Output:**
    *   `BinnedPixelTable_Global`: A single table or list containing all diffuse observations from all wedges, each associated with a `voxel_idx`.
    *   `ScalingModel_initial_list`: A list of initial `ScalingModel` objects, one per wedge `w`.
*   **Testing (Module 3.1.D):**
    *   **Input Data:**
        *   Sample `CorrectedDiffusePixelList_w` (programmatically constructed with known q-vectors and values).
        *   A `GlobalGrid` (constructed).
    *   **Execution:** Call the binning function.
    *   **Verification:** Assert observations in `BinnedPixelTable_Global` are correctly assigned to `voxel_idx`.

**Module 3.2.D: Relative Scaling of Binned Observations**
*   **Action:** Iteratively refine `ScalingModel` parameters (`a,b,c,d` control points) for each wedge `w`.
*   **Input:**
    *   `BinnedPixelTable_Global`.
    *   `ScalingModel_initial_list`.
*   **Process (Iterative, using a `MultiBatchScaler`-like approach):** (Details as per previous draft)
*   **Output:** `ScalingModel_refined_list`.
*   **Testing (Module 3.2.D):**
    *   **Input Data:**
        *   A representative `BinnedPixelTable_Global` (programmatically constructed with known systematic variations).
        *   `ScalingModel_initial_list`.
    *   **Execution:** Run the relative scaling process.
    *   **Verification:** Assert `ScalingModel_refined_list` parameters correct known variations. Check residuals.

**Module 3.3.D: Merging Relatively Scaled Data into Voxels**
*   **Action:** Apply final refined scales and merge observations into `GlobalGrid` voxels.
*   **Input:**
    *   `BinnedPixelTable_Global`.
    *   `ScalingModel_refined_list`.
    *   `GlobalGrid`.
*   **Process:** (Details as per previous draft)
*   **Output:**
    *   `VoxelData_relative`: Table/array: `(voxel_idx, H_center, K_center, L_center, s_mag_center, I_merged_relative, Sigma_merged_relative, num_obs_in_voxel)`.
*   **Testing (Module 3.3.D):**
    *   **Input Data:**
        *   `BinnedPixelTable_Global` (constructed).
        *   `ScalingModel_refined_list` (constructed with known scaling factors).
        *   `GlobalGrid`.
    *   **Execution:** Call the merging function.
    *   **Verification:** Assert `I_merged_relative` and `Sigma_merged_relative` in `VoxelData_relative` are correct.

---

**Phase 4: Absolute Scaling**

**Module 4.1.D: Absolute Scaling and Incoherent Subtraction**
*   **Action:** Convert `VoxelData_relative` to absolute units and subtract theoretical incoherent scattering.
*   **Input:**
    *   `VoxelData_relative` (from Module 3.3.D).
    *   `unitCellInventory`: Atomic composition of the unit cell.
    *   `Crystal`: Average crystal model (from `GlobalGrid.ref_crystal` or similar).
    *   `Reflections_dials_w` (all wedges): Used to create `I_bragg_relative(HKL_asu, s_mag)` by applying `ScalingModel_refined_list`.
*   **Process:** (Details as per previous draft, using `s_mag_center(v)` for incoherent subtraction per voxel)
*   **Output:**
    *   `VoxelData_absolute`: Final diffuse map `(voxel_idx, H_center, K_center, L_center, s_mag_center, I_abs_diffuse, Sigma_abs_diffuse)`.
    *   `ScalingModel_final_list`: Updated scaling models.
*   **Testing (Module 4.1.D):**
    *   **Input Data:**
        *   `VoxelData_relative` (constructed with known relative intensities).
        *   Known `unitCellInventory` and `Crystal` model.
        *   Constructed list of Bragg reflections with known relative intensities.
    *   **Execution:** Call the absolute scaling function.
    *   **Verification:** Assert `Scale_Abs` is correct. Assert `I_abs_diffuse` in `VoxelData_absolute` matches expected values.

---
```
