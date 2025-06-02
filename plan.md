# Diffuse Scattering Processing Plan (DIALS-Integrated)

**Nomenclature:**

*   `I_raw(px, py, f)`: Raw intensity at detector pixel `(px, py)` for frame `f`.
*   `t_exp(f)`: Exposure time for frame `f`.
*   `Experiment_dials_w`: DIALS `ExperimentList` object for wedge `w` (from `.expt` file).
*   `Reflections_dials_w`: DIALS `reflection_table` for wedge `w` (from `.refl` file).
*   `BraggMask_dials_w(px, py)`: Boolean mask from DIALS (`.pickle` file) for wedge `w`.
*   `IS_w`: `io.ImageSeries` object for wedge `w` (for accessing raw image data).
*   `Mask_pixel(px, py)`: Static bad pixel mask (e.g., beamstop, bad pixels).
*   `v`: Voxel index in a 3D reciprocal space grid (defined in Phase 3).
*   `p`: Detector pixel index `(px,py)`.
*   `s`: Scattering vector magnitude `|s|`.
*   `H,K,L`: Integer Miller indices.
*   `h,k,l`: Continuous (fractional) Miller indices.
*   `w`: Wedge index.

---

**Phase 1: DIALS-Based Initial Processing & Masking**
*(This phase replaces original `plan.md` Modules 1.0, 1.1, 1.2)*

**Module 1.D: DIALS Processing Orchestration (New Module)**
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
    *   `BraggMask_dials_w`: Boolean mask for Bragg peaks (from `bragg_mask.pickle`).
*   **Consistency Check:** Successful completion of all DIALS steps (check logs). Reasonableness of refined geometry and indexing statistics.

**Module 1.1 (Revised): Static Pixel Masking**
*   **Action:** Create a static 2D detector mask based on detector properties and known bad regions, independent of diffraction data.
*   **Input:**
    *   `Experiment_dials_w.detector` (for detector geometry).
    *   Optionally, a representative raw image `I_raw(px, py, first_frame)` for dynamic masking (e.g., negative counts).
*   **Process:**
    1.  `Mask_static(px,py)`: From detector properties (e.g., panel gaps, edges).
    2.  `Mask_beamstop(px,py)`: User-defined or geometrically derived beamstop mask.
    3.  `Mask_virtual_corrected(px,py)`: Mask for virtual or corrected pixels if applicable to detector type.
    4.  `Mask_negative_counts(px,py)`: If using a raw image, `(I_raw(px,py,first_frame) >= 0)`.
    5.  `Mask_pixel(px,py) = Mask_static & Mask_beamstop & Mask_virtual_corrected & Mask_negative_counts`.
*   **Output:** `Mask_pixel(px,py)` (a 2D boolean array).
*   **Consistency Check:** Visual inspection of `Mask_pixel`. Ensure it covers known bad regions without excessively masking good data.

---

**Phase 2: Diffuse Intensity Extraction and Pixel-Level Correction**
*(This phase revises original `plan.md` Modules 2.1, 2.2)*

**Module 2.1.D: Pixel-Based Diffuse Data Extraction (New/Replaces original `plan.md` 2.1)**
*   **Action:** For each wedge `w`, read raw images, apply combined masks (`BraggMask_dials_w` and `Mask_pixel`), calculate q-vector for each unmasked pixel, and extract its raw intensity.
*   **Input:**
    *   `Experiment_dials_w` (for geometry).
    *   `IS_w` (`io.ImageSeries` for accessing raw image data for wedge `w`).
    *   `BraggMask_dials_w` (from Module 1.D).
    *   `Mask_pixel` (from Module 1.1).
*   **Process (for each frame `f` in `IS_w`, for each pixel `p=(px,py)` on each panel):**
    1.  If `NOT BraggMask_dials_w(p, panel_idx)` AND `Mask_pixel(p, panel_idx)`:
        a.  `I_raw_pf = I_raw(p,f)` (from `IS_w`).
        b.  Calculate `q_vector(p,f)`:
            i.  Get laboratory coordinates of pixel `p` on its panel using `Experiment_dials_w.detector[panel_idx]`.
            ii. Get incident beam vector `s_incident` from `Experiment_dials_w.beam`.
            iii.Calculate scattered vector `s_scattered` from sample to pixel.
            iv. `q_vector(p,f) = s_scattered - s_incident`.
        c.  Store `(q_vector(p,f), I_raw_pf, frame_index=f, panel_index=panel_idx, pixel_coords=(px,py))`.
*   **Output (for each wedge `w`):**
    *   `DiffusePixelList_w`: A list of tuples, where each tuple is `(q_vector, raw_intensity, frame_idx, panel_idx, px, py)` for all selected diffuse pixels in the wedge.
*   **Consistency Check:** Sanity check on the number of extracted pixels. Visual inspection of q-vector distribution (e.g., as a 2D projection) if feasible.

**Module 2.2.D: Pixel-Based Correction Factor Application (Revises original `plan.md` 2.2)**
*   **Action:** Apply geometric, experimental, and background corrections to each extracted diffuse pixel's intensity.
*   **Input:**
    *   `Experiment_dials_w` (for geometry, source, spindle information).
    *   `IS_w` (for `t_exp(f)`).
    *   `DiffusePixelList_w` (from Module 2.1.D).
    *   Optionally, background image series `IS_bkg_w` and their exposure times.
*   **Process (for each entry `(q_vec, I_raw_val, f_idx, panel_idx, px, py)` in `DiffusePixelList_w`):**
    1.  **Calculate Per-Pixel Corrections `CF(p,f_idx)` for pixel `p=(px,py)` at frame `f_idx`:**
        *   `d(p), cosω(p)`: Distance and angle from `Experiment_dials_w.detector[panel_idx]` and pixel `(px,py)`.
        *   `SA(p) = geom.Corrections.solidAngle(Detector, px, py)`
        *   `Pol(p,f_idx) = geom.Corrections.polarization(Source, Detector, Spindle_at_frame_f_idx, px, py)`
        *   `Eff(p) = geom.Corrections.efficiency(Detector, px, py)`
        *   `Att(p) = geom.Corrections.attenuation(Detector, px, py)` (e.g., air path)
        *   `TotalGeomCorr(p,f_idx) = SA(p) * Pol(p,f_idx) * Eff(p) * Att(p)`
    2.  **Background Subtraction (if applicable):**
        *   `I_bkg_pixel_rate(p)`: Pre-calculate average background rate per pixel if background images `IS_bkg_w` are provided.
            `I_bkg_pixel_rate(p) = (Σ_{f_bkg in IS_bkg_w} I_raw_bkg(p, f_bkg)) / (Σ_{f_bkg in IS_bkg_w} t_exp_bkg(f_bkg))`
        *   `I_raw_per_sec = I_raw_val / t_exp(f_idx)`
        *   `I_bkg_subtracted_per_sec = I_raw_per_sec - I_bkg_pixel_rate(p)` (if background is used, else `I_bkg_subtracted_per_sec = I_raw_per_sec`)
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
    *   `CorrectedDiffusePixelList_w`: List of tuples `(q_vector, corrected_intensity, corrected_sigma, sx, sy, sz, ix, iy, iz_frame, panel_idx)`.
*   **Consistency Check:** Corrected intensities should be physically reasonable (e.g., mostly positive). Sigma values should be positive.

---

**Phase 3: Voxelization, Scaling (Relative), and Merging**
*(This phase involves major revisions to original `plan.md` Modules 3.1, 3.2, 3.3)*

**Module 3.0.D: Global Voxel Grid Definition (New/Adapted from original `plan.md` 1.2)**
*   **Action:** Define a common 3D reciprocal space grid for merging diffuse data from all wedges.
*   **Input:**
    *   `Experiment_dials_w` (from all wedges, to determine overall HKL range from crystal models).
    *   Alternatively, `CorrectedDiffusePixelList_w` (from all wedges, to determine q-space coverage).
    *   Options: `s_max` (maximum scattering vector magnitude), `ndiv` (number of divisions per HKL unit).
*   **Process:**
    1.  Determine the overall minimum and maximum `(h,k,l)` ranges covered by all wedges. This can be derived from the `q_vector` components in all `CorrectedDiffusePixelList_w` by transforming them to fractional Miller indices using an average or reference crystal model.
    2.  Create `GlobalGrid = grid.Sub3dRef()` (or equivalent grid object):
        *   `GlobalGrid.ref`: Defines the reference unit cell for the grid (e.g., from an average of all `Experiment_dials_w.crystal`).
        *   `GlobalGrid.hkl_range`: Min/max H, K, L integer indices based on observed range and `s_max`.
        *   `GlobalGrid.ndiv = [n_h, n_k, n_l]` (subdivisions per unit cell, from options).
*   **Output:** `GlobalGrid` (a single grid object for the entire dataset).

**Module 3.1.D: Binning Corrected Diffuse Pixels into Global Voxel Grid (Replaces original `plan.md` 3.1's voxel-based export)**
*   **Action:** For each wedge `w`, assign each entry in `CorrectedDiffusePixelList_w` to a voxel in `GlobalGrid`.
*   **Input:**
    *   `CorrectedDiffusePixelList_w` (for all wedges).
    *   `GlobalGrid` (from Module 3.0.D).
*   **Process (for each entry `(q_vec, I_corr, Sigma_corr, sx, sy, sz, ix, iy, iz_frame, panel_idx)` in each `CorrectedDiffusePixelList_w`):**
    1.  Transform `q_vec` to fractional Miller indices `(h,k,l)` using `GlobalGrid.ref` crystal model.
    2.  `voxel_idx = GlobalGrid.hkl2index(h,k,l)` (get the index of the voxel in `GlobalGrid` that `(h,k,l)` falls into).
    3.  Store the observation: `(voxel_idx, I_corr, Sigma_corr, sx, sy, sz, ix, iy, iz_frame, panel_idx, wedge_idx=w)`.
*   **Output:**
    *   `BinnedPixelTable_Global`: A single table or list containing all diffuse observations from all wedges, each associated with a `voxel_idx` in `GlobalGrid`. Each entry is a tuple: `(voxel_idx, I_corr, Sigma_corr, sx, sy, sz, ix_orig, iy_orig, iz_frame, panel_idx_orig, wedge_idx)`.
    *   `ScalingModel_initial_list`: A list of initial `ScalingModel` objects, one per wedge `w`, with default control points for parameters `a,b,c,d`. (Similar to original `plan.md`).
*   **Consistency Check:** Ensure `voxel_idx` is valid for `GlobalGrid`.

**Module 3.2.D: Relative Scaling of Binned Observations (Logic similar to original `plan.md` 3.2, but operates on `BinnedPixelTable_Global`)**
*   **Action:** Iteratively refine `ScalingModel` parameters (`a,b,c,d` control points) for each wedge `w` to bring observations onto a common relative scale.
*   **Input:**
    *   `BinnedPixelTable_Global`.
    *   `ScalingModel_initial_list`.
*   **Process (Iterative, using a `MultiBatchScaler`-like approach):**
    1.  **Reference Intensity Calculation:** For each `voxel_idx` in `GlobalGrid`, calculate a reference intensity `I_ref(voxel_idx)`. This is typically the weighted average of all currently scaled observations falling into that voxel:
        `I_ref(voxel_idx) = Σ_obs_in_voxel ( (I_corr(obs) / Scale_mult(obs) - Offset_add(obs)) * Weight(obs) ) / Σ_obs_in_voxel (Weight(obs))`
        where `Scale_mult` and `Offset_add` are derived from the *current* `ScalingModel` for the wedge of `obs`, and `Weight(obs) = 1 / Sigma_corr(obs)^2`. This step is iterative as `ScalingModel`s are refined.
    2.  **Fit ScalingModel Parameters:** For each wedge `j` and its `ScalingModel_j`:
        Minimize a target function, e.g., for parameter `b` (overall scale vs. frame/`iz`):
        `χ² = Σ_{obs in wedge j} [ (I_corr(obs) / (a_j*d_j) - (c_j/b_j + I_ref(voxel_idx_obs))) / (Sigma_corr(obs)/(a_j*d_j)) ]^2 + Regularization(b_j_control_points)`
        The parameters `a_j, b_j, c_j, d_j` are functions (e.g., splines) of observation properties (`ix_orig`, `iy_orig`, `iz_frame`, `panel_idx_orig`, `s_mag` derived from `q_vec`) evaluated from the control points of `ScalingModel_j`.
        This is done iteratively for all parameters `a,b,c,d` for all wedges until convergence.
*   **Output:** `ScalingModel_refined_list`.
*   **Consistency Check:** Convergence of the fit. Smoothness and physical plausibility of the refined `a,b,c,d` scaling functions. Reduction in residuals.

**Module 3.3.D: Merging Relatively Scaled Data into Voxels (Logic similar to original `plan.md` 3.3, but uses `BinnedPixelTable_Global`)**
*   **Action:** Apply the final refined scales from `ScalingModel_refined_list` to all observations in `BinnedPixelTable_Global` and merge them into the `GlobalGrid` voxels.
*   **Input:**
    *   `BinnedPixelTable_Global`.
    *   `ScalingModel_refined_list`.
    *   `GlobalGrid`.
*   **Process:**
    1.  For each observation `obs = (voxel_idx, I_corr, Sigma_corr, ... wedge_idx=j)` in `BinnedPixelTable_Global`:
        *   Retrieve `ScalingModel_j` from `ScalingModel_refined_list`.
        *   Evaluate `a_j(obs), b_j(obs), c_j(obs), d_j(obs)` based on `obs` properties (ix, iy, iz_frame, panel, s_mag).
        *   `Scale_mult_obs = a_j(obs) * b_j(obs) * d_j(obs)`
        *   `Offset_add_obs = c_j(obs) / b_j(obs)`
        *   `I_rel_obs = I_corr(obs) / Scale_mult_obs - Offset_add_obs`
        *   `Sigma_rel_obs = Sigma_corr(obs) / Scale_mult_obs` (error propagation)
        *   Store `(voxel_idx, I_rel_obs, Sigma_rel_obs, sx, sy, sz)` for merging.
    2.  For each unique `voxel_idx` in `GlobalGrid`:
        *   Collect all `(I_rel_obs, Sigma_rel_obs, sx, sy, sz)` that map to this `voxel_idx`.
        *   `w_rel_obs = 1 / Sigma_rel_obs^2` (weight for each observation).
        *   `I_merged_relative(voxel_idx) = Σ(I_rel_obs * w_rel_obs) / Σ(w_rel_obs)`
        *   `Sigma_merged_relative(voxel_idx) = sqrt(1 / Σ(w_rel_obs))`
        *   `sx_avg(voxel_idx) = Σ(sx * w_rel_obs) / Σ(w_rel_obs)` (similarly for `sy`, `sz`).
        *   `s_mag_center(voxel_idx) = sqrt(sx_avg^2 + sy_avg^2 + sz_avg^2)`.
        *   `(H_center, K_center, L_center)_voxel = GlobalGrid.index2hkl(voxel_idx)` (center HKL of the voxel).
*   **Output:**
    *   `VoxelData_relative`: A table or array containing, for each populated voxel in `GlobalGrid`: `(voxel_idx, H_center, K_center, L_center, s_mag_center, I_merged_relative, Sigma_merged_relative, num_observations_in_voxel)`.
*   **Consistency Check:** R-merge statistics. Distribution of residuals `(I_rel_obs - I_merged_relative(voxel_idx_obs)) / Sigma_rel_obs`. Smoothness of the merged map.

---

**Phase 4: Absolute Scaling**
*(This phase is largely unchanged in logic from original `plan.md` Module 4.1, but operates on `VoxelData_relative` and uses Bragg data from DIALS)*

**Module 4.1.D: Absolute Scaling and Incoherent Subtraction**
*   **Action:** Convert the relatively scaled merged diffuse map (`VoxelData_relative`) to absolute units and subtract theoretical incoherent (Compton) scattering.
*   **Input:**
    *   `VoxelData_relative` (from Module 3.3.D).
    *   `unitCellInventory`: Details of atomic composition of the unit cell (e.g., list of `Molecules`, `occupancies`).
    *   `Crystal`: Average crystal model (e.g., from an average of all `Experiment_dials_w.crystal`).
    *   `Reflections_dials_w` (all wedges): Used to create a merged, relatively scaled Bragg intensity list `I_bragg_relative(HKL_asu, s_mag)`. This requires applying the same relative scaling models (`ScalingModel_refined_list`) to the integrated Bragg intensities from DIALS.
*   **Process:**
    1.  **Theoretical Scattering Calculation (for various `s` values corresponding to radial shells):**
        *   `f_0_atom(s,Z)`: Atomic coherent scattering factor (from `model.atom.ScatteringFactor`).
        *   `I_incoh_atomic(s,Z)`: Atomic incoherent scattering intensity (from `model.atom.ScatteringFactor`).
        *   `I_coh_UC_theo(s) = Σ_atoms_in_UC (occupancy_atom * f_0_atom(s, Z_atom)^2)`
        *   `I_incoh_UC_theo(s) = Σ_atoms_in_UC (occupancy_atom * I_incoh_atomic(s, Z_atom))`
        *   `N_elec_UC = Σ_atoms_in_UC (occupancy_atom * Z_atom)`
    2.  **Radial Averaging (using a `StatisticsVsRadius`-like utility):**
        *   `I_obs_total_avg(s_shell) = RadiallyAvg(I_merged_relative(from_VoxelData_relative) + I_bragg_relative(from_Reflections_dials))`
        *   `I_theo_total_avg(s_shell) = RadiallyAvg(I_coh_UC_theo(s) + I_incoh_UC_theo(s))`
    3.  **Determine Absolute Scale Factor `Scale_Abs`:**
        *   `V_cell = Crystal.UnitCell.vCell`
        *   `Cumul_I_obs(s_cutoff) = V_cell * Integral_0^s_cutoff (I_obs_total_avg(s) * 4πs^2 ds)`
        *   `Cumul_I_theo(s_cutoff) = -N_elec_UC^2 + V_cell * Integral_0^s_cutoff (I_theo_total_avg(s) * 4πs^2 ds)` (includes forward scattering term)
        *   `Scale_Abs = Cumul_I_theo(s_cutoff) / Cumul_I_obs(s_cutoff)` (for a chosen `s_cutoff`).
    4.  **Apply Absolute Scale and Subtract Incoherent Scattering (to each entry in `VoxelData_relative`):**
        *   For each voxel `v` with `s_mag_center(v)`:
            *   `I_abs_diffuse(v) = I_merged_relative(v) * Scale_Abs - I_incoh_UC_theo(s_mag_center(v))`
            *   `Sigma_abs_diffuse(v) = Sigma_merged_relative(v) * Scale_Abs` (error propagation)
*   **Output:**
    *   `VoxelData_absolute`: The final diffuse map data, typically `(voxel_idx, H_center, K_center, L_center, s_mag_center, I_abs_diffuse, Sigma_abs_diffuse)`.
    *   `ScalingModel_final_list`: The `ScalingModel_refined_list` where parameters `b` and `c` are adjusted by `Scale_Abs` and `I_incoh_UC_theo` respectively, for consistency.
*   **Consistency Check:** `Scale_Abs` should be a reasonable positive number. `I_abs_diffuse` should be largely positive. Wilson plot of `I_abs_diffuse` (or `I_merged_relative * Scale_Abs`) vs `s` should align with `I_coh_UC_theo(s)` at high `s` after accounting for `I_incoh_UC_theo`.

---
