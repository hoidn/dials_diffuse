**Phase Adaptations for Stills Processing**

The fundamental goal remains the same: convert raw images to an absolutely scaled diffuse map. However, the absence of continuous rotation and the shot-to-shot heterogeneity of stills necessitate significant changes.

**Overarching Changes for Stills:**

*   **No Wedges:** Each still is processed individually or in small, strategically grouped batches. The concept of a "wedge" `w` is replaced by a "still index" or "shot ID" `i`.
*   **Per-Still Orientation:** Crystal orientation `U_i` (and potentially cell parameters) must be determined for *each still* `i` via indexing (likely an external step).
*   **Partiality:** Bragg reflections on stills are almost always partial. A partiality factor `P_spot` for each observed Bragg spot is essential for quantitative analysis and scaling. This also likely comes from external indexing/integration software.
*   **Data Sparsity:** A single still provides a very sparse sampling of reciprocal space. Merging data from many stills is crucial.

---

**Revised Phase Breakdown for Stills Processing:**

**Phase 1: Per-Still Geometry, Indexing, and Initial Masking**

**Step 0: Define Experimental Geometry and Image Series (Per Still)**
*   **Action:** Load/define `geom.DiffractionExperiment` (DE_base - common detector, source) and `io.ImageSeries` (IS_stills - list of still images).
*   **Math:**
    *   `DE_base`: Contains `Detector`, `Source`. `Spindle` is not used for rotation.
    *   `IS_stills`: Manages access to individual still image files `I_raw(px, py, i)`.

**Step 1: Indexing and Bragg Spot Identification (External or New Module - CRITICAL)**
*   **Action:** For each still `i`:
    *   Find Bragg spots.
    *   Determine crystal orientation `U_i` and unit cell `Cell_i`. (Tools like CrystFEL's `indexamajig` or DIALS for stills).
    *   Integrate Bragg spots, obtaining `I_bragg_obs(spot)`, `Sigma_bragg_obs(spot)`, and partiality `P_spot`.
*   **Math:**
    *   Output for still `i`: `{U_i, Cell_i, List_of_Spots[(HKL_spot, I_bragg_obs_spot, Sigma_bragg_obs_spot, P_spot, px_spot, py_spot)]}`.

**Step 1b (was 1a): Pixel Masking (Per Still, if needed, or global)**
*   **Action:** Apply static bad pixel masks.
*   **Math:** `Mask_pixel(px, py)` (as before, but applied to each still image).

**Step 1c (was 1d, adapted): Bragg Peak Masking on 2D Detector Image (Per Still)**
*   **Action:** For each still `i`, create a 2D mask `Mask_bragg_2D_i(px, py)` to exclude regions around indexed Bragg spots.
*   **Math:**
    *   For each `spot` in `List_of_Spots_i`:
        *   Predict spot shape/extent on detector based on `(px_spot, py_spot)`, `U_i`, `Cell_i`, beam divergence, mosaicity.
        *   `Mask_bragg_2D_i(pixels_near_spot) = true`.
*   **Output:** `Mask_total_2D_i(px,py) = Mask_pixel(px,py) & ~Mask_bragg_2D_i(px,py)`. This mask identifies pixels containing primarily diffuse scattering for still `i`.

---

**Phase 2: Per-Still Intensity Correction and Diffuse Extraction**

**Step 2 (was 2 & 3a): Per-Still Diffuse Pixel Correction and Extraction**
*   **Action (per still `i`):** Extract and correct diffuse pixel intensities.
*   **Math (for each pixel `p = (px,py)` in still `i`):**
    *   If `Mask_total_2D_i(p)` is true (i.e., good diffuse pixel):
        *   `TotalGeomCorr_i(p) = geom.Corrections.total(Source, Detector_at_still_i)` (Detector geometry might have slight per-still refinements if available from indexing).
        *   `I_diffuse_pixel_raw_i(p) = I_raw(p,i) / (TotalGeomCorr_i(p) * t_exp(i))`
        *   `Sigma_diffuse_pixel_raw_i(p) = sqrt(I_raw(p,i)) / (TotalGeomCorr_i(p) * t_exp(i))` (simplified error)
        *   `(h_i(p), k_i(p), l_i(p)) = DE_base.phi2hkl(0)` using `Crystal_i` (with orientation `U_i`) and `Spindle` (with `phi=0` as reference for still).
        *   `s_mag_i(p) = |s_vector_for_pixel_p_still_i|`
        *   `panel_idx_i(p) = Detector.chipIndex(px,py)`
*   **Output:** For each still `i`, a list of diffuse pixels: `DiffusePixels_i = [{h,k,l, s_mag, I_raw, Sigma_raw, panel_idx, px, py, ...}]_i`.
*   **Dedicated Background Subtraction:** If blank shots `I_raw_bkg(p, shot_bkg)` are available and can be reliably matched or averaged:
    *   `I_bkg_pixel_rate(p) = [sum(I_raw_bkg(p,shot_bkg))/t_exp_bkg] / TotalGeomCorr_i(p)`
    *   `I_diffuse_pixel_raw_i(p) = I_diffuse_pixel_raw_i(p) - I_bkg_pixel_rate(p)` (propagate errors).

---

**Phase 3: Scaling (Relative and Absolute) and Merging of Stills**

**Step 3 (was 4b & 5): Define Ordering Parameter and Initialize Scaling Models**
*   **Action:**
    *   Choose an ordering parameter `p_order(i)` for each still `i` (e.g., timestamp, total intensity, refined crystal volume from indexing, FEL pulse energy).
    *   Initialize `ScalingModel_initial_list` (one per still `i`, or one per group if stills are pre-clustered).
    *   The `iz` dimension of interpolators in `ScalingModel` is now conceptually `p_order`.
*   **Math:**
    *   `ScalingModel_i.izLim = [min(p_order), max(p_order)]` (or range for the specific group).
    *   `ScalingModel_i.a, .b, .c` control points are initialized (e.g., `a=1, b=1, c=0`).
    *   `ScalingModel_i.d` (panel scales) can be global.

**Step 4 (was 5 & 6): Relative Scaling of Stills using `p_order`**
*   **Action:** Iteratively refine `ScalingModel_i` parameters.
*   **Math (conceptual for one iteration):**
    1.  **Reference Generation (Crucial Adaptation):**
        *   **Bragg:** `I_bragg_merged_ref(HKL) = WeightedAvg_i [ I_bragg_obs_spot(HKL,i) / (P_spot * b_i * a_i(px_spot,py_spot,p_order(i)) * d_i(panel_spot)) ]` (This is complex; `b_i, a_i, d_i` are current estimates. Partiality `P_spot` is key).
        *   **Diffuse:** For each voxel `v` in a common 3D grid:
            `I_diffuse_merged_ref(v) = WeightedAvg_i [ (I_diffuse_pixel_raw_i(p) - c_i(s_mag(p), p_order(i))/b_i(p_order(i)) ) / (a_i(px,py,p_order(i)) * d_i(panel)) ]` for all pixels `p` from still `i` that map to voxel `v` (using orientation `U_i`).
    2.  **Refine `ScalingModel_i` (for each still `i` or group):**
        *   Minimize residuals against the *current* `I_bragg_merged_ref` and/or `I_diffuse_merged_ref`.
        *   The fitting for `b_i(p_order(i))`, `c_i(|s|, p_order(i))`, `a_i(px,py,p_order(i))` now uses `p_order(i)` as the input to the "frame-like" dimension of the interpolators.
        *   Regularization ensures smoothness of scale factors with respect to `p_order`.
*   **Output:** `ScalingModel_refined_list`.

**Step 5 (was 6): Merging Relatively Scaled Still Data**
*   **Action:** Apply final refined scales and merge diffuse data onto a 3D grid. Merge Bragg data.
*   **Math:**
    *   **Diffuse (for each pixel `p` in still `i`):**
        *   `Scale_mult_i(p) = a_i(px,py,p_order(i)) * b_i(p_order(i)) * d(panel_idx(p))`
        *   `Offset_add_i(p) = c_i(s_mag(p), p_order(i)) / b_i(p_order(i))`
        *   `I_diffuse_pixel_scaled_i(p) = I_diffuse_pixel_raw_i(p) / Scale_mult_i(p) - Offset_add_i(p)`
        *   `Sigma_diffuse_pixel_scaled_i(p) = Sigma_diffuse_pixel_raw_i(p) / Scale_mult_i(p)`
        *   Map these to a 3D grid `Grid_3D_diffuse` using orientation `U_i` and average contributions to each voxel `v`:
            `I_merged_diffuse_relative(v) = WeightedAvg_i,p_maps_to_v [I_diffuse_pixel_scaled_i(p)]`
            `Sigma_merged_diffuse_relative(v) = error_propagation(...)`
    *   **Bragg (for each spot in still `i`):**
        *   `I_bragg_corrected_i(spot) = I_bragg_obs_spot / (P_spot * Scale_mult_i(at_spot_center))` (offset usually not applied to Bragg).
        *   Merge these `I_bragg_corrected_i` across all stills for each unique HKL.
*   **Output:** `I_merged_diffuse_relative` (3D array), `hklMerge_Bragg_relative` (table).

---

**Phase 4: Absolute Scaling (Largely Similar, but on Merged Still Data)**

**Step 6 (was 7): Absolute Scaling**
*   **Action:** Scale the merged diffuse map and merged Bragg intensities to absolute units.
*   **Math:**
    *   **7a. Theoretical Scattering Calculation:** Same as before (using `unitCellInventory`).
        `I_coh_UC(s)`, `I_incoh_UC(s)`, `N_elec_UC`.
    *   **7b. Radial Averaging:**
        *   `I_obs_total_avg(s_shell) = RadiallyAvg(I_merged_diffuse_relative(v) + I_merged_Bragg_relative(HKL))`
        *   `I_theo_total_avg(s_shell)`: Same as before.
    *   **7c. Determine Absolute Scale Factor `Scale_Absolute`:** Same comparison of cumulative sums.
    *   **7d. Apply Absolute Scale and Subtract Incoherent (to 3D diffuse map):**
        *   `I_abs_diffuse_map(v) = I_merged_diffuse_relative(v) * Scale_Absolute - I_incoh_UC(s_at_voxel_v)`
        *   `Sigma_abs_diffuse_map(v) = Sigma_merged_diffuse_relative(v) * Scale_Absolute`
    *   **Output:** `DiffuseMap_absolute` (3D array).
    *   **7e. Update ScalingModel (Optional for Stills):** The per-still scaling models could be updated by `Scale_Absolute` and for `I_incoh_UC` if they are to be reused or represent the final model.

**Key Adaptations Summarized for Stills:**

1.  **Indexing and Partiality:** Essential upstream steps, likely external.
2.  **2D Bragg Masking:** Per-still, replacing 3D voxel filtering.
3.  **Diffuse Extraction:** From unmasked 2D pixels per still.
4.  **Scaling Model Dimension:** The "frame" (`iz`) dimension of interpolators for `a, b, c` is replaced by an "ordering parameter" `p_order(i)` specific to stills. This `p_order` must be chosen carefully to reflect expected smooth variations. If no such smooth variation is expected for a parameter (e.g., overall scale `b`), it becomes a per-still (or per-group) discrete parameter rather than an interpolated one.
5.  **Reference for Scaling:** Must be built from all stills (iteratively) or be an external model, due to data sparsity per still.
6.  **Merging:** Involves mapping 2D diffuse data from many differently oriented stills onto a common 3D grid.

This adapted workflow is significantly more complex, especially in the scaling and merging stages, due to the independent nature of each still shot. The success of using interpolation for scaling parameters hinges critically on finding a meaningful `p_order` that captures a dominant source of systematic variation across the still dataset.
