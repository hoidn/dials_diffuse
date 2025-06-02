**Detailed Plan for Diffuse Scattering Processing**

**Nomenclature:**

*   `I_raw(px, py, f)`: Raw intensity at detector pixel `(px, py)` for frame `f`.
*   `t_exp(f)`: Exposure time for frame `f`.
*   `DE_w`: `geom.DiffractionExperiment` object for wedge `w`.
*   `IS_w`: `io.ImageSeries` object for wedge `w`.
*   `Grid_w`: `grid.Sub3dRef` object for wedge `w`.
*   `Mask_pixel(px, py)`: Boolean mask for good detector pixels.
*   `v`: Voxel index in `Grid_w`.
*   `p`: Detector pixel index.
*   `s`: Scattering vector magnitude.
*   `H,K,L`: Integer Miller indices.
*   `h,k,l`: Continuous (fractional) Miller indices.
*   `w`: Wedge index.
*   `i`: Still/shot index (for potential stills adaptation, but primarily `w` for rotation).
*   `obs`: Observation index in a combined table.

---

**Phase 1: Geometry, Grid Definition, and Initial Masking**

**Module 1.0: Experimental Setup Definition**
*   **Action:** Define/load experimental geometry and image series information for each data wedge.
*   **Input:** Raw image files, XDS output files (e.g., `XDS.INP`, `INTEGRATE.LP`), or CBF headers.
*   **Process:**
    1.  Utilize `+proc/@Batch/xds2geom.m` or `+proc/@Batch/cbf2geom.m`.
    2.  Parse input files to populate `geom.DiffractionExperiment` objects (`DE_w`) for each wedge `w`, containing:
        *   `DE_w.Source`: Wavelength `λ`, incident beam direction `s⃗₀`, polarization.
        *   `DE_w.Spindle`: Rotation axis, starting angle `φ_start`, oscillation per frame `Δφ`.
        *   `DE_w.Crystal`: Unit cell parameters (`a,b,c,α,β,γ`), space group, initial orientation matrix `U_w`.
        *   `DE_w.Detector`: Pixel size (`q_x, q_y`), dimensions (`N_px, N_py`), distance `D`, origin (`org_x, org_y`), orientation `E_det`.
    3.  Define `io.ImageSeries` objects (`IS_w`) for each wedge, linking to raw image files `I_raw(px, py, f)`.
*   **Output:** `DE_w` (list), `IS_w` (list).
*   **Consistency Check:** Verify reasonableness of geometric parameters.

**Module 1.1: Pixel Masking**
*   **Action:** Create a static 2D detector mask.
*   **Input:** `DE_w.Detector`, `I_raw(px, py, first_frame)`.
*   **Process:**
    1.  `Mask_static(px,py)`: From detector properties (gaps, edges, e.g., `geom.Pilatus6m.isGapPixel()`).
    2.  `Mask_beamstop(px,py)`: User-defined or from geometry.
    3.  `Mask_virtual_corrected(px,py)`: If applicable (e.g., `geom.Pilatus6m.isVirtualPixel()`).
    4.  `Mask_negative_counts(px,py) = (I_raw(px,py,first_frame) >= 0)`.
    5.  `Mask_pixel(px,py) = Mask_static & Mask_beamstop & Mask_virtual_corrected & Mask_negative_counts`.
*   **Output:** `Mask_pixel(px,py)`.
*   **Consistency Check:** Visually inspect the mask. Ensure it covers known bad regions without excessively masking good data.

**Module 1.2: Voxel Grid Definition and Statistical Filtering (per wedge `w`)**
*   **Action:** Define a 3D reciprocal space grid and identify outlier voxels (e.g., Bragg peaks).
*   **Input:** `DE_w`, `IS_w`, `Mask_pixel`. Options: `s_max`, `ndiv`, `filter_window`, `max_count_hist`.
*   **Process:**
    1.  **HKL Prediction:** For each frame `f` in `IS_w`, for each pixel `p=(px,py)`:
        `(h(p,f), k(p,f), l(p,f)) = DE_w.frame2hkl(f)` (using pixel `p` coords).
    2.  **Grid Initialization:** Create `Grid_w = grid.Sub3dRef()`:
        *   `Grid_w.ref`: Set of unique integer `(H,K,L)` covered by `(h(p,f), k(p,f), l(p,f))` within `s_max`.
        *   `Grid_w.ndiv = [n_h, n_k, n_l]` (from options).
    3.  **Count Histogramming:** For each voxel `v` in `Grid_w`:
        *   `idx_voxel(p,f) = Grid_w.hkl2index(h(p,f), k(p,f), l(p,f))` (returns `rowIndex_ref`, `fractionIndex`).
        *   `CountHist_w(v, C+1) = Σ_{p,f where idx_voxel(p,f)=v AND Mask_pixel(p) AND I_raw(p,f)=C} (1)` for `C ≤ max_count_hist`.
        *   `Overflow_w(v) = Σ_{p,f where idx_voxel(p,f)=v AND Mask_pixel(p) AND I_raw(p,f)>max_count_hist} (I_raw(p,f))`.
    4.  **Statistical Filtering (using `proc.Filter`):**
        *   For each voxel `v`, define `Neighborhood(v)` using `Grid_w.movingWindow(filter_window, v)`.
        *   `MeanCountRate(v') = (Σ_{C=0}^{max} C * CountHist_w(v',C+1)) / (Σ_{C=0}^{max} CountHist_w(v',C+1))`.
        *   `Λ_v = WeightedMedian_{v' in Neighborhood(v)} (MeanCountRate(v'))` (weights are total pixels in voxel `v'`).
        *   `P_poisson(C | Λ_v) = exp(-Λ_v) * Λ_v^C / C!`.
        *   `P_obs(C | Neighborhood(v)) = (Σ_{v' in Neighborhood(v)} CountHist_w(v',C+1)) / (Σ_{C'} Σ_{v' in Neighborhood(v)} CountHist_w(v',C'+1))`.
        *   `DKL_v = Σ_C P_obs(C | Neighborhood(v)) * log(P_obs(C | Neighborhood(v)) / P_poisson(C | Λ_v))`.
        *   Iteratively identify `isOutlier_w(v)` by checking if removing voxel `v` from its neighborhoods reduces the sum of `DKL`s for those neighborhoods.
    5.  **Voxel Masking:** `Grid_w.voxelMask(v) = isOutlier_w(v) OR (Overflow_w(v) > overflow_threshold)`. (The `negateMask` property of `Grid_w` determines if `true` means include or exclude).
*   **Output:** `Grid_w` (list, one per wedge, with `voxelMask` populated).
*   **Consistency Check:** Visualize `Grid_w.voxelMask` in reciprocal space slices. Ensure Bragg peaks are masked. Check fraction of masked voxels.

---

**Phase 2: Intensity Integration and Correction**

**Module 2.1: Intensity Integration (per wedge `w`)**
*   **Action:** Sum raw intensities into voxels, respecting masks.
*   **Input:** `DE_w`, `IS_w`, `Grid_w`, `Mask_pixel`. Option: `binExcludedVoxels`.
*   **Process (for each voxel `v` in `Grid_w`):**
    *   `idx_voxel(p,f)` as in 1.2.3.
    *   If `NOT Grid_w.voxelMask(v)` (or `Grid_w.voxelMask(v)` if `negateMask` is false):
        *   `Counts_w(v) = Σ_{p,f where idx_voxel(p,f)=v} (I_raw(p,f) * Mask_pixel(p))`
        *   `Pixels_w(v) = Σ_{p,f where idx_voxel(p,f)=v} (Mask_pixel(p))`
        *   `N_frame_w(v) = Σ_{p,f where idx_voxel(p,f)=v} (f * Mask_pixel(p))`
        *   `iz_avg_w(v) = N_frame_w(v) / Pixels_w(v)`
    *   If `binExcludedVoxels` and voxel `v` *is* masked by `Grid_w.voxelMask`:
        *   Similar sums into `CountsExcl_w(v)`, `PixelsExcl_w(v)`, `iz_avg_Excl_w(v)`.
*   **Output:** `bin_list{w} = table(Counts_w, Pixels_w, iz_avg_w)`, `binExcl_list{w}`.
*   **Consistency Check:** `Counts_w(v) > 0` only if `Pixels_w(v) > 0`. `iz_avg_w(v)` within frame range of wedge.

**Module 2.2: Correction Factor Calculation (per wedge `w`)**
*   **Action:** Calculate geometric, experimental, and background corrections per pixel, then average per voxel.
*   **Input:** `DE_w`, `IS_w` (for `t_exp(f)`), `Grid_w`, `Mask_pixel`, `DE_bkg_w`, `IS_bkg_w` (if background run exists).
*   **Process:**
    1.  **Per-Pixel Corrections `CF(p)` for detector `p=(px,py)`:**
        *   `d(p), cosω(p) = geom.Corrections.calcGeometry(DE_w.Detector)`
        *   `SA(p) = geom.Corrections.solidAngle(DE_w.Detector.qx, DE_w.Detector.qy, d(p), cosω(p))`
        *   `Pol(p) = geom.Corrections.polarization(DE_w.Source.wavevector, ..., p_frac, pn_vec, x_lab(p), y_lab(p), z_lab(p), d(p))`
        *   `Eff(p) = geom.Corrections.efficiency(DE_w.Detector.sensorMaterial, ..., cosω(p))`
        *   `Att(p) = geom.Corrections.attenuation("air", ..., d(p))`
        *   `d3s_pix(p) = geom.Corrections.d3s(DE_w.Source, DE_w.Detector, DE_w.Spindle)`
        *   `sx_pix(p), sy_pix(p), sz_pix(p) = DE_w.s()` (scattering vector for pixel `p`)
        *   `ix_pix(p), iy_pix(p)` (pixel indices themselves)
    2.  **Background Image (if applicable):**
        *   `I_bkg_sum_w(p) = Σ_{f_bkg in IS_bkg_w} (I_raw_bkg(p, f_bkg))`
        *   `t_bkg_total_w = Σ_{f_bkg in IS_bkg_w} (t_exp_bkg(f_bkg))`
    3.  **Average Corrections per Voxel `v`:**
        *   `Multiplicity_w(v,p) = proc.Integrate.pixelMultiplicity(Grid_w, ...)`
        *   `CF_avg_w(v) = [Σ_p (Multiplicity_w(v,p) * CF(p))] / [Σ_p Multiplicity_w(v,p)]` (for SA, Pol, Eff, Att, d3s_pix, sx_pix, sy_pix, sz_pix, ix_pix, iy_pix).
        *   `dt_w = mean(t_exp(f) for f in IS_w)` (average signal exposure time for wedge).
        *   `Bkg_avg_counts_w(v) = [Σ_p (Multiplicity_w(v,p) * I_bkg_sum_w(p))] / [Σ_p Multiplicity_w(v,p)]`
        *   `BkgErr_avg_counts_w(v) = sqrt[Σ_p (Multiplicity_w(v,p)^2 * I_bkg_sum_w(p))] / [Σ_p Multiplicity_w(v,p)]`
*   **Output:** `corr_list{w}` table of averaged corrections, `dt_w`, `BkgDt_w = t_bkg_total_w`. `corrExcl_list{w}` if applicable.
*   **Consistency Check:** Correction factor maps should be smooth and physically reasonable.

---

**Phase 3: Scaling (Relative and Absolute) and Merging**

**Module 3.1: Initial Export & Per-Voxel Intensity Calculation**
*   **Action:** Combine integrated counts and corrections.
*   **Input:** `bin_list`, `corr_list`, `AverageGeometry` (averaged from all `DE_w`).
*   **Process (for each voxel `v` in wedge `w`):**
    1.  `TotalGeomCorr_w(v) = SA_avg_w(v) * Pol_avg_w(v) * Eff_avg_w(v) * Att_avg_w(v)`
    2.  `Rate_signal_w(v) = Counts_w(v) / (Pixels_w(v) * dt_w)`
    3.  `Rate_bkg_w(v) = Bkg_avg_counts_w(v) / BkgDt_w`
    4.  `I_obs_w(v) = (Rate_signal_w(v) - Rate_bkg_w(v)) / TotalGeomCorr_w(v)`
    5.  `Sigma_signal_raw_w(v) = sqrt(Counts_w(v)) / (Pixels_w(v) * dt_w)`
    6.  `Sigma_bkg_raw_w(v) = BkgErr_avg_counts_w(v) / BkgDt_w`
    7.  `Sigma_obs_w(v) = sqrt( (Sigma_signal_raw_w(v)/TotalGeomCorr_w(v))^2 + (Sigma_bkg_raw_w(v)/TotalGeomCorr_w(v))^2 )`
    8.  `Fraction_w(v) = (d3s_avg_w(v) * Pixels_w(v)) / (V_cell_recip / prod(Grid_w.ndiv))`
    9.  `(H_asu,K_asu,L_asu)_v = AverageCrystal.hkl2asu(H_grid(v), K_grid(v), L_grid(v))`
    10. `s_mag_v = sqrt(sx_avg_w(v)^2 + sy_avg_w(v)^2 + sz_avg_w(v)^2)`
    11. `panel_idx_v = AverageDetector.chipIndex(ix_avg_w(v), iy_avg_w(v))`
*   **Output:** `DiffuseTable_combined` (all `I_obs_w`, `Sigma_obs_w`, and derived coordinates/indices from all wedges). `ScalingModel_initial_list` (one per batch, initialized with default control points).
*   **Consistency Check:** `I_obs` generally positive. `Sigma_obs` reasonable.

**Module 3.2: Relative Scaling**
*   **Action:** Refine `ScalingModel` parameters (`a,b,c,d` control points) for each batch.
*   **Input:** `DiffuseTable_combined`, `ScalingModel_initial_list`.
*   **Process (Iterative, using `MultiBatchScaler`):**
    1.  **Reference:** `I_ref(HKL_asu) = WeightedAvg_obs [I_scaled_obs(HKL_asu)]` (current best merged intensity).
    2.  **Fit (e.g., for `b` in batch `j`'s `ScalingModel_j`):**
        Minimize `Σ_obs_in_batch_j [ (I_obs(obs) / (a_j*d_j) - (c_j/b_j + I_ref(HKL_asu_obs))) / (Sigma_obs(obs)/(a_j*d_j)) ]^2 + Regularization(b_j_control_points)`
        where `a_j,b_j,c_j,d_j` are functions of `(ix,iy,iz,panel,s_mag)` evaluated from `ScalingModel_j`.
*   **Output:** `ScalingModel_refined_list`.
*   **Consistency Check:** Convergence of fit. Smoothness and physical plausibility of `a,b,c,d` functions.

**Module 3.3: Merging Relatively Scaled Data**
*   **Action:** Apply refined scales and merge unique `(H_asu,K_asu,L_asu)`.
*   **Input:** `DiffuseTable_combined`, `ScalingModel_refined_list`.
*   **Process:**
    1.  For each `obs` in `DiffuseTable_combined` from batch `j`:
        *   `Scale_mult_obs = a_j(obs) * b_j(obs) * d_j(obs)`
        *   `Offset_add_obs = c_j(obs) / b_j(obs)`
        *   `I_rel_obs = I_obs(obs) / Scale_mult_obs - Offset_add_obs`
        *   `Sigma_rel_obs = Sigma_obs(obs) / Scale_mult_obs`
    2.  For each unique `(H_asu,K_asu,L_asu)`:
        *   `w_rel_obs = 1 / Sigma_rel_obs^2`
        *   `I_merged_relative(HKL_asu) = Σ(I_rel_obs * w_rel_obs) / Σ(w_rel_obs)`
        *   `Sigma_merged_relative(HKL_asu) = sqrt(1 / Σ(w_rel_obs))`
*   **Output:** `hklMerge_relative` table.
*   **Consistency Check:** R-merge statistics. Distribution of residuals `(I_rel_obs - I_merged_relative) / Sigma_rel_obs`.

---

**Phase 4: Absolute Scaling**

**Module 4.1: Absolute Scaling and Incoherent Subtraction**
*   **Action:** Convert to absolute units and subtract Compton scattering.
*   **Input:** `hklMerge_relative`, `unitCellInventory` (`Molecules`, `occupancies`, `Crystal`), `braggTable_relative` (if available).
*   **Process:**
    1.  **Theoretical Scattering (for `s` values corresponding to radial shells):**
        *   `f_0_atom(s,Z)` from `model.atom.ScatteringFactor`.
        *   `I_incoh_atomic(s,Z)` from `model.atom.ScatteringFactor`.
        *   `I_coh_UC(s) = Σ_atoms (occ * f_0_atom^2)`
        *   `I_incoh_UC_theo(s) = Σ_atoms (occ * I_incoh_atomic)`
        *   `N_elec_UC = Σ_atoms (occ * Z)`
    2.  **Radial Averaging (using `StatisticsVsRadius`):**
        *   `I_obs_total_avg(s_shell) = RadiallyAvg(I_merged_relative + I_bragg_relative)`
        *   `I_theo_total_avg(s_shell) = RadiallyAvg(I_coh_UC(s) + I_incoh_UC_theo(s))`
    3.  **Absolute Scale Factor:**
        *   `V_cell = Crystal.UnitCell.vCell`
        *   `Cumul_I_obs(s_cut) = V_cell * Integral_0^s_cut (I_obs_total_avg(s) * 4πs^2 ds)`
        *   `Cumul_I_theo(s_cut) = -N_elec_UC^2 + V_cell * Integral_0^s_cut (I_theo_total_avg(s) * 4πs^2 ds)`
        *   `Scale_Abs = Cumul_I_theo(scutoff) / Cumul_I_obs(scutoff)`
    4.  **Final Diffuse Map (for each `HKL_asu` in `hklMerge_relative`):**
        *   `I_abs_diffuse(HKL_asu) = I_merged_relative(HKL_asu) * Scale_Abs - I_incoh_UC_theo(s_at_HKL_asu)`
        *   `Sigma_abs_diffuse(HKL_asu) = Sigma_merged_relative(HKL_asu) * Scale_Abs`
*   **Output:** `hklMerge_absolute` (final diffuse map data). `ScalingModel_final_list` (updated with `Scale_Abs`).
*   **Consistency Check:** `Scale_Abs` should be a reasonable positive number. `I_abs_diffuse` should be largely positive. Wilson plot of `I_abs_diffuse` might be inspected.

