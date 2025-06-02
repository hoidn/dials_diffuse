
**Assumptions:**

*   We are following the rotation data processing pipeline outlined by the MATLAB package.
*   `I_raw(px, py, f)`: Raw intensity at detector pixel `(px, py)` for frame `f`.
*   `t_exp(f)`: Exposure time for frame `f`.
*   `t_bkg_total`: Total exposure time for the summed background image.
*   `(H, K, L)`: Integer Miller indices of a reciprocal lattice point (RLP).
*   `(h, k, l)`: Continuous coordinates in reciprocal space (often fractional Miller indices).
*   `s`: Scattering vector, `s = k_scattered - k_incident`. `|s| = 2sin(θ)/λ`.
*   `v`: Index for a voxel in the 3D reciprocal space grid.
*   `p`: Index for a pixel on the 2D detector.
*   `w`: Index for a wedge of data.

---

**Phase 1: Geometry Definition and Initial Data Preparation**

**Step 0: Define Experimental Geometry and Image Series (Conceptual Pre-step)**
*   **Action:** Load/define `geom.DiffractionExperiment` (DE) and `io.ImageSeries` (IS).
*   **Math:** Not an explicit calculation on intensities, but defines parameters used later.
    *   `DE_w`: `DiffractionExperiment` object for wedge `w` (includes `Crystal_w`, `Detector_w`, `Source_w`, `Spindle_w`).
    *   `IS_w`: `ImageSeries` object for wedge `w`.

**Step 1: Voxel Grid Definition and Filtering (`+proc/@Batch/filter.m`)**

*   **1a. Pixel Masking:**
    *   **Action:** Identify bad/untrusted detector pixels.
    *   **Math:** `Mask_pixel(px, py)` = boolean (true if pixel is good).
        *   `Mask_pixel = Mask_static & Mask_beamstop & Mask_gaps & Mask_virtual_corrected & (I_raw_first_frame >= 0)`
*   **1b. HKL Prediction & Grid Initialization (per wedge `w`):**
    *   **Action:** Determine reciprocal space coverage for `DE_w` and define `grid.Sub3dRef` object `Grid_w`.
    *   **Math (conceptual):**
        *   For each frame `f` in `IS_w`, for each detector pixel `(px, py)`:
            `(h(px,py,f), k(px,py,f), l(px,py,f)) = DE_w.frame2hkl(f)` using pixel `(px,py)`.
        *   `Grid_w.ref`: Set of unique integer `(H,K,L)` covered.
        *   `Grid_w.ndiv`: Subdivisions `[n_h, n_k, n_l]`.
*   **1c. Count Histogramming (per wedge `w`, for `Grid_w`):**
    *   **Action:** For each voxel `v` in `Grid_w`, count how many pixels `p` mapping to it have a certain intensity `I_raw(p,f)`.
    *   **Math:**
        *   `idx_voxel(p,f) = Grid_w.hkl2index(h(p,f), k(p,f), l(p,f))` (maps pixel `p` at frame `f` to a voxel index).
        *   `CountHist_w(v, count_val+1) = sum_{p,f where idx_voxel(p,f)=v and I_raw(p,f)=count_val} (1)` if `I_raw(p,f) <= maxCount`.
        *   `Overflow_w(v) = sum_{p,f where idx_voxel(p,f)=v and I_raw(p,f)>maxCount} (I_raw(p,f))`.
*   **1d. Statistical Filtering (`proc.Filter` on `CountHist_w`):**
    *   **Action:** Identify outlier voxels (likely Bragg peaks).
    *   **Math (conceptual for voxel `v` in wedge `w`):**
        *   `Lambda_v = WeightedMedian_{v' in Neighborhood(v)} ( MeanCountRate(v') )`
        *   `P_poisson(c | Lambda_v) = exp(-Lambda_v) * Lambda_v^c / c!`
        *   `P_observed(c | Neighborhood(v)) = CountHist_Neighborhood(v, c+1) / sum(CountHist_Neighborhood(v, :))`
        *   `DKL_v = sum_c P_observed(c) * log(P_observed(c) / P_poisson(c | Lambda_v))`
        *   Iteratively identify `isOutlier_w(v)` based on `DKL_v` and change in DKL upon removal.
    *   `Grid_w.voxelMask(v)` = `isOutlier_w(v)` OR (`Overflow_w(v) > threshold`).

---

**Phase 2: Intensity Integration and Correction**

**Step 2: Intensity Integration into Voxels (`+proc/@Batch/integrate.m`)**

*   **Action (per wedge `w`):** Sum intensities from `IS_w` into voxels of `Grid_w`, respecting `Grid_w.voxelMask`.
*   **Math (for voxel `v` in wedge `w`):**
    *   `Counts_w(v) = sum_{p,f where idx_voxel(p,f)=v AND NOT Grid_w.voxelMask(v)} (I_raw(p,f) * Mask_pixel(p))`
    *   `Pixels_w(v) = sum_{p,f where idx_voxel(p,f)=v AND NOT Grid_w.voxelMask(v)} (Mask_pixel(p))`
    *   `N_frame_weighted_w(v) = sum_{p,f where idx_voxel(p,f)=v AND NOT Grid_w.voxelMask(v)} (f * Mask_pixel(p))`
    *   `iz_avg_w(v) = N_frame_weighted_w(v) / Pixels_w(v)`
    *   Similar sums for `CountsExcl_w(v)`, `PixelsExcl_w(v)` if `Grid_w.voxelMask(v)` is true.
    *   **Output:** `bin{w} = table(Counts_w, Pixels_w, iz_avg_w)`

**Step 3: Calculation of Correction Factors (`+proc/@Batch/correct.m`)**

*   **3a. Per-Pixel Geometric Corrections (per wedge `w`, for `Detector_w`, `Source_w`):**
    *   **Action:** Calculate standard corrections for each pixel `p`.
    *   **Math:**
        *   `SolidAngle_w(p) = geom.Corrections.solidAngle(...)`
        *   `Polarization_w(p) = geom.Corrections.polarization(...)`
        *   `Efficiency_w(p) = geom.Corrections.efficiency(...)`
        *   `Attenuation_w(p) = geom.Corrections.attenuation(...)` (typically for air)
        *   `d3s_w(p) = geom.Corrections.d3s(Source_w, Detector_w, Spindle_w)` (swept volume per pixel per frame)
        *   `(sx_w(p), sy_w(p), sz_w(p)) = DE_w.s()` using pixel `p`.
*   **3b. Background Image Processing (per wedge `w`, if `readBackground`):**
    *   **Action:** Read and sum dedicated background images corresponding to `IS_w`.
    *   **Math:**
        *   `I_bkg_sum_w(p) = sum_{f_bkg in IS_bkg_w} (I_raw_bkg(p, f_bkg))`
        *   `t_bkg_total_w = sum_{f_bkg in IS_bkg_w} (t_exp_bkg(f_bkg))`
*   **3c. Averaging Corrections per Voxel (per wedge `w`):**
    *   **Action:** Average per-pixel corrections over all pixels `p` contributing to each voxel `v`.
    *   **Math (for a generic correction `CF(p)` and voxel `v`):**
        *   `Multiplicity_w(v,p)`: Number of times pixel `p` maps to voxel `v` in wedge `w` (from `Integrater.pixelMultiplicity`).
        *   `CF_avg_w(v) = [sum_p (Multiplicity_w(v,p) * CF(p))] / [sum_p Multiplicity_w(v,p)]`
        *   This is done for `SolidAngle_avg_w(v)`, `Polarization_avg_w(v)`, etc., and also for:
            *   `sx_avg_w(v)`, `sy_avg_w(v)`, `sz_avg_w(v)`
            *   `ix_avg_w(v)`, `iy_avg_w(v)` (average detector pixel contributing)
            *   `d3s_avg_w(v)`
            *   `Bkg_avg_counts_w(v) = [sum_p (Multiplicity_w(v,p) * I_bkg_sum_w(p))] / [sum_p Multiplicity_w(v,p)]`
            *   `BkgErr_avg_counts_w(v) = sqrt[sum_p (Multiplicity_w(v,p)^2 * I_bkg_sum_w(p))] / [sum_p Multiplicity_w(v,p)]` (error of weighted mean)
    *   **Output:** `corr{w}` table containing these averaged corrections, `dt_w` (avg exposure of signal frames), `BkgDt_w` (total background exposure).

---

**Phase 3: Scaling (Relative and Absolute) and Merging**

**Step 4: Initial Export and Combination of Batches (`+proc/@Batch/export.m`, then `+proc/@Batch/combine.m`)**

*   **4a. Per-Wedge Intensity Calculation (`exportScript`):**
    *   **Action:** Calculate initial intensity per voxel before inter-wedge scaling.
    *   **Math (for voxel `v` in wedge `w`):**
        *   `TotalGeomCorr_w(v) = SolidAngle_avg_w(v) * Polarization_avg_w(v) * Efficiency_avg_w(v) * Attenuation_avg_w(v)`
        *   `Intensity_raw_w(v) = Counts_w(v) / (Pixels_w(v) * TotalGeomCorr_w(v) * dt_w)`
        *   `Sigma_raw_w(v) = sqrt(Counts_w(v)) / (Pixels_w(v) * TotalGeomCorr_w(v) * dt_w)`
        *   `BkgIntensity_raw_w(v) = Bkg_avg_counts_w(v) / (TotalGeomCorr_w(v) * BkgDt_w)` (assuming same geom. corr. for bkg)
        *   `BkgSigma_raw_w(v) = BkgErr_avg_counts_w(v) / (TotalGeomCorr_w(v) * BkgDt_w)`
        *   `I_obs_w(v) = Intensity_raw_w(v) - BkgIntensity_raw_w(v)`
        *   `Sigma_obs_w(v) = sqrt(Sigma_raw_w(v)^2 + BkgSigma_raw_w(v)^2)`
        *   `Fraction_w(v) = (d3s_avg_w(v) * Pixels_w(v)) / GridVoxelVolume_w(v)`
    *   **Output:** `diffuseTable_w` with `(H,K,L)`, `I_obs_w`, `Sigma_obs_w`, `sx_avg_w`, `sy_avg_w`, `sz_avg_w`, `ix_avg_w`, `iy_avg_w`, `iz_avg_w`, `Fraction_w`, `wedge_idx=w`.
*   **4b. Combine Batch Data (`combineScript`):**
    *   **Action:** Concatenate all `diffuseTable_w` into one `DiffuseTable_combined`.
    *   **Math:**
        *   `(H_asu, K_asu, L_asu)_v = AverageCrystal.hkl2asu((H,K,L)_v)`
        *   `s_mag_v = sqrt(sx_avg_v^2 + sy_avg_v^2 + sz_avg_v^2)`
        *   `panel_idx_v = AverageDetector.chipIndex(ix_avg_v, iy_avg_v)`
    *   **Output:** `DiffuseTable_combined` (with new columns), `ScalingModel_initial_list` (one per original batch, initialized).

**Step 5: Relative Scaling (`+proc/@Batch/scale.m`)**

*   **Action:** Refine `ScalingModel_initial_list` parameters (`a,b,c,d` control points) iteratively.
*   **Math (conceptual for one iteration and one batch `j`'s `ScalingModel_j`):**
    *   `I_merged_reference(H_asu,K_asu,L_asu)` is the current best estimate of merged intensities (from `MultiBatchScaler.merge()`).
    *   For each observation `obs` in batch `j` (mapping to `HKL_asu`):
        *   `a_obs = ScalingModel_j.aVal(ix_obs, iy_obs, iz_obs)`
        *   `b_obs = ScalingModel_j.bVal(iz_obs)`
        *   `c_obs = ScalingModel_j.cVal(s_mag_obs, iz_obs)`
        *   `d_obs = ScalingModel_j.dVal(panel_idx_obs)`
        *   `Scale_mult_obs = a_obs * b_obs * d_obs`
        *   `Offset_add_obs = c_obs / b_obs`
        *   `I_predicted_obs = Scale_mult_obs * (I_merged_reference(HKL_asu) + Offset_add_obs)`
        *   **Fitting `b` (example):** Minimize `sum_obs [ (I_obs(obs)/ (a_obs*d_obs) - (Offset_add_obs_using_b_current_guess + I_merged_reference(HKL_asu))) / (Sigma_obs(obs)/(a_obs*d_obs)) ]^2` by adjusting control points of `b`. Regularization terms (smoothness) are added to the minimization target. Similar for `a, c, d`.
    *   **Output:** `ScalingModel_refined_list`.

**Step 6: Merging Relatively Scaled Data (`+proc/@Batch/merge.m`)**

*   **Action:** Apply refined scales and merge.
*   **Math (for each observation `obs` in `DiffuseTable_combined` belonging to batch `j`):**
    *   `a_obs, b_obs, c_obs, d_obs` from `ScalingModel_refined_list(j)`.
    *   `Scale_mult_obs = a_obs * b_obs * d_obs`
    *   `Offset_add_obs = c_obs / b_obs`
    *   `I_scaled_obs = I_obs(obs) / Scale_mult_obs - Offset_add_obs`
    *   `Sigma_scaled_obs = Sigma_obs(obs) / Scale_mult_obs`
    *   **Merge (for each unique `HKL_asu`):**
        *   `w_obs = 1 / Sigma_scaled_obs^2`
        *   `I_merged(HKL_asu) = sum(I_scaled_obs * w_obs) / sum(w_obs)`
        *   `Sigma_merged(HKL_asu) = sqrt(1 / sum(w_obs))`
    *   (Outlier rejection can be done by re-weighting based on residuals against `I_merged` and re-merging).
    *   **Output:** `hklMerge_relative` table.

---

**Phase 4: Absolute Scaling**

**Step 7: Absolute Scaling (`+proc/@Batch/rescale.m`)**

*   **7a. Theoretical Scattering Calculation:**
    *   **Action:** Calculate theoretical coherent and incoherent scattering from `unitCellInventory`.
    *   **Math (for given `s`):**
        *   `f_0_atom(s, Z) = model.atom.ScatteringFactor(atom_symbol).f_coh(s)`
        *   `I_incoh_atomic(s, Z) = model.atom.ScatteringFactor(atom_symbol).I_incoh(s)`
        *   `I_coh_UC(s) = sum_atoms_in_UC (occupancy_atom * f_0_atom(s, Z_atom)^2)`
        *   `I_incoh_UC(s) = sum_atoms_in_UC (occupancy_atom * I_incoh_atomic(s, Z_atom))`
        *   `(Optionally) Ibond_UC(s) = ...` (from bonding interference terms)
        *   `N_elec_UC = sum_atoms_in_UC (occupancy_atom * Z_atom)`
*   **7b. Radial Averaging of Observed and Theoretical Data:**
    *   **Action:** Use `proc.script.StatisticsVsRadius`.
    *   **Math:**
        *   `I_obs_total_avg(s_shell) = RadiallyAvg(I_merged_diffuse(HKL_asu) + I_merged_Bragg(HKL_asu))` (from `hklMerge_relative` and `hklMergeBragg_relative`).
        *   `I_theo_total_avg(s_shell) = RadiallyAvg(I_coh_UC(s) [+ Ibond_UC(s)] + I_incoh_UC(s))`
*   **7c. Determine Absolute Scale Factor:**
    *   **Action:** Compare cumulative sums.
    *   **Math:**
        *   `V_cell = Crystal.UnitCell.vCell`
        *   `Cumul_I_obs(s_max_cutoff) = V_cell * sum_{s_shell <= s_max_cutoff} ( I_obs_total_avg(s_shell) * Volume_shell(s_shell) )`
        *   `Cumul_I_theo(s_max_cutoff) = -N_elec_UC^2 + V_cell * sum_{s_shell <= s_max_cutoff} ( I_theo_total_avg(s_shell) * Volume_shell(s_shell) )` (Forward scattering theorem term `-N_elec_UC^2`)
        *   `Scale_Absolute = Cumul_I_theo(scutoff) / Cumul_I_obs(scutoff)`
*   **7d. Apply Absolute Scale and Subtract Incoherent:**
    *   **Action:** Create final diffuse map.
    *   **Math (for each `HKL_asu` in `hklMerge_relative`):**
        *   `I_abs_diffuse(HKL_asu) = I_merged_diffuse(HKL_asu) * Scale_Absolute - I_incoh_UC(s_at_HKL_asu)`
        *   `Sigma_abs_diffuse(HKL_asu) = Sigma_merged_diffuse(HKL_asu) * Scale_Absolute`
    *   **Output:** `hklMerge_absolute` table (the final diffuse map data).
*   **7e. Update ScalingModel (for consistency if used later):**
    *   `ScalingModel_final(j).b = ScalingModel_refined_list(j).b / Scale_Absolute`
    *   `ScalingModel_final(j).c = ScalingModel_refined_list(j).c + I_incoh_UC_on_c_grid * (ScalingModel_refined_list(j).b / Scale_Absolute)`

This detailed trace shows the transformation of raw pixel intensities through various corrections, averaging, scaling (relative and absolute), and finally incoherent subtraction to yield the diffuse scattering map on an absolute scale. Each step involves specific mathematical operations and relies on the geometric and physical models defined within the package.
