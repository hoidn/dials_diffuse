Agent Task: Implement Phase 3 of plan.md - Voxelization, Relative Scaling, and Merging
  Overall Goal for Agent (Phase 3): To take per-still, corrected diffuse pixel data from Phase 2, map it to a common 3D reciprocal space grid, apply relative scaling between stills to account for
  experimental variations, and merge the data into a final, relatively-scaled 3D diffuse scattering map.
  Checklist Usage Instructions for Agent:
  Copy this entire checklist into your working memory or a dedicated scratchpad area.
  Context Priming: Before starting a new major section (e.g., Module), carefully read all "Context Priming" items for that section.
  Sequential Execution: Address checklist items in the order presented.
  Update State: As you work on an item, change its state field:
  [ ] Open -> [P] In Progress when you start.
  [P] In Progress -> [D] Done when completed successfully.
  [P] In Progress -> [B] Blocked if you encounter a blocker. Add a note explaining the blocker in the "Details/Notes/Path" column.
  Record Details (in the "Details/Notes/Path" column):
  If a step requires creating or modifying a file, add the full relative path to that file (e.g., src/diffusepipe/voxelization/global_voxel_grid.py).
  If a significant design decision or clarification is made, note it briefly.
  For "IDL Definition/Review" tasks, summarize key interface aspects (Inputs, Outputs, Behavior, Errors).
  Include specific API function calls or class names from libdocs/dials/ where relevant.
  Iterative Review: Periodically re-read completed sections and notes.
  Save State: Ensure this checklist with current states and notes is saved if pausing.
  Phase 3: Voxelization, Relative Scaling, and Merging of Diffuse Data
  Item ID    Task Description    State    Details/Notes/Path (API Refs from libdocs/dials/)
  3.A    Context Priming (Phase 3)    [ ]
  3.A.1    Re-read plan.md Module 3.S.1: Global Voxel Grid Definition.    [ ]
  3.A.2    Re-read plan.md Module 3.S.2: Binning Corrected Diffuse Pixels.    [ ]    Focus on HDF5 backend, Welford's, ASU mapping.
  3.A.3    Re-read plan.md Module 3.S.3: Relative Scaling of Binned Observations.    [ ]    Focus on custom DiffuseScalingModel, v1 parameter-guarded model, DIALS components, partiality strategy.
  3.A.4    Re-read plan.md Module 3.S.4: Merging Relatively Scaled Data.    [ ]    Focus on error propagation and weighted merge.
  3.A.5    Review plan.md "Implementation Note on Efficient Pixel Data Handling" (updated parts for Phase 3).    [ ]
  3.A.6    Review src/diffusepipe/types/types_IDL.md and .py (esp. RelativeScalingConfig, StillsPipelineConfig).    [ ]
  3.A.7    Review relevant API docs: libdocs/dials/dxtbx_models.md (B.3 Crystal Model), crystallographic_calculations.md (C.2, C.6, C.7), dials_scaling.md (D.0), flex_arrays.md (D.3, D.4).    [ ]
  3.A.8    Understand Goal: Implement voxel grid creation, pixel binning (HDF5), custom relative scaling model, and merging.    [ ]
  3.B    Module 3.S.1: Global Voxel Grid Definition    [ ]
  3.B.idl    Define/Review IDL for GlobalVoxelGrid    [ ]    Path: src/diffusepipe/voxelization/global_voxel_grid_IDL.md (New) <br>Purpose: Encapsulate the 3D grid definition. <br>Inputs (constructor):
  List of Experiment_dials_i (for crystal models), List of CorrectedDiffusePixelData_i (for q-ranges), grid config (d_min_target, d_max_target, ndiv_h,k,l). <br>Attributes: Crystal_avg_ref, A_avg_ref,
  HKL range, subdivisions. <br>Methods: hkl_to_voxel_idx, voxel_idx_to_hkl_center, get_q_vector_for_voxel_center.
  3.B.1    Implement GlobalVoxelGrid class structure.    [ ]    Path: src/diffusepipe/voxelization/global_voxel_grid.py (New)
  3.B.2    Implement robust averaging of Crystal_i.get_unit_cell():    [ ]    API Ref: crystallographic_calculations.md (C.6 average_unit_cells using cctbx.uctbx.unit_cell). dxtbx_models.md (B.3
  crystal.get_unit_cell()).
  3.B.3    Implement robust averaging of Crystal_i.get_U() matrices to get U_avg_ref:    [ ]    Note: libdocs/dials/ doesn't specify quaternion averaging. May need direct scitbx or custom robust rotation
   matrix averaging. C.6 in crystallographic_calculations.md is for unit cells, not U matrices directly.
  3.B.4    Calculate B_avg_ref from averaged unit cell:    [ ]    API Ref: crystallographic_calculations.md (C.6 avg_B_matrix = matrix.sqr(avg_unit_cell.fractionalization_matrix()).transpose()).
  3.B.5    Calculate A_avg_ref = U_avg_ref * B_avg_ref.    [ ]
  3.B.6    Implement diagnostic: RMSD of Δhkl for Bragg reflections using (A_avg_ref)⁻¹.    [ ]
  3.B.7    Implement diagnostic: RMS misorientation of U_i vs U_avg_ref.    [ ]
  3.B.8    Transform all q_vectors (from Phase 2 output) to fractional HKLs using (A_avg_ref)⁻¹:    [ ]    API Ref: crystallographic_calculations.md (C.2 q_to_miller_indices_static(crystal_avg_ref,
  q_vector)).
  3.B.9    Determine overall HKL range for grid boundaries considering d_min_target, d_max_target.    [ ]
  3.B.10    Store grid parameters (HKL range, ndiv_h,k,l) and implement conversion methods (hkl_to_voxel_idx, voxel_idx_to_hkl_center, etc.).    [ ]
  3.B.T    Tests for GlobalVoxelGrid    [ ]    Path: tests/voxelization/test_global_voxel_grid.py (New)
  3.B.T.1    Test: Unit cell and U-matrix averaging with known inputs.    [ ]
  3.B.T.2    Test: HKL range determination from q-vectors.    [ ]
  3.B.T.3    Test: hkl_to_voxel_idx and voxel_idx_to_hkl_center conversions.    [ ]
  3.C    Module 3.S.2: Binning Corrected Diffuse Pixels into Global Voxel Grid    [ ]
  3.C.idl    Define/Review IDL for VoxelAccumulator    [ ]    Path: src/diffusepipe/voxelization/voxel_accumulator_IDL.md (New) <br>Purpose: Bin pixels, manage HDF5 storage. <br>Inputs
  (add_observations): voxel_idx, intensities_array, sigmas_array, still_ids_array, q_vectors_lab_array. <br>Methods: add_observations, get_observations_for_voxel, get_all_binned_data_for_scaling,
  finalize. <br>Behavior: Uses HDF5 (h5py, zstd).
  3.C.1    Implement VoxelAccumulator class with HDF5 backend (h5py, zstd).    [ ]    Path: src/diffusepipe/voxelization/voxel_accumulator.py (New)
  3.C.2    Implement binning logic: For each observation (q_lab_i, I_corr, Sigma_corr, still_id):    [ ]    Input: Per-still arrays from Phase 2 (final_q_vectors_still_i, final_I_corrected_still_i, etc.)
  3.C.2.a    Transform q_lab_i to hkl_frac using GlobalVoxelGrid.A_avg_ref.inverse():    [ ]    API Ref: crystallographic_calculations.md (C.2).
  3.C.2.b    Map hkl_frac to ASU using GlobalVoxelGrid.Crystal_avg_ref.get_space_group().info().map_to_asu():    [ ]    API Ref: crystallographic_calculations.md (C.7). dxtbx_models.md (B.3 for
  crystal.get_space_group()).
  3.C.2.c    Determine voxel_idx using GlobalVoxelGrid.hkl_to_voxel_idx().    [ ]
  3.C.2.d    Store (I_corr, Sigma_corr, still_id, q_lab_i) to VoxelAccumulator for that voxel_idx.    [ ]    This is BinnedPixelData_Global as per plan.md.
  3.C.3    Implement logic to create ScalingModel_initial_list (list of initial scaling model objects, one per still/group).    [ ]    Scaling models will be DIALS-based, e.g., instances of a
  PerStillMultiplierComponent.
  3.C.T    Tests for VoxelAccumulator    [ ]    Path: tests/voxelization/test_voxel_accumulator.py (New)
  3.C.T.1    Test: Adding observations to HDF5 and retrieving them.    [ ]
  3.C.T.2    Test: Correct HKL transformation, ASU mapping, and voxel indexing for sample q-vectors.    [ ]
  3.D    Module 3.S.3: Relative Scaling of Binned Observations    [ ]
  3.D.idl    Define/Review IDLs for DiffuseScalingModel, PerStillMultiplierComponent, ResolutionSmootherComponent    [ ]    Path: src/diffusepipe/scaling/diffuse_scaling_model_IDL.md (New) <br>Purpose:
  Custom scaling model for diffuse data. <br>Inheritance: DiffuseScalingModel(ScalingModelBase), components from ScaleComponentBase. <br>Inputs: BinnedPixelData_Global, Reflections_dials_i,
  RelativeScalingConfig. <br>Methods: refine_parameters, get_scales_for_observation.
  3.D.1    Implement PerStillMultiplierComponent(ScaleComponentBase) for b_i parameter.    [ ]    Path: src/diffusepipe/scaling/components/per_still_multiplier.py (New). API Ref: dials_scaling.md (D.0
  ScaleComponentBase, example SimpleMultiplicativeComponent).
  3.D.2    Implement ResolutionSmootherComponent(ScaleComponentBase) for `a(    q    )usingGaussianSmoother1D`.
  3.D.3    Implement DiffuseScalingModel(ScalingModelBase) to aggregate components.    [ ]    Path: src/diffusepipe/scaling/diffuse_scaling_model.py (New). V1: b_i and optional `a(
  3.D.4    Integrate with DIALS active parameter manager:    [ ]    API Ref: dials_scaling.md (D.0 active_parameter_manager, multi_active_parameter_manager).
  3.D.5    Implement Bragg reference generation:    [ ]    Filter Reflections_dials_i by P_spot >= P_min_thresh (from RelativeScalingConfig). API Ref: flex_arrays.md (D.3 access "partiality"). Weighted
  average I_bragg_obs_spot scaled by current ScalingModel_i.
  3.D.6    Implement diffuse reference generation from VoxelAccumulator data (weighted average, scaled by current ScalingModel_i).    [ ]
  3.D.7    Implement DiffuseScalingTarget function for residuals: Res = (I_obs / M_i) - I_ref.    [ ]
  3.D.8    Integrate with DIALS/ScitBX minimizer (e.g., Levenberg-Marquardt).    [ ]    API Ref: dials_scaling.md (D.0 example normal_eqns_solving.levenberg_marquardt_iterations from scitbx.lstbx).
  3.D.9    Implement iterative refinement loop with convergence checks.    [ ]
  3.D.T    Tests for Relative Scaling    [ ]    Path: tests/scaling/test_diffuse_scaling_model.py (New)
  3.D.T.1    Test: Individual scaling components (PerStillMultiplier, ResolutionSmoother).    [ ]
  3.D.T.2    Test: Bragg and diffuse reference generation logic.    [ ]
  3.D.T.3    Test: Scaling refinement with synthetic data (known scales, check recovery).    [ ]
  3.D.T.4    Test: Correct use of P_spot as quality filter (not divisor).    [ ]
  3.E    Module 3.S.4: Merging Relatively Scaled Data into Voxels    [ ]
  3.E.idl    Define/Review IDL for DiffuseDataMerger (or as methods in scaling model/orchestrator)    [ ]    Path: src/diffusepipe/merging/merger_IDL.md (New) or integrate into existing. <br>Purpose:
  Apply scales, merge. <br>Inputs: BinnedPixelData_Global (from VoxelAccumulator), ScalingModel_refined_list, GlobalVoxelGrid. <br>Output: VoxelData_relative (structured array with I_merged_relative,
  Sigma_merged_relative, etc.).
  3.E.1    Implement logic to apply refined scales M_i (from DiffuseScalingModel.get_scales()) to each observation:    [ ]    API Ref: dials_scaling.md (D.0 model.get_scales()). I_final_relative = I_corr
   / M_i.
  3.E.2    Implement error propagation: Sigma_final_relative = Sigma_corr / abs(M_i). Verify C_i (additive offset) is effectively zero for v1.    [ ]
  3.E.3    Implement weighted merge for each voxel: weight = 1 / Sigma_final_relative^2.    [ ]    API Ref: flex_arrays.md (D.4 flex.sum, arithmetic).
  3.E.4    Calculate voxel center q-vector attributes (q_center_x,y,z, `    q    _center) usingGlobalVoxelGridmethods andA_avg_ref`.
  3.E.5    Structure and save VoxelData_relative output (e.g., NumPy structured array or reflection table).    [ ]
  3.E.T    Tests for Merging    [ ]    Path: tests/merging/test_merger.py (New) or tests/scaling/test_merging.py
  3.E.T.1    Test: Correct application of scale factors to observations.    [ ]
  3.E.T.2    Test: Weighted merging logic and error propagation for sample voxels.    [ ]
  3.E.T.3    Test: Correct calculation of voxel center q-attributes.    [ ]
  3.F    Orchestration, Configuration, and QC    [ ]
  3.F.1    Update StillsPipelineOrchestrator to manage Phase 3 flow, data passing, and calls to new components.    [ ]
  3.F.2    Define RelativeScalingConfig in types_IDL.py and .md. Add to StillsPipelineConfig.    [ ]
  3.F.3    Implement QC metrics/plots for Phase 3 as per plan.md (Overall R-factor, scale factor plots, redundancy map, mean intensity vs. resolution).    [ ]
  3.F.T    Integration Tests for Phase 3 Orchestration    [ ]    Path: tests/integration/test_phase3_workflow.py (New)
  3.F.T.1    Test: End-to-end Phase 3 workflow with mock Phase 2 output and synthetic scaling factors. Verify final VoxelData_relative.    [ ]
  3.Z    Phase 3 Review & Next Steps    [ ]
  3.Z.1    Self-Review: All Phase 3 items addressed? IDLs created/updated? Tests passing?    [ ]
  3.Z.2    Context Refresh: Re-read plan.md sections for Phase 4 (Absolute Scaling).    [ ]
  3.Z.3    Decision: Proceed to Phase 4 Checklist.
