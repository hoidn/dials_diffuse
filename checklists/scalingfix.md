### **Agent Implementation Checklist: Scaling and Voxelization Fixes**

**Overall Goal:** Correct the three critical remaining bugs in the Phase 3 scaling and voxelization implementation to align the code with the project plan (`plan.md`) and standard DIALS/CCTBX practices.

**Instructions:**
1.  Copy this checklist into your working memory.
2.  Update the `State` for each item as you progress: `[ ]` (Open) -> `[P]` (In Progress) -> `[D]` (Done).
3.  Follow the API guidance in the `How/Why & API Guidance` column for precise implementation.

---

### **Issue #1: Address Placeholder Refinement in `DiffuseScalingModel`**

**Context:** The current `refine_parameters` method in `DiffuseScalingModel` is a simplified placeholder. It does not use the robust, iterative Levenberg-Marquardt minimizer required by `plan.md` and the IDL contract, which can lead to unstable or incorrect scaling results. The immediate fix is to prevent its misuse and confirm obsolete code is removed.

| ID | Task Description | State | How/Why & API Guidance |
| :--- | :--- | :--- | :--- |
| **1.A** | **Locate and Prepare File** | `[ ]` | **File:** `src/diffusepipe/scaling/diffuse_scaling_model.py`. |
| **1.B** | **Confirm Warning in `refine_parameters`** | `[ ]` | **Why:** To make it explicit that the current implementation is a placeholder and not a production-ready minimizer, preventing misuse. <br> **How:** Verify that the `refine_parameters` method begins with this exact `logger.warning()` call: <br> `logger.warning("The current 'refine_parameters' implementation is a simplified placeholder and does not use a robust DIALS/ScitBX minimizer. The results may not be optimal.")` |
| **1.C** | **Verify Removal of Obsolete Helpers** | `[ ]` | **Why:** The helpers `_refine_step` and `_calculate_gradient_for_still` are obsolete and will be replaced when a proper refiner is implemented. Removing them now cleans the code. <br> **How:** Search for and confirm that the method definitions for `_refine_step` and `_calculate_gradient_for_still` do not exist in `DiffuseScalingModel`. |
| **1.D** | **Verify IDL Compliance** | `[ ]` | **Why:** To confirm the IDL accurately specifies the *intended* behavior, which the code currently lacks. <br> **How:** Check `src/diffusepipe/scaling/diffuse_scaling_model_IDL.md`. The IDL correctly states "Update parameters via Levenberg-Marquardt or similar." No change is needed to the IDL, as it reflects the correct target state. |

---

### **Issue #2: Fix `ResolutionSmootherComponent` Implementation**

**Context:** The `ResolutionSmootherComponent` is a placeholder that does not correctly use the underlying `GaussianSmoother1D` from DIALS. It uses incorrect logic for both scale factors and their derivatives, which will lead to incorrect refinement of the resolution-dependent scaling term.

| ID | Task Description | State | How/Why & API Guidance |
| :--- | :--- | :--- | :--- |
| **2.A** | **Locate and Prepare Files** | `[ ]` | **Files:** `src/diffusepipe/scaling/components/resolution_smoother.py` and its test file `tests/scaling/test_diffuse_scaling_model.py`. |
| **2.B** | **Implement `calculate_scales_and_derivatives`** | `[ ]` | **Why:** To replace the dummy logic with the correct DIALS smoother API, which provides both the interpolated scales and the analytical Jacobian required for refinement. <br> **How:** <br> 1. In `calculate_scales_and_derivatives`, first extract q-magnitudes and convert to a `flex.double` array. <br> 2. Set the current parameters on the smoother object: `self._smoother.set_parameters(self.parameters)`. <br> 3. Call the correct DIALS API method `self._smoother.value_weight(positions=q_locations)`. This returns a `value_weight_result` object. <br> 4. Extract `scales = value_weight_result.value` and `derivatives = value_weight_result.weight`. The `weight` attribute contains the analytical derivatives. <br> 5. Return the `scales` and `derivatives`. |
| **2.C** | **Update `get_scale_for_q`** | `[ ]` | **Why:** To use the correct evaluation method for consistency with the main calculation. <br> **How:** Update the method to also use `self._smoother.set_parameters(self.parameters)` and `self._smoother.value_weight(positions=flex.double([q_magnitude]))` to evaluate the scale at a single point, then extract the value from the result. |
| **2.D** | **Update Tests** | `[ ]` | **Why:** To verify the new, correct behavior and ensure analytical derivatives are being calculated. <br> **How:** In `tests/scaling/test_diffuse_scaling_model.py`, within `TestResolutionSmootherComponent.test_scale_calculation`, assert that the returned `derivatives` array has the correct shape (`n_obs`, `n_params`) and is not all zeros. |

---

### **Issue #3: Correct ASU Mapping Precision Loss in `VoxelAccumulator`**

**Context:** The `VoxelAccumulator._map_to_asu` method prematurely rounds fractional HKL coordinates to integers before mapping them to the asymmetric unit (ASU). This causes a significant loss of precision and leads to incorrect voxel binning for diffuse data.

| ID | Task Description | State | How/Why & API Guidance |
| :--- | :--- | :--- | :--- |
| **3.A** | **Locate and Prepare Files** | `[ ]` | **Files:** `src/diffusepipe/voxelization/voxel_accumulator.py` and its test file `tests/voxelization/test_voxel_accumulator.py`. |
| **3.B** | **Rewrite `_map_to_asu` Method** | `[ ]` | **Why:** To fix the precision loss from premature rounding and to use the more direct CCTBX API for fractional coordinates. <br> **How:** <br> 1. In `_map_to_asu`, remove any line that performs rounding (e.g., `np.round(...)`). <br> 2. Convert the input NumPy array of fractional HKLs to a `flex.vec3_double`: `hkl_flex = flex.vec3_double(hkl_array)`. <br> 3. Get the `space_group_info` object: `sg_info = self.space_group.info()`. <br> 4. Call `sg_info.map_to_asu(hkl_flex)` on the `flex.vec3_double` array directly. This method correctly handles fractional coordinates. <br> 5. Convert the result back to a NumPy array: `return hkl_asu_flex.as_numpy_array()`. |
| **3.C** | **Update `voxel_accumulator_IDL.md`** | `[ ]` | **Why:** To make the contract more robust and less tied to a specific, incorrect implementation detail (`cctbx.miller.set` implies integers). <br> **How:** In `src/diffusepipe/voxelization/voxel_accumulator_IDL.md`, change the `Behavior` description for ASU mapping from "...using `cctbx.miller.set`..." to the more general **"Map fractional HKL coordinates to the asymmetric unit using CCTBX symmetry operations."** |
| **3.D** | **Update and Enhance Tests** | `[ ]` | **Why:** To verify the correctness of the new ASU mapping with fractional data. <br> **How:** In `tests/voxelization/test_voxel_accumulator.py`: <br> 1. In `test_asu_mapping`, modify the test to use fractional inputs (e.g., `np.array([[1.1, 2.3, 3.7]])`). <br> 2. In `test_asu_mapping_with_p2_symmetry`, verify that symmetry-equivalent *fractional* HKLs (e.g., `[1.1, 2.3, 3.7]` and `[-1.1, 2.3, -3.7]`) are correctly mapped to the exact same ASU HKL coordinate using `np.allclose`. |
