### **Agent Implementation Checklist: Scaling and Merging Fixes**

**Overall Goal:** Implement the four identified fixes to bring the Phase 3 scaling implementation into alignment with the project plan (`plan.md`) and standard DIALS/CCTBX practices, correcting bugs and completing placeholder logic.

**Instructions:**
1.  Copy this checklist into your working memory.
2.  Update the `State` for each item as you progress: `[ ]` (Open) -> `[P]` (In Progress) -> `[D]` (Done).
3.  Follow the API guidance in the `How/Why & API Guidance` column.

---

### **Issue #1: Fix `ResolutionSmootherComponent` Implementation**

| ID | Task Description | State | How/Why & API Guidance |
| :--- | :--- | :--- | :--- |
| **1.A** | **Locate and Prepare Files** | `[ ]` | **Files:** `src/diffusepipe/scaling/components/resolution_smoother.py` and its test file `tests/scaling/test_diffuse_scaling_model.py`. |
| **1.B** | **Update `ResolutionSmootherComponent.__init__`** | `[ ]` | **Why:** To decouple the component from the parameter manager, following standard DIALS patterns. <br> **How:** <br> 1. Modify the constructor signature to `__init__(self, n_control_points: int, resolution_range: tuple[float, float])`. <br> 2. Call the base class constructor `super().__init__()`. <br> 3. Instantiate the `GaussianSmoother1D` from `dials.algorithms.scaling.model.components.smooth_scale_components`. <br> 4. Initialize its parameters to a `flex.double` array of `1.0`s. Store the number of parameters (`self.n_params`). |
| **1.C** | **Implement `calculate_scales_and_derivatives`** | `[ ]` | **Why:** To replace the dummy derivative logic with a correct implementation that provides the analytical Jacobian required for refinement. <br> **How:** <br> 1. Get the current parameters (`self.parameters`). <br> 2. Call the DIALS `_smoother.value_weight(q_locations, self.parameters)` method. This single call correctly returns both the interpolated `scales` and the analytical `derivatives`. <br> 3. Return both `scales` and `derivatives`. <br> **API Guidance:** `libdocs/dials/dials_scaling.md` (D.0, Example 7) demonstrates this pattern: `weight_result = self.smoother.value_weight(coord)` and shows how the result contains the necessary derivatives. |
| **1.D** | **Update `get_scale_for_q`** | `[ ]` | **Why:** To use the correct evaluation method. <br> **How:** Use `self._smoother.value_weight(flex.double([q_magnitude]), self.parameters)` to evaluate the scale at a single point. |
| **1.E** | **Update `diffuse_scaling_model_IDL.md`** | `[ ]` | **Why:** To reflect the improved, decoupled constructor signature in the contract. <br> **How:** In `src/diffusepipe/scaling/diffuse_scaling_model_IDL.md`, change the constructor signature for `ResolutionSmootherComponent` to `__init__(self, n_control_points: int, resolution_range: tuple[float, float])`. |
| **1.F** | **Update Tests** | `[ ]` | **Why:** To verify the new, correct behavior. <br> **How:** In `tests/scaling/test_diffuse_scaling_model.py`, update the instantiation of `ResolutionSmootherComponent`. Add an assertion to check that the returned `derivatives` array has the correct shape (`n_obs`, `n_params`) and is not all zeros. |

---

### **Issue #2: Address Placeholder Refinement in `DiffuseScalingModel`**

| ID | Task Description | State | How/Why & API Guidance |
| :--- | :--- | :--- | :--- |
| **2.A** | **Locate and Prepare File** | `[ ]` | **File:** `src/diffusepipe/scaling/diffuse_scaling_model.py`. |
| **2.B** | **Add Warning to `refine_parameters`** | `[ ]` | **Why:** The current method is a placeholder and does not meet the plan's requirement for a robust minimizer. This must be made explicit to prevent misuse. <br> **How:** Add a `logger.warning()` at the beginning of the `refine_parameters` method, stating that the implementation is a simplified placeholder and not a production-ready DIALS/ScitBX minimizer. |
| **2.C** | **Remove Unused Helper Methods** | `[ ]` | **Why:** The placeholder helpers `_refine_step` and `_calculate_gradient_for_still` will be obsolete once a proper refiner is implemented. Removing them now cleans the code. <br> **How:** Delete the method definitions for `_refine_step` and `_calculate_gradient_for_still`. |
| **2.D** | **Verify IDL Compliance** | `[ ]` | **Why:** To confirm the IDL accurately specifies the *intended* behavior, which the code currently lacks. <br> **How:** Check `src/diffusepipe/scaling/diffuse_scaling_model_IDL.md`. The IDL correctly states "Update parameters via Levenberg-Marquardt or similar." No change is needed to the IDL. The code is what needs to eventually be brought into compliance. <br> **API Guidance:** `libdocs/dials/dials_scaling.md` (D.0, Example 6) shows the target implementation using `scitbx.lstbx.normal_eqns_solving.levenberg_marquardt_iterations`. This is the API to be used in a future task to replace the placeholder. |

---

### **Issue #3: Refactor `PerStillMultiplierComponent` Initialization**

| ID | Task Description | State | How/Why & API Guidance |
| :--- | :--- | :--- | :--- |
| **3.A** | **Locate and Prepare Files** | `[ ]` | **Files:** `src/diffusepipe/scaling/components/per_still_multiplier.py` and `src/diffusepipe/scaling/diffuse_scaling_model.py`. |
| **3.B** | **Update `PerStillMultiplierComponent.__init__`** | `[ ]` | **Why:** To remove the direct dependency on the parameter manager, making the component more modular and compliant with DIALS patterns. <br> **How:** <br> 1. Change the constructor signature to `__init__(self, still_ids: List[int], initial_values: Optional[Dict[int, float]] = None)`. <br> 2. Call `super().__init__()`. <br> 3. Set `self.parameters` directly using the provided initial values. |
| **3.C** | **Update `DiffuseScalingModel._initialize_components`** | `[ ]` | **Why:** The parent model must now take responsibility for registering the component's parameters. <br> **How:** <br> 1. Instantiate `active_parameter_manager` in the model. <br> 2. Instantiate `PerStillMultiplierComponent` *without* passing the manager. <br> 3. Explicitly add the component's parameters to the manager: `self.active_parameter_manager.add_active_parameters(self.per_still_component.parameters, 'per_still')`. <br> **API Guidance:** `libdocs/dials/dials_scaling.md` (D.0, Example 5) shows the parent model creating the `active_parameter_manager` and configuring its components, supporting this pattern of parental responsibility. |
| **3.D** | **Update `diffuse_scaling_model_IDL.md`** | `[ ]` | **Why:** The IDL contract must reflect the new, cleaner constructor signature. <br> **How:** In `src/diffusepipe/scaling/diffuse_scaling_model_IDL.md`, change the constructor for `PerStillMultiplierComponent` to `__init__(self, still_ids: list[int], initial_values: dict = None)`. |
| **3.E** | **Update Tests** | `[ ]` | **Why:** Test cases must be updated to reflect the new instantiation logic for the components. <br> **How:** In `tests/scaling/test_diffuse_scaling_model.py`, change how `PerStillMultiplierComponent` and `DiffuseScalingModel` are instantiated. |

---

### **Issue #4: Correct ASU Mapping in `VoxelAccumulator`**

| ID | Task Description | State | How/Why & API Guidance |
| :--- | :--- | :--- | :--- |
| **4.A** | **Locate and Prepare Files** | `[ ]` | **Files:** `src/diffusepipe/voxelization/voxel_accumulator.py` and its test file `tests/voxelization/test_voxel_accumulator.py`. |
| **4.B** | **Rewrite `_map_to_asu` Method** | `[ ]` | **Why:** To fix the precision loss from premature rounding and to use the more direct CCTBX API as intended by the plan. <br> **How:** <br> 1. Convert the input NumPy array of fractional HKLs to a `flex.vec3_double`. <br> 2. Get the `space_group_info` object from the grid's average crystal model. <br> 3. Call `sg_info.map_to_asu(hkl_flex)` on the `flex.vec3_double` array directly. <br> 4. Convert the result back to a NumPy array. |
| **4.C** | **Update `voxel_accumulator_IDL.md`** | `[ ]` | **Why:** To make the contract more robust and less tied to a specific implementation detail. <br> **How:** In `src/diffusepipe/voxelization/voxel_accumulator_IDL.md`, change the `Behavior` description for ASU mapping from "...using `cctbx.miller.set`..." to the more general "Map fractional HKL coordinates to the asymmetric unit using CCTBX symmetry operations." |
| **4.D** | **Update and Enhance Tests** | `[ ]` | **Why:** To verify the correctness of the new ASU mapping. <br> **How:** In `tests/voxelization/test_voxel_accumulator.py`: <br> 1. Update the existing ASU mapping test to use fractional inputs. <br> 2. Add a new test case with a non-trivial space group (e.g., "P2") to confirm that symmetry-equivalent reflections are correctly mapped to the same ASU HKL. <br> **API Guidance:** `libdocs/dials/crystallographic_calculations.md` (C.7) shows the use of `sgtbx.space_group_info` and `miller.set` for ASU mapping, confirming this is the correct toolkit. The proposed fix is a more direct and precise application of this API. |
