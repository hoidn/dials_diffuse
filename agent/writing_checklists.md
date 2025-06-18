# Guide to Writing Agent Implementation Checklists

### 1. Purpose & Philosophy

This document provides a template and guidelines for creating comprehensive, documentation-driven implementation checklists. The goal is to ensure that all development work is systematically planned, executed, and verified against its specification *before* and *after* implementation.

**Core Philosophy: The IDL is the Contract.**
-   **Specification First:** All implementation tasks must begin with a thorough understanding of the relevant high-level plans (`plan.md`, architecture docs) and the specific component contracts defined in **Interface Definition Language (IDL)** files (`*_IDL.md`).
-   **Traceability:** Every implementation step should be traceable back to a requirement in the IDL or a design decision in the plan.
-   **Verification:** The final implementation must be explicitly verified against the IDL contract.

Using this structured approach prevents architectural drift, reduces bugs, and ensures the final code is a faithful and correct implementation of its design.

### 2. Implementation Checklist Template

Copy and use this template for every new implementation task.

```markdown
### **Agent Implementation Checklist: [Task Title]**

**Overall Goal:** [A brief, one-sentence summary of the task's primary objective.]

**Instructions:**
1.  Copy this entire checklist into your working memory.
2.  Update the `State` for each item as you progress: `[ ]` (Open) -> `[P]` (In Progress) -> `[D]` (Done).
3.  Follow the API guidance in the `How/Why & API Guidance` column.

| ID | Task Description | State | How/Why & API Guidance |
| :--- | :--- | :--- | :--- |
| **Phase 0: Preparation & Understanding (Documentation-Driven Planning)** |
| 0.A | **Review High-Level Context** | `[ ]` | **Why:** To understand the task's purpose and constraints within the project. <br> **Files:** `plan.md`, `docs/ARCHITECTURE/overview.md`, relevant `docs/LESSONS_LEARNED.md`. |
| 0.B | **Identify and Review IDL Contract(s)** | `[ ]` | **Why:** The IDL is the primary specification. This step is mandatory. <br> **Files:** [List all relevant `*_IDL.md` files here]. |
| 0.C | **Summarize IDL Contract** | `[ ]` | **Why:** To confirm understanding of the component's public interface and required behavior. <br> **Summary:** <br> - **Component(s):** [List class/module names] <br> - **Inputs:** [List key inputs/parameters] <br> - **Outputs:** [List key return values/types] <br> - **Behavior:** [Summarize the core logic described in the IDL `Behavior` sections] <br> - **Errors:** [List key error conditions from `@raises_error`] |
| 0.D | **Review Implementation Rules** | `[ ]` | **Why:** To ensure the implementation will adhere to project-wide standards. <br> **Files:** `docs/02_IMPLEMENTATION_RULES.md`, `docs/03_PROJECT_RULES.md`. |
| 0.E | **Outline Testing Strategy** | `[ ]` | **Why:** To plan for verification from the start. <br> **Strategy:** [e.g., "Integration test for Component A using a real Component B. Mock external API X. Add performance test for method Y."]. |
| **Phase 1: Implementation & Core Testing (Contract Fulfillment)** |
| 1.A | **Create/Locate Files** | `[ ]` | **Path(s):** [List all source and test files to be created/modified]. |
| 1.B | **Implement [Feature/Method 1]** | `[ ]` | **Why:** [Connect this feature to a specific part of the IDL contract]. <br> **How:** [Describe implementation steps]. <br> **API:** [Reference specific internal or external APIs, e.g., `dials.scaling.model.ScalingModelBase`]. |
| 1.C | **Write Test for [Feature/Method 1]** | `[ ]` | **Why:** To verify the implementation against the IDL's postconditions and error conditions. |
| 1.D | **Implement [Feature/Method 2]** | `[ ]` | **Why:** ... |
| ... | ... | `[ ]` | ... |
| **Phase 2: Finalization & Verification (Contract Conformance)** |
| 2.A | **Format and Lint Code** | `[ ]` | **Why:** To ensure code style consistency. <br> **How:** Run project's standard formatting and linting tools (e.g., `make format`, `make lint`). |
| 2.B | **Run Full Test Suite** | `[ ]` | **Why:** To catch any regressions introduced by the changes. |
| 2.C | **Perform Final Sanity Check** | `[ ]` | **Why:** A final self-review to ensure all requirements were met. <br> **Check:** <br> 1. Does the code fulfill every aspect of the IDL contract (methods, behavior, errors)? <br> 2. Are all relevant points from `plan.md` addressed? <br> 3. Does the implementation align with `LESSONS_LEARNED.md`? |
| 2.D | **Update IDL & Documentation** | `[ ]` | **Why:** If the implementation required a necessary deviation or clarification of the contract, the IDL must be updated to remain the source of truth. <br> **How:** Modify the relevant `*_IDL.md` files. Check if any other documentation (`plan.md`, etc.) needs updating. If no changes are needed, mark as done. |
```

### 3. Guidelines for Filling Out the Checklist

#### Phase 0: Preparation & Understanding
This is the most critical phase for ensuring a documentation-driven workflow.
-   **0.A (High-Level Context):** Always start here. A feature might be technically correct but architecturally wrong if it violates the master plan.
-   **0.B (Identify IDLs):** Your first action is to find the specification. If one doesn't exist for the code you're modifying, your first task should be to create one by reducing the existing code to its essential contract.
-   **0.C (Summarize Contract):** Do not skip this. This step forces you to internalize the requirements *before* writing code. Your summary should be a concise version of the IDL's `Preconditions`, `Postconditions`, `Behavior`, and `@raises_error` tags.
-   **0.E (Testing Strategy):** This ensures testability is considered part of the design, not an afterthought.

#### Phase 1: Implementation
-   Break the task into the smallest logical steps possible. Each "feature" should ideally correspond to implementing a single method or a small, related group of methods from the IDL.
-   Explicitly state the "Why" for each implementation step, linking it back to the IDL (e.g., "Why: To fulfill the `calculate_scales_and_derivatives` contract defined in the IDL.").
-   Use the "API" section to note the specific functions, classes, or methods from libraries (`DIALS`, `CCTBX`, `NumPy`) or internal modules that you plan to use. This is where you cross-reference the `libdocs/` or other API documentation.

#### Phase 2: Finalization & Verification
-   **2.C (Sanity Check):** This is a crucial final review. Reread the IDL and the plan one last time and compare it against your finished code. It's the last chance to catch deviations from the specification.
-   **2.D (Update Documentation):** This is the step that ensures our documentation remains alive and accurate. If you had to change the public-facing behavior of a component, its IDL **must** be updated. If the change has broader architectural implications, `plan.md` or an ADR might need an update.

### 4. Example Filled-Out Checklist (For Fixing `ResolutionSmootherComponent`)

Here's how the template would be applied to the first fix identified in the review:

#### **Agent Implementation Checklist: Fix `ResolutionSmootherComponent`**

**Overall Goal:** Refactor the `ResolutionSmootherComponent` to correctly wrap the DIALS `GaussianSmoother1D` and provide analytical derivatives, ensuring it complies with its IDL contract.

| ID | Task Description | State | How/Why & API Guidance |
| :--- | :--- | :--- | :--- |
| **Phase 0: Preparation & Understanding** |
| 0.A | **Review High-Level Context** | `[D]` | **Why:** To understand the role of scaling components in the plan. <br> **Files:** `plan.md` (Module 3.S.3). |
| 0.B | **Identify and Review IDL Contract(s)** | `[D]` | **Why:** To understand the required interface and behavior. <br> **Files:** `src/diffusepipe/scaling/diffuse_scaling_model_IDL.md`. |
| 0.C | **Summarize IDL Contract** | `[D]` | **Summary:** <br> - **Component:** `ResolutionSmootherComponent` <br> - **Inputs:** `active_parameter_manager`, `n_control_points`, `resolution_range`. <br> - **Outputs:** `(scales, derivatives)` tuple of `flex.double` arrays. <br> - **Behavior:** Provides a smooth, resolution-dependent multiplicative scale factor. <br> - **Errors:** `ParameterLimitExceeded`. |
| 0.D | **Review Implementation Rules** | `[D]` | **Why:** To check for rules on DIALS integration. <br> **Files:** `docs/02_IMPLEMENTATION_RULES.md`, `docs/06_DIALS_DEBUGGING_GUIDE.md`. |
| 0.E | **Outline Testing Strategy** | `[D]` | **Strategy:** Update existing tests. Verify the shape and content of the returned `derivatives` array to ensure it's not a dummy placeholder. |
| **Phase 1: Implementation & Core Testing** |
| 1.A | **Update `__init__`** | `[D]` | **Why:** To decouple from the parameter manager. <br> **How:** Change signature to `__init__(self, n_control_points, resolution_range)`. Instantiate `GaussianSmoother1D` and initialize `self.parameters`. <br> **Path:** `src/diffusepipe/scaling/components/resolution_smoother.py`. |
| 1.B | **Implement `calculate_scales_and_derivatives`** | `[D]` | **Why:** To provide correct analytical derivatives. <br> **How:** Use `self._smoother.value_weight(q_locations, self.parameters)` to get both `scales` and `derivatives`. <br> **API:** `libdocs/dials/dials_scaling.md` (D.0, Example 7). <br> **Path:** `src/diffusepipe/scaling/components/resolution_smoother.py`. |
| 1.C | **Update `get_scale_for_q`** | `[D]` | **Why:** To use the correct evaluation method. <br> **How:** Call `self._smoother.value_weight()` to get the scale. <br> **Path:** `src/diffusepipe/scaling/components/resolution_smoother.py`. |
| **Phase 2: Finalization & Verification** |
| 2.A | **Format and Lint Code** | `[D]` | **How:** Ran `make format` and `make lint`. |
| 2.B | **Run Full Test Suite** | `[D]` | **Why:** To check for regressions. All tests passed. |
| 2.C | **Perform Final Sanity Check** | `[D]` | **Check:** The new implementation correctly provides scales and a non-dummy derivatives matrix, fulfilling the IDL contract. |
| 2.D | **Update IDL & Documentation** | `[D]` | **Why:** To reflect the improved constructor. <br> **How:** Changed `__init__` signature in `src/diffusepipe/scaling/diffuse_scaling_model_IDL.md` to remove `active_parameter_manager`. |
