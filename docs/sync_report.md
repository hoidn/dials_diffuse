# Documentation Synchronization Report (Reflecting Q-Vector Validation Priority)

## Executive Summary

This report documents inconsistencies found between `plan.md` (highest authority) and other documentation/implementation files. It now reflects the decision to **prioritize the implementation and debugging of Q-vector based validation** as the primary method for geometric consistency checks, rather than adopting pixel-based validation.

## Documentation Hierarchy Applied
1.  `plan.md` (highest authority)
2.  IDL files (`*_IDL.md`)
3.  Other `.md` files
4.  Implementation files (`.py`)

## Inconsistencies Found and Resolutions

### 1. Incomplete `plan.md` Integration with `plan_adaptation.md`

*   **Issue:** `plan.md` has only partially integrated the changes specified in `plan_adaptation.md` for dual DIALS processing modes.
*   **Found:**
    *   Adapter descriptions were added to section 0.6 for dual processing.
    *   Testing sections were updated for dual processing mode support.
    *   **However, Module 1.S.0 (CBF Data Type Detection) is completely missing from `plan.md`.**
    *   Module 1.S.1 processing logic in `plan.md` still lacks the Route A/Route B structure specified in `plan_adaptation.md`.
*   **Status:** `plan.md` does NOT yet fully support both DIALS processing modes as intended by `plan_adaptation.md`.
*   **Resolution Required:** Complete the `plan.md` update by:
    1.  Adding Module 1.S.0 (CBF Data Type Detection).
    2.  Updating Module 1.S.1 processing logic with the dual routing (Route A/B) as specified in `plan_adaptation.md`.

### 2. Missing Adapter IDL Files

*   **Issue:** `plan.md` references adapter components but no IDL files exist for them.
*   **Adapters:** `DIALSStillsProcessAdapter`, `DIALSSequenceProcessAdapter`, `DIALSGenerateMaskAdapter`, `DXTBXIOAdapter`.
*   **Found:** Implementations exist in `src/diffusepipe/adapters/` but lack IDL specifications.
*   **Resolution Required:** Create IDL files for all adapter components, detailing their contracts.

### 3. Missing Dual Processing Mode Configuration in `types_IDL.md`

*   **Issue:** `types_IDL.md` lacks configuration fields for dual DIALS processing mode support as specified in `plan_adaptation.md`.
*   **Missing Fields in `DIALSStillsProcessConfig`:**
    *   `force_processing_mode`: Optional[str]
    *   `sequence_processing_phil_overrides`: Optional[List[str]]
    *   `data_type_detection_enabled`: bool
*   **Resolution Required:** Update `types_IDL.md` (and `types_IDL.py`) to include these fields in `DIALSStillsProcessConfig`.

### 4. Missing Masking Module IDL Files

*   **Issue:** `plan.md` references masking modules (1.S.2, 1.S.3) but no IDL files exist.
*   **Modules:**
    *   Module 1.S.2: Static and Dynamic Pixel Mask Generation (`pixel_mask_generator.py` exists)
    *   Module 1.S.3: Combined Per-Still Bragg and Pixel Mask Generation (`bragg_mask_generator.py` exists)
*   **Resolution Required:** Create IDL files for these masking components.

### 5. Missing Scaling and Merging IDL Files (Phase 3 & 4)

*   **Issue:** `plan.md` defines Modules 3.S.1-3.S.4 and 4.S.1 but no corresponding IDL files exist.
*   **Resolution Required:** Create IDL specifications for all Phase 3 and 4 components when their development begins.

### 6. Inconsistent DIALS Integration Approach in Documentation

*   **Issue:** High-level documentation does not consistently reflect the dual DIALS processing mode.
*   **Found:**
    *   `00_START_HERE.md` (lines 98-102) mentions only `dials.stills_process` integration.
    *   `CLAUDE.md` has been partially updated with sequence processing information.
*   **Resolution Required:** Update `00_START_HERE.md` section "Integration with DIALS via Python API" to accurately describe the dual processing capability (once Module 1.S.0 and routing are fully planned and implemented).

### 7. Missing `VoxelAccumulator` Component IDL

*   **Issue:** `plan.md` Module 3.S.2 references a `VoxelAccumulator` class but no IDL exists.
*   **Resolution Required:** Create IDL specification for `VoxelAccumulator` when Phase 3 development begins.

### 8. Incomplete `StillsPipelineOrchestrator` IDL

*   **Issue:** The `stills_pipeline_orchestrator_IDL.md` may not reflect the new routing logic for dual DIALS processing modes.
*   **Resolution Required:** Review and update `stills_pipeline_orchestrator_IDL.md` to include data type detection and routing logic once Module 1.S.0 is integrated into `plan.md` and its implementation in the orchestrator is defined.

## Technical Consistency Issues

### 1. PHIL Parameter Specifications

*   **Issue:** Critical PHIL parameters for sequence processing are not documented centrally or organized as configuration files.
*   **Found:**
    *   `plan.md` (via `plan_adaptation.md`) specifies critical parameters in the adapter section.
    *   No dedicated PHIL files for sequence processing steps exist in `src/diffusepipe/config/`.
*   **Resolution Required:**
    1.  Create base PHIL files in `src/diffusepipe/config/` for each step of the sequential DIALS workflow (`import.phil`, `find_spots_sequence.phil`, `index_sequence.phil`, `integrate_sequence.phil`).
    2.  Ensure `DIALSSequenceProcessAdapter` can use these base PHILs and apply overrides.

### 2. Correction Factor Conventions

*   **Status:** Correction factor conventions appear consistent across documentation (multiplicative factors). **No action needed.**

### 3. Missing Critical PHIL Parameters in `DIALSSequenceProcessAdapter` Implementation

*   **Issue:** The `DIALSSequenceProcessAdapter` code is missing critical PHIL parameters specified in `plan.md` (via `plan_adaptation.md`).
*   **Missing Parameters in code:**
    *   `indexing.method=fft3d`
    *   `geometry.convert_sequences_to_stills=false`
*   **Found Parameters in code:**
    *   `spotfinder.filter.min_spot_size=3` ✓
    *   `spotfinder.threshold.algorithm=dispersion` ✓
*   **Resolution Required:** Update `dials_sequence_process_adapter.py` to include *all* critical PHIL parameters as specified in the plan.

### 4. Testing Strategy Alignment

*   **Status:** Testing strategies align between `00_START_HERE.md` and `plan.md` (emphasis on integration tests). **No action needed.**

## Recommendations

1.  **Immediate Actions (Documentation & Planning):**
    *   **Fully integrate `plan_adaptation.md` into `plan.md`**: Add Module 1.S.0 and the Route A/B logic for Module 1.S.1.
    *   **Update `plan.md` Validation Logic (Module 1.S.1.Validation)**: Ensure it clearly describes the Q-vector consistency check (`q_bragg` vs. `q_pixel_recalculated`) as the primary method. Remove or mark as secondary/debug-only any references to pixel-based validation in this core plan.
    *   Update `types_IDL.md` (and `.py`) with dual processing configuration fields.
    *   Update `00_START_HERE.md` to mention dual DIALS processing capabilities.
2.  **Immediate Actions (Code & IDLs):**
    *   **Prioritize fixing `ModelValidator._check_q_consistency`**: Implement the Q-vector comparison correctly, removing the placeholder logic, and ensure it uses `extraction_config.q_consistency_tolerance_angstrom_inv`.
    *   Create missing IDL files for all existing adapter components (`DIALSStillsProcessAdapter`, `DIALSSequenceProcessAdapter`, `DIALSGenerateMaskAdapter`, `DXTBXIOAdapter`).
    *   Implement missing PHIL parameters in `DIALSSequenceProcessAdapter`.
3.  **Short-term Actions:**
    *   Implement Module 1.S.0 (CBF Data Type Detection) and routing logic in `StillProcessorAndValidatorComponent`.
    *   Refactor `DIALSStillsProcessAdapter` to correctly wrap the `dials.stills_process.Processor` API and use appropriate PHILs for true stills.
    *   Create IDL files for masking components (Module 1.S.2, 1.S.3).
    *   Create base PHIL files for sequence processing steps in `src/diffusepipe/config/`.
    *   Review and update `stills_pipeline_orchestrator_IDL.md`.
4.  **Future Actions (as development proceeds):**
    *   Create IDL files for Phase 3 and 4 components.
    *   Create IDL for `VoxelAccumulator`.

## Critical Implementation Inconsistencies (Reflecting Q-Vector Validation Priority)

### 1. Hardcoded Sequence Processing Contradicts Dual Processing Design

*   **Issue:** The `StillProcessorAndValidatorComponent` hardcodes `DIALSSequenceProcessAdapter`, bypassing the planned dual processing design.
*   **Resolution Required:** Implement Module 1.S.0 data type detection in `plan.md` and `StillProcessorAndValidatorComponent`, and update the component to use routing logic to select the appropriate DIALS adapter.

### 2. `DIALSStillsProcessAdapter` Implementation Issues

*   **Issue:** The `dials_stills_process_adapter.py` re-implements DIALS internal steps and uses inappropriate sequence-specific PHIL parameters.
*   **Resolution Required:** Refactor `DIALSStillsProcessAdapter` to be a thin wrapper around the `dials.command_line.stills_process.Processor` Python API, using PHIL parameters suitable for true stills processing.

### 3. Critical Validation Logic Implementation (MAJOR - NOW FOCUSED ON Q-VECTOR)

*   **Issue:** The Q-vector based validation in `ModelValidator._check_q_consistency` is currently non-functional (uses a placeholder based on `len(reflections)`), and the calculated |Δq| values are too large, indicating issues in the Q-vector calculation path within this method.
*   **Found:**
    *   `plan.md` specifies "Internal Q-Vector Consistency Check."
    *   `06_DIALS_DEBUGGING_GUIDE.md` noted issues with Q-vector calcs and suggested pixel validation as a *fix*, but the current decision is to make Q-vector validation work.
    *   `simple_validation_fix.py` (pixel-based) is now considered a fallback or debug tool, not the primary validation method.
    *   `src/diffusepipe/diagnostics/consistency_checker.py` contains functional Q-vector comparison logic for diagnostic purposes.
*   **Resolution Required:**
    1.  **Debug and Fix Q-vector calculations within `ModelValidator._check_q_consistency`**: Identify why |Δq| values are large. Compare its calculation logic meticulously with `consistency_checker.py`.
    2.  **Remove Placeholder**: Once Q-vector calculations are reliable, remove the placeholder logic that bases pass/fail on `len(reflections)`.
    3.  **Implement Tolerance Check**: Ensure the method compares the mean/max of calculated |Δq| values against `extraction_config.q_consistency_tolerance_angstrom_inv` to determine pass/fail.
    4.  Update `plan.md` and `checklists/phase1.md` to ensure they clearly describe the Q-vector validation method and its parameters.

### 4. Incomplete Partiality Generation for Sequence Data

*   **Issue:** The sequence adapter may not generate partiality values consistently.
*   **Resolution Required:** Verify partiality generation by `dials.integrate` in sequence mode. If inconsistent, update `plan.md` regarding how partiality is handled/filtered, ensuring the "Universal `P_spot` Strategy" (use as quality filter, not divisor) is clearly documented and applied for data from both processing routes.

## Unresolved Issues Requiring Human Decision (Updated)

1.  **Module 1.S.0 Numbering:** How to number/name the new CBF Data Type Detection module in `plan.md` (e.g., 0.S.1 or renumber 1.S.x).
2.  **PHIL File Organization (Sequence PHILs):** Confirm approach for base PHILs in `src/diffusepipe/config/` and how adapters use them.
3.  **Adapter IDL Granularity:** One comprehensive IDL for all adapters, or separate ones? (Separate is usually clearer).
4.  **Data Type Detection Location:** Confirm if `StillProcessorAndValidatorComponent` is the right place for Module 1.S.0 logic.
5.  **(Resolved by new decision)** Validation Logic Strategy: Decision is to fix and use Q-vector validation.
6.  **Adapter Architecture (Stills Adapter):** Confirm `DIALSStillsProcessAdapter` should be a thin wrapper.

## Summary

**Major Finding:** The project requires a concerted effort to:
1.  **Implement the planned dual DIALS processing architecture** (Module 1.S.0, routing, correct adapter implementations).
2.  **Fix and enable the Q-vector based validation** in `ModelValidator` as the primary geometric check.
3.  Synchronize `plan.md` to accurately reflect these two critical aspects.
4.  Create the numerous missing IDL files.

Addressing these will provide a much more consistent and robust foundation.
