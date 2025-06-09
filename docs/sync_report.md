# Documentation Synchronization Report

## Executive Summary

This report documents inconsistencies found between plan.md (highest authority) and other documentation/implementation files, along with resolutions made according to the documentation hierarchy.

## Documentation Hierarchy Applied
1. plan.md (highest authority)
2. IDL files (*_IDL.md)
3. Other .md files
4. Implementation files (.py)

## Inconsistencies Found and Resolutions

### 1. Incomplete plan.md Integration with plan_adaptation.md

**Issue:** plan.md has only partially integrated the changes specified in plan_adaptation.md.

**Found:** 
- Adapter descriptions were added to section 0.6 for dual processing
- Testing sections were updated for dual processing mode support
- **However, Module 1.S.0 (CBF Data Type Detection) is completely missing**
- Module 1.S.1 processing logic still lacks the Route A/Route B structure specified in plan_adaptation.md

**Status:** plan.md does NOT yet fully support both modes as intended - it needs complete integration of plan_adaptation.md specifications.

**Resolution Required:** Complete the plan.md update by adding Module 1.S.0 and updating Module 1.S.1 with the dual routing logic specified in plan_adaptation.md.

### 2. Missing Adapter IDL Files

**Issue:** plan.md references adapter components but no IDL files exist for:
- DIALSStillsProcessAdapter
- DIALSSequenceProcessAdapter (newly required)
- DIALSGenerateMaskAdapter
- DXTBXIOAdapter

**Found:** Adapter implementations exist in `src/diffusepipe/adapters/` but lack corresponding IDL specifications.

**Resolution Required:** Create IDL files for all adapter components.

### 3. Missing Dual Processing Mode Configuration

**Issue:** types_IDL.md lacks configuration fields for dual processing mode support as specified in plan_adaptation.md.

**Missing Fields:**
- `force_processing_mode`: Optional[str]
- `sequence_processing_phil_overrides`: Optional[List[str]]
- `data_type_detection_enabled`: bool

**Resolution Required:** Update types_IDL.md to include these fields in DIALSStillsProcessConfig.

### 4. Missing Masking Module IDL Files

**Issue:** plan.md references masking modules but no IDL files exist for:
- Module 1.S.2: Static and Dynamic Pixel Mask Generation (pixel_mask_generator.py exists)
- Module 1.S.3: Combined Per-Still Bragg and Pixel Mask Generation (bragg_mask_generator.py exists)

**Resolution Required:** Create IDL files for masking components.

### 5. Missing Scaling and Merging IDL Files

**Issue:** plan.md defines Modules 3.S.1-3.S.4 and 4.S.1 but no corresponding IDL files exist for:
- Module 3.S.1: Global Voxel Grid Definition
- Module 3.S.2: Binning Corrected Diffuse Pixels
- Module 3.S.3: Relative Scaling
- Module 3.S.4: Merging Relatively Scaled Data
- Module 4.S.1: Absolute Scaling

**Resolution Required:** Create IDL specifications for Phase 3 and 4 components.

### 6. Inconsistent DIALS Integration Approach

**Issue:** Documentation inconsistency between stills-only vs dual processing mode.

**Found:** 
- 00_START_HERE.md (lines 98-102) mentions only `dials.stills_process` integration
- CLAUDE.md has been updated with sequence processing information
- plan.md now supports both modes but Module 1.S.0 is missing

**Resolution Required:** Update 00_START_HERE.md section "Integration with DIALS via Python API" to mention both stills and sequence processing capabilities.

### 7. Missing VoxelAccumulator Component

**Issue:** plan.md Module 3.S.2 references a VoxelAccumulator class with HDF5 backend but no IDL exists.

**Resolution Required:** Create IDL specification for VoxelAccumulator.

### 8. Incomplete StillsPipelineOrchestrator IDL

**Issue:** The orchestrator IDL may not reflect the new routing logic for dual processing modes.

**Resolution Required:** Review and update stills_pipeline_orchestrator_IDL.md to include data type detection and routing.

## Technical Consistency Issues

### 1. PHIL Parameter Specifications

**Issue:** Critical PHIL parameters for sequence processing are not documented in a central location.

**Found:**
- plan.md specifies critical parameters in adapter section
- No PHIL files exist for sequence processing in `src/diffusepipe/config/`

**Resolution Required:** Create sequence processing PHIL configuration files.

### 2. Correction Factor Conventions

**Status:** Correction factor conventions appear consistent across documentation (multiplicative factors).

### 3. Missing Critical PHIL Parameters in Sequence Adapter

**Issue:** The DIALSSequenceProcessAdapter implementation is missing critical PHIL parameters specified in plan.md.

**Missing Parameters:**
- `indexing.method=fft3d` (not implemented)
- `geometry.convert_sequences_to_stills=false` (not implemented)

**Found Parameters:**
- `spotfinder.filter.min_spot_size=3` ✓ (implemented)
- `spotfinder.threshold.algorithm=dispersion` ✓ (implemented)

**Resolution Required:** Update dials_sequence_process_adapter.py to include all critical PHIL parameters.

### 3. Testing Strategy Alignment

**Status:** Testing strategies align between 00_START_HERE.md and plan.md (emphasis on integration tests).

## Recommendations

1. **Immediate Actions:**
   - Complete the Module 1.S.0 addition to plan.md
   - Create missing IDL files for all adapters
   - Update types_IDL.md with dual processing configuration
   - Update 00_START_HERE.md to mention dual processing capability

2. **Short-term Actions:**
   - Create IDL files for all Phase 3 and 4 components
   - Create PHIL configuration files for sequence processing
   - Review and update existing orchestrator IDLs

3. **Documentation Improvements:**
   - Create a central PHIL parameter reference document
   - Add sequence processing examples to documentation
   - Create architecture diagrams showing dual processing flow

4. **Process Improvements:**
   - Establish a checklist for ensuring IDL files are created when new components are added
   - Create templates for common IDL patterns (adapters, processors, etc.)

## Critical Implementation Inconsistencies

### 1. Hardcoded Sequence Processing Contradicts Dual Processing Design

**Issue:** The main processing component contradicts the planned dual processing approach.

**Found:** 
- `StillProcessorAndValidatorComponent` in `still_processing_and_validation.py` hardcodes `DIALSSequenceProcessAdapter`
- No data type detection logic exists anywhere in the codebase
- No routing logic to choose between stills vs sequence processing
- This completely bypasses the planned Module 1.S.0 data type detection and dual processing design

**Impact:** Current implementation processes ALL data as sequences, ignoring the critical distinction between true stills and oscillation data.

**Resolution Required:** Implement Module 1.S.0 data type detection and update the processor to use routing logic.

### 2. DIALSStillsProcessAdapter Implementation Issues

**Issue:** The stills adapter implementation contradicts its intended purpose.

**Found:** 
- `dials_stills_process_adapter.py` re-implements internal steps rather than using `Processor.process_experiments()` directly
- Contains sequence-specific PHIL parameters (`geometry.convert_sequences_to_stills=false`, `indexing.method=fft3d`) which are inappropriate for the `dials.stills_process` Python API
- This suggests confusion about the adapter's role - it should interface with the existing DIALS Python API, not reimplement it

**Resolution Required:** Refactor the stills adapter to properly use `dials.stills_process.Processor` and remove inappropriate sequence-specific parameters.

### 3. Critical Validation Logic Discrepancy (MAJOR)

**Issue:** Major inconsistency between planned validation approach and recommended/implemented approach.

**Found:** 
- plan.md specifies "Internal Q-Vector Consistency Check" with complex q-vector calculations
- `06_DIALS_DEBUGGING_GUIDE.md` and `simple_validation_fix.py` recommend "Simple Pixel Validation" (`|observed_px - calculated_px|`)
- Current `ModelValidator._check_q_consistency` is just a placeholder
- This represents a fundamental disagreement about validation methodology

**Resolution Required:** Decide on validation approach and update either plan.md or implementation to match. The simpler pixel validation may be more reliable for debugging coordinate system errors.

### 4. Incomplete Partiality Generation for Sequence Data

**Issue:** The sequence adapter may not generate partiality values required by downstream modules.

**Found:** The adapter validates partiality exists but `dials.integrate` in sequence mode may not generate it consistently.

**Resolution Required:** Verify partiality generation or implement fallback calculation for sequence processing.

## Unresolved Issues Requiring Human Decision

1. **Module Naming Convention:** Should Module 1.S.0 be inserted, renumbering subsequent modules, or added as Module 0.S.1?

2. **PHIL File Organization:** Should sequence processing PHIL parameters be in separate files or integrated into existing ones?

3. **Adapter IDL Granularity:** Should there be one comprehensive adapter IDL or separate ones for each adapter type?

4. **Data Type Detection Location:** Should data type detection be in the orchestrator, a separate module, or the processor?

5. **Validation Logic Strategy:** Should the project use complex q-vector consistency checks (plan.md) or simpler pixel position validation (debugging guide recommendation)?

6. **Adapter Architecture:** Should adapters directly wrap DIALS APIs or reimplement internal logic for consistency?

## Summary

**Major Finding:** The documentation review reveals deeper issues than initially identified. While documentation gaps exist (missing IDL files, incomplete plan.md integration), the more critical problems are:

1. **Fundamental Implementation Conflicts:** The current code contradicts the planned dual processing design by hardcoding sequence processing for all data.

2. **Validation Methodology Disagreement:** There's a fundamental split between the complex validation approach in plan.md and the simpler approach recommended in debugging documentation.

3. **Adapter Design Confusion:** The stills adapter doesn't properly use the DIALS Python API it's supposed to wrap.

**Priority Actions:**
1. **CRITICAL:** Resolve the validation logic discrepancy (plan.md vs debugging guide)
2. **CRITICAL:** Decide whether to implement true dual processing or standardize on sequence processing
3. **HIGH:** Complete plan.md integration with plan_adaptation.md
4. **HIGH:** Fix adapter implementations to match their intended roles
5. **MEDIUM:** Create missing IDL files for proper documentation

The project needs strategic decisions on validation methodology and processing architecture before documentation can be fully synchronized.

----

# decisions 
**1. DIALS Processing Strategy: Implement Full Dual-Path (Decision 1 - Option A)**

*   **Recommendation:** Commit to implementing the full dual-path strategy as outlined in `plan_adaptation.md`.
    *   **Rationale:**
        *   **Correctness:** DIALS tools are specialized. `dials.stills_process` is designed for true stills, while the sequential workflow (import, find_spots, index, integrate) is standard for oscillation/sequence data. Using the correct tool for the data type is more likely to yield optimal results and is more aligned with DIALS best practices.
        *   **Future-Proofing:** This provides a more flexible architecture if the pipeline needs to handle diverse datasets with varying characteristics.
        *   **Clarity:** Explicitly handling different data types leads to clearer processing logic and easier debugging.
    *   **Implementation Steps:**
        1.  Implement Module 1.S.0 (CBF Data Type Detection) in `StillProcessorAndValidatorComponent`.
        2.  Refactor `StillProcessorAndValidatorComponent` to instantiate and use the appropriate adapter (`DIALSStillsProcessAdapter` or `DIALSSequenceProcessAdapter`) based on the detected data type.
        3.  **Critically refactor `DIALSStillsProcessAdapter`**:
            *   It **must** be a thin wrapper around the `dials.command_line.stills_process.Processor` Python API.
            *   It should directly call `Processor.process_experiments()`.
            *   Its `_generate_phil_parameters` method should create PHIL parameters appropriate *only* for the `dials.stills_process` Python API (i.e., remove sequence-specific overrides like `geometry.convert_sequences_to_stills=false` and `indexing.method=fft3d` from *this* adapter, as these are for the CLI sequence).
        4.  Ensure `DIALSSequenceProcessAdapter` correctly implements all critical PHIL parameters specified in `plan_adaptation.md` for the CLI sequence.

**2. Validation Methodology: Formally Adopt Pixel-Based Validation (Decision 2)**

*   **Recommendation:** Formally adopt the pixel-based validation method (from `simple_validation_fix.py`) as the standard for Module 1.S.1.Validation.
    *   **Rationale:**
        *   **Robustness:** The `06_DIALS_DEBUGGING_GUIDE.md` indicates issues with complex Q-vector calculations due to coordinate system errors. Pixel-based validation is simpler and less prone to these specific errors.
        *   **Pragmatism:** It provides a reliable check on the geometric consistency of indexing. If pixel positions match, the derived Q-vectors are likely consistent.
        *   **Directly Actionable:** The logic is already available in `simple_validation_fix.py`.
    *   **Implementation Steps:**
        1.  Integrate the logic from `simple_validation_fix.py` into `ModelValidator._check_q_consistency` as previously detailed.
        2.  Update `plan.md` (Module 1.S.1.Validation) and `checklists/phase1.md` (Item 1.B.V.2) to describe this pixel-based method.
        3.  Add a new configuration field `pixel_position_tolerance_px: float` (e.g., default 2.0) to `ExtractionConfig` in `types_IDL.md` and `types_IDL.py`.
        4.  Update `create_default_extraction_config()` to include this new field.
        5.  Ensure `ModelValidator.validate_geometry` uses this new config field when calling `_check_q_consistency`.

**3. Role and Implementation of `DIALSStillsProcessAdapter` (Decision 3 - Follows from Decision 1)**

*   **Recommendation:** Refactor `DIALSStillsProcessAdapter` as described in Decision 1, Point 3.
    *   **Rationale:** Its current implementation is confusing and doesn't correctly leverage the `dials.stills_process.Processor` API. A clean wrapper is needed for the dual-path strategy.

**4. Partiality Handling for Sequence Data (Decision 4)**

*   **Recommendation:**
    1.  **Verify Output:** First, confirm whether `dials.integrate` (as called by `DIALSSequenceProcessAdapter`) *can* be configured to reliably output a `"partiality"` column. Standard DIALS integration should be capable of this.
    2.  **Ensure Configuration:** If it can, ensure `DIALSSequenceProcessAdapter` configures `dials.integrate` to do so.
    3.  **Adopt Universal `P_spot` Strategy:** Regardless of whether it's from `stills_process` or `sequence_process`, adopt the "Critical Partiality Handling Strategy" from `plan.md` (Module 3.S.3) universally:
        *   Use the `"partiality"` column (`P_spot`) **primarily as a quality filter** (e.g., `P_spot >= P_min_thresh`).
        *   **Do not use `P_spot` as a quantitative divisor** for intensity correction in the main diffuse data path or for the primary absolute scaling method (Krogh-Moe/Norman).
        *   The Wilson plot for *diagnostic* absolute scaling can still consider using high-partiality reflections as an approximation.
    *   **Rationale:** This simplifies the pipeline by having a consistent (and more cautious) approach to partiality, acknowledging its potential unreliability, especially for true stills and potentially for sequence data if not perfectly modeled. It shifts the burden of absolute scaling to methods that don't directly depend on precise per-reflection partiality correction of diffuse intensities.
    *   **Documentation:** Clearly document this universal partiality handling strategy in `plan.md`.

**5. PHIL File Organization and Centralization (Decision 5)**

*   **Recommendation:**
    1.  **Base PHIL Files:** Create base PHIL files in `src/diffusepipe/config/` for each step of the sequential DIALS workflow (`import.phil`, `find_spots_sequence.phil`, `index_sequence.phil`, `integrate_sequence.phil`). These should contain sensible defaults and the critical parameters identified.
    2.  **Adapter Usage:**
        *   `DIALSSequenceProcessAdapter` should load these base PHIL files for each respective CLI call and then apply any further specific overrides from `DIALSStillsProcessConfig.sequence_processing_phil_overrides` or hardcoded essential parameters.
        *   `DIALSStillsProcessAdapter` should also be able to take a `stills_process_phil_path` from `DIALSStillsProcessConfig` as its base, and then programmatically apply specific overrides relevant to the `Processor` API.
    3.  **Central Reference:** Consider creating a `docs/PHIL_PARAMETERS.md` or similar document that explains the key PHIL parameters used by each adapter and why, referencing the files in `src/diffusepipe/config/`.
    *   **Rationale:** This improves organization, makes default configurations more transparent, and allows easier modification of base settings without altering adapter code.

**Summary of Rationale for Recommendations:**

*   **Prioritize Correctness and Robustness:** The dual-path DIALS strategy and pixel-based validation aim for more reliable processing based on lessons learned.
*   **Adhere to DIALS Design:** Using DIALS tools as intended (stills vs. sequence) is generally better.
*   **Simplify Where Possible:** The universal partiality strategy simplifies one aspect of data correction.
*   **Improve Maintainability:** Clearer adapter roles and organized PHIL files enhance long-term maintainability.
*   **Iterative Improvement:** These decisions provide a solid foundation. The pipeline can be further optimized once this baseline is robust.

