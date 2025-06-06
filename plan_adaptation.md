# Plan Adaptation: Supporting Both Stills and Sequence Data

## Overview

This document outlines the necessary adaptations to `plan.md` based on the critical discovery that CBF files with oscillation data (Angle_increment > 0°) cannot be processed using `dials.stills_process` and require a sequential DIALS workflow instead.

## Root Cause of Required Changes

**Original Assumption (Incorrect):**
- All CBF data can be processed using `dials.stills_process`
- Single processing pathway for all input data

**Reality:**
- CBF files with `Angle_increment > 0°` are **sequence data**, not stills
- `dials.stills_process` **fails** on oscillation data due to fundamental algorithmic assumptions
- Two distinct processing pathways required based on data type

## Impact Assessment

### Scope of Changes: **Limited and Surgical**

**What Remains Unchanged (85% of plan):**
- Core 4-phase pipeline structure
- All of Phase 2 (diffuse extraction and correction)
- All of Phase 3 (voxelization, scaling, merging)
- All of Phase 4 (absolute scaling)
- Output data structures and interfaces
- Testing framework and principles
- Most of Module 1.S.1 validation logic

**What Requires Adaptation (15% of plan):**
- Module 1.S.1 processing logic
- Addition of data type detection
- Adapter layer implementations
- Some testing scenarios

## Detailed Adaptation Requirements

### 1. New Module: Data Type Detection

**Insert before Module 1.S.1:**

```markdown
**Module 1.S.0: CBF Data Type Detection and Processing Route Selection**
*   **Action:** Analyze CBF file headers to determine if data is true stills (Angle_increment = 0°) or sequence data (Angle_increment > 0°), then route to appropriate processing pathway.
*   **Input (per still `i`):**
    *   File path to raw CBF image
*   **Process:**
    1.  Parse CBF header to extract `Angle_increment` value
    2.  Determine data type:
        *   IF `Angle_increment = 0.0`: Route to stills processing pathway
        *   IF `Angle_increment > 0.0`: Route to sequence processing pathway
    3.  Log data type determination for debugging
*   **Output:** 
    *   `data_type`: String ("stills" or "sequence")
    *   `processing_route`: Enum or string indicating which adapter to use
*   **Testing:**
    *   **Input:** Sample CBF files with known `Angle_increment` values
    *   **Verification:** Assert correct data type classification and routing decisions
```

### 2. Module 1.S.1 Process Section Adaptations

**Replace lines 95-107 in current plan:**

```markdown
*   **Process (Orchestrated per still `i` by the `StillsPipelineOrchestrator`, which determines processing route based on Module 1.S.0 output):**
    
    **Route A: True Stills Processing (Angle_increment = 0°):**
    1.  Initialize `dials.command_line.stills_process.Processor` via `DIALSStillsProcessAdapter`
    2.  Call `do_import()` using image file path and base geometry
    3.  Invoke `processor.process_experiments()` for spot finding, indexing, refinement, integration
    4.  Collect output `integrated_experiments` and `integrated_reflections`
    
    **Route B: Sequence Processing (Angle_increment > 0°):**
    1.  Use `DIALSSequenceProcessAdapter` with CLI-based sequential workflow
    2.  Execute `dials.import` with sequence-appropriate parameters
    3.  Execute `dials.find_spots` with critical PHIL parameters:
        *   `spotfinder.filter.min_spot_size=3`
        *   `spotfinder.threshold.algorithm=dispersion`
    4.  Execute `dials.index` with parameters:
        *   `indexing.method=fft3d`
        *   `geometry.convert_sequences_to_stills=false`
    5.  Execute `dials.integrate` with sequence integration
    6.  Load output experiment and reflection objects from generated files
    
    **Common Continuation:**
    5.  If processing reports success, proceed to Sub-Module 1.S.1.Validation
    6.  Retrieve `Experiment_dials_i` and `Reflections_dials_i` objects (identical structure regardless of processing route)
```

### 3. Adapter Layer Enhancements

**Update Section 0.6 (Adapter Layer):**

```markdown
**0.6 Adapter Layer Enhancement for Dual Processing Modes:**
External DIALS processing **shall be wrapped** in two complementary adapter implementations:

*   **`DIALSStillsProcessAdapter`:** Wraps `dials.stills_process` Python API for true still images (Angle_increment = 0°)
*   **`DIALSSequenceProcessAdapter`:** Implements CLI-based sequential workflow for oscillation data (Angle_increment > 0°)

Both adapters **must** produce identical output interfaces (`Experiment` and `reflection_table` objects) to ensure downstream compatibility. The choice between adapters is determined by Module 1.S.0 data type detection.

**Critical PHIL Parameters for Sequence Processing:**
The `DIALSSequenceProcessAdapter` must apply the following non-default parameters:
*   `spotfinder.filter.min_spot_size=3` (not default 2)
*   `spotfinder.threshold.algorithm=dispersion` (not default)
*   `indexing.method=fft3d` (not fft1d)
*   `geometry.convert_sequences_to_stills=false` (preserve oscillation)

These parameters were determined through systematic comparison with working manual DIALS processing.
```

### 4. Configuration Updates

**Enhance `DIALSStillsProcessConfig` in types_IDL.md:**

```markdown
**Additional Fields for Dual Processing Support:**
*   `force_processing_mode`: Optional[str] = None  # "stills", "sequence", or None for auto-detection
*   `sequence_processing_phil_overrides`: Optional[List[str]] = None  # PHIL parameters specific to sequence processing
*   `data_type_detection_enabled`: bool = True  # Enable/disable automatic data type detection
```

### 5. Testing Adaptations

**Add to Module 1.S.1 Testing section:**

```markdown
**Testing for Dual Processing Mode Support:**
*   **Data Type Detection Testing:**
    *   **Input:** CBF files with known `Angle_increment` values (0.0°, 0.1°, 0.5°)
    *   **Verification:** Assert correct routing to stills vs sequence processing
    
*   **Sequence Processing Adapter Testing:**
    *   **Input:** CBF file with oscillation data, sequence processing configuration
    *   **Execution:** Call `DIALSSequenceProcessAdapter.process_still()`
    *   **Verification:** Assert successful processing and correct output object types
    
*   **Processing Route Integration Testing:**
    *   **Input:** Mixed dataset with both stills and sequence CBF files
    *   **Verification:** Assert each file is processed with correct adapter and produces valid results
    
*   **PHIL Parameter Validation Testing:**
    *   **Input:** Sequence data with incorrect PHIL parameters (default values)
    *   **Verification:** Assert processing failure, then success with correct parameters
```

### 6. Validation Logic Adaptations

**Module 1.S.1.Validation remains largely unchanged**, but add diagnostic information:

```markdown
**Enhanced Validation Reporting:**
*   Include `processing_route_used` in validation output
*   Log critical PHIL parameters used for sequence processing
*   Add specific checks for sequence processing quality metrics (e.g., minimum reflections expected)
```

### 7. Error Handling and Logging

**Enhance error handling for dual processing modes:**

```markdown
**Processing Mode Error Handling:**
*   If auto-detected processing mode fails, log the failure with specific error details
*   For sequence processing, capture and report DIALS CLI error messages
*   Add fallback mechanisms (e.g., try alternative PHIL parameters if initial sequence processing fails)
*   Enhanced logging to distinguish between stills vs sequence processing failures
```

## Implementation Strategy

### Phase 1: Core Infrastructure
1. Implement `DIALSSequenceProcessAdapter` based on working CLI approach
2. Add data type detection logic (CBF header parsing)
3. Update `StillsPipelineOrchestrator` to route based on data type

### Phase 2: Integration and Testing
1. Integrate both adapters into Module 1.S.1 processing logic
2. Implement comprehensive test suite for both processing modes
3. Validate that output objects are identical regardless of processing route

### Phase 3: Configuration and Documentation
1. Update configuration classes and IDL specifications
2. Enhance error handling and logging for dual processing modes
3. Update documentation to reflect dual processing capability

## Backward Compatibility

**Maintaining Existing Functionality:**
- All existing stills processing functionality remains unchanged
- Original `DIALSStillsProcessAdapter` preserved for true stills data
- Default behavior maintains auto-detection with fallback to stills processing
- Configuration option to force specific processing mode if needed

## Success Criteria

**The adaptation is successful when:**
1. Both stills (Angle_increment = 0°) and sequence (Angle_increment > 0°) data process correctly
2. Output objects (`Experiment_dials_i`, `Reflections_dials_i`) are identical in structure regardless of processing route
3. All downstream modules (Phases 2-4) work unchanged with both data types
4. Processing failures are correctly diagnosed and logged with appropriate error messages
5. Test suite validates both processing pathways comprehensively

## Risk Mitigation

**Potential Risks and Mitigations:**
1. **API Stability:** Sequence processing uses CLI calls, reducing dependency on unstable Python APIs
2. **Performance Impact:** CLI-based processing may be slower but is more reliable
3. **Configuration Complexity:** Auto-detection minimizes user configuration burden
4. **Output Compatibility:** Extensive testing ensures identical output object structures

## Future Considerations

**Potential Future Enhancements:**
1. Optimize sequence processing performance through batch CLI calls
2. Add support for micro-rotation data (small but non-zero oscillations)
3. Implement hybrid processing modes for edge cases
4. Add advanced data type classification beyond simple Angle_increment checking

## Conclusion

The required adaptations to `plan.md` are **surgical rather than fundamental**. The core insight is that both processing pathways produce identical output objects, meaning 85% of the pipeline remains unchanged. The primary adaptation is adding intelligent routing logic and a robust sequence processing adapter while preserving all existing stills processing functionality.

This adaptation transforms the pipeline from a single-mode processor into a **data-type-aware processor** that automatically selects the appropriate DIALS workflow based on the experimental data characteristics.