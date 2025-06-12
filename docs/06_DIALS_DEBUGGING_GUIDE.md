# DIALS Integration Debugging Guide

## Overview

This guide provides comprehensive troubleshooting strategies for DIALS crystallography software integration issues based on real debugging experiences. It covers the most common failure modes and their solutions.

## Critical Issue: Stills vs Sequences Processing

### Problem Identification

**Symptoms:**
- DIALS processing fails with "no spots found" or indexing errors
- CBF files that work with manual DIALS commands fail in Python integration
- Inconsistent results between CLI DIALS and Python API usage

**Root Cause:**
CBF files with oscillation data (Angle_increment > 0) require sequence processing, not stills processing.

### Solution

**1. Check CBF Header:**
```bash
# Look for this key indicator in CBF headers:
# Angle_increment 0.1000 deg.  # Sequence data - use sequential workflow
# Angle_increment 0.0000 deg.  # Still data - use stills_process
```

**2. Use Correct Processing Mode:**
- **Stills data (0° oscillation):** Use `dials.stills_process`
- **Sequence data (>0° oscillation):** Use sequential CLI workflow

**3. Sequential Workflow Implementation:**
```python
# Correct approach for oscillation data:
subprocess.run(["dials.import", cbf_file])
subprocess.run(["dials.find_spots", "imported.expt"])  
subprocess.run(["dials.index", "imported.expt", "strong.refl"])
subprocess.run(["dials.integrate", "indexed.expt", "indexed.refl"])
```

## PHIL Parameter Issues

### Critical Parameters for Oscillation Data

```python
# Essential PHIL parameters that differ from defaults:
phil_overrides = [
    "spotfinder.filter.min_spot_size=3",          # Not default 2
    "spotfinder.threshold.algorithm=dispersion",   # Not default
    "indexing.method=fft3d",                       # Not fft1d  
    "geometry.convert_sequences_to_stills=false"   # Preserve oscillation
]
```

### Debugging PHIL Parameters

**1. Compare with Working Logs:**
- Check existing working DIALS processing directories
- Extract PHIL parameters from successful runs
- Compare against current parameters

**2. Test Parameter Changes Incrementally:**
- Change one parameter at a time
- Monitor spot finding results (should find ~100+ spots for good data)
- Validate indexing success

## Configuration Object Issues

### Common Configuration Errors

**Error Pattern:**
```
AttributeError: 'dict' object has no attribute 'spotfinder_threshold_algorithm'
```

**Root Cause:** Configuration object structure mismatch - code expects a structured object but receives a dictionary.

**Debugging Steps:**
1. **Verify Configuration Object Creation:**
   ```python
   # Check type of config object
   print(f"Config type: {type(config)}")
   print(f"Config contents: {config}")
   
   # Expected: <class 'diffusepipe.types.types_IDL.DIALSSequenceProcessConfig'>
   # Problem: <class 'dict'>
   ```

2. **Fix Configuration Construction:**
   ```python
   # WRONG - passing raw dict
   config = {"spotfinder_threshold_algorithm": "dispersion"}
   
   # CORRECT - using proper configuration class
   from diffusepipe.types.types_IDL import DIALSSequenceProcessConfig
   config = DIALSSequenceProcessConfig(
       spotfinder_threshold_algorithm="dispersion"
   )
   ```

3. **Validate Configuration Fields:**
   - Check that all required fields are present
   - Verify field names match the configuration class definition
   - Ensure proper type conversion (strings, floats, booleans)

### Data Type Detection Failures

**Log Pattern:**
```
num stills: 0
sweep: 1
```

**Interpretation:** DIALS import detects sequence data (oscillation), not still data.

**Common Issues:**
1. **Incorrect Adapter Selection:**
   - Code attempts to use `DIALSStillsProcessAdapter` on sequence data
   - Should use `DIALSSequenceProcessAdapter` for oscillation data

2. **CBF Header Detection Failure:**
   - Module 1.S.0 data type detection may be bypassed or failing
   - Manual check: `grep "Angle_increment" your_file.cbf`

3. **Configuration Override Issues:**
   - `force_processing_mode` setting may be incorrect
   - Routing logic may have bugs in condition checking

**Debugging Approach:**
```python
# Add explicit logging in data type detection
logger.info(f"CBF Angle_increment detected: {angle_increment}")
logger.info(f"Processing route selected: {processing_route}")
logger.info(f"Adapter type being used: {type(adapter)}")
```

## DIALS Python API Issues

### Import Problems

**Common Import Failures:**
```python
# This frequently breaks:
from dials.algorithms.indexing.indexer import Indexer  # WRONG

# Correct import:
from dials.algorithms.indexing import indexer
indexer_obj = indexer.Indexer.from_parameters(reflections, experiments, params)
```

**Solution Strategy:**
1. Check DIALS documentation: `libdocs/dials/DIALS_Python_API_Reference.md`
2. Test imports independently before integration
3. Use CLI-based adapters as fallback for unstable APIs

### API Stability Issues

**Problem:** DIALS Python API changes frequently between versions.

**Solutions:**
- Prefer CLI-based subprocess calls for production code
- Use Python API only for data access, not processing
- Implement fallback mechanisms

### PHIL Scope Import Changes

**Common Failure:**
```python
# This breaks in newer DIALS versions:
from dials.command_line.stills_process import master_phil_scope

# Correct approach:
from dials.command_line.stills_process import phil_scope
```

**Fix Strategy:**
1. Check DIALS version and documentation
2. Update import statements systematically
3. Test imports before main logic execution
4. Use try-catch blocks for version compatibility

## Validation Issues

### Primary Validation Method: Q-Vector Consistency

**Official Project Strategy:** The project uses **Q-vector consistency checking as the primary geometric validation method** (Module 1.S.1.Validation). This compares `q_model` (from DIALS-refined crystal models) with `q_observed` (recalculated from pixel coordinates).

**Key Implementation:**
- Calculate `q_model = s1 - s0` from DIALS reflection table
- Calculate `q_observed` by converting pixel coordinates to lab frame  
- Compare `|Δq| = |q_model - q_observed|` against `q_consistency_tolerance_angstrom_inv`
- Goal: `|Δq|` typically < 0.01 Å⁻¹ for good data

### Q-Vector Debugging Strategies

**Common Q-Vector Issues:**
- Physically impossible Q-vector magnitudes (>1.0 Å⁻¹)
- Large `|Δq|` discrepancies (>0.05 Å⁻¹)
- Coordinate system transformation errors

**Debugging Approach:**
1. **Verify coordinate transformations** in q_observed calculations
2. **Check reflection table columns** (`s1`, `xyzobs.px.value`, panel assignments)
3. **Validate detector geometry** and beam parameters

### Alternative Diagnostic: Pixel-Based Validation

**Use Case:** Pixel-based validation serves as a **diagnostic tool** or **simpler fallback** when Q-vector validation proves persistently problematic. It is **not** the primary validation method.

**Implementation:**
```python
def simple_position_validation(reflections, tolerance=2.0):
    """Use pixel position differences. Useful for debugging or as a simpler check."""
    obs_pos = reflections['xyzobs.px.value']
    calc_pos = reflections['xyzcal.px']
    
    # Calculate pixel distance differences
    dx = obs_pos[:, 0] - calc_pos[:, 0]
    dy = obs_pos[:, 1] - calc_pos[:, 1]
    distances = np.sqrt(dx*dx + dy*dy)
    
    # Reasonable tolerance: 1-2 pixels
    passed = np.mean(distances) <= tolerance
    return passed, distances
```
This pixel-based check avoids complex coordinate transformations and can help isolate whether the geometric model itself is poor or if the Q-vector calculations are problematic. However, strive to make the Q-vector validation the primary pass/fail criterion.

## Debugging Workflow

### Step-by-Step Approach

**1. Verify Data Type:**
```bash
# Check CBF header for oscillation
grep "Angle_increment" your_file.cbf
```

**2. Compare with Working Approach:**
```bash
# Find existing working logs
ls -la *_dials_processing/
# Compare PHIL parameters and workflow
```

**3. Test Incremental Changes:**
- Start with working PHIL parameters
- Test spot finding first (should find 100+ spots)
- Validate indexing success
- Check integration results

**4. Use Diagnostic Outputs:**
```python
# Log key metrics at each step
logger.info(f"Found {len(reflections)} spots")
logger.info(f"Indexed {len(indexed_reflections)} reflections")
logger.info(f"Integrated {len(integrated_reflections)} reflections")
```

### Common Debugging Commands

```bash
# Test manual DIALS workflow
dials.import your_file.cbf
dials.find_spots imported.expt
dials.index imported.expt strong.refl
dials.integrate indexed.expt indexed.refl

# Check logs for errors
cat dials.find_spots.log
cat dials.index.log
cat dials.integrate.log

# Verify output files
ls -la *.expt *.refl
```

## Performance Optimization

### CLI vs Python API

**When to Use CLI:**
- Production processing pipelines
- When API stability is a concern
- For complex DIALS workflows

**When to Use Python API:**
- Data access and analysis
- Simple operations on DIALS objects
- When tight integration is needed

### Error Recovery

**Graceful Fallbacks:**
```python
try:
    # Try Python API first
    result = dials_python_api_call()
except ImportError:
    # Fall back to CLI
    result = dials_cli_subprocess_call()
```

## Key Files and Locations

**Working Examples:**
- `lys_nitr_10_6_0491_dials_processing/` - Successful processing logs
- `lys_nitr_8_2_0110_dials_processing/` - Another working example

**Implementation Files:**
- `src/diffusepipe/adapters/dials_stills_process_adapter.py` - Python API adapter
- `src/diffusepipe/adapters/dials_sequence_process_adapter.py` - CLI adapter
- `src/diffusepipe/crystallography/still_processing_and_validation.py` - Validation logic

**Configuration:**
- `src/diffusepipe/config/find_spots.phil` - Spot finding parameters
- `src/diffusepipe/config/refine_detector.phil` - Refinement parameters

## Quick Reference

### Decision Tree

1. **Check CBF Angle_increment**
   - = 0°: Use stills_process
   - > 0°: Use sequential workflow

2. **If Sequential Workflow Fails:**
   - Check PHIL parameters against working logs
   - Verify spot finding (>100 spots expected)
   - Test DIALS API imports

3. **If Q-Vector Validation Fails:**
   - Debug coordinate transformations in q_observed calculation
   - Check reflection table data quality (s1 vectors, pixel positions)
   - Use pixel position validation as diagnostic tool
   - Verify detector geometry and beam parameters

### Success Criteria

- **Spot Finding:** 100+ spots found
- **Indexing:** Crystal model successfully determined
- **Integration:** 500+ reflections integrated
- **Q-Vector Validation:** Mean |Δq| < 0.01 Å⁻¹ (primary criterion)
- **Pixel Validation:** Pixel differences < 2 pixels (diagnostic/fallback)

This guide should be consulted whenever DIALS integration issues arise, as it captures hard-learned lessons from real debugging sessions.