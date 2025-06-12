# System-Wide Shared Type Definitions [Type:DiffusePipe:1.0]

> This document is the authoritative source for system-wide shared data structures and type definitions used across the DiffusePipe crystallography processing pipeline.
>
> These types are defined in the IDL specification file `src/diffusepipe/types/types_IDL.md` and are referenced by multiple components throughout the pipeline.
>
> Cross-reference: For implementation details and complete IDL specifications, see `src/diffusepipe/types/types_IDL.md`.

---

## Core Operation Types

### OperationOutcome [Type:DiffusePipe:OperationOutcome:1.0]

Generic outcome for operations within components, providing standardized return type for component methods.

```python
# Implementation: Pydantic BaseModel in src/diffusepipe/types/
class OperationOutcome:
    status: str                          # Must be "SUCCESS", "FAILURE", or "WARNING"
    message: Optional[str]               # Human-readable message about the outcome
    error_code: Optional[str]            # Machine-readable code for specific error types
    output_artifacts: Optional[Dict[str, str]]  # Map of artifact names to file paths
```

**Usage**: Returned by all major component methods to indicate success/failure with structured details.

### StillProcessingOutcome [Type:DiffusePipe:StillProcessingOutcome:1.0]

Comprehensive outcome for processing a single still image through the entire pipeline.

```python
class StillProcessingOutcome:
    input_cbf_path: str                  # Path to the original CBF file
    status: str                          # Pipeline status (see Status Values below)
    message: Optional[str]               # Overall processing message
    working_directory: str               # Path to dedicated working directory
    dials_outcome: OperationOutcome      # DIALS processing results
    extraction_outcome: OperationOutcome # DataExtractor results
    consistency_outcome: Optional[OperationOutcome]  # ConsistencyChecker results (if run)
    q_calc_outcome: Optional[OperationOutcome]       # QValueCalculator results (if run)
```

**Status Values**:
- `"SUCCESS_ALL"`: All stages completed successfully
- `"SUCCESS_DIALS_ONLY"`: DIALS succeeded, subsequent stages failed
- `"SUCCESS_EXTRACTION_ONLY"`: DIALS and extraction succeeded, diagnostics failed
- `"FAILURE_DIALS"`: DIALS processing failed
- `"FAILURE_EXTRACTION"`: DIALS succeeded but extraction failed
- `"FAILURE_DIAGNOSTICS"`: Processing succeeded but diagnostics failed

---

## Configuration Types

### DIALSStillsProcessConfig [Type:DiffusePipe:DIALSStillsProcessConfig:1.0]

Configuration for DIALS crystallographic processing, supporting both stills and sequence data.

```python
class DIALSStillsProcessConfig:
    stills_process_phil_path: Optional[str]          # Path to PHIL configuration file
    force_processing_mode: Optional[str]             # "stills", "sequence", or None for auto-detection
    sequence_processing_phil_overrides: Optional[List[str]]  # PHIL overrides for sequence processing
    data_type_detection_enabled: Optional[bool]     # Enable automatic data type detection (default: True)
    known_unit_cell: Optional[str]                  # Unit cell parameters for indexing
    known_space_group: Optional[str]                # Space group for indexing
    spotfinder_threshold_algorithm: Optional[str]   # Spot finding algorithm (e.g., "dispersion")
    min_spot_area: Optional[int]                    # Minimum spot area for detection
    output_shoeboxes: Optional[bool]                # Save shoeboxes for Bragg masking
    calculate_partiality: Optional[bool]            # Calculate partialities (default: True)
```

**Key Behavior**: 
- `force_processing_mode` overrides automatic CBF header detection
- `sequence_processing_phil_overrides` applied only when sequence processing is used
- Partialities used as quality threshold, not quantitative divisors

### ExtractionConfig [Type:DiffusePipe:ExtractionConfig:1.0]

Configuration for diffuse scattering data extraction and correction.

```python
class ExtractionConfig:
    # Resolution and intensity filtering
    min_res: Optional[float]                         # Low-resolution limit (max d-spacing in Å)
    max_res: Optional[float]                         # High-resolution limit (min d-spacing in Å)
    min_intensity: Optional[float]                   # Minimum pixel intensity threshold
    max_intensity: Optional[float]                   # Maximum pixel intensity threshold
    
    # Processing parameters
    gain: float                                      # Detector gain factor (required)
    pixel_step: int                                  # Process every Nth pixel (required)
    
    # Geometric validation tolerances
    cell_length_tol: float                           # Cell length tolerance (fractional)
    cell_angle_tol: float                            # Cell angle tolerance (degrees)
    orient_tolerance_deg: float                      # Orientation tolerance (degrees)
    q_consistency_tolerance_angstrom_inv: float      # Q-vector consistency tolerance (Å⁻¹)
    
    # Correction and background options
    lp_correction_enabled: bool                      # Apply Lorentz-Polarization correction
    subtract_measured_background_path: Optional[str] # Path to background map
    subtract_constant_background_value: Optional[float]  # Constant background value
    
    # Output and diagnostics
    plot_diagnostics: bool                           # Generate diagnostic plots
    verbose: bool                                    # Enable verbose logging
```

**Key Validation Tolerances**:
- `q_consistency_tolerance_angstrom_inv`: Used in Module 1.S.1.Validation for geometric model validation
- Typical values: `cell_length_tol=0.01` (1%), `q_consistency_tolerance_angstrom_inv=0.01` (0.01 Å⁻¹)

### RelativeScalingConfig [Type:DiffusePipe:RelativeScalingConfig:1.0]

Configuration for the relative scaling model (future implementation).

```python
class RelativeScalingConfig:
    refine_per_still_scale: bool                     # Refine multiplicative scale per still (default: True)
    refine_resolution_scale_multiplicative: bool     # Refine resolution-dependent scaling (default: False)
    resolution_scale_bins: Optional[int]            # Number of resolution bins if enabled
    refine_additive_offset: bool                     # MUST be False for v1 implementation
    min_partiality_threshold: float                  # P_spot threshold for Bragg reference (default: 0.1)
```

**Critical Note**: `refine_additive_offset` must be `False` in v1 to avoid parameter correlation issues.

---

## Pipeline Configuration

### StillsPipelineConfig [Type:DiffusePipe:StillsPipelineConfig:1.0]

Overall pipeline configuration encapsulating all processing stages.

```python
class StillsPipelineConfig:
    dials_stills_process_config: DIALSStillsProcessConfig  # DIALS processing settings
    extraction_config: ExtractionConfig                   # Diffuse extraction settings
    relative_scaling_config: RelativeScalingConfig        # Future scaling settings
    run_consistency_checker: bool                          # Enable q-vector consistency checking
    run_q_calculator: bool                                 # Enable q-map calculation
```

---

## File Management Types

### ComponentInputFiles [Type:DiffusePipe:ComponentInputFiles:1.0]

Standardized file path container for component dependencies.

```python
class ComponentInputFiles:
    cbf_image_path: Optional[str]        # Primary CBF image file
    dials_expt_path: Optional[str]       # DIALS experiment list (.expt)
    dials_refl_path: Optional[str]       # DIALS reflection table (.refl)
    bragg_mask_path: Optional[str]       # DIALS-generated Bragg mask (.pickle)
    external_pdb_path: Optional[str]     # External PDB for validation
```

**Usage Pattern**: Fields are optional to allow flexibility for different components. Components validate required fields in their preconditions.

---

## Processing Route Constants

### Data Type Detection [Type:DiffusePipe:ProcessingConstants:1.0]

```python
# Processing route identifiers (Module 1.S.0)
PROCESSING_ROUTE_STILLS = "stills"      # For Angle_increment = 0.0°
PROCESSING_ROUTE_SEQUENCE = "sequence"  # For Angle_increment > 0.0°

# Status constants
SUCCESS_STATUS = "SUCCESS"
FAILURE_STATUS = "FAILURE" 
WARNING_STATUS = "WARNING"
```

---

## Type Dependencies and Relationships

### Cross-Component Usage

1. **OperationOutcome**: Used by all major components (adapters, extractors, validators)
2. **ComponentInputFiles**: Standard input pattern for file-dependent components
3. **Configuration Types**: Passed down through orchestrator to individual components
4. **StillProcessingOutcome**: Aggregates results from all pipeline stages

### Implementation Notes

- All types implemented as Pydantic BaseModel classes for validation
- Optional fields use `Optional[Type]` annotation
- Required fields have no default values and must be explicitly provided
- Configuration validation occurs at component initialization

---

For detailed behavioral specifications and implementation requirements, refer to the complete IDL definitions in `src/diffusepipe/types/types_IDL.md`.