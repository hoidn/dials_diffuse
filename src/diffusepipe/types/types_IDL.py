"""
Python implementation of types defined in types_IDL.md.

This module implements Pydantic models corresponding to the IDL struct definitions
for type safety and validation throughout the pipeline.
"""

from typing import Dict, Optional, Any
from pydantic import BaseModel, Field


class DIALSStillsProcessConfig(BaseModel):
    """Configuration for DIALS stills_process Python API execution by the orchestrator."""
    
    stills_process_phil_path: Optional[str] = Field(
        None, 
        description="Path to an existing, readable PHIL file containing comprehensive parameters for dials.stills_process"
    )
    known_unit_cell: Optional[str] = Field(
        None,
        description="Known unit cell for indexing, e.g., 'a,b,c,alpha,beta,gamma'"
    )
    known_space_group: Optional[str] = Field(
        None,
        description="Known space group for indexing, e.g., 'P1', 'C2'"
    )
    spotfinder_threshold_algorithm: Optional[str] = Field(
        None,
        description="Spot finding algorithm, e.g., 'dispersion'"
    )
    min_spot_area: Optional[int] = Field(
        None,
        description="Minimum spot area for spot finding"
    )
    output_shoeboxes: Optional[bool] = Field(
        None,
        description="If true, ensures shoeboxes are saved by dials.stills_process"
    )
    calculate_partiality: Optional[bool] = Field(
        True,
        description="If true, ensures partialities are calculated and output by dials.stills_process"
    )


class ExtractionConfig(BaseModel):
    """Parameters for the DataExtractor component."""
    
    min_res: Optional[float] = Field(
        None,
        description="Low-resolution limit (maximum d-spacing in Angstroms)"
    )
    max_res: Optional[float] = Field(
        None,
        description="High-resolution limit (minimum d-spacing in Angstroms)"
    )
    min_intensity: Optional[float] = Field(
        None,
        description="Minimum pixel intensity (after gain, corrections, background subtraction) to be included"
    )
    max_intensity: Optional[float] = Field(
        None,
        description="Maximum pixel intensity (after gain, corrections, background subtraction) to be included"
    )
    gain: float = Field(
        description="Detector gain factor applied to raw pixel intensities"
    )
    cell_length_tol: float = Field(
        description="Fractional tolerance for comparing DIALS-derived cell lengths with an external PDB reference"
    )
    cell_angle_tol: float = Field(
        description="Tolerance in degrees for comparing DIALS-derived cell angles with an external PDB reference"
    )
    orient_tolerance_deg: float = Field(
        description="Tolerance in degrees for comparing DIALS-derived crystal orientation with an external PDB reference"
    )
    q_consistency_tolerance_angstrom_inv: float = Field(
        description="Tolerance in Å⁻¹ for q-vector consistency checks in geometric model validation"
    )
    pixel_position_tolerance_px: float = Field(
        2.0,
        description="Tolerance in pixels for reflection position consistency checks."
    )
    pixel_step: int = Field(
        description="Process every Nth pixel (e.g., 1 for all pixels, 2 for every other)"
    )
    lp_correction_enabled: bool = Field(
        description="If true, Lorentz-Polarization correction is applied using DIALS Corrections API"
    )
    subtract_measured_background_path: Optional[str] = Field(
        None,
        description="Path to a pre-processed background image/map to be subtracted pixel-wise"
    )
    subtract_constant_background_value: Optional[float] = Field(
        None,
        description="A constant value to be subtracted from all pixels"
    )
    plot_diagnostics: bool = Field(
        description="If true, diagnostic plots are generated"
    )
    verbose: bool = Field(
        description="If true, enables verbose logging output during extraction"
    )


class RelativeScalingConfig(BaseModel):
    """Configuration for the relative scaling model."""
    
    refine_per_still_scale: bool = Field(
        True,
        description="If true, refines a per-still (or per-group) overall multiplicative scale factor"
    )
    refine_resolution_scale_multiplicative: bool = Field(
        False,
        description="If true, refines a 1D resolution-dependent multiplicative scale factor"
    )
    resolution_scale_bins: Optional[int] = Field(
        None,
        description="Number of bins for resolution-dependent scaling if enabled"
    )
    refine_additive_offset: bool = Field(
        False,
        description="If true, refines additive offset components (e.g., background terms)"
    )
    min_partiality_threshold: float = Field(
        0.1,
        description="Minimum P_spot threshold for including Bragg reflections in reference generation"
    )


class StillsPipelineConfig(BaseModel):
    """Overall pipeline configuration for processing stills."""
    
    dials_stills_process_config: DIALSStillsProcessConfig
    extraction_config: ExtractionConfig
    relative_scaling_config: RelativeScalingConfig
    run_consistency_checker: bool = Field(
        description="If true, ConsistencyChecker is run after successful extraction"
    )
    run_q_calculator: bool = Field(
        description="If true, QValueCalculator is run after successful extraction"
    )


class ComponentInputFiles(BaseModel):
    """Represents a set of related input file paths for a component."""
    
    cbf_image_path: Optional[str] = Field(
        None,
        description="Path to the primary CBF image file being processed"
    )
    dials_expt_path: Optional[str] = Field(
        None,
        description="Path to the DIALS experiment list JSON file (.expt)"
    )
    dials_refl_path: Optional[str] = Field(
        None,
        description="Path to the DIALS reflection table file (.refl)"
    )
    bragg_mask_path: Optional[str] = Field(
        None,
        description="Path to the DIALS-generated Bragg mask pickle file"
    )
    external_pdb_path: Optional[str] = Field(
        None,
        description="Path to an external PDB file used for consistency checks"
    )


class OperationOutcome(BaseModel):
    """Generic outcome for operations within components."""
    
    status: str = Field(
        description="Must be one of 'SUCCESS', 'FAILURE', 'WARNING'"
    )
    message: Optional[str] = Field(
        None,
        description="Human-readable message about the outcome"
    )
    error_code: Optional[str] = Field(
        None,
        description="A machine-readable code for specific error types"
    )
    output_artifacts: Optional[Dict[str, Any]] = Field(
        None,
        description="A map where keys are artifact names and values are their file paths or objects"
    )


class StillProcessingOutcome(BaseModel):
    """Outcome for processing a single still image through the main pipeline."""
    
    input_cbf_path: str = Field(
        description="Path to the original CBF file"
    )
    status: str = Field(
        description="Must be one of 'SUCCESS_ALL', 'SUCCESS_DIALS_ONLY', 'SUCCESS_EXTRACTION_ONLY', 'FAILURE_DIALS', 'FAILURE_EXTRACTION', 'FAILURE_DIAGNOSTICS'"
    )
    message: Optional[str] = Field(
        None,
        description="Overall message for this image's processing"
    )
    working_directory: str = Field(
        description="Path to the dedicated working directory"
    )
    dials_outcome: OperationOutcome = Field(
        description="Outcome of the DIALS processing steps"
    )
    extraction_outcome: OperationOutcome = Field(
        description="Outcome of the DataExtractor"
    )
    consistency_outcome: Optional[OperationOutcome] = Field(
        None,
        description="Outcome of the ConsistencyChecker, if run"
    )
    q_calc_outcome: Optional[OperationOutcome] = Field(
        None,
        description="Outcome of the QValueCalculator, if run"
    )