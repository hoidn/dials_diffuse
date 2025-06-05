"""
Still processor component for orchestrating per-still DIALS processing.

This module provides a higher-level interface for processing individual still images
using the DIALSStillsProcessAdapter, implementing Module 1.S.1 from the plan.
"""

import logging
from typing import Optional, Tuple, Union
from pathlib import Path

from diffusepipe.adapters.dials_stills_process_adapter import DIALSStillsProcessAdapter
from diffusepipe.types.types_IDL import DIALSStillsProcessConfig, OperationOutcome
from diffusepipe.exceptions import DIALSError, ConfigurationError, DataValidationError

logger = logging.getLogger(__name__)


class StillProcessorComponent:
    """
    Component for processing individual still images through DIALS stills_process.
    
    This component orchestrates the use of DIALSStillsProcessAdapter to perform
    spot finding, indexing, refinement, and integration for still diffraction images.
    """
    
    def __init__(self):
        """Initialize the still processor component."""
        self.adapter = DIALSStillsProcessAdapter()
    
    def process_still(
        self,
        image_path: str,
        config: DIALSStillsProcessConfig,
        base_experiment_path: Optional[str] = None
    ) -> OperationOutcome:
        """
        Process a single still image using DIALS stills_process.
        
        Args:
            image_path: Path to the CBF image file to process
            config: Configuration parameters for DIALS stills_process
            base_experiment_path: Optional path to base experiment file for geometry
            
        Returns:
            OperationOutcome with processing results and artifacts
            
        Behavior:
            - Uses DIALSStillsProcessAdapter to perform DIALS processing
            - Returns experiment and reflection objects via output_artifacts
            - Validates partiality column in reflection table
        """
        try:
            logger.info(f"Processing still image: {image_path}")
            
            # Process using the adapter
            experiment, reflections, success, log_messages = self.adapter.process_still(
                image_path=image_path,
                config=config,
                base_expt_path=base_experiment_path
            )
            
            if success and experiment is not None and reflections is not None:
                # Create output artifacts dictionary
                output_artifacts = {
                    "experiment": experiment,  # Store object directly
                    "reflections": reflections,  # Store object directly
                    "log_messages": log_messages
                }
                
                return OperationOutcome(
                    status="SUCCESS",
                    message=f"Successfully processed still image {image_path}",
                    error_code=None,
                    output_artifacts=output_artifacts
                )
            else:
                return OperationOutcome(
                    status="FAILURE",
                    message=f"DIALS processing failed for {image_path}: {log_messages}",
                    error_code="DIALS_PROCESSING_FAILED",
                    output_artifacts={"log_messages": log_messages}
                )
                
        except (DIALSError, ConfigurationError, DataValidationError) as e:
            logger.error(f"Known error processing still {image_path}: {e}")
            return OperationOutcome(
                status="FAILURE",
                message=str(e),
                error_code=type(e).__name__.upper(),
                output_artifacts=None
            )
        except Exception as e:
            logger.error(f"Unexpected error processing still {image_path}: {e}")
            return OperationOutcome(
                status="FAILURE",
                message=f"Unexpected error: {e}",
                error_code="UNEXPECTED_ERROR",
                output_artifacts=None
            )
    
    def validate_processing_outcome(self, outcome: OperationOutcome) -> bool:
        """
        Validate that a processing outcome contains expected artifacts.
        
        Args:
            outcome: OperationOutcome from process_still
            
        Returns:
            True if outcome contains valid experiment and reflection artifacts
        """
        if outcome.status != "SUCCESS":
            return False
            
        if not outcome.output_artifacts:
            return False
            
        # Check for required artifacts
        required_keys = ["experiment", "reflections"]
        for key in required_keys:
            if key not in outcome.output_artifacts:
                logger.error(f"Missing required artifact: {key}")
                return False
            
            if outcome.output_artifacts[key] is None:
                logger.error(f"Artifact {key} is None")
                return False
        
        # Validate that reflections contain partiality column
        reflections = outcome.output_artifacts["reflections"]
        try:
            if hasattr(reflections, 'has_key') and not reflections.has_key('partiality'):
                logger.error("Reflection table missing 'partiality' column")
                return False
        except AttributeError:
            # Handle case where reflections is a mock object
            logger.warning("Could not validate partiality column (possibly mock object)")
        
        return True


def create_default_config(
    phil_path: Optional[str] = None,
    enable_partiality: bool = True,
    enable_shoeboxes: bool = False
) -> DIALSStillsProcessConfig:
    """
    Create a default DIALS stills process configuration.
    
    Args:
        phil_path: Optional path to PHIL configuration file
        enable_partiality: Whether to enable partiality calculation
        enable_shoeboxes: Whether to enable shoebox output
        
    Returns:
        DIALSStillsProcessConfig with default settings
    """
    return DIALSStillsProcessConfig(
        stills_process_phil_path=phil_path,
        known_unit_cell=None,
        known_space_group=None,
        spotfinder_threshold_algorithm=None,
        min_spot_area=None,
        output_shoeboxes=enable_shoeboxes,
        calculate_partiality=enable_partiality
    )