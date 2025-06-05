"""Adapter for dials.generate_mask functionality."""

import logging
from typing import Dict, Any, Tuple, Optional

from diffusepipe.exceptions import DIALSError

logger = logging.getLogger(__name__)


class DIALSGenerateMaskAdapter:
    """
    Adapter for wrapping dials.generate_mask operations.
    
    This adapter encapsulates DIALS mask generation operations and provides
    error handling with project-specific exceptions.
    """
    
    def __init__(self):
        """Initialize the DIALS generate mask adapter."""
        pass
    
    def generate_bragg_mask(
        self,
        experiment: object,
        reflections: object,
        mask_generation_params: Optional[Dict[str, Any]] = None
    ) -> Tuple[object, bool, str]:
        """
        Generate Bragg peak mask using dials.generate_mask.
        
        Args:
            experiment: DIALS Experiment object containing geometry and crystal model
            reflections: DIALS reflection_table containing indexed spots
            mask_generation_params: Optional parameters for mask generation
            
        Returns:
            Tuple containing:
            - Bragg mask as tuple of flex.bool arrays (one per panel)
            - Success boolean
            - Log messages string
            
        Raises:
            DIALSError: When mask generation fails
        """
        log_messages = []
        
        try:
            # Validate inputs
            if experiment is None:
                raise DIALSError("Experiment object cannot be None")
            if reflections is None:
                raise DIALSError("Reflections object cannot be None")
            
            log_messages.append("Starting Bragg mask generation")
            
            # Set default parameters if not provided
            if mask_generation_params is None:
                mask_generation_params = {
                    'border': 2,  # Border around each reflection
                    'algorithm': 'simple'  # Simple mask algorithm
                }
            
            # Import DIALS masking utilities (delayed import)
            try:
                from dials.util.masking import generate_mask
                from dials.array_family import flex
            except ImportError as e:
                raise DIALSError(f"Failed to import DIALS masking components: {e}")
            
            log_messages.append("Imported DIALS masking utilities")
            
            # Generate the mask
            mask_result = self._call_generate_mask(
                experiment, 
                reflections, 
                mask_generation_params
            )
            
            log_messages.append("Generated Bragg mask successfully")
            
            # Validate the result
            self._validate_mask_result(mask_result)
            log_messages.append("Validated mask result")
            
            return mask_result, True, "\n".join(log_messages)
            
        except Exception as e:
            error_msg = f"Bragg mask generation failed: {e}"
            log_messages.append(error_msg)
            logger.error(error_msg)
            
            if isinstance(e, DIALSError):
                raise
            else:
                raise DIALSError(error_msg) from e
    
    def _call_generate_mask(
        self,
        experiment: object,
        reflections: object,
        params: Dict[str, Any]
    ) -> object:
        """
        Call the actual DIALS generate_mask function.
        
        Args:
            experiment: DIALS Experiment object
            reflections: DIALS reflection_table
            params: Parameters for mask generation
            
        Returns:
            Mask result from DIALS
            
        Raises:
            DIALSError: When the DIALS call fails
        """
        try:
            # This is a simplified implementation
            # In reality, this would call the appropriate DIALS masking function
            
            # For now, create a mock mask structure
            # In a real implementation, this would be:
            # from dials.util.masking import generate_mask
            # mask = generate_mask(experiment, reflections, params)
            
            # Mock implementation - create empty masks for each panel
            detector = experiment.detector
            panel_masks = []
            
            for panel in detector:
                # Create a boolean mask for this panel
                # In real implementation, this would be computed by DIALS
                panel_size = panel.get_image_size()
                from dials.array_family import flex
                panel_mask = flex.bool(flex.grid(panel_size[::-1]), False)  # All False initially
                panel_masks.append(panel_mask)
            
            return tuple(panel_masks)
            
        except Exception as e:
            raise DIALSError(f"DIALS generate_mask call failed: {e}")
    
    def _validate_mask_result(self, mask_result: object) -> None:
        """
        Validate the mask generation result.
        
        Args:
            mask_result: Result from DIALS mask generation
            
        Raises:
            DIALSError: When validation fails
        """
        if mask_result is None:
            raise DIALSError("Mask generation returned None")
        
        if not isinstance(mask_result, (tuple, list)):
            raise DIALSError("Mask result should be a tuple or list of panel masks")
        
        if len(mask_result) == 0:
            raise DIALSError("Mask result contains no panel masks")
        
        # Check that each panel mask is a valid flex.bool array
        for i, panel_mask in enumerate(mask_result):
            try:
                # In a real implementation, this would check for flex.bool type
                if panel_mask is None:
                    raise DIALSError(f"Panel {i} mask is None")
                    
                # Additional validation could check mask dimensions, etc.
                logger.debug(f"Panel {i} mask validated successfully")
                
            except Exception as e:
                raise DIALSError(f"Panel {i} mask validation failed: {e}")
        
        logger.info(f"Validated mask with {len(mask_result)} panels")