"""Adapter for dials.stills_process Python API."""

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

from diffusepipe.exceptions import DIALSError, ConfigurationError, DataValidationError
from diffusepipe.types.types_IDL import DIALSStillsProcessConfig

logger = logging.getLogger(__name__)


class DIALSStillsProcessAdapter:
    """
    Thin wrapper around dials.stills_process.Processor for true still images.
    
    This adapter should only be used for CBF files with Angle_increment = 0.0°.
    For oscillation data (Angle_increment > 0.0°), use DIALSSequenceProcessAdapter instead.
    """
    
    def __init__(self):
        """Initialize the DIALS stills process adapter."""
        self._processor = None
        self._extracted_params = None
        
    def process_still(
        self,
        image_path: str,
        config: DIALSStillsProcessConfig,
        base_expt_path: Optional[str] = None
    ) -> Tuple[Optional[object], Optional[object], bool, str]:
        """
        Process a single still image using dials.stills_process.Processor.
        
        Args:
            image_path: Path to the CBF image file to process (must be true still)
            config: Configuration parameters for dials.stills_process
            base_expt_path: Optional path to base experiment file for geometry
            
        Returns:
            Tuple containing:
            - Experiment object (or None if failed)
            - Reflection table object (or None if failed) 
            - Success boolean
            - Log messages string
            
        Raises:
            DIALSError: When DIALS operations fail
            ConfigurationError: When configuration is invalid
            DataValidationError: When partiality data is missing
        """
        log_messages = []
        
        try:
            # Validate inputs
            if not Path(image_path).exists():
                raise ConfigurationError(f"Image file does not exist: {image_path}")
                
            # Generate PHIL parameters for true stills processing
            logger.info("Generating PHIL parameters for stills processing...")
            phil_params = self._generate_phil_parameters(config)
            self._extracted_params = phil_params.extract()
            log_messages.append(f"Generated PHIL parameters for {image_path}")
            
            # Import DIALS components
            try:
                logger.info("Importing DIALS components...")
                from dials.command_line.stills_process import Processor
                from dials.command_line.stills_process import do_import
                logger.info("Successfully imported DIALS components")
            except ImportError as e:
                raise DIALSError(f"Failed to import DIALS components: {e}")
            
            # Step 1: Import experiments
            logger.info("Step 1: Importing experiments...")
            if base_expt_path and Path(base_expt_path).exists():
                from dxtbx.model.experiment_list import ExperimentListFactory
                experiments = ExperimentListFactory.from_json_file(base_expt_path)
                log_messages.append(f"Loaded base experiment from {base_expt_path}")
            else:
                experiments = do_import([image_path])
                log_messages.append(f"Imported experiment from {image_path}")
            
            if not experiments or len(experiments) == 0:
                raise DIALSError(f"Failed to import experiments from {image_path}")
            
            # Step 2: Initialize and run the Processor
            logger.info("Step 2: Initializing stills_process.Processor...")
            self._processor = Processor(params=self._extracted_params)
            
            # Step 3: Process experiments using the Processor
            logger.info("Step 3: Processing experiments with stills_process.Processor...")
            tag = Path(image_path).stem  # Use filename as tag
            self._processor.process_experiments(tag=tag, experiments=experiments)
            
            # Step 4: Extract results
            logger.info("Step 4: Extracting results...")
            integrated_experiments = self._processor.all_integrated_experiments
            integrated_reflections = self._processor.all_integrated_reflections
            
            if not integrated_experiments or len(integrated_experiments) == 0:
                raise DIALSError("stills_process produced no integrated experiments")
                
            if not integrated_reflections or len(integrated_reflections) == 0:
                raise DIALSError("stills_process produced no integrated reflections")
            
            log_messages.append("Completed DIALS stills processing")
            
            # Extract single experiment and reflections
            experiment = self._extract_experiment(integrated_experiments)
            reflections = self._extract_reflections(integrated_reflections)
            
            # Validate partiality column
            self._validate_partiality(reflections)
            log_messages.append("Validated partiality data")
            
            return experiment, reflections, True, "\n".join(log_messages)
            
        except Exception as e:
            error_msg = f"DIALS stills processing failed: {e}"
            log_messages.append(error_msg)
            logger.error(error_msg)
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Exception details: {str(e)}")
            
            if isinstance(e, (DIALSError, ConfigurationError, DataValidationError)):
                raise
            else:
                raise DIALSError(error_msg) from e
    
    def _generate_phil_parameters(self, config: DIALSStillsProcessConfig) -> object:
        """
        Generate PHIL parameters for true stills processing.
        
        This method generates parameters appropriate for dials.stills_process,
        which is designed for true still images (Angle_increment = 0.0°).
        
        Args:
            config: Configuration object containing DIALS parameters
            
        Returns:
            PHIL parameter object for dials.stills_process
            
        Raises:
            ConfigurationError: When PHIL generation fails
        """
        try:
            from dials.command_line.stills_process import phil_scope as master_phil_scope
            from libtbx.phil import parse
            
            # Start with DIALS stills_process master scope
            phil_scope = master_phil_scope
            
            # Apply base PHIL file if provided
            if config.stills_process_phil_path:
                phil_path = Path(config.stills_process_phil_path)
                if not phil_path.exists():
                    raise ConfigurationError(f"PHIL file not found: {config.stills_process_phil_path}")
                
                with open(phil_path, 'r') as f:
                    phil_content = f.read()
                user_phil = parse(phil_content)
                phil_scope = phil_scope.fetch(user_phil)
            
            # Apply configuration overrides for stills processing
            phil_overrides = []
            
            if config.known_unit_cell:
                logger.info(f"Adding known unit cell: {config.known_unit_cell}")
                phil_overrides.append(f"indexing.known_symmetry.unit_cell={config.known_unit_cell}")
                
            if config.known_space_group:
                logger.info(f"Adding known space group: {config.known_space_group}")
                phil_overrides.append(f"indexing.known_symmetry.space_group={config.known_space_group}")
                
            # Use default stills_process parameters (don't override with sequence parameters)
            if config.spotfinder_threshold_algorithm:
                phil_overrides.append(f"spotfinder.threshold.algorithm={config.spotfinder_threshold_algorithm}")
                
            if config.min_spot_area is not None:
                phil_overrides.append(f"spotfinder.filter.min_spot_area={config.min_spot_area}")
                
            if config.output_shoeboxes is not None:
                if config.output_shoeboxes:
                    phil_overrides.append("integration.lookup.mask=pixels shoeboxes")
                
            if config.calculate_partiality is not None:
                phil_overrides.append(f"integration.summation.estimate_partiality={config.calculate_partiality}")
            
            # Combine base PHIL with overrides
            if phil_overrides:
                override_text = "\n".join(phil_overrides)
                logger.info(f"Applying PHIL overrides:\n{override_text}")
                if phil_scope:
                    phil_scope = phil_scope.fetch(parse(override_text))
                else:
                    phil_scope = parse(override_text)
            
            # Debug: show final PHIL structure
            if phil_scope:
                logger.debug(f"Final PHIL scope: {phil_scope.as_str()}")
            
            return phil_scope
            
        except Exception as e:
            raise ConfigurationError(f"Failed to generate PHIL parameters: {e}")
    
    def _extract_experiment(self, integrated_experiments: object) -> Optional[object]:
        """
        Extract single experiment from integrated experiments.
        
        Args:
            integrated_experiments: Result from stills processing
            
        Returns:
            Single Experiment object or None
        """
        if integrated_experiments and len(integrated_experiments) > 0:
            return integrated_experiments[0]
        return None
    
    def _extract_reflections(self, integrated_reflections: object) -> Optional[object]:
        """
        Extract reflection table from integrated reflections.
        
        Args:
            integrated_reflections: Result from stills processing
            
        Returns:
            Reflection table object or None
        """
        return integrated_reflections
    
    def _validate_partiality(self, reflections: Optional[object]) -> None:
        """
        Validate that reflection table contains partiality column.
        
        Args:
            reflections: Reflection table to validate
            
        Raises:
            DataValidationError: When partiality column is missing
        """
        if reflections is None:
            return
            
        try:
            # Check for partiality column - this would be actual DIALS reflection table
            if not hasattr(reflections, 'has_key') or not reflections.has_key('partiality'):
                raise DataValidationError("Reflection table missing required 'partiality' column")
                
            # Validate partiality values are reasonable
            partialities = reflections['partiality']
            if len(partialities) == 0:
                logger.warning("No reflections with partiality values found")
            else:
                logger.info(f"Found {len(partialities)} reflections with partiality values")
                
        except AttributeError:
            # In case of mock objects during testing
            logger.warning("Could not validate partiality column (possibly mock object)")
            pass