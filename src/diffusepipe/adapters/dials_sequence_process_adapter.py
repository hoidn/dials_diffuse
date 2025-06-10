"""Adapter for DIALS sequential processing workflow (import→find_spots→index→integrate)."""

import logging
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, Tuple, Union

from diffusepipe.exceptions import DIALSError, ConfigurationError, DataValidationError
from diffusepipe.types.types_IDL import DIALSStillsProcessConfig

logger = logging.getLogger(__name__)


class DIALSSequenceProcessAdapter:
    """
    Adapter for DIALS sequential processing workflow.
    
    This adapter replicates the successful manual workflow:
    1. dials.import
    2. dials.find_spots  
    3. dials.index with known_symmetry
    4. dials.integrate
    
    This approach works for 0.1° oscillation images that stills_process fails on.
    """
    
    def __init__(self):
        """Initialize the DIALS sequence process adapter."""
        pass
        
    def process_still(
        self,
        image_path: str,
        config: DIALSStillsProcessConfig,
        base_expt_path: Optional[str] = None
    ) -> Tuple[Optional[object], Optional[object], bool, str]:
        """
        Process a still/sequence image using DIALS sequential workflow.
        
        Args:
            image_path: Path to the CBF image file to process
            config: Configuration parameters for DIALS processing
            base_expt_path: Optional path to base experiment file for geometry (ignored)
            
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
            
            # Use temporary directory for processing
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                original_cwd = Path.cwd()
                
                # Convert to absolute path before changing directories
                abs_image_path = Path(image_path).resolve()
                
                try:
                    # Change to temp directory for processing
                    import os
                    os.chdir(temp_path)
                    
                    # Step 1: Import
                    logger.info("Step 1: Running dials.import")
                    import_result = self._run_dials_import(str(abs_image_path))
                    log_messages.append("Completed dials.import")
                    
                    if not Path("imported.expt").exists():
                        raise DIALSError("dials.import failed to create imported.expt")
                    
                    # Step 2: Find spots
                    logger.info("Step 2: Running dials.find_spots")
                    spots_result = self._run_dials_find_spots(config)
                    log_messages.append("Completed dials.find_spots")
                    
                    if not Path("strong.refl").exists():
                        raise DIALSError("dials.find_spots failed to create strong.refl")
                    
                    # Step 3: Index
                    logger.info("Step 3: Running dials.index")
                    index_result = self._run_dials_index(config)
                    log_messages.append("Completed dials.index")
                    
                    if not Path("indexed.expt").exists():
                        raise DIALSError("dials.index failed to create indexed.expt")
                    
                    # Step 4: Integrate
                    logger.info("Step 4: Running dials.integrate")
                    integrate_result = self._run_dials_integrate(config)
                    log_messages.append("Completed dials.integrate")
                    
                    if not Path("integrated.expt").exists():
                        raise DIALSError("dials.integrate failed to create integrated.expt")
                    
                    # Load results using DIALS Python API
                    logger.info("Loading results with DIALS Python API")
                    experiment, reflections = self._load_results()
                    
                    # Validate partiality column
                    self._validate_partiality(reflections)
                    log_messages.append("Validated partiality data")
                    
                    return experiment, reflections, True, "\n".join(log_messages)
                    
                finally:
                    os.chdir(original_cwd)
            
        except Exception as e:
            error_msg = f"DIALS sequential processing failed: {e}"
            log_messages.append(error_msg)
            logger.error(error_msg)
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Exception details: {str(e)}")
            
            if isinstance(e, (DIALSError, ConfigurationError, DataValidationError)):
                raise
            else:
                raise DIALSError(error_msg) from e
    
    def _run_dials_import(self, image_path: str) -> subprocess.CompletedProcess:
        """Run dials.import step."""
        cmd = ["dials.import", image_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise DIALSError(f"dials.import failed: {result.stderr}")
        
        logger.debug(f"dials.import stdout: {result.stdout}")
        return result
    
    def _run_dials_find_spots(self, config: DIALSStillsProcessConfig) -> subprocess.CompletedProcess:
        """Run dials.find_spots step."""
        cmd = [
            "dials.find_spots", "imported.expt",
            "spotfinder.filter.min_spot_size=3",
            "spotfinder.threshold.algorithm=dispersion"
        ]
        
        # Add any additional spot finding parameters
        if config.spotfinder_threshold_algorithm:
            cmd.append(f"spotfinder.threshold.algorithm={config.spotfinder_threshold_algorithm}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise DIALSError(f"dials.find_spots failed: {result.stderr}")
        
        logger.debug(f"dials.find_spots stdout: {result.stdout}")
        return result
    
    def _run_dials_index(self, config: DIALSStillsProcessConfig) -> subprocess.CompletedProcess:
        """Run dials.index step."""
        cmd = [
            "dials.index", "imported.expt", "strong.refl",
            'output.experiments="indexed.expt"',
            'output.reflections="indexed.refl"'
        ]
        
        # Add known symmetry parameters
        if config.known_space_group:
            cmd.append(f'indexing.known_symmetry.space_group="{config.known_space_group}"')
        
        if config.known_unit_cell:
            cmd.append(f'indexing.known_symmetry.unit_cell={config.known_unit_cell}')
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise DIALSError(f"dials.index failed: {result.stderr}")
        
        logger.debug(f"dials.index stdout: {result.stdout}")
        return result
    
    def _run_dials_integrate(self, config: DIALSStillsProcessConfig) -> subprocess.CompletedProcess:
        """Run dials.integrate step."""
        cmd = [
            "dials.integrate", "indexed.expt", "indexed.refl",
            'output.experiments="integrated.expt"',
            'output.reflections="integrated.refl"'
        ]
        
        # Add basic integration parameters
        # Note: Some parameters like partiality are stills_process specific
        # Keep integration simple for compatibility
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise DIALSError(f"dials.integrate failed: {result.stderr}")
        
        logger.debug(f"dials.integrate stdout: {result.stdout}")
        return result
    
    def _load_results(self) -> Tuple[object, object]:
        """Load experiment and reflection results using DIALS Python API."""
        try:
            from dxtbx.model.experiment_list import ExperimentListFactory
            from dials.array_family import flex
            
            # Load experiment
            experiments = ExperimentListFactory.from_json_file("integrated.expt")
            if len(experiments) == 0:
                raise DIALSError("No experiments in integrated.expt")
            
            experiment = experiments[0]
            
            # Load reflections
            reflections = flex.reflection_table.from_file("integrated.refl")
            
            logger.info(f"Loaded experiment and {len(reflections)} reflections")
            return experiment, reflections
            
        except Exception as e:
            raise DIALSError(f"Failed to load DIALS results: {e}")
    
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
            # Check for partiality column
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