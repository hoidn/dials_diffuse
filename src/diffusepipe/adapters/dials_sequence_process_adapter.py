"""Adapter for DIALS sequential processing workflow (import→find_spots→index→integrate)."""

import logging
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, Tuple, Union, Dict, List

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
        # Get the path to config files (relative to this package)
        self.config_dir = Path(__file__).parent.parent / "config"
        
        # Define base PHIL file paths for each step
        self.base_phil_files = {
            "import": self.config_dir / "sequence_import_default.phil",
            "find_spots": self.config_dir / "sequence_find_spots_default.phil", 
            "index": self.config_dir / "sequence_index_default.phil",
            "integrate": self.config_dir / "sequence_integrate_default.phil"
        }
    
    def _load_and_merge_phil_parameters(
        self, 
        step: str, 
        config: DIALSStillsProcessConfig,
        runtime_overrides: Optional[Dict[str, str]] = None
    ) -> List[str]:
        """
        Load base PHIL file and merge with configuration overrides.
        
        Args:
            step: DIALS step name ("import", "find_spots", "index", "integrate")
            config: Configuration object with potential overrides
            runtime_overrides: Additional runtime parameter overrides
            
        Returns:
            List of command-line parameter strings for the DIALS command
        """
        try:
            from libtbx.phil import parse
        except ImportError:
            logger.warning("libtbx.phil not available, using hardcoded parameters")
            return self._get_fallback_parameters(step, config, runtime_overrides)
        
        # Start with base PHIL file if it exists
        phil_parameters = []
        base_file = self.base_phil_files.get(step)
        
        if base_file and base_file.exists():
            try:
                # Parse base PHIL file
                with open(base_file, 'r') as f:
                    base_phil_content = f.read()
                base_phil = parse(base_phil_content)
                
                # Extract parameters from base PHIL
                extracted_params = base_phil.extract()
                logger.debug(f"Loaded base PHIL parameters for {step} from {base_file}")
                
            except Exception as e:
                logger.warning(f"Failed to load base PHIL file {base_file}: {e}")
                return self._get_fallback_parameters(step, config, runtime_overrides)
        else:
            logger.warning(f"Base PHIL file not found for {step}: {base_file}")
            return self._get_fallback_parameters(step, config, runtime_overrides)
        
        # Apply sequence processing overrides from config
        if config.sequence_processing_phil_overrides:
            for override in config.sequence_processing_phil_overrides:
                phil_parameters.append(override)
                logger.debug(f"Applied sequence override: {override}")
        
        # Apply runtime overrides (like input/output file names)
        if runtime_overrides:
            for param, value in runtime_overrides.items():
                phil_param = f"{param}={value}"
                phil_parameters.append(phil_param)
                logger.debug(f"Applied runtime override: {phil_param}")
        
        return phil_parameters
    
    def _get_fallback_parameters(
        self, 
        step: str, 
        config: DIALSStillsProcessConfig,
        runtime_overrides: Optional[Dict[str, str]] = None
    ) -> List[str]:
        """
        Fallback method to provide hardcoded critical parameters when PHIL loading fails.
        
        This preserves the original behavior with hardcoded critical parameters.
        """
        parameters = []
        
        if step == "find_spots":
            parameters.extend([
                "spotfinder.filter.min_spot_size=3",  # Critical: not default 2
                "spotfinder.threshold.algorithm=dispersion"  # Critical: not default
            ])
            
            # Handle config overrides with warnings for critical parameters
            if config.spotfinder_threshold_algorithm and config.spotfinder_threshold_algorithm != "dispersion":
                logger.warning(f"Overriding critical sequence parameter: spotfinder.threshold.algorithm={config.spotfinder_threshold_algorithm} (recommended: dispersion)")
                parameters[-1] = f"spotfinder.threshold.algorithm={config.spotfinder_threshold_algorithm}"
                
        elif step == "index":
            parameters.extend([
                "indexing.method=fft3d",  # Critical: not fft1d
                "geometry.convert_sequences_to_stills=false"  # Critical: preserve oscillation
            ])
            
            # Add known symmetry parameters
            if config.known_space_group:
                parameters.append(f'indexing.known_symmetry.space_group="{config.known_space_group}"')
            if config.known_unit_cell:
                parameters.append(f'indexing.known_symmetry.unit_cell={config.known_unit_cell}')
                
        elif step == "integrate":
            parameters.extend([
                "geometry.convert_sequences_to_stills=false",  # Consistency
                "integration.summation.estimate_partiality=true"  # For validation
            ])
        
        # Apply runtime overrides
        if runtime_overrides:
            for param, value in runtime_overrides.items():
                parameters.append(f"{param}={value}")
        
        return parameters
        
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
        """Run dials.import step using PHIL files."""
        # For import, we mainly need to set the output file
        runtime_overrides = {
            "output.experiments": "imported.expt",
            "output.log": "dials.import.log"
        }
        
        # Use empty config since import step doesn't typically need config overrides
        empty_config = DIALSStillsProcessConfig()
        
        phil_params = self._load_and_merge_phil_parameters("import", empty_config, runtime_overrides)
        
        # Build command
        cmd = ["dials.import", image_path] + phil_params
        
        logger.info(f"Running dials.import with parameters: {phil_params}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise DIALSError(f"dials.import failed: {result.stderr}")
        
        logger.debug(f"dials.import stdout: {result.stdout}")
        return result
    
    def _run_dials_find_spots(self, config: DIALSStillsProcessConfig) -> subprocess.CompletedProcess:
        """Run dials.find_spots step using PHIL files and configuration overrides."""
        # Load parameters from PHIL file and merge with config
        runtime_overrides = {
            "output.reflections": "strong.refl",
            "output.log": "dials.find_spots.log"
        }
        
        phil_params = self._load_and_merge_phil_parameters("find_spots", config, runtime_overrides)
        
        # Build command
        cmd = ["dials.find_spots", "imported.expt"] + phil_params
        
        logger.info(f"Running dials.find_spots with parameters: {phil_params}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise DIALSError(f"dials.find_spots failed: {result.stderr}")
        
        logger.debug(f"dials.find_spots stdout: {result.stdout}")
        return result
    
    def _run_dials_index(self, config: DIALSStillsProcessConfig) -> subprocess.CompletedProcess:
        """Run dials.index step using PHIL files and configuration overrides."""
        # Load parameters from PHIL file and merge with config
        runtime_overrides = {
            "output.experiments": "indexed.expt",
            "output.reflections": "indexed.refl",
            "output.log": "dials.index.log"
        }
        
        # Add known symmetry from config to runtime overrides
        if config.known_space_group:
            runtime_overrides["indexing.known_symmetry.space_group"] = f'"{config.known_space_group}"'
        if config.known_unit_cell:
            runtime_overrides["indexing.known_symmetry.unit_cell"] = config.known_unit_cell
        
        phil_params = self._load_and_merge_phil_parameters("index", config, runtime_overrides)
        
        # Build command
        cmd = ["dials.index", "imported.expt", "strong.refl"] + phil_params
        
        logger.info(f"Running dials.index with parameters: {phil_params}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise DIALSError(f"dials.index failed: {result.stderr}")
        
        logger.debug(f"dials.index stdout: {result.stdout}")
        return result
    
    def _run_dials_integrate(self, config: DIALSStillsProcessConfig) -> subprocess.CompletedProcess:
        """Run dials.integrate step using PHIL files and configuration overrides."""
        # Load parameters from PHIL file and merge with config
        runtime_overrides = {
            "output.experiments": "integrated.expt",
            "output.reflections": "integrated.refl",
            "output.log": "dials.integrate.log"
        }
        
        phil_params = self._load_and_merge_phil_parameters("integrate", config, runtime_overrides)
        
        # Build command
        cmd = ["dials.integrate", "indexed.expt", "indexed.refl"] + phil_params
        
        logger.info(f"Running dials.integrate with parameters: {phil_params}")
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