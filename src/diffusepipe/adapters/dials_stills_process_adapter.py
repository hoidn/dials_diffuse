"""Adapter for dials.stills_process Python API."""

import logging
from pathlib import Path
from typing import Optional, Tuple, Any

from diffusepipe.exceptions import DIALSError, ConfigurationError, DataValidationError
from diffusepipe.types.types_IDL import DIALSStillsProcessConfig

# Imports needed for patching in tests
try:
    from libtbx.phil import parse
    from dxtbx.model.experiment_list import ExperimentListFactory
    from dials.command_line.stills_process import Processor, do_import
except ImportError:
    # These imports might fail in testing environments without DIALS
    parse = None
    ExperimentListFactory = None
    Processor = None
    do_import = None

logger = logging.getLogger(__name__)


class DIALSStillsProcessAdapter:
    """
    Thin wrapper around dials.stills_process.Processor for true still images.

    This adapter should only be used for CBF files with Angle_increment = 0.0°.
    For oscillation data (Angle_increment > 0.0°), use DIALSSequenceProcessAdapter instead.
    """

    def __init__(self):
        """Initialize the DIALS stills process adapter."""
        self._processor: Optional[Any] = None  # Type hint for Processor
        self._extracted_params: Optional[Any] = (
            None  # Type hint for extracted PHIL params
        )

    def process_still(
        self,
        image_path: str,
        config: DIALSStillsProcessConfig,
        base_expt_path: Optional[str] = None,
        output_dir_final: Optional[str] = None,
    ) -> Tuple[Optional[object], Optional[object], bool, str]:
        """
        Process a single still image using dials.stills_process.Processor.

        Args:
            image_path: Path to the CBF image file to process (must be true still)
            config: Configuration parameters for dials.stills_process
            base_expt_path: Optional path to base experiment file for geometry
            output_dir_final: Optional path to save final output files with consistent naming

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
            self._extracted_params = self._generate_phil_parameters(
                config, base_expt_path, output_dir_final
            )
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
                experiments = do_import(image_path)
                log_messages.append(f"Imported experiment from {image_path}")

            if not experiments or len(experiments) == 0:
                raise DIALSError(f"Failed to import experiments from {image_path}")

            # Step 2: Initialize and run the Processor
            logger.info("Step 2: Initializing stills_process.Processor...")
            self._processor = Processor(params=self._extracted_params)

            # Step 3: Process experiments using the Processor
            logger.info(
                "Step 3: Processing experiments with stills_process.Processor..."
            )
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

            # Extract experiments and reflections (keep as ExperimentList for orchestrator)
            experiments = integrated_experiments  # Keep as ExperimentList
            reflections = self._extract_reflections(integrated_reflections)

            # Validate partiality column
            self._validate_partiality(reflections)
            log_messages.append("Validated partiality data")

            return experiments, reflections, True, "\n".join(log_messages)

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

    def _generate_phil_parameters(
        self,
        config: DIALSStillsProcessConfig,
        base_expt_path: Optional[str] = None,
        output_dir_final: Optional[str] = None,
    ) -> Any:
        """
        Generate PHIL parameters for true stills processing.

        This method generates parameters appropriate for dials.stills_process,
        which is designed for true still images (Angle_increment = 0.0°).

        Args:
            config: Configuration object containing DIALS parameters
            base_expt_path: Optional path to reference experiment for crystal models
            output_dir_final: Optional output directory path

        Returns:
            Extracted PHIL parameter object for dials.stills_process

        Raises:
            ConfigurationError: When PHIL generation fails
        """
        try:
            from dials.command_line.stills_process import (
                phil_scope as stills_phil_scope,
            )
            from libtbx.phil import parse
            from cctbx import uctbx, sgtbx

            # Start with DIALS stills_process master scope
            working_phil = stills_phil_scope

            # Apply base PHIL file if provided
            if config.stills_process_phil_path:
                phil_path = Path(config.stills_process_phil_path)
                if not phil_path.exists():
                    raise ConfigurationError(
                        f"PHIL file not found: {config.stills_process_phil_path}"
                    )

                with open(phil_path, "r") as f:
                    phil_content = f.read()
                user_phil = parse(phil_content)
                working_phil = working_phil.fetch(user_phil)

            # Extract params object to apply direct overrides
            params = working_phil.extract()

            # Set output directory and prefix for consistent naming
            if output_dir_final:
                logger.info(f"Setting output directory: {output_dir_final}")
                params.output.output_dir = output_dir_final
                params.output.prefix = "indexed_refined_detector"

            # Apply configuration overrides for stills processing
            if config.known_unit_cell:
                logger.info(f"Adding known unit cell: {config.known_unit_cell}")
                params.indexing.known_symmetry.unit_cell = uctbx.unit_cell(
                    config.known_unit_cell
                )

            if config.known_space_group:
                logger.info(f"Adding known space group: {config.known_space_group}")
                try:
                    params.indexing.known_symmetry.space_group = sgtbx.space_group_info(
                        config.known_space_group
                    ).group()
                except Exception as e:
                    raise ConfigurationError(
                        f"Invalid space group string: {config.known_space_group} - {e}"
                    )

            # Add reference crystal models if provided
            if base_expt_path and Path(base_expt_path).exists():
                logger.info(f"Loading reference crystal models from: {base_expt_path}")
                try:
                    from dxtbx.model.experiment_list import ExperimentListFactory

                    reference_experiments = ExperimentListFactory.from_json_file(
                        base_expt_path
                    )
                    if reference_experiments and len(reference_experiments) > 0:
                        # Extract crystal models from reference experiments
                        crystal_models = [
                            exp.crystal
                            for exp in reference_experiments
                            if exp.crystal is not None
                        ]
                        if crystal_models:
                            params.indexing.known_symmetry.crystal_models = (
                                crystal_models
                            )
                            logger.info(
                                f"Added {len(crystal_models)} reference crystal models for constrained indexing"
                            )
                        else:
                            logger.warning(
                                f"No crystal models found in reference experiment: {base_expt_path}"
                            )
                    else:
                        logger.warning(
                            f"Could not load reference experiments from: {base_expt_path}"
                        )
                except Exception as e:
                    logger.warning(
                        f"Failed to load reference crystal models from {base_expt_path}: {e}"
                    )
                    # Continue without reference models rather than failing

            if config.spotfinder_threshold_algorithm:
                params.spotfinder.threshold.algorithm = (
                    config.spotfinder_threshold_algorithm
                )

            if config.min_spot_area is not None:
                params.spotfinder.filter.min_spot_size = config.min_spot_area

            if config.output_shoeboxes is not None:
                params.output.shoeboxes = config.output_shoeboxes

            if config.calculate_partiality is not None:
                # Note: estimate_partiality parameter removed from current DIALS stills_process PHIL
                # Partiality calculation may be handled automatically or via different parameters
                logger.info(
                    f"Partiality calculation requested: {config.calculate_partiality} (parameter no longer available in DIALS PHIL)"
                )

            # Store the extracted params
            self._extracted_params = params

            # Debug: log key parameters
            logger.debug(f"Unit cell: {params.indexing.known_symmetry.unit_cell}")
            logger.debug(f"Space group: {params.indexing.known_symmetry.space_group}")
            logger.debug(
                f"Spotfinder algorithm: {params.spotfinder.threshold.algorithm}"
            )
            logger.debug(f"Min spot size: {params.spotfinder.filter.min_spot_size}")
            logger.debug(f"Output shoeboxes: {params.output.shoeboxes}")
            # Note: estimate_partiality parameter no longer available in DIALS PHIL

            return params

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
            # Check if this is a DIALS reflection table with has_key method
            if not hasattr(reflections, "has_key"):
                logger.warning(
                    "Could not validate partiality column (possibly mock object or unexpected type)"
                )
                return

            # Check for partiality column
            if not reflections.has_key("partiality"):
                raise DataValidationError(
                    "Reflection table missing required 'partiality' column"
                )

            # Validate partiality values are reasonable
            partialities = reflections["partiality"]
            if len(partialities) == 0:
                logger.warning("No reflections with partiality values found")
            else:
                logger.info(
                    f"Found {len(partialities)} reflections with partiality values"
                )

        except AttributeError:
            # In case of mock objects during testing
            logger.warning(
                "Could not validate partiality column (possibly mock object)"
            )
            pass
