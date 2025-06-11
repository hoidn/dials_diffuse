"""Adapter for DXTBX/DIALS file I/O operations."""

import logging
from pathlib import Path
from typing import Union

from diffusepipe.exceptions import FileSystemError, DIALSError

# Imports needed for patching in tests
try:
    from dxtbx.model.experiment_list import ExperimentListFactory, ExperimentListDumper
    from dials.array_family import flex
except ImportError:
    # These imports might fail in testing environments without DIALS
    ExperimentListFactory = None
    ExperimentListDumper = None
    flex = None

logger = logging.getLogger(__name__)


class DXTBXIOAdapter:
    """
    Adapter for standardizing DXTBX/DIALS file I/O operations.

    This adapter provides a consistent interface for loading and saving
    DIALS/DXTBX objects while handling errors appropriately.
    """

    def __init__(self):
        """Initialize the DXTBX I/O adapter."""
        pass

    def load_experiment_list(self, expt_path: Union[str, Path]) -> object:
        """
        Load an ExperimentList from a JSON file.

        Args:
            expt_path: Path to the .expt file

        Returns:
            DIALS ExperimentList object

        Raises:
            FileSystemError: When file operations fail
            DIALSError: When DIALS loading fails
        """
        expt_path = Path(expt_path)

        try:
            if not expt_path.exists():
                raise FileSystemError(f"Experiment file does not exist: {expt_path}")

            if not expt_path.is_file():
                raise FileSystemError(f"Path is not a file: {expt_path}")

            logger.info(f"Loading experiment list from {expt_path}")

            # Use top-level import (supports patching in tests)
            if ExperimentListFactory is None:
                raise DIALSError("Failed to import DXTBX components: ExperimentListFactory not available")

            # Load the experiment list
            experiments = ExperimentListFactory.from_json_file(str(expt_path))

            if len(experiments) == 0:
                logger.warning(f"Loaded empty experiment list from {expt_path}")
            else:
                logger.info(f"Loaded {len(experiments)} experiments from {expt_path}")

            return experiments

        except (FileSystemError, DIALSError):
            raise
        except Exception as e:
            raise FileSystemError(
                f"Failed to load experiment list from {expt_path}: {e}"
            ) from e

    def load_reflection_table(self, refl_path: Union[str, Path]) -> object:
        """
        Load a reflection table from a file.

        Args:
            refl_path: Path to the .refl file

        Returns:
            DIALS reflection_table object

        Raises:
            FileSystemError: When file operations fail
            DIALSError: When DIALS loading fails
        """
        refl_path = Path(refl_path)

        try:
            if not refl_path.exists():
                raise FileSystemError(f"Reflection file does not exist: {refl_path}")

            if not refl_path.is_file():
                raise FileSystemError(f"Path is not a file: {refl_path}")

            logger.info(f"Loading reflection table from {refl_path}")

            # Use top-level import (supports patching in tests)
            if flex is None:
                raise DIALSError("Failed to import DIALS components: flex not available")

            # Load the reflection table
            reflections = flex.reflection_table.from_file(str(refl_path))

            if len(reflections) == 0:
                logger.warning(f"Loaded empty reflection table from {refl_path}")
            else:
                logger.info(f"Loaded {len(reflections)} reflections from {refl_path}")

            return reflections

        except (FileSystemError, DIALSError):
            raise
        except Exception as e:
            raise FileSystemError(
                f"Failed to load reflection table from {refl_path}: {e}"
            ) from e

    def save_experiment_list(
        self, experiments: object, expt_path: Union[str, Path]
    ) -> None:
        """
        Save an ExperimentList to a JSON file.

        Args:
            experiments: DIALS ExperimentList object to save
            expt_path: Path where to save the .expt file

        Raises:
            FileSystemError: When file operations fail
            DIALSError: When DIALS saving fails
        """
        expt_path = Path(expt_path)

        try:
            if experiments is None:
                raise DIALSError("Cannot save None experiment list")

            # Create parent directory if it doesn't exist
            expt_path.parent.mkdir(parents=True, exist_ok=True)

            logger.info(f"Saving experiment list to {expt_path}")

            # Use top-level import (supports patching in tests)
            if ExperimentListDumper is None:
                raise DIALSError("Failed to import DXTBX components: ExperimentListDumper not available")

            # Save the experiment list
            dumper = ExperimentListDumper(experiments)
            dumper.as_json(str(expt_path))

            if not expt_path.exists():
                raise FileSystemError(f"Failed to create experiment file: {expt_path}")

            logger.info(
                f"Successfully saved {len(experiments)} experiments to {expt_path}"
            )

        except (FileSystemError, DIALSError):
            raise
        except Exception as e:
            raise FileSystemError(
                f"Failed to save experiment list to {expt_path}: {e}"
            ) from e

    def save_reflection_table(
        self, reflections: object, refl_path: Union[str, Path]
    ) -> None:
        """
        Save a reflection table to a file.

        Args:
            reflections: DIALS reflection_table object to save
            refl_path: Path where to save the .refl file

        Raises:
            FileSystemError: When file operations fail
            DIALSError: When DIALS saving fails
        """
        refl_path = Path(refl_path)

        try:
            if reflections is None:
                raise DIALSError("Cannot save None reflection table")

            # Create parent directory if it doesn't exist
            refl_path.parent.mkdir(parents=True, exist_ok=True)

            logger.info(f"Saving reflection table to {refl_path}")

            # Save the reflection table
            # This assumes reflections has a as_file method
            if hasattr(reflections, "as_file"):
                reflections.as_file(str(refl_path))
            else:
                raise DIALSError("Reflection table does not support file saving")

            if not refl_path.exists():
                raise FileSystemError(f"Failed to create reflection file: {refl_path}")

            logger.info(
                f"Successfully saved {len(reflections)} reflections to {refl_path}"
            )

        except (FileSystemError, DIALSError):
            raise
        except Exception as e:
            raise FileSystemError(
                f"Failed to save reflection table to {refl_path}: {e}"
            ) from e
