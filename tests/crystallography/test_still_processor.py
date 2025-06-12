"""
Integration tests for StillProcessorComponent.

These tests focus on the integration between StillProcessorComponent and 
DIALSStillsProcessAdapter, following the testing principles from plan.md.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from diffusepipe.crystallography.still_processing_and_validation import (
    StillProcessorComponent,
    create_default_config,
)
from diffusepipe.types.types_IDL import DIALSStillsProcessConfig, OperationOutcome
from diffusepipe.exceptions import DIALSError, ConfigurationError, DataValidationError


class TestStillProcessorComponent:
    """Test suite for StillProcessorComponent integration."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = StillProcessorComponent()
        self.test_image_path = "tests/data/minimal_still.cbf"
        self.test_phil_path = "tests/data/minimal_stills_process.phil"

    def test_process_single_still_successfully(self):
        """Test successful processing of a single still image."""
        # Arrange
        config = create_default_config(
            phil_path=self.test_phil_path, enable_partiality=True
        )

        # Mock the adapter to return successful results
        mock_experiment = Mock()
        mock_reflections = Mock()
        mock_reflections.has_key = Mock(return_value=True)  # Has partiality column

        # Mock the routing to return stills adapter and the process_still method
        with patch.object(self.processor, "_determine_processing_route") as mock_route:
            with patch.object(self.processor.stills_adapter, "process_still") as mock_process:
                mock_route.return_value = ("stills", self.processor.stills_adapter)
                mock_process.return_value = (
                    mock_experiment,
                    mock_reflections,
                    True,
                    "Processing completed successfully",
                )

                # Act
                outcome = self.processor.process_still(
                    image_path=self.test_image_path, config=config
                )

                # Assert
                assert outcome.status == "SUCCESS"
                assert "DIALS processing only (legacy path)" in outcome.message
                assert outcome.error_code is None
                assert outcome.output_artifacts is not None

                # Verify required artifacts are present
                assert "experiment" in outcome.output_artifacts
                assert "reflections" in outcome.output_artifacts
                assert outcome.output_artifacts["experiment"] is mock_experiment
                assert outcome.output_artifacts["reflections"] is mock_reflections

                # Verify adapter was called with correct parameters
                mock_process.assert_called_once_with(
                    image_path=self.test_image_path, config=config, base_expt_path=None
                )

    def test_process_still_with_base_experiment(self):
        """Test processing with a base experiment file."""
        # Arrange
        config = create_default_config()
        base_expt_path = "tests/data/base_experiment.expt"

        mock_experiment = Mock()
        mock_reflections = Mock()
        mock_reflections.has_key = Mock(return_value=True)

        with patch.object(self.processor, "_determine_processing_route") as mock_route:
            with patch.object(self.processor.stills_adapter, "process_still") as mock_process:
                mock_route.return_value = ("stills", self.processor.stills_adapter)
                mock_process.return_value = (
                    mock_experiment,
                    mock_reflections,
                    True,
                    "Success",
                )

                # Act
                outcome = self.processor.process_still(
                    image_path=self.test_image_path,
                    config=config,
                    base_experiment_path=base_expt_path,
                )

                # Assert
                assert outcome.status == "SUCCESS"
                mock_process.assert_called_once_with(
                    image_path=self.test_image_path,
                    config=config,
                    base_expt_path=base_expt_path,
                )

    def test_process_still_indexing_failure(self):
        """Test handling of indexing failure."""
        # Arrange
        config = create_default_config()

        with patch.object(self.processor, "_determine_processing_route") as mock_route:
            with patch.object(self.processor.stills_adapter, "process_still") as mock_process:
                mock_route.return_value = ("stills", self.processor.stills_adapter)
                mock_process.return_value = (None, None, False, "Indexing failed")

                # Act
                outcome = self.processor.process_still(
                    image_path=self.test_image_path, config=config
                )

                # Assert
                assert outcome.status == "FAILURE"
                assert "DIALS processing failed (legacy path)" in outcome.message
                assert outcome.error_code == "DIALS_PROCESSING_FAILED"

    def test_process_still_dials_error(self):
        """Test handling of DIALSError exception."""
        # Arrange
        config = create_default_config()

        with patch.object(self.processor, "_determine_processing_route") as mock_route:
            with patch.object(self.processor.stills_adapter, "process_still") as mock_process:
                mock_route.return_value = ("stills", self.processor.stills_adapter)
                mock_process.side_effect = DIALSError("DIALS internal error")

                # Act & Assert - legacy method doesn't catch exceptions
                with pytest.raises(DIALSError, match="DIALS internal error"):
                    self.processor.process_still(
                        image_path=self.test_image_path, config=config
                    )

    def test_process_still_configuration_error(self):
        """Test handling of ConfigurationError exception."""
        # Arrange
        config = create_default_config()

        with patch.object(self.processor, "_determine_processing_route") as mock_route:
            with patch.object(self.processor.stills_adapter, "process_still") as mock_process:
                mock_route.return_value = ("stills", self.processor.stills_adapter)
                mock_process.side_effect = ConfigurationError("Invalid configuration")

                # Act & Assert - legacy method doesn't catch exceptions
                with pytest.raises(ConfigurationError, match="Invalid configuration"):
                    self.processor.process_still(
                        image_path=self.test_image_path, config=config
                    )

    def test_process_still_unexpected_error(self):
        """Test handling of unexpected exceptions."""
        # Arrange
        config = create_default_config()

        with patch.object(self.processor, "_determine_processing_route") as mock_route:
            with patch.object(self.processor.stills_adapter, "process_still") as mock_process:
                mock_route.return_value = ("stills", self.processor.stills_adapter)
                mock_process.side_effect = RuntimeError("Unexpected runtime error")

                # Act & Assert - legacy method doesn't catch exceptions
                with pytest.raises(RuntimeError, match="Unexpected runtime error"):
                    self.processor.process_still(
                        image_path=self.test_image_path, config=config
                    )



class TestCreateDefaultConfig:
    """Test suite for create_default_config function."""

    def test_create_default_config_basic(self):
        """Test creation of basic default configuration."""
        # Act
        config = create_default_config()

        # Assert
        assert isinstance(config, DIALSStillsProcessConfig)
        assert config.stills_process_phil_path is None
        assert config.calculate_partiality is True
        assert config.output_shoeboxes is False

    def test_create_default_config_with_phil_path(self):
        """Test creation of configuration with PHIL path."""
        # Arrange
        phil_path = "tests/data/test.phil"

        # Act
        config = create_default_config(phil_path=phil_path)

        # Assert
        assert config.stills_process_phil_path == phil_path

    def test_create_default_config_with_shoeboxes(self):
        """Test creation of configuration with shoeboxes enabled."""
        # Act
        config = create_default_config(enable_shoeboxes=True)

        # Assert
        assert config.output_shoeboxes is True

    def test_create_default_config_disable_partiality(self):
        """Test creation of configuration with partiality disabled."""
        # Act
        config = create_default_config(enable_partiality=False)

        # Assert
        assert config.calculate_partiality is False
