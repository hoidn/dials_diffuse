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
    create_default_config
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
            phil_path=self.test_phil_path,
            enable_partiality=True
        )
        
        # Mock the adapter to return successful results
        mock_experiment = Mock()
        mock_reflections = Mock()
        mock_reflections.has_key = Mock(return_value=True)  # Has partiality column
        
        with patch.object(self.processor.adapter, 'process_still') as mock_process:
            mock_process.return_value = (
                mock_experiment, 
                mock_reflections, 
                True, 
                "Processing completed successfully"
            )
            
            # Act
            outcome = self.processor.process_still(
                image_path=self.test_image_path,
                config=config
            )
            
            # Assert
            assert outcome.status == "SUCCESS"
            assert "Successfully processed" in outcome.message
            assert outcome.error_code is None
            assert outcome.output_artifacts is not None
            
            # Verify required artifacts are present
            assert "experiment" in outcome.output_artifacts
            assert "reflections" in outcome.output_artifacts
            assert outcome.output_artifacts["experiment"] is mock_experiment
            assert outcome.output_artifacts["reflections"] is mock_reflections
            
            # Verify adapter was called with correct parameters
            mock_process.assert_called_once_with(
                image_path=self.test_image_path,
                config=config,
                base_expt_path=None
            )
    
    def test_process_still_with_base_experiment(self):
        """Test processing with a base experiment file."""
        # Arrange
        config = create_default_config()
        base_expt_path = "tests/data/base_experiment.expt"
        
        mock_experiment = Mock()
        mock_reflections = Mock()
        mock_reflections.has_key = Mock(return_value=True)
        
        with patch.object(self.processor.adapter, 'process_still') as mock_process:
            mock_process.return_value = (mock_experiment, mock_reflections, True, "Success")
            
            # Act
            outcome = self.processor.process_still(
                image_path=self.test_image_path,
                config=config,
                base_experiment_path=base_expt_path
            )
            
            # Assert
            assert outcome.status == "SUCCESS"
            mock_process.assert_called_once_with(
                image_path=self.test_image_path,
                config=config,
                base_expt_path=base_expt_path
            )
    
    def test_process_still_indexing_failure(self):
        """Test handling of indexing failure."""
        # Arrange
        config = create_default_config()
        
        with patch.object(self.processor.adapter, 'process_still') as mock_process:
            mock_process.return_value = (None, None, False, "Indexing failed")
            
            # Act
            outcome = self.processor.process_still(
                image_path=self.test_image_path,
                config=config
            )
            
            # Assert
            assert outcome.status == "FAILURE"
            assert "DIALS processing failed" in outcome.message
            assert outcome.error_code == "DIALS_PROCESSING_FAILED"
            assert "Indexing failed" in outcome.message
    
    def test_process_still_dials_error(self):
        """Test handling of DIALSError exception."""
        # Arrange
        config = create_default_config()
        
        with patch.object(self.processor.adapter, 'process_still') as mock_process:
            mock_process.side_effect = DIALSError("DIALS internal error")
            
            # Act
            outcome = self.processor.process_still(
                image_path=self.test_image_path,
                config=config
            )
            
            # Assert
            assert outcome.status == "FAILURE"
            assert "DIALS internal error" in outcome.message
            assert outcome.error_code == "DIALSERROR"
    
    def test_process_still_configuration_error(self):
        """Test handling of ConfigurationError exception."""
        # Arrange
        config = create_default_config()
        
        with patch.object(self.processor.adapter, 'process_still') as mock_process:
            mock_process.side_effect = ConfigurationError("Invalid configuration")
            
            # Act
            outcome = self.processor.process_still(
                image_path=self.test_image_path,
                config=config
            )
            
            # Assert
            assert outcome.status == "FAILURE"
            assert "Invalid configuration" in outcome.message
            assert outcome.error_code == "CONFIGURATIONERROR"
    
    def test_process_still_unexpected_error(self):
        """Test handling of unexpected exceptions."""
        # Arrange
        config = create_default_config()
        
        with patch.object(self.processor.adapter, 'process_still') as mock_process:
            mock_process.side_effect = RuntimeError("Unexpected runtime error")
            
            # Act
            outcome = self.processor.process_still(
                image_path=self.test_image_path,
                config=config
            )
            
            # Assert
            assert outcome.status == "FAILURE"
            assert "Unexpected error" in outcome.message
            assert outcome.error_code == "UNEXPECTED_ERROR"
    
    def test_validate_processing_outcome_success(self):
        """Test validation of successful processing outcome."""
        # Arrange
        mock_experiment = Mock()
        mock_reflections = Mock()
        mock_reflections.has_key = Mock(return_value=True)
        
        outcome = OperationOutcome(
            status="SUCCESS",
            message="Success",
            error_code=None,
            output_artifacts={
                "experiment": mock_experiment,
                "reflections": mock_reflections
            }
        )
        
        # Act
        is_valid = self.processor.validate_processing_outcome(outcome)
        
        # Assert
        assert is_valid is True
    
    def test_validate_processing_outcome_failure_status(self):
        """Test validation rejects failed outcomes."""
        # Arrange
        outcome = OperationOutcome(
            status="FAILURE",
            message="Failed",
            error_code="SOME_ERROR",
            output_artifacts=None
        )
        
        # Act
        is_valid = self.processor.validate_processing_outcome(outcome)
        
        # Assert
        assert is_valid is False
    
    def test_validate_processing_outcome_missing_artifacts(self):
        """Test validation rejects outcomes with missing artifacts."""
        # Arrange
        outcome = OperationOutcome(
            status="SUCCESS",
            message="Success",
            error_code=None,
            output_artifacts={"experiment": Mock()}  # Missing reflections
        )
        
        # Act
        is_valid = self.processor.validate_processing_outcome(outcome)
        
        # Assert
        assert is_valid is False
    
    def test_validate_processing_outcome_none_artifacts(self):
        """Test validation rejects outcomes with None artifacts."""
        # Arrange
        outcome = OperationOutcome(
            status="SUCCESS",
            message="Success",
            error_code=None,
            output_artifacts={
                "experiment": None,  # None artifact
                "reflections": Mock()
            }
        )
        
        # Act
        is_valid = self.processor.validate_processing_outcome(outcome)
        
        # Assert
        assert is_valid is False
    
    def test_validate_processing_outcome_missing_partiality(self):
        """Test validation rejects reflections without partiality column."""
        # Arrange
        mock_experiment = Mock()
        mock_reflections = Mock()
        mock_reflections.has_key = Mock(return_value=False)  # No partiality column
        
        outcome = OperationOutcome(
            status="SUCCESS",
            message="Success",
            error_code=None,
            output_artifacts={
                "experiment": mock_experiment,
                "reflections": mock_reflections
            }
        )
        
        # Act
        is_valid = self.processor.validate_processing_outcome(outcome)
        
        # Assert
        assert is_valid is False


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