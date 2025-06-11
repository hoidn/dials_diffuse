"""Tests for DIALSStillsProcessAdapter."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from diffusepipe.adapters.dials_stills_process_adapter import DIALSStillsProcessAdapter
from diffusepipe.exceptions import DIALSError, ConfigurationError, DataValidationError


@pytest.fixture
def adapter():
    """Create a DIALSStillsProcessAdapter instance for testing."""
    return DIALSStillsProcessAdapter()


@pytest.fixture
def mock_config():
    """Create a mock configuration object."""
    config = Mock()
    config.stills_process_phil_path = None
    config.known_unit_cell = None
    config.known_space_group = None
    config.spotfinder_threshold_algorithm = None
    config.min_spot_area = None
    config.output_shoeboxes = None
    config.calculate_partiality = True
    return config


@pytest.fixture
def mock_config_with_phil(tmp_path):
    """Create a mock configuration with PHIL file."""
    phil_file = tmp_path / "test.phil"
    phil_file.write_text("spotfinder.threshold.dispersion.gain=1.0")
    
    config = Mock()
    config.stills_process_phil_path = str(phil_file)
    config.known_unit_cell = "10,10,10,90,90,90"
    config.known_space_group = "P1"
    config.spotfinder_threshold_algorithm = "dispersion"
    config.min_spot_area = 3
    config.output_shoeboxes = True
    config.calculate_partiality = True
    return config


class TestDIALSStillsProcessAdapter:
    """Test cases for DIALSStillsProcessAdapter."""
    
    def test_init(self, adapter):
        """Test adapter initialization."""
        assert adapter._processor is None
    
    def test_process_still_nonexistent_file(self, adapter, mock_config):
        """Test processing with non-existent image file."""
        with pytest.raises(ConfigurationError, match="Image file does not exist"):
            adapter.process_still("/nonexistent/file.cbf", mock_config)
    
    @patch('diffusepipe.adapters.dials_stills_process_adapter.Path.exists')
    def test_phil_parameter_generation_basic(self, mock_exists, adapter, mock_config):
        """Test basic PHIL parameter generation."""
        mock_exists.return_value = True
        
        with patch('diffusepipe.adapters.dials_stills_process_adapter.parse') as mock_parse:
            mock_parse.return_value = Mock()
            result = adapter._generate_phil_parameters(mock_config)
            assert result is not None
    
    @patch('diffusepipe.adapters.dials_stills_process_adapter.Path.exists')
    def test_phil_parameter_generation_with_file(self, mock_exists, adapter, mock_config_with_phil):
        """Test PHIL parameter generation with file."""
        mock_exists.return_value = True
        
        with patch('diffusepipe.adapters.dials_stills_process_adapter.parse') as mock_parse, \
             patch('builtins.open', mock_open_read="test phil content"):
            mock_phil_scope = Mock()
            mock_phil_scope.fetch.return_value = Mock()
            mock_parse.return_value = mock_phil_scope
            
            result = adapter._generate_phil_parameters(mock_config_with_phil)
            assert result is not None
    
    def test_phil_parameter_generation_missing_file(self, adapter):
        """Test PHIL parameter generation with missing file."""
        config = Mock()
        config.stills_process_phil_path = "/nonexistent/file.phil"
        config.known_unit_cell = None
        config.known_space_group = None
        config.spotfinder_threshold_algorithm = None
        config.min_spot_area = None
        config.output_shoeboxes = None
        config.calculate_partiality = None
        
        with pytest.raises(ConfigurationError, match="PHIL file not found"):
            adapter._generate_phil_parameters(config)
    
    @patch('diffusepipe.adapters.dials_stills_process_adapter.DataBlockFactory')
    @patch('diffusepipe.adapters.dials_stills_process_adapter.ExperimentListFactory')
    def test_do_import_success(self, mock_exp_factory, mock_db_factory, adapter):
        """Test successful experiment import."""
        # Mock the import process
        mock_datablocks = [Mock()]
        mock_db_factory.from_filenames.return_value = mock_datablocks
        
        mock_experiments = [Mock()]
        mock_exp_factory.from_datablocks.return_value = mock_experiments
        
        result = adapter._do_import("/path/to/image.cbf")
        assert result == mock_experiments
    
    @patch('diffusepipe.adapters.dials_stills_process_adapter.DataBlockFactory')
    @patch('diffusepipe.adapters.dials_stills_process_adapter.ExperimentListFactory')
    def test_do_import_failure(self, mock_exp_factory, mock_db_factory, adapter):
        """Test experiment import failure."""
        mock_db_factory.from_filenames.return_value = [Mock()]
        mock_exp_factory.from_datablocks.return_value = []  # Empty list
        
        with pytest.raises(DIALSError, match="Failed to create experiment"):
            adapter._do_import("/path/to/image.cbf")
    
    def test_process_experiments_no_processor(self, adapter):
        """Test processing experiments without initialized processor."""
        mock_experiments = Mock()
        
        with pytest.raises(DIALSError, match="Processor not initialized"):
            adapter._process_experiments(mock_experiments)
    
    def test_process_experiments_success(self, adapter):
        """Test successful experiment processing."""
        mock_experiments = Mock()
        adapter._processor = Mock()  # Set up processor
        
        result_exps, result_refls = adapter._process_experiments(mock_experiments)
        
        # In the current implementation, it returns the input experiments as placeholder
        assert result_exps == mock_experiments
        assert result_refls is None
    
    def test_extract_experiment_success(self, adapter):
        """Test successful experiment extraction."""
        mock_exp1 = Mock()
        mock_exp2 = Mock()
        mock_experiments = [mock_exp1, mock_exp2]
        
        result = adapter._extract_experiment(mock_experiments)
        assert result == mock_exp1
    
    def test_extract_experiment_empty(self, adapter):
        """Test experiment extraction with empty list."""
        result = adapter._extract_experiment([])
        assert result is None
    
    def test_extract_experiment_none(self, adapter):
        """Test experiment extraction with None input."""
        result = adapter._extract_experiment(None)
        assert result is None
    
    def test_extract_reflections(self, adapter):
        """Test reflection extraction."""
        mock_reflections = Mock()
        result = adapter._extract_reflections(mock_reflections)
        assert result == mock_reflections
    
    def test_validate_partiality_none(self, adapter):
        """Test partiality validation with None reflections."""
        # Should not raise an exception
        adapter._validate_partiality(None)
    
    def test_validate_partiality_missing_column(self, adapter):
        """Test partiality validation with missing column."""
        mock_reflections = Mock()
        mock_reflections.has_key.return_value = False
        
        with pytest.raises(DataValidationError, match="missing required 'partiality' column"):
            adapter._validate_partiality(mock_reflections)
    
    def test_validate_partiality_success(self, adapter):
        """Test successful partiality validation."""
        mock_reflections = Mock()
        mock_reflections.has_key.return_value = True
        mock_reflections.__getitem__.return_value = [0.5, 0.8, 1.0]  # Mock partiality values
        
        # Should not raise an exception
        adapter._validate_partiality(mock_reflections)
    
    def test_validate_partiality_no_has_key_attribute(self, adapter):
        """Test partiality validation with object lacking has_key method."""
        mock_reflections = Mock()
        del mock_reflections.has_key  # Remove the has_key attribute
        
        # Should not raise an exception (warning logged instead)
        adapter._validate_partiality(mock_reflections)
    
    @patch('diffusepipe.adapters.dials_stills_process_adapter.Path.exists')
    @patch('diffusepipe.adapters.dials_stills_process_adapter.Processor')
    @patch('diffusepipe.adapters.dials_stills_process_adapter.ExperimentListFactory')
    def test_process_still_import_error(self, mock_exp_factory, mock_processor, mock_exists, adapter, mock_config):
        """Test process_still with DIALS import error."""
        mock_exists.return_value = True
        
        # Mock import error
        with patch('diffusepipe.adapters.dials_stills_process_adapter.Processor', side_effect=ImportError("DIALS not found")):
            experiment, reflections, success, logs = adapter.process_still("/test/image.cbf", mock_config)
            
            assert experiment is None
            assert reflections is None
            assert success is False
            assert "Failed to import DIALS components" in logs


def mock_open_read(content):
    """Helper function to create a mock open context manager."""
    from unittest.mock import mock_open
    return mock_open(read_data=content)