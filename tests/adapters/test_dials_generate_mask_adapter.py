"""Tests for DIALSGenerateMaskAdapter."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from diffusepipe.adapters.dials_generate_mask_adapter import DIALSGenerateMaskAdapter
from diffusepipe.exceptions import DIALSError


@pytest.fixture
def adapter():
    """Create a DIALSGenerateMaskAdapter instance for testing."""
    return DIALSGenerateMaskAdapter()


@pytest.fixture
def mock_experiment():
    """Create a mock DIALS experiment object."""
    experiment = Mock()
    
    # Mock detector with panels
    panel1 = Mock()
    panel1.get_image_size.return_value = (100, 100)
    panel2 = Mock()
    panel2.get_image_size.return_value = (100, 100)
    
    detector = Mock()
    detector.__iter__.return_value = iter([panel1, panel2])
    detector.__len__.return_value = 2
    
    experiment.detector = detector
    return experiment


@pytest.fixture
def mock_reflections():
    """Create a mock DIALS reflection table."""
    reflections = Mock()
    reflections.__len__.return_value = 100  # Mock 100 reflections
    return reflections


@pytest.fixture
def mock_mask_params():
    """Create mock mask generation parameters."""
    return {
        'border': 3,
        'algorithm': 'simple'
    }


class TestDIALSGenerateMaskAdapter:
    """Test cases for DIALSGenerateMaskAdapter."""
    
    def test_init(self, adapter):
        """Test adapter initialization."""
        assert adapter is not None
    
    def test_generate_bragg_mask_none_experiment(self, adapter, mock_reflections):
        """Test mask generation with None experiment."""
        with pytest.raises(DIALSError, match="Experiment object cannot be None"):
            adapter.generate_bragg_mask(None, mock_reflections)
    
    def test_generate_bragg_mask_none_reflections(self, adapter, mock_experiment):
        """Test mask generation with None reflections."""
        with pytest.raises(DIALSError, match="Reflections object cannot be None"):
            adapter.generate_bragg_mask(mock_experiment, None)
    
    @patch('diffusepipe.adapters.dials_generate_mask_adapter.flex')
    def test_generate_bragg_mask_success(self, mock_flex, adapter, mock_experiment, mock_reflections):
        """Test successful mask generation."""
        # Mock flex.bool and flex.grid
        mock_grid = Mock()
        mock_flex.grid.return_value = mock_grid
        mock_bool_array = Mock()
        mock_flex.bool.return_value = mock_bool_array
        
        mask_result, success, logs = adapter.generate_bragg_mask(
            mock_experiment, 
            mock_reflections
        )
        
        assert success is True
        assert isinstance(mask_result, tuple)
        assert len(mask_result) == 2  # Two panels
        assert "Generated Bragg mask successfully" in logs
    
    @patch('diffusepipe.adapters.dials_generate_mask_adapter.flex')
    def test_generate_bragg_mask_with_params(self, mock_flex, adapter, mock_experiment, mock_reflections, mock_mask_params):
        """Test mask generation with custom parameters."""
        # Mock flex.bool and flex.grid
        mock_grid = Mock()
        mock_flex.grid.return_value = mock_grid
        mock_bool_array = Mock()
        mock_flex.bool.return_value = mock_bool_array
        
        mask_result, success, logs = adapter.generate_bragg_mask(
            mock_experiment, 
            mock_reflections,
            mock_mask_params
        )
        
        assert success is True
        assert isinstance(mask_result, tuple)
    
    def test_generate_bragg_mask_import_error(self, adapter, mock_experiment, mock_reflections):
        """Test mask generation with DIALS import error."""
        with patch('diffusepipe.adapters.dials_generate_mask_adapter.generate_mask', side_effect=ImportError("DIALS not found")):
            with pytest.raises(DIALSError, match="Failed to import DIALS masking components"):
                adapter.generate_bragg_mask(mock_experiment, mock_reflections)
    
    @patch('diffusepipe.adapters.dials_generate_mask_adapter.flex')
    def test_call_generate_mask_success(self, mock_flex, adapter, mock_experiment, mock_reflections):
        """Test successful _call_generate_mask."""
        # Mock flex components
        mock_grid = Mock()
        mock_flex.grid.return_value = mock_grid
        mock_bool_array = Mock()
        mock_flex.bool.return_value = mock_bool_array
        
        params = {'border': 2, 'algorithm': 'simple'}
        result = adapter._call_generate_mask(mock_experiment, mock_reflections, params)
        
        assert isinstance(result, tuple)
        assert len(result) == 2  # Two panels from mock_experiment
    
    @patch('diffusepipe.adapters.dials_generate_mask_adapter.flex')
    def test_call_generate_mask_failure(self, mock_flex, adapter, mock_experiment, mock_reflections):
        """Test _call_generate_mask with failure."""
        # Make flex.bool raise an exception
        mock_flex.bool.side_effect = Exception("Flex error")
        
        params = {'border': 2, 'algorithm': 'simple'}
        
        with pytest.raises(DIALSError, match="DIALS generate_mask call failed"):
            adapter._call_generate_mask(mock_experiment, mock_reflections, params)
    
    def test_validate_mask_result_none(self, adapter):
        """Test mask validation with None result."""
        with pytest.raises(DIALSError, match="Mask generation returned None"):
            adapter._validate_mask_result(None)
    
    def test_validate_mask_result_not_tuple_or_list(self, adapter):
        """Test mask validation with invalid type."""
        with pytest.raises(DIALSError, match="Mask result should be a tuple or list"):
            adapter._validate_mask_result("invalid_type")
    
    def test_validate_mask_result_empty(self, adapter):
        """Test mask validation with empty result."""
        with pytest.raises(DIALSError, match="Mask result contains no panel masks"):
            adapter._validate_mask_result(tuple())
    
    def test_validate_mask_result_none_panel(self, adapter):
        """Test mask validation with None panel mask."""
        mask_result = (Mock(), None)  # Second panel is None
        
        with pytest.raises(DIALSError, match="Panel 1 mask is None"):
            adapter._validate_mask_result(mask_result)
    
    def test_validate_mask_result_success(self, adapter):
        """Test successful mask validation."""
        panel1_mask = Mock()
        panel2_mask = Mock()
        mask_result = (panel1_mask, panel2_mask)
        
        # Should not raise an exception
        adapter._validate_mask_result(mask_result)
    
    @patch('diffusepipe.adapters.dials_generate_mask_adapter.flex')
    def test_generate_bragg_mask_integration(self, mock_flex, adapter):
        """Test integration of the full generate_bragg_mask workflow."""
        # Set up mocks
        mock_grid = Mock()
        mock_flex.grid.return_value = mock_grid
        mock_bool_array = Mock()
        mock_flex.bool.return_value = mock_bool_array
        
        # Create experiment with detector
        experiment = Mock()
        panel = Mock()
        panel.get_image_size.return_value = (100, 100)
        detector = Mock()
        detector.__iter__.return_value = iter([panel])
        detector.__len__.return_value = 1
        experiment.detector = detector
        
        # Create reflections
        reflections = Mock()
        
        # Test the full workflow
        mask_result, success, logs = adapter.generate_bragg_mask(experiment, reflections)
        
        assert success is True
        assert isinstance(mask_result, tuple)
        assert len(mask_result) == 1
        assert "Starting Bragg mask generation" in logs
        assert "Generated Bragg mask successfully" in logs
        assert "Validated mask result" in logs