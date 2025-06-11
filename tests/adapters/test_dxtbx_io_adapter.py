"""Tests for DXTBXIOAdapter."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from diffusepipe.adapters.dxtbx_io_adapter import DXTBXIOAdapter
from diffusepipe.exceptions import FileSystemError, DIALSError


@pytest.fixture
def adapter():
    """Create a DXTBXIOAdapter instance for testing."""
    return DXTBXIOAdapter()


@pytest.fixture
def mock_experiments():
    """Create a mock ExperimentList."""
    experiments = Mock()
    experiments.__len__.return_value = 2
    return experiments


@pytest.fixture
def mock_reflections():
    """Create a mock reflection table."""
    reflections = Mock()
    reflections.__len__.return_value = 100
    reflections.as_file = Mock()
    return reflections


class TestDXTBXIOAdapter:
    """Test cases for DXTBXIOAdapter."""
    
    def test_init(self, adapter):
        """Test adapter initialization."""
        assert adapter is not None
    
    def test_load_experiment_list_nonexistent_file(self, adapter):
        """Test loading non-existent experiment file."""
        with pytest.raises(FileSystemError, match="Experiment file does not exist"):
            adapter.load_experiment_list("/nonexistent/file.expt")
    
    @patch('diffusepipe.adapters.dxtbx_io_adapter.Path.exists')
    @patch('diffusepipe.adapters.dxtbx_io_adapter.Path.is_file')
    def test_load_experiment_list_not_file(self, mock_is_file, mock_exists, adapter):
        """Test loading experiment from non-file path."""
        mock_exists.return_value = True
        mock_is_file.return_value = False
        
        with pytest.raises(FileSystemError, match="Path is not a file"):
            adapter.load_experiment_list("/path/to/directory")
    
    @patch('diffusepipe.adapters.dxtbx_io_adapter.Path.exists')
    @patch('diffusepipe.adapters.dxtbx_io_adapter.Path.is_file')
    @patch('diffusepipe.adapters.dxtbx_io_adapter.ExperimentListFactory')
    def test_load_experiment_list_success(self, mock_factory, mock_is_file, mock_exists, adapter, mock_experiments):
        """Test successful experiment list loading."""
        mock_exists.return_value = True
        mock_is_file.return_value = True
        mock_factory.from_json_file.return_value = mock_experiments
        
        result = adapter.load_experiment_list("/path/to/file.expt")
        
        assert result == mock_experiments
        mock_factory.from_json_file.assert_called_once_with("/path/to/file.expt")
    
    @patch('diffusepipe.adapters.dxtbx_io_adapter.Path.exists')
    @patch('diffusepipe.adapters.dxtbx_io_adapter.Path.is_file')
    @patch('diffusepipe.adapters.dxtbx_io_adapter.ExperimentListFactory')
    def test_load_experiment_list_empty(self, mock_factory, mock_is_file, mock_exists, adapter):
        """Test loading empty experiment list."""
        mock_exists.return_value = True
        mock_is_file.return_value = True
        
        empty_experiments = Mock()
        empty_experiments.__len__.return_value = 0
        mock_factory.from_json_file.return_value = empty_experiments
        
        result = adapter.load_experiment_list("/path/to/file.expt")
        
        assert result == empty_experiments
    
    @patch('diffusepipe.adapters.dxtbx_io_adapter.Path.exists')
    @patch('diffusepipe.adapters.dxtbx_io_adapter.Path.is_file')
    def test_load_experiment_list_import_error(self, mock_is_file, mock_exists, adapter):
        """Test experiment loading with import error."""
        mock_exists.return_value = True
        mock_is_file.return_value = True
        
        with patch('diffusepipe.adapters.dxtbx_io_adapter.ExperimentListFactory', side_effect=ImportError("DXTBX not found")):
            with pytest.raises(DIALSError, match="Failed to import DXTBX components"):
                adapter.load_experiment_list("/path/to/file.expt")
    
    def test_load_reflection_table_nonexistent_file(self, adapter):
        """Test loading non-existent reflection file."""
        with pytest.raises(FileSystemError, match="Reflection file does not exist"):
            adapter.load_reflection_table("/nonexistent/file.refl")
    
    @patch('diffusepipe.adapters.dxtbx_io_adapter.Path.exists')
    @patch('diffusepipe.adapters.dxtbx_io_adapter.Path.is_file')
    @patch('diffusepipe.adapters.dxtbx_io_adapter.flex')
    def test_load_reflection_table_success(self, mock_flex, mock_is_file, mock_exists, adapter, mock_reflections):
        """Test successful reflection table loading."""
        mock_exists.return_value = True
        mock_is_file.return_value = True
        mock_flex.reflection_table.from_file.return_value = mock_reflections
        
        result = adapter.load_reflection_table("/path/to/file.refl")
        
        assert result == mock_reflections
        mock_flex.reflection_table.from_file.assert_called_once_with("/path/to/file.refl")
    
    @patch('diffusepipe.adapters.dxtbx_io_adapter.Path.exists')
    @patch('diffusepipe.adapters.dxtbx_io_adapter.Path.is_file')
    def test_load_reflection_table_import_error(self, mock_is_file, mock_exists, adapter):
        """Test reflection loading with import error."""
        mock_exists.return_value = True
        mock_is_file.return_value = True
        
        with patch('diffusepipe.adapters.dxtbx_io_adapter.flex', side_effect=ImportError("DIALS not found")):
            with pytest.raises(DIALSError, match="Failed to import DIALS components"):
                adapter.load_reflection_table("/path/to/file.refl")
    
    def test_save_experiment_list_none(self, adapter):
        """Test saving None experiment list."""
        with pytest.raises(DIALSError, match="Cannot save None experiment list"):
            adapter.save_experiment_list(None, "/path/to/file.expt")
    
    @patch('diffusepipe.adapters.dxtbx_io_adapter.Path.exists')
    @patch('diffusepipe.adapters.dxtbx_io_adapter.Path.mkdir')
    @patch('diffusepipe.adapters.dxtbx_io_adapter.ExperimentListDumper')
    def test_save_experiment_list_success(self, mock_dumper_class, mock_mkdir, mock_exists, adapter, mock_experiments):
        """Test successful experiment list saving."""
        mock_exists.return_value = True
        mock_dumper = Mock()
        mock_dumper_class.return_value = mock_dumper
        
        adapter.save_experiment_list(mock_experiments, "/path/to/file.expt")
        
        mock_dumper_class.assert_called_once_with(mock_experiments)
        mock_dumper.as_json.assert_called_once_with("/path/to/file.expt")
    
    @patch('diffusepipe.adapters.dxtbx_io_adapter.Path.exists')
    @patch('diffusepipe.adapters.dxtbx_io_adapter.Path.mkdir')
    @patch('diffusepipe.adapters.dxtbx_io_adapter.ExperimentListDumper')
    def test_save_experiment_list_file_not_created(self, mock_dumper_class, mock_mkdir, mock_exists, adapter, mock_experiments):
        """Test experiment saving when file is not created."""
        mock_exists.return_value = False  # File doesn't exist after saving
        mock_dumper = Mock()
        mock_dumper_class.return_value = mock_dumper
        
        with pytest.raises(FileSystemError, match="Failed to create experiment file"):
            adapter.save_experiment_list(mock_experiments, "/path/to/file.expt")
    
    def test_save_experiment_list_import_error(self, adapter, mock_experiments):
        """Test experiment saving with import error."""
        with patch('diffusepipe.adapters.dxtbx_io_adapter.ExperimentListDumper', side_effect=ImportError("DXTBX not found")):
            with pytest.raises(DIALSError, match="Failed to import DXTBX components"):
                adapter.save_experiment_list(mock_experiments, "/path/to/file.expt")
    
    def test_save_reflection_table_none(self, adapter):
        """Test saving None reflection table."""
        with pytest.raises(DIALSError, match="Cannot save None reflection table"):
            adapter.save_reflection_table(None, "/path/to/file.refl")
    
    @patch('diffusepipe.adapters.dxtbx_io_adapter.Path.exists')
    @patch('diffusepipe.adapters.dxtbx_io_adapter.Path.mkdir')
    def test_save_reflection_table_success(self, mock_mkdir, mock_exists, adapter, mock_reflections):
        """Test successful reflection table saving."""
        mock_exists.return_value = True
        
        adapter.save_reflection_table(mock_reflections, "/path/to/file.refl")
        
        mock_reflections.as_file.assert_called_once_with("/path/to/file.refl")
    
    @patch('diffusepipe.adapters.dxtbx_io_adapter.Path.exists')
    @patch('diffusepipe.adapters.dxtbx_io_adapter.Path.mkdir')
    def test_save_reflection_table_no_as_file_method(self, mock_mkdir, mock_exists, adapter):
        """Test reflection saving when object lacks as_file method."""
        mock_exists.return_value = True
        reflections_no_as_file = Mock()
        del reflections_no_as_file.as_file  # Remove the as_file attribute
        
        with pytest.raises(DIALSError, match="Reflection table does not support file saving"):
            adapter.save_reflection_table(reflections_no_as_file, "/path/to/file.refl")
    
    @patch('diffusepipe.adapters.dxtbx_io_adapter.Path.exists')
    @patch('diffusepipe.adapters.dxtbx_io_adapter.Path.mkdir')
    def test_save_reflection_table_file_not_created(self, mock_mkdir, mock_exists, adapter, mock_reflections):
        """Test reflection saving when file is not created."""
        mock_exists.return_value = False  # File doesn't exist after saving
        
        with pytest.raises(FileSystemError, match="Failed to create reflection file"):
            adapter.save_reflection_table(mock_reflections, "/path/to/file.refl")
    
    @patch('diffusepipe.adapters.dxtbx_io_adapter.Path')
    def test_pathlib_path_handling(self, mock_path_class, adapter, mock_experiments):
        """Test that adapter handles Path objects correctly."""
        # Create a mock Path instance
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = True
        mock_path_instance.is_file.return_value = True
        mock_path_instance.parent.mkdir = Mock()
        mock_path_class.return_value = mock_path_instance
        
        # Test with pathlib.Path object
        path_obj = Path("/test/path.expt")
        
        with patch('diffusepipe.adapters.dxtbx_io_adapter.ExperimentListFactory') as mock_factory:
            mock_factory.from_json_file.return_value = mock_experiments
            result = adapter.load_experiment_list(path_obj)
            assert result == mock_experiments