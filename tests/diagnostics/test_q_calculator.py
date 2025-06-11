"""Tests for QValueCalculator."""

import pytest
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock

from diffusepipe.diagnostics.q_calculator import QValueCalculator
from diffusepipe.types.types_IDL import ComponentInputFiles, OperationOutcome


@pytest.fixture
def calculator():
    """Create a QValueCalculator instance for testing."""
    return QValueCalculator()


@pytest.fixture
def mock_inputs():
    """Create mock ComponentInputFiles."""
    return ComponentInputFiles(
        dials_expt_path="/path/to/test.expt"
    )


@pytest.fixture
def mock_experiment():
    """Create a mock DIALS experiment with beam and detector."""
    experiment = MagicMock()
    
    # Mock beam model
    beam = MagicMock()
    beam.get_wavelength.return_value = 1.0  # 1 Angstrom
    beam.get_s0.return_value = [0.0, 0.0, 1.0]  # Beam along +z direction
    experiment.beam = beam
    
    # Mock detector with single panel
    panel = MagicMock()
    panel.get_image_size.return_value = (100, 100)  # 100x100 pixels
    
    # Mock get_pixel_lab_coord to return predictable coordinates
    def mock_get_pixel_lab_coord(pixel_coord):
        fast_idx, slow_idx = pixel_coord
        # Return coordinates that simulate a simple flat detector
        # positioned at z = 100mm with pixel size 0.1mm
        x = (fast_idx - 50) * 0.1  # Center at x=0
        y = (slow_idx - 50) * 0.1  # Center at y=0
        z = 100.0  # Fixed distance
        return [x, y, z]
    
    panel.get_pixel_lab_coord.side_effect = mock_get_pixel_lab_coord
    
    detector = MagicMock()
    detector.__len__.return_value = 1
    detector.__iter__.return_value = iter([panel])
    detector.__getitem__.return_value = panel
    experiment.detector = detector
    
    return experiment


@pytest.fixture
def mock_experiment_list(mock_experiment):
    """Create a mock DIALS experiment list."""
    exp_list = MagicMock()
    exp_list.__len__.return_value = 1
    exp_list.__getitem__.return_value = mock_experiment
    return exp_list


class TestQValueCalculator:
    """Test cases for QValueCalculator."""

    def test_init(self, calculator):
        """Test calculator initialization."""
        assert calculator is not None

    def test_calculate_q_map_missing_expt_path(self, calculator):
        """Test calculation with missing experiment path."""
        inputs = ComponentInputFiles()  # No dials_expt_path
        
        result = calculator.calculate_q_map(inputs, "test_output")
        
        assert result.status == "FAILURE"
        assert result.error_code == "InputFileError"
        assert "not provided" in result.message

    def test_calculate_q_map_file_not_found(self, calculator):
        """Test calculation with non-existent experiment file."""
        inputs = ComponentInputFiles(dials_expt_path="/nonexistent/file.expt")
        
        result = calculator.calculate_q_map(inputs, "test_output")
        
        assert result.status == "FAILURE"
        assert result.error_code == "InputFileError"
        assert "not found" in result.message

    @patch('diffusepipe.diagnostics.q_calculator.os.path.exists')
    @patch('dxtbx.model.ExperimentList.from_file')
    @patch('numpy.save')
    def test_calculate_q_map_success_single_panel(
        self, mock_np_save, mock_from_file, mock_exists, calculator, mock_inputs, mock_experiment_list
    ):
        """Test successful q-map calculation for single panel detector."""
        # Setup mocks
        mock_exists.return_value = True
        mock_from_file.return_value = mock_experiment_list
        
        # Execute
        result = calculator.calculate_q_map(mock_inputs, "test_output")
        
        # Verify
        assert result.status == "SUCCESS"
        assert "Successfully generated q-maps" in result.message
        
        # Check output artifacts
        assert "qx_map_path" in result.output_artifacts
        assert "qy_map_path" in result.output_artifacts
        assert "qz_map_path" in result.output_artifacts
        
        # Verify np.save was called for each component
        assert mock_np_save.call_count == 3
        
        # Check file paths
        assert "test_output_qx.npy" in result.output_artifacts["qx_map_path"]
        assert "test_output_qy.npy" in result.output_artifacts["qy_map_path"]
        assert "test_output_qz.npy" in result.output_artifacts["qz_map_path"]

    @patch('diffusepipe.diagnostics.q_calculator.os.path.exists')
    @patch('dxtbx.model.ExperimentList.from_file')
    @patch('numpy.save')
    def test_calculate_q_map_success_multi_panel(
        self, mock_np_save, mock_from_file, mock_exists, calculator, mock_inputs
    ):
        """Test successful q-map calculation for multi-panel detector."""
        # Setup multi-panel experiment
        experiment = MagicMock()
        beam = MagicMock()
        beam.get_wavelength.return_value = 1.0
        beam.get_s0.return_value = [0.0, 0.0, 1.0]
        experiment.beam = beam
        
        # Create two panels
        panel1 = MagicMock()
        panel1.get_image_size.return_value = (50, 50)
        panel1.get_pixel_lab_coord.return_value = [0.0, 0.0, 100.0]
        
        panel2 = MagicMock()
        panel2.get_image_size.return_value = (50, 50)
        panel2.get_pixel_lab_coord.return_value = [0.0, 0.0, 100.0]
        
        detector = MagicMock()
        detector.__len__.return_value = 2
        detector.__iter__.return_value = iter([panel1, panel2])
        experiment.detector = detector
        
        exp_list = MagicMock()
        exp_list.__len__.return_value = 1
        exp_list.__getitem__.return_value = experiment
        
        mock_exists.return_value = True
        mock_from_file.return_value = exp_list
        
        # Execute
        result = calculator.calculate_q_map(mock_inputs, "test_output")
        
        # Verify
        assert result.status == "SUCCESS"
        
        # Check that panel-specific paths are created
        assert "panel0_qx_map_path" in result.output_artifacts
        assert "panel1_qx_map_path" in result.output_artifacts
        
        # Verify np.save was called for both panels (3 components × 2 panels = 6)
        assert mock_np_save.call_count == 6

    @patch('diffusepipe.diagnostics.q_calculator.os.path.exists')
    @patch('dxtbx.model.ExperimentList.from_file')
    def test_calculate_q_map_empty_experiment_list(
        self, mock_from_file, mock_exists, calculator, mock_inputs
    ):
        """Test calculation with empty experiment list."""
        mock_exists.return_value = True
        empty_exp_list = MagicMock()
        empty_exp_list.__len__.return_value = 0
        mock_from_file.return_value = empty_exp_list
        
        result = calculator.calculate_q_map(mock_inputs, "test_output")
        
        assert result.status == "FAILURE"
        assert result.error_code == "DIALSModelError"
        assert "No experiments found" in result.message

    @patch('diffusepipe.diagnostics.q_calculator.os.path.exists')
    def test_calculate_q_map_import_error(self, mock_exists, calculator, mock_inputs):
        """Test calculation when DIALS import fails."""
        mock_exists.return_value = True
        
        with patch('dxtbx.model.ExperimentList.from_file', side_effect=ImportError("DIALS not available")):
            result = calculator.calculate_q_map(mock_inputs, "test_output")
            
            assert result.status == "FAILURE"
            assert result.error_code == "DIALSModelError"
            assert "DIALS not available" in result.message

    @patch('diffusepipe.diagnostics.q_calculator.os.path.exists')
    @patch('dxtbx.model.ExperimentList.from_file')
    @patch('numpy.save')
    def test_calculate_q_map_save_error(
        self, mock_np_save, mock_from_file, mock_exists, calculator, mock_inputs, mock_experiment_list
    ):
        """Test calculation when numpy save fails."""
        mock_exists.return_value = True
        mock_from_file.return_value = mock_experiment_list
        mock_np_save.side_effect = IOError("Permission denied")
        
        result = calculator.calculate_q_map(mock_inputs, "test_output")
        
        assert result.status == "FAILURE"
        assert result.error_code == "OutputWriteError"

    def test_load_dials_experiment_import_error(self, calculator):
        """Test _load_dials_experiment with import error."""
        with patch('builtins.__import__', side_effect=ImportError("Module not found")):
            with pytest.raises(ImportError, match="Failed to import dxtbx.model.ExperimentList"):
                calculator._load_dials_experiment("/path/to/test.expt")

    @patch('dxtbx.model.ExperimentList.from_file')
    def test_load_dials_experiment_file_error(self, mock_from_file, calculator):
        """Test _load_dials_experiment with file loading error."""
        mock_from_file.side_effect = Exception("File corrupted")
        
        with pytest.raises(Exception, match="Failed to load DIALS experiment"):
            calculator._load_dials_experiment("/path/to/test.expt")

    @patch('dxtbx.model.ExperimentList.from_file')
    def test_load_dials_experiment_success(self, mock_from_file, calculator, mock_experiment_list):
        """Test successful _load_dials_experiment."""
        mock_from_file.return_value = mock_experiment_list
        
        result = calculator._load_dials_experiment("/path/to/test.expt")
        
        assert result == mock_experiment_list
        mock_from_file.assert_called_once_with("/path/to/test.expt")

    def test_calculate_panel_q_vectors(self, calculator):
        """Test _calculate_panel_q_vectors with predictable inputs."""
        # Create mock beam and panel
        beam = MagicMock()
        beam.get_wavelength.return_value = 1.0  # 1 Angstrom
        beam.get_s0.return_value = [0.0, 0.0, 1.0]  # Beam along +z
        
        panel = MagicMock()
        panel.get_image_size.return_value = (2, 2)  # Small 2x2 panel for testing
        
        # Mock lab coordinates for a simple case
        def mock_lab_coord(pixel_coord):
            fast_idx, slow_idx = pixel_coord
            return [fast_idx, slow_idx, 100.0]  # Simple coordinates
        
        panel.get_pixel_lab_coord.side_effect = mock_lab_coord
        
        # Execute
        qx_map, qy_map, qz_map = calculator._calculate_panel_q_vectors(beam, panel)
        
        # Verify
        assert qx_map.shape == (2, 2)
        assert qy_map.shape == (2, 2)
        assert qz_map.shape == (2, 2)
        
        # Check that q-vectors have reasonable values
        # (detailed numerical validation would require specific geometry)
        assert np.all(np.isfinite(qx_map))
        assert np.all(np.isfinite(qy_map))
        assert np.all(np.isfinite(qz_map))

    def test_calculate_panel_q_vectors_realistic_geometry(self, calculator):
        """Test q-vector calculation with more realistic detector geometry."""
        # Create realistic beam model
        beam = MagicMock()
        beam.get_wavelength.return_value = 0.97680  # Wavelength from CBF header
        beam.get_s0.return_value = [0.0, 0.0, 1.0]  # Beam along +z
        
        # Create realistic panel model
        panel = MagicMock()
        panel.get_image_size.return_value = (10, 10)  # Small panel for fast testing
        
        # Mock realistic lab coordinates (detector at 230mm, pixel size 0.172mm)
        def mock_realistic_lab_coord(pixel_coord):
            fast_idx, slow_idx = pixel_coord
            pixel_size = 0.172  # mm
            detector_distance = 230.0  # mm
            
            # Center coordinates at beam center (assume beam at 5, 5)
            x = (fast_idx - 5) * pixel_size
            y = (slow_idx - 5) * pixel_size
            z = detector_distance
            
            return [x, y, z]
        
        panel.get_pixel_lab_coord.side_effect = mock_realistic_lab_coord
        
        # Execute
        qx_map, qy_map, qz_map = calculator._calculate_panel_q_vectors(beam, panel)
        
        # Verify shape and finite values
        assert qx_map.shape == (10, 10)
        assert qy_map.shape == (10, 10)
        assert qz_map.shape == (10, 10)
        
        assert np.all(np.isfinite(qx_map))
        assert np.all(np.isfinite(qy_map))
        assert np.all(np.isfinite(qz_map))
        
        # Check that q-vectors have expected magnitude order
        q_magnitude = np.sqrt(qx_map**2 + qy_map**2 + qz_map**2)
        k_magnitude = 2 * np.pi / 0.97680
        
        # Q-vectors should be less than 2*k (forward scattering limit)
        assert np.all(q_magnitude < 2 * k_magnitude)
        
        # Q-vectors at the center should be smallest (forward scattering)
        center_q = q_magnitude[5, 5]
        assert center_q < np.mean(q_magnitude)