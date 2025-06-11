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
    experiment = MagicMock()

    # Mock detector with panels
    panel1 = MagicMock()
    panel1.get_image_size.return_value = (100, 100)
    panel2 = MagicMock()
    panel2.get_image_size.return_value = (100, 100)

    detector = MagicMock()
    detector.__iter__.return_value = iter([panel1, panel2])
    detector.__len__.return_value = 2

    experiment.detector = detector
    return experiment


@pytest.fixture
def mock_reflections():
    """Create a mock DIALS reflection table."""
    reflections = MagicMock()
    reflections.__len__.return_value = 100  # Mock 100 reflections
    return reflections


@pytest.fixture
def mock_mask_params():
    """Create mock mask generation parameters."""
    return {"border": 3, "algorithm": "simple"}


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

    @patch("dials.util.masking.generate_mask")
    def test_generate_bragg_mask_success(
        self, mock_generate_mask, adapter, mock_experiment, mock_reflections
    ):
        """Test successful mask generation."""
        # Mock generate_mask to return a tuple of mock panel masks
        mock_panel_mask1 = Mock()
        mock_panel_mask2 = Mock()
        mock_generate_mask.return_value = (mock_panel_mask1, mock_panel_mask2)

        mask_result, success, logs = adapter.generate_bragg_mask(
            mock_experiment, mock_reflections
        )

        assert success is True
        assert isinstance(mask_result, tuple)
        assert len(mask_result) == 2  # Two panels
        assert "Generated Bragg mask successfully" in logs
        
        # Verify generate_mask was called with correct parameters
        mock_generate_mask.assert_called_once_with(
            mock_experiment, mock_reflections, border=2, algorithm="simple"
        )

    @patch("dials.util.masking.generate_mask")
    def test_generate_bragg_mask_with_params(
        self, mock_generate_mask, adapter, mock_experiment, mock_reflections, mock_mask_params
    ):
        """Test mask generation with custom parameters."""
        # Mock generate_mask to return a tuple of mock panel masks
        mock_panel_mask1 = Mock()
        mock_panel_mask2 = Mock()
        mock_generate_mask.return_value = (mock_panel_mask1, mock_panel_mask2)

        mask_result, success, logs = adapter.generate_bragg_mask(
            mock_experiment, mock_reflections, mock_mask_params
        )

        assert success is True
        assert isinstance(mask_result, tuple)
        
        # Verify generate_mask was called with custom parameters
        mock_generate_mask.assert_called_once_with(
            mock_experiment, mock_reflections, **mock_mask_params
        )

    @patch("dials.util.masking.generate_mask")
    def test_generate_bragg_mask_dials_failure(
        self, mock_generate_mask, adapter, mock_experiment, mock_reflections
    ):
        """Test mask generation when DIALS generate_mask fails."""
        # Make generate_mask raise an exception
        mock_generate_mask.side_effect = Exception("DIALS masking failed")

        with pytest.raises(DIALSError, match="DIALS generate_mask call failed"):
            adapter.generate_bragg_mask(mock_experiment, mock_reflections)

    def test_generate_bragg_mask_import_error(
        self, adapter, mock_experiment, mock_reflections
    ):
        """Test mask generation with DIALS import error."""
        with patch(
            "dials.util.masking.generate_mask",
            side_effect=ImportError("DIALS not found"),
        ):
            with pytest.raises(
                DIALSError, match="DIALS generate_mask call failed"
            ):
                adapter.generate_bragg_mask(mock_experiment, mock_reflections)

    @patch("dials.util.masking.generate_mask")
    def test_call_generate_mask_success(
        self, mock_generate_mask, adapter, mock_experiment, mock_reflections
    ):
        """Test successful _call_generate_mask."""
        # Mock generate_mask to return a tuple of mock panel masks
        mock_panel_mask1 = Mock()
        mock_panel_mask2 = Mock()
        mock_generate_mask.return_value = (mock_panel_mask1, mock_panel_mask2)

        params = {"border": 2, "algorithm": "simple"}
        result = adapter._call_generate_mask(mock_experiment, mock_reflections, params)

        assert isinstance(result, tuple)
        assert len(result) == 2  # Two panels from mock_experiment
        
        # Verify generate_mask was called with correct parameters
        mock_generate_mask.assert_called_once_with(
            mock_experiment, mock_reflections, **params
        )

    @patch("dials.util.masking.generate_mask")
    def test_call_generate_mask_failure(
        self, mock_generate_mask, adapter, mock_experiment, mock_reflections
    ):
        """Test _call_generate_mask with failure."""
        # Make generate_mask raise an exception
        mock_generate_mask.side_effect = Exception("DIALS masking error")

        params = {"border": 2, "algorithm": "simple"}

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

    @patch("dials.util.masking.generate_mask")
    def test_generate_bragg_mask_integration(self, mock_generate_mask, adapter):
        """Test integration of the full generate_bragg_mask workflow."""
        # Mock generate_mask to return a single panel mask
        mock_panel_mask = Mock()
        mock_generate_mask.return_value = (mock_panel_mask,)

        # Create experiment with detector
        experiment = MagicMock()
        panel = MagicMock()
        panel.get_image_size.return_value = (100, 100)
        detector = MagicMock()
        detector.__iter__.return_value = iter([panel])
        detector.__len__.return_value = 1
        experiment.detector = detector

        # Create reflections
        reflections = MagicMock()

        # Test the full workflow
        mask_result, success, logs = adapter.generate_bragg_mask(
            experiment, reflections
        )

        assert success is True
        assert isinstance(mask_result, tuple)
        assert len(mask_result) == 1
        assert "Starting Bragg mask generation" in logs
        assert "Generated Bragg mask successfully" in logs
        assert "Validated mask result" in logs
        
        # Verify generate_mask was called with default parameters
        mock_generate_mask.assert_called_once_with(
            experiment, reflections, border=2, algorithm="simple"
        )
