"""Tests for DIALSGenerateMaskAdapter."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from diffusepipe.adapters.dials_generate_mask_adapter import DIALSGenerateMaskAdapter
from diffusepipe.exceptions import DIALSError

try:
    from dials.array_family import flex
except ImportError:
    # For environments without DIALS, create a minimal mock
    class MockFlex:
        class bool:
            def __init__(self, grid, value):
                self.grid = grid
                self.value = value

        class grid:
            def __init__(self, x, y):
                self.x = x
                self.y = y

        class vec3_double:
            def __init__(self, values):
                self.values = values

        class int:
            def __init__(self, values, default=0):
                self.values = values

    flex = MockFlex()


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

    # Setup for _call_generate_mask tests
    reflections.__contains__.side_effect = lambda k: k in ["xyzobs.px.value", "panel"]

    # Mock centroid data
    mock_centroids = flex.vec3_double([(10.0, 10.0, 0.0)])
    mock_panels = flex.int([0])

    reflections.__getitem__.side_effect = lambda key: {
        "xyzobs.px.value": mock_centroids,
        "panel": mock_panels,
    }.get(key)

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

    def test_generate_bragg_mask_success(
        self, adapter, mock_experiment, mock_reflections
    ):
        """Test successful mask generation."""

        # Create a proper ExperimentList mock class that works with isinstance
        class MockExperimentList:
            def __init__(self, experiments):
                self.experiments = experiments

            def __len__(self):
                return len(self.experiments)

            def __getitem__(self, index):
                return self.experiments[index]

        with patch.object(
            adapter, "_call_generate_mask"
        ) as mock_call_generate_mask_method:
            # Mock the _call_generate_mask method to return panel masks
            mock_panel_mask1 = Mock()
            mock_panel_mask2 = Mock()
            mock_call_generate_mask_method.return_value = (
                mock_panel_mask1,
                mock_panel_mask2,
            )

            with patch("dxtbx.model.ExperimentList", MockExperimentList):
                mask_result, success, logs = adapter.generate_bragg_mask(
                    mock_experiment, mock_reflections
                )

                assert success is True
                assert isinstance(mask_result, tuple)
                assert len(mask_result) == 2  # Two panels
                assert "Generated Bragg mask successfully" in logs

                # Verify _call_generate_mask was called with correct parameters
                # The first argument should be an instance of MockExperimentList
                mock_call_generate_mask_method.assert_called_once()
                call_args = mock_call_generate_mask_method.call_args[0]
                assert isinstance(call_args[0], MockExperimentList)
                assert call_args[1] is mock_reflections

    def test_generate_bragg_mask_with_params(
        self, adapter, mock_experiment, mock_reflections, mock_mask_params
    ):
        """Test mask generation with custom parameters."""

        # Create a proper ExperimentList mock class that works with isinstance
        class MockExperimentList:
            def __init__(self, experiments):
                self.experiments = experiments

            def __len__(self):
                return len(self.experiments)

            def __getitem__(self, index):
                return self.experiments[index]

        with patch.object(
            adapter, "_call_generate_mask"
        ) as mock_call_generate_mask_method:
            # Mock the _call_generate_mask method to return panel masks
            mock_panel_mask1 = Mock()
            mock_panel_mask2 = Mock()
            mock_call_generate_mask_method.return_value = (
                mock_panel_mask1,
                mock_panel_mask2,
            )

            with patch("dxtbx.model.ExperimentList", MockExperimentList):
                mask_result, success, logs = adapter.generate_bragg_mask(
                    mock_experiment, mock_reflections, mock_mask_params
                )

                assert success is True
                assert isinstance(mask_result, tuple)

                # Verify _call_generate_mask was called with correct parameters
                mock_call_generate_mask_method.assert_called_once()
                call_args = mock_call_generate_mask_method.call_args[0]
                assert isinstance(call_args[0], MockExperimentList)
                assert call_args[1] is mock_reflections

    def test_generate_bragg_mask_dials_failure(
        self, adapter, mock_experiment, mock_reflections
    ):
        """Test mask generation when DIALS generate_mask fails."""

        # Create a proper ExperimentList mock class that works with isinstance
        class MockExperimentList:
            def __init__(self, experiments):
                self.experiments = experiments

            def __len__(self):
                return len(self.experiments)

            def __getitem__(self, index):
                return self.experiments[index]

        with patch.object(
            adapter, "_call_generate_mask"
        ) as mock_call_generate_mask_method:
            # Make _call_generate_mask raise an exception
            mock_call_generate_mask_method.side_effect = DIALSError(
                "Mocked DIALS masking failed"
            )

            with patch("dxtbx.model.ExperimentList", MockExperimentList):
                with pytest.raises(DIALSError, match="Mocked DIALS masking failed"):
                    adapter.generate_bragg_mask(mock_experiment, mock_reflections)

    def test_generate_bragg_mask_import_error(
        self, adapter, mock_experiment, mock_reflections
    ):
        """Test mask generation with DIALS import error."""
        import sys

        # Mock the import by removing the module from sys.modules temporarily
        original_module = sys.modules.get("dxtbx.model")
        if "dxtbx.model" in sys.modules:
            del sys.modules["dxtbx.model"]

        try:
            with patch("builtins.__import__") as mock_import:

                def side_effect(name, *args, **kwargs):
                    if name == "dxtbx.model":
                        raise ImportError("Mocked ExperimentList import error")
                    return __import__(name, *args, **kwargs)

                mock_import.side_effect = side_effect

                with pytest.raises(
                    DIALSError,
                    match="Failed to import ExperimentList: Mocked ExperimentList import error",
                ):
                    adapter.generate_bragg_mask(mock_experiment, mock_reflections)
        finally:
            # Restore the original module
            if original_module is not None:
                sys.modules["dxtbx.model"] = original_module

    def test_call_generate_mask_success(self, adapter):
        """Test successful _call_generate_mask."""
        # Setup mock reflections with proper structure for the actual implementation
        mock_reflections = MagicMock()
        mock_reflections.__len__.return_value = 1
        mock_reflections.__contains__.side_effect = lambda k: k in [
            "xyzobs.px.value",
            "panel",
        ]

        # Mock centroid and panel data
        mock_centroids = [MagicMock()]
        mock_centroids[0].__getitem__.side_effect = lambda i: [10.0, 10.0, 0.0][i]
        mock_panels = [0]

        mock_reflections.__getitem__.side_effect = lambda key: {
            "xyzobs.px.value": mock_centroids,
            "panel": mock_panels,
        }.get(key)

        # Setup mock experiment list with proper structure
        mock_panel = MagicMock()
        mock_panel.get_image_size.return_value = (100, 100)

        # Create a mock flex.bool that behaves like the real one
        mock_mask = MagicMock()
        mock_mask.all.return_value = (100, 100)  # height, width
        mock_mask.__setitem__ = MagicMock()

        # Mock the entire flex module
        with patch("dials.array_family.flex") as mock_flex:
            # Setup flex.bool mock
            mock_flex.bool.return_value = mock_mask
            mock_flex.grid = MagicMock()
            mock_flex.int = MagicMock(return_value=[0])  # For panel assignments

            mock_experiment = MagicMock()
            mock_experiment.detector = MagicMock()
            mock_experiment.detector.__iter__.return_value = iter([mock_panel])
            mock_experiment.detector.__len__.return_value = 1

            mock_experiment_list = MagicMock()
            mock_experiment_list.__len__.return_value = 1
            mock_experiment_list.__getitem__.return_value = mock_experiment

            # Setup mock phil params
            mock_params = MagicMock()
            mock_params.border = 2

            result = adapter._call_generate_mask(
                mock_experiment_list, mock_reflections, mock_params
            )

            assert isinstance(result, tuple)
            assert len(result) == 1  # One panel from mock_experiment

    def test_call_generate_mask_failure(self, adapter, mock_reflections):
        """Test _call_generate_mask with failure."""
        # Setup empty experiment list to trigger failure
        mock_experiment_list = MagicMock()
        mock_experiment_list.__len__.return_value = 0

        mock_params = MagicMock()
        mock_params.border = 2

        with pytest.raises(
            DIALSError, match="Bragg mask generation failed: ExperimentList is empty"
        ):
            adapter._call_generate_mask(
                mock_experiment_list, mock_reflections, mock_params
            )

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

    def test_generate_bragg_mask_integration(self, adapter):
        """Test integration of the full generate_bragg_mask workflow."""

        # Create a proper ExperimentList mock class that works with isinstance
        class MockExperimentList:
            def __init__(self, experiments):
                self.experiments = experiments

            def __len__(self):
                return len(self.experiments)

            def __getitem__(self, index):
                return self.experiments[index]

        with patch.object(
            adapter, "_call_generate_mask"
        ) as mock_call_generate_mask_method:
            # Mock the _call_generate_mask method to return a single panel mask
            mock_panel_mask = Mock()
            mock_call_generate_mask_method.return_value = (mock_panel_mask,)

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

            with patch("dxtbx.model.ExperimentList", MockExperimentList):
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

                # Verify _call_generate_mask was called
                mock_call_generate_mask_method.assert_called_once()
                call_args = mock_call_generate_mask_method.call_args[0]
                assert isinstance(call_args[0], MockExperimentList)
                assert call_args[1] is reflections
