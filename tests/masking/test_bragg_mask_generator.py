"""
Unit tests for BraggMaskGenerator.

These tests focus on the Bragg mask generation functions, including both
dials.generate_mask (Option A) and shoebox-based (Option B) approaches.
"""

import pytest
from unittest.mock import Mock, patch

from diffusepipe.masking.bragg_mask_generator import (
    BraggMaskGenerator,
    create_default_bragg_mask_config,
    create_expanded_bragg_mask_config,
    validate_mask_compatibility,
)
from diffusepipe.exceptions import BraggMaskError


class TestBraggMaskGenerator:
    """Test suite for BraggMaskGenerator class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.generator = BraggMaskGenerator()

    def test_initialization(self):
        """Test BraggMaskGenerator initialization."""
        generator = BraggMaskGenerator()
        assert generator.dials_adapter is not None

    def test_generate_bragg_mask_from_spots_success(self):
        """Test successful Bragg mask generation using dials.generate_mask."""
        # Arrange
        mock_experiment = Mock()
        mock_reflections = Mock()
        mock_config = {"border": 2}

        # Mock successful adapter response
        mock_bragg_mask = (Mock(), Mock())  # Two panels
        with patch.object(
            self.generator.dials_adapter, "generate_bragg_mask"
        ) as mock_generate:
            mock_generate.return_value = (mock_bragg_mask, True, "Success")

            # Act
            result = self.generator.generate_bragg_mask_from_spots(
                experiment=mock_experiment,
                reflections=mock_reflections,
                config=mock_config,
            )

            # Assert
            assert result is mock_bragg_mask
            mock_generate.assert_called_once_with(
                experiment=mock_experiment,
                reflections=mock_reflections,
                mask_generation_params=mock_config,
            )

    def test_generate_bragg_mask_from_spots_failure(self):
        """Test handling of dials.generate_mask failure."""
        # Arrange
        mock_experiment = Mock()
        mock_reflections = Mock()

        # Mock failed adapter response
        with patch.object(
            self.generator.dials_adapter, "generate_bragg_mask"
        ) as mock_generate:
            mock_generate.return_value = (None, False, "DIALS error")

            # Act & Assert
            with pytest.raises(BraggMaskError) as exc_info:
                self.generator.generate_bragg_mask_from_spots(
                    experiment=mock_experiment, reflections=mock_reflections
                )

            assert "DIALS mask generation failed" in str(exc_info.value)

    def test_generate_bragg_mask_from_spots_exception(self):
        """Test handling of unexpected exceptions in spots-based generation."""
        # Arrange
        mock_experiment = Mock()
        mock_reflections = Mock()

        # Mock adapter exception
        with patch.object(
            self.generator.dials_adapter, "generate_bragg_mask"
        ) as mock_generate:
            mock_generate.side_effect = RuntimeError("Unexpected error")

            # Act & Assert
            with pytest.raises(BraggMaskError) as exc_info:
                self.generator.generate_bragg_mask_from_spots(
                    experiment=mock_experiment, reflections=mock_reflections
                )

            assert "Failed to generate Bragg mask from spots" in str(exc_info.value)

    def test_generate_bragg_mask_from_shoeboxes_success(self):
        """Test successful Bragg mask generation using shoebox data (integration test with real DIALS objects)."""
        # Arrange
        mock_detector = Mock()
        mock_panel = Mock()
        mock_panel.get_image_size.return_value = (4, 4)  # (fast, slow)
        mock_detector.__iter__ = Mock(return_value=iter([mock_panel]))

        # Mock reflections with shoeboxes
        mock_reflections = Mock()
        mock_reflections.__len__ = Mock(return_value=1)
        mock_reflections.__iter__ = Mock(
            return_value=iter([{"shoebox": Mock(), "panel": 0}])
        )
        mock_reflections.has_key.return_value = True

        # Act
        result = self.generator.generate_bragg_mask_from_shoeboxes(
            reflections=mock_reflections, detector=mock_detector
        )

        # Assert - Check that we got a valid flex.bool object
        from dials.array_family import flex

        assert result is not None
        assert len(result) == 1
        assert isinstance(result[0], flex.bool)
        assert result[0].size() == 16  # 4x4 panel
        mock_reflections.has_key.assert_called_with("shoebox")

    def test_generate_bragg_mask_from_shoeboxes_no_reflections(self):
        """Test shoebox-based generation with no reflections (integration test with real DIALS objects)."""
        # Arrange
        mock_detector = Mock()
        mock_panel = Mock()
        mock_panel.get_image_size.return_value = (4, 4)
        mock_detector.__iter__ = Mock(return_value=iter([mock_panel]))

        mock_reflections = Mock()
        mock_reflections.__len__ = Mock(return_value=0)

        # Act
        result = self.generator.generate_bragg_mask_from_shoeboxes(
            reflections=mock_reflections, detector=mock_detector
        )

        # Assert - Check that we got a valid flex.bool object
        from dials.array_family import flex

        assert result is not None
        assert len(result) == 1
        assert isinstance(result[0], flex.bool)
        assert result[0].size() == 16  # 4x4 panel

    @patch("diffusepipe.masking.bragg_mask_generator.flex")
    def test_generate_bragg_mask_from_shoeboxes_missing_shoebox_column(self, mock_flex):
        """Test error handling when shoebox column is missing."""
        # Arrange
        mock_detector = Mock()
        mock_panel = Mock()
        mock_panel.get_image_size.return_value = (4, 4)
        mock_detector.__iter__ = Mock(return_value=iter([mock_panel]))

        mock_reflections = Mock()
        mock_reflections.__len__ = Mock(return_value=1)
        mock_reflections.has_key.return_value = False  # No shoebox column

        mock_mask = Mock()
        mock_flex.bool.return_value = mock_mask
        mock_flex.grid.return_value = Mock()

        # Act & Assert
        with pytest.raises(BraggMaskError) as exc_info:
            self.generator.generate_bragg_mask_from_shoeboxes(
                reflections=mock_reflections, detector=mock_detector
            )

        assert "missing required 'shoebox' column" in str(exc_info.value)

    def test_get_total_mask_for_still_success(self):
        """Test successful combination of Bragg and pixel masks."""
        # Arrange
        mock_bragg_mask = Mock()
        mock_pixel_mask = Mock()
        mock_inverted_bragg = Mock()
        mock_total_mask = Mock()

        # Mock logical operations
        mock_bragg_mask.__invert__ = Mock(return_value=mock_inverted_bragg)
        mock_pixel_mask.__and__ = Mock(return_value=mock_total_mask)

        # Mock count operations for logging
        mock_pixel_mask.count.return_value = 90
        mock_bragg_mask.count.return_value = 10
        mock_total_mask.count.return_value = 85
        mock_total_mask.__len__ = Mock(return_value=100)

        bragg_mask = (mock_bragg_mask,)
        global_pixel_mask = (mock_pixel_mask,)

        # Act
        result = self.generator.get_total_mask_for_still(bragg_mask, global_pixel_mask)

        # Assert
        assert result is not None
        assert len(result) == 1
        assert result[0] is mock_total_mask

        # Verify logical operations were performed correctly
        mock_bragg_mask.__invert__.assert_called_once()
        mock_pixel_mask.__and__.assert_called_once_with(mock_inverted_bragg)

    def test_get_total_mask_for_still_panel_count_mismatch(self):
        """Test error handling when panel counts don't match."""
        # Arrange
        bragg_mask = (Mock(), Mock())  # 2 panels
        global_pixel_mask = (Mock(),)  # 1 panel

        # Act & Assert
        with pytest.raises(BraggMaskError) as exc_info:
            self.generator.get_total_mask_for_still(bragg_mask, global_pixel_mask)

        assert "different panel counts" in str(exc_info.value)

    def test_get_total_mask_for_still_exception_handling(self):
        """Test handling of unexpected exceptions in mask combination."""
        # Arrange
        mock_bragg_mask = Mock()
        mock_pixel_mask = Mock()

        # Mock operation to raise exception
        mock_bragg_mask.__invert__ = Mock(
            side_effect=RuntimeError("Logical operation failed")
        )

        bragg_mask = (mock_bragg_mask,)
        global_pixel_mask = (mock_pixel_mask,)

        # Act & Assert
        with pytest.raises(BraggMaskError) as exc_info:
            self.generator.get_total_mask_for_still(bragg_mask, global_pixel_mask)

        assert "Failed to combine masks" in str(exc_info.value)

    @patch("diffusepipe.masking.bragg_mask_generator.MaskCode")
    def test_process_reflection_shoebox_basic(self, mock_maskcode):
        """Test processing of a single reflection's shoebox."""
        # Arrange
        mock_maskcode.Foreground = 1
        mock_maskcode.Strong = 2

        # Create mock reflection with shoebox
        mock_shoebox = Mock()
        mock_shoebox.bbox = (0, 0, 0, 1, 2, 2)  # z1, y1, x1, z2, y2, x2
        mock_shoebox.mask = [1, 0, 1, 0]  # Some foreground pixels

        reflection = {"shoebox": mock_shoebox, "panel": 0}

        # Create mock panel masks
        mock_panel_mask = Mock()
        mock_panel_mask.__len__ = Mock(return_value=4)  # 2x2 panel
        mock_panel_mask.__getitem__ = Mock()
        mock_panel_mask.__setitem__ = Mock()
        panel_masks = [mock_panel_mask]

        # Act
        result = self.generator._process_reflection_shoebox(
            reflection, 0, panel_masks, 0
        )

        # Assert
        assert isinstance(result, int)
        assert result >= 0

    def test_process_reflection_shoebox_invalid_panel(self):
        """Test error handling for invalid panel ID in reflection."""
        # Arrange
        mock_shoebox = Mock()
        mock_shoebox.bbox = (0, 0, 0, 1, 1, 1)
        reflection = {"shoebox": mock_shoebox, "panel": 5}  # Invalid panel ID
        panel_masks = [Mock()]  # Only one panel available

        # Act & Assert
        with pytest.raises(Exception) as exc_info:
            self.generator._process_reflection_shoebox(reflection, 0, panel_masks, 0)

        assert "exceeds available panels" in str(exc_info.value)


class TestUtilityFunctions:
    """Test suite for utility functions."""

    def test_create_default_bragg_mask_config(self):
        """Test creation of default Bragg mask configuration."""
        config = create_default_bragg_mask_config()

        assert isinstance(config, dict)
        assert "border" in config
        assert "algorithm" in config
        assert config["border"] == 2
        assert config["algorithm"] == "simple"

    def test_create_expanded_bragg_mask_config(self):
        """Test creation of expanded Bragg mask configuration."""
        config = create_expanded_bragg_mask_config(border=5)

        assert isinstance(config, dict)
        assert config["border"] == 5
        assert config["algorithm"] == "simple"

    def test_create_expanded_bragg_mask_config_default(self):
        """Test creation of expanded configuration with default border."""
        config = create_expanded_bragg_mask_config()

        assert config["border"] == 3

    def test_validate_mask_compatibility_success(self):
        """Test successful mask compatibility validation."""
        # Arrange
        mock_bragg_panel = Mock()
        mock_pixel_panel = Mock()
        mock_bragg_panel.__len__ = Mock(return_value=100)
        mock_pixel_panel.__len__ = Mock(return_value=100)

        bragg_mask = (mock_bragg_panel,)
        pixel_mask = (mock_pixel_panel,)

        # Act
        result = validate_mask_compatibility(bragg_mask, pixel_mask)

        # Assert
        assert result is True

    def test_validate_mask_compatibility_panel_count_mismatch(self):
        """Test mask compatibility validation with panel count mismatch."""
        # Arrange
        bragg_mask = (Mock(), Mock())  # 2 panels
        pixel_mask = (Mock(),)  # 1 panel

        # Act
        result = validate_mask_compatibility(bragg_mask, pixel_mask)

        # Assert
        assert result is False

    def test_validate_mask_compatibility_size_mismatch(self):
        """Test mask compatibility validation with panel size mismatch."""
        # Arrange
        mock_bragg_panel = Mock()
        mock_pixel_panel = Mock()
        mock_bragg_panel.__len__ = Mock(return_value=100)
        mock_pixel_panel.__len__ = Mock(return_value=200)  # Different size

        bragg_mask = (mock_bragg_panel,)
        pixel_mask = (mock_pixel_panel,)

        # Act
        result = validate_mask_compatibility(bragg_mask, pixel_mask)

        # Assert
        assert result is False

    def test_validate_mask_compatibility_exception_handling(self):
        """Test mask compatibility validation with exception."""
        # Arrange
        mock_bragg_panel = Mock()
        mock_bragg_panel.__len__ = Mock(side_effect=RuntimeError("Access error"))

        bragg_mask = (mock_bragg_panel,)
        pixel_mask = (Mock(),)

        # Act
        result = validate_mask_compatibility(bragg_mask, pixel_mask)

        # Assert
        assert result is False


class TestIntegrationScenarios:
    """Integration-style tests for complete Bragg masking workflows."""

    def setup_method(self):
        """Set up test fixtures."""
        self.generator = BraggMaskGenerator()

    def test_complete_workflow_option_a(self):
        """Test complete workflow using dials.generate_mask (Option A)."""
        # Arrange
        mock_experiment = Mock()
        mock_reflections = Mock()
        mock_global_pixel_mask = Mock()

        # Mock successful Bragg mask generation
        mock_bragg_mask = Mock()
        mock_bragg_mask.count.return_value = 10
        mock_bragg_mask.__len__ = Mock(return_value=100)

        # Mock mask combination
        mock_inverted_bragg = Mock()
        mock_total_mask = Mock()
        mock_bragg_mask.__invert__ = Mock(return_value=mock_inverted_bragg)
        mock_global_pixel_mask.__and__ = Mock(return_value=mock_total_mask)
        mock_global_pixel_mask.count.return_value = 95
        mock_total_mask.count.return_value = 88
        mock_total_mask.__len__ = Mock(return_value=100)

        with patch.object(
            self.generator.dials_adapter, "generate_bragg_mask"
        ) as mock_generate:
            mock_generate.return_value = ((mock_bragg_mask,), True, "Success")

            # Act
            # Step 1: Generate Bragg mask
            bragg_mask = self.generator.generate_bragg_mask_from_spots(
                experiment=mock_experiment, reflections=mock_reflections
            )

            # Step 2: Combine with pixel mask
            total_mask = self.generator.get_total_mask_for_still(
                bragg_mask=bragg_mask, global_pixel_mask=(mock_global_pixel_mask,)
            )

            # Assert
            assert bragg_mask is not None
            assert len(bragg_mask) == 1
            assert total_mask is not None
            assert len(total_mask) == 1
            assert total_mask[0] is mock_total_mask

    @patch("diffusepipe.masking.bragg_mask_generator.flex")
    @patch("diffusepipe.masking.bragg_mask_generator.MaskCode")
    def test_complete_workflow_option_b(self, mock_maskcode, mock_flex):
        """Test complete workflow using shoeboxes (Option B)."""
        # Arrange
        mock_detector = Mock()
        mock_panel = Mock()
        mock_panel.get_image_size.return_value = (4, 4)
        mock_detector.__iter__ = Mock(return_value=iter([mock_panel]))

        mock_reflections = Mock()
        mock_reflections.__len__ = Mock(return_value=0)  # No reflections for simplicity
        mock_reflections.has_key.return_value = True

        mock_global_pixel_mask = Mock()

        # Mock flex operations
        mock_bragg_mask = Mock()
        mock_bragg_mask.count.return_value = 5
        mock_bragg_mask.__len__ = Mock(return_value=16)
        mock_flex.bool.return_value = mock_bragg_mask
        mock_flex.grid.return_value = Mock()

        # Mock mask combination
        mock_inverted_bragg = Mock()
        mock_total_mask = Mock()
        mock_bragg_mask.__invert__ = Mock(return_value=mock_inverted_bragg)
        mock_global_pixel_mask.__and__ = Mock(return_value=mock_total_mask)
        mock_global_pixel_mask.count.return_value = 15
        mock_total_mask.count.return_value = 12
        mock_total_mask.__len__ = Mock(return_value=16)

        # Act
        # Step 1: Generate Bragg mask from shoeboxes
        bragg_mask = self.generator.generate_bragg_mask_from_shoeboxes(
            reflections=mock_reflections, detector=mock_detector
        )

        # Step 2: Combine with pixel mask
        total_mask = self.generator.get_total_mask_for_still(
            bragg_mask=bragg_mask, global_pixel_mask=(mock_global_pixel_mask,)
        )

        # Assert
        assert bragg_mask is not None
        assert len(bragg_mask) == 1
        assert total_mask is not None
        assert len(total_mask) == 1
        assert total_mask[0] is mock_total_mask
