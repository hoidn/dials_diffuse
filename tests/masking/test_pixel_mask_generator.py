"""
Unit tests for PixelMaskGenerator.

These tests focus on individual functions within the PixelMaskGenerator class,
using mock objects to isolate the masking logic.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from diffusepipe.masking.pixel_mask_generator import (
    PixelMaskGenerator,
    StaticMaskParams,
    DynamicMaskParams,
    Circle,
    Rectangle,
    create_circular_beamstop,
    create_rectangular_beamstop,
    create_default_static_params,
    create_default_dynamic_params,
)
from diffusepipe.exceptions import MaskGenerationError


class TestPixelMaskGenerator:
    """Test suite for PixelMaskGenerator class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.generator = PixelMaskGenerator()

    def test_generate_static_mask_basic(self):
        """Test basic static mask generation."""
        # Arrange
        mock_detector = Mock()
        mock_panel = Mock()
        mock_panel.get_image_size.return_value = (100, 50)  # (fast, slow)
        mock_panel.get_trusted_range.return_value = (-10, 65535)
        mock_detector.__iter__ = Mock(return_value=iter([mock_panel]))
        mock_detector.__len__ = Mock(return_value=1)

        static_params = StaticMaskParams()

        # Act
        result = self.generator.generate_static_mask(mock_detector, static_params)

        # Assert
        assert result is not None
        assert len(result) == 1
        # Check that we got a valid flex.bool object
        from dials.array_family import flex
        assert isinstance(result[0], flex.bool)
        # Check the mask has the expected size (50 * 100 = 5000 pixels)
        assert result[0].size() == 5000

    @patch("diffusepipe.masking.pixel_mask_generator.flex")
    def test_generate_static_mask_with_beamstop(self, mock_flex):
        """Test static mask generation with circular beamstop."""
        # Arrange
        mock_detector = Mock()
        mock_panel = Mock()
        mock_panel.get_image_size.return_value = (100, 50)
        mock_detector.__iter__ = Mock(return_value=iter([mock_panel]))
        mock_detector.__len__ = Mock(return_value=1)

        # Mock flex operations
        mock_mask = Mock()
        mock_beamstop_mask = Mock()
        mock_flex.bool.side_effect = [mock_mask, mock_beamstop_mask]
        mock_flex.grid.return_value = Mock()

        # Mock the logical operations
        mock_mask.__and__ = Mock(return_value=mock_mask)
        mock_beamstop_mask.__invert__ = Mock(return_value=mock_beamstop_mask)

        beamstop = Circle(center_x=50, center_y=25, radius=10)
        static_params = StaticMaskParams(beamstop=beamstop)

        # Act
        result = self.generator.generate_static_mask(mock_detector, static_params)

        # Assert
        assert result is not None
        assert len(result) == 1
        # Verify beamstop mask was applied
        mock_mask.__and__.assert_called()

    @patch("diffusepipe.masking.pixel_mask_generator.flex")
    def test_generate_static_mask_with_untrusted_panels(self, mock_flex):
        """Test static mask generation with untrusted panels."""
        # Arrange
        mock_detector = Mock()
        mock_panel1 = Mock()
        mock_panel1.get_image_size.return_value = (100, 50)
        mock_panel2 = Mock()
        mock_panel2.get_image_size.return_value = (100, 50)
        mock_detector.__iter__ = Mock(return_value=iter([mock_panel1, mock_panel2]))
        mock_detector.__len__ = Mock(return_value=2)

        mock_mask1 = Mock()
        mock_mask2 = Mock()
        mock_false_mask = Mock()
        mock_flex.bool.side_effect = [mock_mask1, mock_mask2, mock_false_mask]
        mock_flex.grid.return_value = Mock()

        static_params = StaticMaskParams(untrusted_panels=[1])  # Panel 1 is untrusted

        # Act
        result = self.generator.generate_static_mask(mock_detector, static_params)

        # Assert
        assert result is not None
        assert len(result) == 2
        assert result[0] is mock_mask1  # Panel 0 uses normal mask
        assert result[1] is mock_false_mask  # Panel 1 gets all-False mask

    def test_generate_dynamic_mask_no_images(self):
        """Test dynamic mask generation with no representative images using real flex arrays."""
        # Arrange
        from dials.array_family import flex
        
        mock_detector = Mock()
        mock_panel = Mock()
        mock_panel.get_image_size.return_value = (100, 50)
        mock_detector.__iter__ = Mock(return_value=iter([mock_panel]))

        representative_images = []
        dynamic_params = DynamicMaskParams()

        # Act
        result = self.generator.generate_dynamic_mask(
            mock_detector, representative_images, dynamic_params
        )

        # Assert
        assert result is not None
        assert len(result) == 1
        # Should return all-True masks when no images provided
        assert isinstance(result[0], flex.bool)
        assert result[0].count(True) == 100 * 50  # All pixels should be True
        assert result[0].count(False) == 0  # No pixels should be False

    @patch("diffusepipe.masking.pixel_mask_generator.np")
    @patch("diffusepipe.masking.pixel_mask_generator.flex")
    def test_generate_dynamic_mask_with_images(self, mock_flex, mock_np):
        """Test dynamic mask generation with representative images."""
        # Arrange
        mock_detector = Mock()
        mock_panel = Mock()
        mock_panel.get_image_size.return_value = (4, 4)  # Small for testing
        mock_detector.__iter__ = Mock(return_value=iter([mock_panel]))

        # Mock image data
        mock_image_set = Mock()
        mock_panel_data = Mock()
        mock_image_set.get_raw_data.return_value = [mock_panel_data]

        # Create test data with some hot and negative pixels
        test_array = np.array(
            [
                [100, 200, 300, 400],
                [150, 1000000, 250, 350],  # Hot pixel at [1,1]
                [175, 225, -50, 375],  # Negative pixel at [2,2]
                [200, 250, 300, 400],
            ],
            dtype=np.float64,
        )

        mock_np.array.return_value = test_array

        # Mock flex operations
        mock_mask = Mock()
        mock_flex.bool.return_value = mock_mask
        mock_flex.grid.return_value = Mock()

        representative_images = [mock_image_set]
        dynamic_params = DynamicMaskParams(
            hot_pixel_thresh=500000, negative_pixel_tolerance=0.0
        )

        # Act
        result = self.generator.generate_dynamic_mask(
            mock_detector, representative_images, dynamic_params
        )

        # Assert
        assert result is not None
        assert len(result) == 1
        mock_image_set.get_raw_data.assert_called_with(0)

    @patch("diffusepipe.masking.pixel_mask_generator.logger")
    def test_combine_masks_success(self, mock_logger):
        """Test successful combination of static and dynamic masks."""
        # Arrange
        mock_static_mask = Mock()
        mock_dynamic_mask = Mock()
        mock_combined_mask = Mock()

        # Mock the logical AND operation
        mock_static_mask.__and__ = Mock(return_value=mock_combined_mask)
        mock_combined_mask.count.return_value = 80  # 80 good pixels out of 100
        mock_static_mask.count.return_value = 90
        mock_dynamic_mask.count.return_value = 85
        mock_combined_mask.__len__ = Mock(return_value=100)

        static_masks = (mock_static_mask,)
        dynamic_masks = (mock_dynamic_mask,)

        # Act
        result = self.generator._combine_masks(static_masks, dynamic_masks)

        # Assert
        assert result is not None
        assert len(result) == 1
        assert result[0] is mock_combined_mask
        mock_static_mask.__and__.assert_called_once_with(mock_dynamic_mask)

    def test_combine_masks_different_panel_counts(self):
        """Test error handling when static and dynamic masks have different panel counts."""
        # Arrange
        static_masks = (Mock(), Mock())  # 2 panels
        dynamic_masks = (Mock(),)  # 1 panel

        # Act & Assert
        with pytest.raises(MaskGenerationError) as exc_info:
            self.generator._combine_masks(static_masks, dynamic_masks)

        assert "different panel counts" in str(exc_info.value)

    def test_apply_beamstop_mask_circular(self):
        """Test application of circular beamstop mask using real flex arrays."""
        # Arrange
        from dials.array_family import flex
        
        mock_panel = Mock()
        mock_panel.get_image_size.return_value = (6, 6)

        # Create real flex arrays
        initial_mask_array = flex.bool(flex.grid(6, 6), True)  # All pixels initially good
        beamstop = Circle(center_x=3, center_y=3, radius=1)

        # Act
        result_mask_array = self.generator._apply_beamstop_mask(
            mock_panel, initial_mask_array, beamstop
        )

        # Assert
        assert isinstance(result_mask_array, flex.bool)
        # Check that some pixels were masked (center should be False)
        assert result_mask_array.count(False) > 0
        # Check that not all pixels were masked
        assert result_mask_array.count(True) > 0
        # Specifically check center pixel is masked
        assert result_mask_array[3, 3] is False

    def test_apply_beamstop_mask_rectangular(self):
        """Test application of rectangular beamstop mask using real flex arrays."""
        # Arrange
        from dials.array_family import flex
        
        mock_panel = Mock()
        mock_panel.get_image_size.return_value = (10, 10)

        # Create real flex arrays
        initial_mask_array = flex.bool(flex.grid(10, 10), True)  # All pixels initially good
        beamstop = Rectangle(min_x=2, max_x=5, min_y=3, max_y=6)

        # Act
        result_mask_array = self.generator._apply_beamstop_mask(
            mock_panel, initial_mask_array, beamstop
        )

        # Assert
        assert isinstance(result_mask_array, flex.bool)
        # Check that some pixels were masked in the rectangular region
        assert result_mask_array.count(False) > 0
        # Check that not all pixels were masked
        assert result_mask_array.count(True) > 0
        # Check specific pixel in rectangle is masked
        assert result_mask_array[4, 3] is False  # Inside rectangle


class TestMaskParameterClasses:
    """Test suite for mask parameter dataclasses and utility functions."""

    def test_circle_creation(self):
        """Test Circle dataclass creation."""
        circle = Circle(center_x=10.5, center_y=20.3, radius=5.0)
        assert circle.center_x == 10.5
        assert circle.center_y == 20.3
        assert circle.radius == 5.0

    def test_rectangle_creation(self):
        """Test Rectangle dataclass creation."""
        rect = Rectangle(min_x=0, max_x=10, min_y=5, max_y=15)
        assert rect.min_x == 0
        assert rect.max_x == 10
        assert rect.min_y == 5
        assert rect.max_y == 15

    def test_static_mask_params_creation(self):
        """Test StaticMaskParams dataclass creation."""
        beamstop = Circle(center_x=50, center_y=60, radius=10)
        rects = [Rectangle(min_x=0, max_x=5, min_y=0, max_y=5)]
        panels = [1, 3]

        params = StaticMaskParams(
            beamstop=beamstop, untrusted_rects=rects, untrusted_panels=panels
        )

        assert params.beamstop is beamstop
        assert params.untrusted_rects == rects
        assert params.untrusted_panels == panels

    def test_dynamic_mask_params_creation(self):
        """Test DynamicMaskParams dataclass creation."""
        params = DynamicMaskParams(
            hot_pixel_thresh=1000000.0,
            negative_pixel_tolerance=5.0,
            max_fraction_bad_pixels=0.2,
        )

        assert params.hot_pixel_thresh == 1000000.0
        assert params.negative_pixel_tolerance == 5.0
        assert params.max_fraction_bad_pixels == 0.2


class TestUtilityFunctions:
    """Test suite for utility functions."""

    def test_create_circular_beamstop(self):
        """Test creation of circular beamstop."""
        beamstop = create_circular_beamstop(center_x=100, center_y=150, radius=25)

        assert isinstance(beamstop, Circle)
        assert beamstop.center_x == 100
        assert beamstop.center_y == 150
        assert beamstop.radius == 25

    def test_create_rectangular_beamstop(self):
        """Test creation of rectangular beamstop."""
        beamstop = create_rectangular_beamstop(min_x=10, max_x=20, min_y=30, max_y=40)

        assert isinstance(beamstop, Rectangle)
        assert beamstop.min_x == 10
        assert beamstop.max_x == 20
        assert beamstop.min_y == 30
        assert beamstop.max_y == 40

    def test_create_default_static_params(self):
        """Test creation of default static mask parameters."""
        params = create_default_static_params()

        assert isinstance(params, StaticMaskParams)
        assert params.beamstop is None
        assert params.untrusted_rects is None
        assert params.untrusted_panels is None

    def test_create_default_dynamic_params(self):
        """Test creation of default dynamic mask parameters."""
        params = create_default_dynamic_params()

        assert isinstance(params, DynamicMaskParams)
        assert params.hot_pixel_thresh == 1e6
        assert params.negative_pixel_tolerance == 0.0
        assert params.max_fraction_bad_pixels == 0.1


class TestIntegrationScenarios:
    """Integration-style tests for complete masking workflows."""

    def setup_method(self):
        """Set up test fixtures."""
        self.generator = PixelMaskGenerator()

    @patch("diffusepipe.masking.pixel_mask_generator.flex")
    @patch("diffusepipe.masking.pixel_mask_generator.np")
    def test_generate_combined_pixel_mask_success(self, mock_np, mock_flex):
        """Test successful generation of combined pixel mask."""
        # Arrange
        mock_detector = Mock()
        mock_panel = Mock()
        mock_panel.get_image_size.return_value = (4, 4)
        mock_detector.__iter__ = Mock(return_value=iter([mock_panel]))
        mock_detector.__len__ = Mock(return_value=1)

        # Mock static mask generation
        mock_static_mask = Mock()
        mock_static_mask.count.return_value = 15
        mock_static_mask.__len__ = Mock(return_value=16)

        # Mock dynamic mask generation
        mock_dynamic_mask = Mock()
        mock_dynamic_mask.count.return_value = 14

        # Mock combined result
        mock_combined_mask = Mock()
        mock_combined_mask.count.return_value = 13
        mock_static_mask.__and__ = Mock(return_value=mock_combined_mask)

        mock_flex.bool.return_value = mock_static_mask
        mock_flex.grid.return_value = Mock()

        static_params = StaticMaskParams()
        representative_images = []  # Empty for simplicity
        dynamic_params = DynamicMaskParams()

        # Act
        result = self.generator.generate_combined_pixel_mask(
            mock_detector, static_params, representative_images, dynamic_params
        )

        # Assert
        assert result is not None
        assert len(result) == 1
        assert result[0] is mock_combined_mask

    def test_generate_combined_pixel_mask_error_handling(self):
        """Test error handling in combined mask generation."""
        # Arrange
        mock_detector = Mock()
        mock_detector.__iter__ = Mock(side_effect=Exception("Detector error"))

        static_params = StaticMaskParams()
        representative_images = []
        dynamic_params = DynamicMaskParams()

        # Act & Assert
        with pytest.raises(MaskGenerationError) as exc_info:
            self.generator.generate_combined_pixel_mask(
                mock_detector, static_params, representative_images, dynamic_params
            )

        assert "Failed to generate combined pixel mask" in str(exc_info.value)
