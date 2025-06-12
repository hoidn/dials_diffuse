"""
Test suite for CBF utilities, specifically CBF data type detection.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from diffusepipe.utils.cbf_utils import (
    get_angle_increment_from_cbf,
    _parse_cbf_header_text,
)


class TestGetAngleIncrementFromCBF:
    """Test CBF data type detection for various file types and edge cases."""

    def test_file_not_found(self):
        """Test handling of non-existent files."""
        with pytest.raises(Exception):  # Should raise an exception
            get_angle_increment_from_cbf("/nonexistent/file.cbf")

    def test_empty_file(self):
        """Test handling of empty CBF files."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".cbf", delete=False) as f:
            temp_path = f.name

        try:
            # Should handle gracefully and return None
            result = get_angle_increment_from_cbf(temp_path)
            assert result is None
        finally:
            os.unlink(temp_path)

    @patch("dxtbx.load")
    def test_dxtbx_stills_data_success(self, mock_dxtbx_load):
        """Test successful dxtbx parsing for stills data (Angle_increment = 0.0)."""
        # Mock successful dxtbx loading for stills
        mock_image = MagicMock()
        mock_scan = MagicMock()
        mock_scan.get_oscillation.return_value = (
            0.0,
            0.0,
        )  # (start_angle, oscillation_width)
        mock_image.get_scan.return_value = mock_scan
        mock_dxtbx_load.return_value = mock_image

        result = get_angle_increment_from_cbf("test_stills.cbf")
        assert result == 0.0
        mock_dxtbx_load.assert_called_once_with("test_stills.cbf")

    @patch("dxtbx.load")
    def test_dxtbx_sequence_data_success(self, mock_dxtbx_load):
        """Test successful dxtbx parsing for sequence data (Angle_increment > 0.0)."""
        # Mock successful dxtbx loading for sequence data
        mock_image = MagicMock()
        mock_scan = MagicMock()
        mock_scan.get_oscillation.return_value = (0.0, 0.1)  # 0.1° oscillation
        mock_image.get_scan.return_value = mock_scan
        mock_dxtbx_load.return_value = mock_image

        result = get_angle_increment_from_cbf("test_sequence.cbf")
        assert result == 0.1
        mock_dxtbx_load.assert_called_once_with("test_sequence.cbf")

    @patch("dxtbx.load")
    def test_dxtbx_no_scan_object(self, mock_dxtbx_load):
        """Test dxtbx parsing when no scan object is available (stills case)."""
        # Mock image with no scan object
        mock_image = MagicMock()
        mock_image.get_scan.return_value = None
        mock_dxtbx_load.return_value = mock_image

        result = get_angle_increment_from_cbf("test_no_scan.cbf")
        assert result == 0.0  # Should default to stills

    @patch("dxtbx.load")
    def test_dxtbx_failure_fallback_to_text_parsing(self, mock_dxtbx_load):
        """Test fallback to text parsing when dxtbx fails."""
        # Mock dxtbx failure
        mock_dxtbx_load.side_effect = Exception("dxtbx failed")

        # Create a temporary CBF file with header
        cbf_content = """###CBF: VERSION 1.5
# Detector: PILATUS3 6M
# Angle_increment 0.1000 deg.
# Exposure_time 0.0990000 s
_array_data.data
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".cbf", delete=False) as f:
            f.write(cbf_content)
            temp_path = f.name

        try:
            result = get_angle_increment_from_cbf(temp_path)
            assert result == 0.1  # Should parse from header text
        finally:
            os.unlink(temp_path)

    @patch("dxtbx.load")
    def test_dxtbx_scan_get_oscillation_fails(self, mock_dxtbx_load):
        """Test handling when scan.get_oscillation() raises an exception."""
        # Mock scan object that fails on get_oscillation
        mock_image = MagicMock()
        mock_scan = MagicMock()
        mock_scan.get_oscillation.side_effect = RuntimeError(
            "Oscillation data unavailable"
        )
        mock_image.get_scan.return_value = mock_scan
        mock_dxtbx_load.return_value = mock_image

        # Should fallback to text parsing
        cbf_content = """###CBF: VERSION 1.5
# Angle_increment 0.2000 deg.
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".cbf", delete=False) as f:
            f.write(cbf_content)
            temp_path = f.name

        try:
            result = get_angle_increment_from_cbf(temp_path)
            assert result == 0.2  # Should parse from header text
        finally:
            os.unlink(temp_path)


class TestParseCBFHeaderText:
    """Test direct CBF header text parsing functionality."""

    def test_parse_stills_header(self):
        """Test parsing stills data from header text."""
        cbf_content = """###CBF: VERSION 1.5
# Detector: PILATUS3 6M, S/N 60-0127
# Angle_increment 0.0000 deg.
# Exposure_time 0.0990000 s
_array_data.data
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".cbf", delete=False) as f:
            f.write(cbf_content)
            temp_path = f.name

        try:
            result = _parse_cbf_header_text(temp_path)
            assert result == 0.0
        finally:
            os.unlink(temp_path)

    def test_parse_sequence_header(self):
        """Test parsing sequence data from header text."""
        cbf_content = """###CBF: VERSION 1.5
# Detector: PILATUS3 6M, S/N 60-0127
# Angle_increment 0.1000 deg.
# Exposure_time 0.0990000 s
_array_data.data
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".cbf", delete=False) as f:
            f.write(cbf_content)
            temp_path = f.name

        try:
            result = _parse_cbf_header_text(temp_path)
            assert result == 0.1
        finally:
            os.unlink(temp_path)

    def test_parse_larger_angle_increment(self):
        """Test parsing larger angle increments."""
        cbf_content = """###CBF: VERSION 1.5
# Angle_increment 0.5000 deg.
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".cbf", delete=False) as f:
            f.write(cbf_content)
            temp_path = f.name

        try:
            result = _parse_cbf_header_text(temp_path)
            assert result == 0.5
        finally:
            os.unlink(temp_path)

    def test_parse_no_angle_increment(self):
        """Test parsing when Angle_increment is not found."""
        cbf_content = """###CBF: VERSION 1.5
# Detector: PILATUS3 6M, S/N 60-0127
# Exposure_time 0.0990000 s
_array_data.data
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".cbf", delete=False) as f:
            f.write(cbf_content)
            temp_path = f.name

        try:
            result = _parse_cbf_header_text(temp_path)
            assert result is None
        finally:
            os.unlink(temp_path)

    def test_parse_malformed_angle_increment_line(self):
        """Test parsing when Angle_increment line is malformed."""
        cbf_content = """###CBF: VERSION 1.5
# Angle_increment invalid_value deg.
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".cbf", delete=False) as f:
            f.write(cbf_content)
            temp_path = f.name

        try:
            result = _parse_cbf_header_text(temp_path)
            assert result is None  # Should handle parsing error gracefully
        finally:
            os.unlink(temp_path)

    def test_parse_case_insensitive_angle_increment(self):
        """Test parsing when case varies in the header."""
        cbf_content = """###CBF: VERSION 1.5
# angle_increment 0.1000 deg.
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".cbf", delete=False) as f:
            f.write(cbf_content)
            temp_path = f.name

        try:
            # Implementation uses re.IGNORECASE, so lowercase should work
            result = _parse_cbf_header_text(temp_path)
            assert result == 0.1
        finally:
            os.unlink(temp_path)

    def test_parse_different_spacing(self):
        """Test parsing with different spacing in header line."""
        cbf_content = """###CBF: VERSION 1.5
#   Angle_increment    0.2500   deg.
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".cbf", delete=False) as f:
            f.write(cbf_content)
            temp_path = f.name

        try:
            result = _parse_cbf_header_text(temp_path)
            assert result == 0.25  # Should handle extra spacing
        finally:
            os.unlink(temp_path)

    def test_parse_negative_angle_increment(self):
        """Test parsing negative angle increment (edge case)."""
        cbf_content = """###CBF: VERSION 1.5
# Angle_increment -0.1000 deg.
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".cbf", delete=False) as f:
            f.write(cbf_content)
            temp_path = f.name

        try:
            result = _parse_cbf_header_text(temp_path)
            assert result == -0.1  # Should parse negative values
        finally:
            os.unlink(temp_path)

    def test_parse_file_read_error(self):
        """Test handling of file read errors."""
        # Test with a directory instead of a file
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(Exception):  # Should raise an exception
                _parse_cbf_header_text(temp_dir)


# Integration tests with both methods
class TestIntegrationCBFDetection:
    """Integration tests for the complete CBF detection workflow."""

    @patch("dxtbx.load")
    def test_full_workflow_dxtbx_success(self, mock_dxtbx_load):
        """Test full workflow when dxtbx succeeds."""
        mock_image = MagicMock()
        mock_scan = MagicMock()
        mock_scan.get_oscillation.return_value = (0.0, 0.1)
        mock_image.get_scan.return_value = mock_scan
        mock_dxtbx_load.return_value = mock_image

        result = get_angle_increment_from_cbf("test.cbf")
        assert result == 0.1

    @patch("dxtbx.load")
    def test_full_workflow_fallback_success(self, mock_dxtbx_load):
        """Test full workflow when dxtbx fails but text parsing succeeds."""
        mock_dxtbx_load.side_effect = Exception("Import error")

        cbf_content = """###CBF: VERSION 1.5
# Angle_increment 0.3000 deg.
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".cbf", delete=False) as f:
            f.write(cbf_content)
            temp_path = f.name

        try:
            result = get_angle_increment_from_cbf(temp_path)
            assert result == 0.3
        finally:
            os.unlink(temp_path)

    @patch("dxtbx.load")
    def test_full_workflow_all_methods_fail(self, mock_dxtbx_load):
        """Test full workflow when both dxtbx and text parsing fail."""
        mock_dxtbx_load.side_effect = Exception("Import error")

        # Create file with no Angle_increment
        cbf_content = """###CBF: VERSION 1.5
# Some other header info
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".cbf", delete=False) as f:
            f.write(cbf_content)
            temp_path = f.name

        try:
            # When all methods fail gracefully, function returns None
            result = get_angle_increment_from_cbf(temp_path)
            assert result is None
        finally:
            os.unlink(temp_path)
