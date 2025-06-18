"""
CBF header parsing utilities for determining data type (stills vs sequence).
"""

import logging
from typing import Optional

# Import needed for patching in tests
try:
    import dxtbx
except ImportError:
    # This import might fail in testing environments without DIALS
    dxtbx = None

logger = logging.getLogger(__name__)


class CBFUtils:
    """Utilities for working with CBF files."""

    def get_angle_increment(self, image_path: str) -> Optional[float]:
        """
        Extract the Angle_increment value from a CBF file header.

        Args:
            image_path: Path to the CBF file

        Returns:
            Angle increment in degrees, or None if not found
        """
        return get_angle_increment_from_cbf(image_path)


def get_angle_increment_from_cbf(image_path: str) -> Optional[float]:
    """
    Extract the Angle_increment value from a CBF file header to determine
    if the data is true stills (0.0°) or sequence data (> 0.0°).

    Uses a two-phase approach:
    1. Primary: dxtbx.load() to get scan information (robust for most files)
    2. Fallback: Direct header text parsing with regex (handles edge cases)

    Args:
        image_path: Path to the CBF file

    Returns:
        Angle increment in degrees, or None if not found/determinable
        - 0.0: True stills data (no oscillation)
        - > 0.0: Sequence data (oscillation per frame)
        - None: Could not determine (caller should default to sequence processing)

    Raises:
        Exception: If file cannot be read or both parsing methods fail critically
    """
    try:
        # First attempt: Use dxtbx to get scan information
        import dxtbx

        logger.debug(f"Attempting to parse CBF header for: {image_path}")

        # Load the image using dxtbx
        image = dxtbx.load(image_path)

        # Try to get scan information with enhanced error handling
        try:
            scan = image.get_scan()
        except AttributeError:
            logger.debug("Image object has no get_scan() method, treating as still")
            return 0.0

        if scan is not None:
            try:
                # Get oscillation information: (start_angle, oscillation_width)
                oscillation = scan.get_oscillation()
                if oscillation is None or len(oscillation) < 2:
                    logger.debug(
                        "Scan object returned None or incomplete oscillation data"
                    )
                    return 0.0
                angle_increment = oscillation[1]  # oscillation width per frame
                logger.info(
                    f"Angle increment from dxtbx scan object: {angle_increment}°"
                )
                return angle_increment
            except (AttributeError, IndexError, TypeError) as e:
                logger.debug(f"Failed to get oscillation from scan object: {e}")
                return 0.0
        else:
            logger.debug("No scan object found, likely a still image")
            return 0.0

    except ImportError:
        logger.warning(
            f"dxtbx not available, falling back to text parsing for {image_path}"
        )
    except Exception as e:
        logger.warning(f"dxtbx method failed for {image_path}: {e}")

        # Fallback: Parse CBF header text directly
        try:
            fallback_result = _parse_cbf_header_text(image_path)
            if fallback_result is not None:
                logger.info(
                    f"Angle increment from header text parsing: {fallback_result}°"
                )
                return fallback_result
            else:
                logger.warning(
                    f"Could not determine Angle_increment for {image_path}. Defaulting to sequence processing. This may lead to incorrect results. Consider using 'force_processing_mode'."
                )
                return None
        except Exception as fallback_error:
            logger.error(
                f"All parsing methods failed for {image_path}: {fallback_error}"
            )
            raise


def _parse_cbf_header_text(image_path: str) -> Optional[float]:
    """
    Fallback method to parse CBF header text directly for Angle_increment.

    Uses regex for more flexible parsing of the Angle_increment line with
    case-insensitive matching and variable spacing.

    Args:
        image_path: Path to the CBF file

    Returns:
        Angle increment in degrees, or None if not found
    """
    import re

    logger.debug(f"Attempting direct header parsing for: {image_path}")

    # Regex pattern for flexible Angle_increment parsing
    # Matches: # (optional spaces) Angle_increment (spaces) number (optional spaces) (optional deg.)
    angle_pattern = re.compile(
        r"^\s*#\s*Angle_increment\s+([+-]?\d*\.?\d+)\s*(?:deg\.?)?", re.IGNORECASE
    )

    try:
        with open(image_path, "r", encoding="utf-8", errors="ignore") as f:
            # Read in chunks to handle large files efficiently
            # Headers are typically in the first 16KB
            chunk_size = 16384
            content = f.read(chunk_size)

            # Split into lines for processing
            for line in content.split("\n"):
                # Stop when we hit the binary data section
                if (
                    "_array_data.data" in line
                    or "Content-Type: application/octet-stream" in line
                ):
                    break

                # Try to match the Angle_increment pattern
                match = angle_pattern.match(line.strip())
                if match:
                    try:
                        angle_increment = float(match.group(1))
                        logger.debug(
                            f"Found angle increment from header text: {angle_increment}°"
                        )
                        return angle_increment
                    except ValueError as e:
                        logger.warning(
                            f"Could not convert angle increment '{match.group(1)}' to float: {e}"
                        )
                        continue

            # If we haven't found it in the first chunk and there's more to read, read another chunk
            if len(content) == chunk_size:
                additional_content = f.read(chunk_size)
                for line in additional_content.split("\n"):
                    if (
                        "_array_data.data" in line
                        or "Content-Type: application/octet-stream" in line
                    ):
                        break
                    match = angle_pattern.match(line.strip())
                    if match:
                        try:
                            angle_increment = float(match.group(1))
                            logger.debug(
                                f"Found angle increment from header text: {angle_increment}°"
                            )
                            return angle_increment
                        except ValueError as e:
                            logger.warning(
                                f"Could not convert angle increment '{match.group(1)}' to float: {e}"
                            )
                            continue

    except IOError as e:
        logger.error(f"Failed to read CBF file {image_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error parsing CBF header for {image_path}: {e}")
        raise

    logger.debug("Angle_increment not found in header text")
    return None
