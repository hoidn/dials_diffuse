"""Logging configuration for the diffusepipe package."""

import logging
import sys
from typing import Optional


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> None:
    """
    Setup basic logging configuration for the diffusepipe package.
    
    Args:
        level: Logging level (e.g., logging.DEBUG, logging.INFO)
        log_file: Optional path to log file. If None, logs to console only.
        format_string: Optional custom format string for log messages.
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=handlers,
        force=True  # Override any existing configuration
    )
    
    # Set specific loggers to appropriate levels
    logging.getLogger("dials").setLevel(logging.WARNING)
    logging.getLogger("dxtbx").setLevel(logging.WARNING)
    logging.getLogger("cctbx").setLevel(logging.WARNING)