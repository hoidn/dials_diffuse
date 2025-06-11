"""Custom exceptions for the diffusepipe package."""


class PipelineError(Exception):
    """Base exception for all pipeline-related errors."""

    pass


class ConfigurationError(PipelineError):
    """Raised when there are configuration-related issues."""

    pass


class DIALSError(PipelineError):
    """Raised when DIALS operations fail."""

    pass


class FileSystemError(PipelineError):
    """Raised when file system operations fail."""

    pass


class DataValidationError(PipelineError):
    """Raised when data validation fails."""

    pass


class NotImplementedYetError(PipelineError):
    """Raised when attempting to use functionality not yet implemented."""

    pass


class MaskGenerationError(PipelineError):
    """Raised when mask generation operations fail."""

    pass


class BraggMaskError(PipelineError):
    """Raised when Bragg mask generation operations fail."""

    pass
