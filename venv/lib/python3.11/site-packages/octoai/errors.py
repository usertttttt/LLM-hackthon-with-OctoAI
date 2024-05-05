"""OctoAI errors."""

from typing import Optional


class OctoAIError(Exception):
    """Base class for all OctoAI errors.

    :param message: error message
    :type message: str
    :param caused_by: Exception source
    :type caused_by: :class:Exception, optional
    """

    def __init__(
        self,
        message: str,
        caused_by: Optional[Exception] = None,
    ):
        """Declare OctoAIError class."""
        self.message = message
        self.caused_by = caused_by

    def __repr__(self) -> str:
        """Format string representation to include wrapped exception message."""
        msg = f"{self.__class__.__name__}({self.message})"
        if self.caused_by is not None:
            msg += f" caused by {self.caused_by.__class__.__name__}({self.caused_by})"
        return msg


class OctoAIClientError(OctoAIError):
    """Raise when the client returns a 4xx error."""


class OctoAIServerError(OctoAIError):
    """Raise when the server returns a 5xx error."""


class OctoAIValidationError(OctoAIError):
    """Raise when inputs are unable to be validated."""


class OctoAIAssetReadyTimeoutError(OctoAIError):
    """Raise when asset was not ready to use by the timeout deadline."""
