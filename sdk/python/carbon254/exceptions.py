"""
Exception classes for 254Carbon SDK.
"""


class CarbonAPIError(Exception):
    """Base exception for API errors."""
    pass


class AuthenticationError(CarbonAPIError):
    """Authentication failed."""
    pass


class RateLimitError(CarbonAPIError):
    """Rate limit exceeded."""
    pass


class NotFoundError(CarbonAPIError):
    """Resource not found."""
    pass

