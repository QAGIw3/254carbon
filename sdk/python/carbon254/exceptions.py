"""
Exception classes for 254Carbon SDK.

The SDK raises a small hierarchy of exceptions to make error handling
predictable and structured. Catch ``CarbonAPIError`` to handle all API
failures in a single place, or catch specific subclasses for finer control.

Typical usage:
    >>> from carbon254 import CarbonClient, AuthenticationError, RateLimitError
    >>> client = CarbonClient(api_key="...")
    >>> try:
    ...     instruments = client.get_instruments(market="power")
    ... except AuthenticationError:
    ...     print("Invalid API key")
    ... except RateLimitError:
    ...     print("Back off and retry later")
"""


class CarbonAPIError(Exception):
    """Base exception for API errors.

    Raised when the server returns a non‑success HTTP status code or an error
    occurs while processing a response. All SDK exceptions inherit from this
    class so callers can handle failures generically.
    """
    pass


class AuthenticationError(CarbonAPIError):
    """Authentication failed or access is forbidden."""
    pass


class RateLimitError(CarbonAPIError):
    """The client exceeded the allowed request rate (HTTP 429)."""
    pass


class NotFoundError(CarbonAPIError):
    """Requested resource was not found (HTTP 404)."""
    pass
