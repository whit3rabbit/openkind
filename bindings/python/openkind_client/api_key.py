"""API keys for explicitly configured servers and clients."""

import secrets


def generate_api_key() -> str:
    """Return a fresh API key without storing it or enabling authentication."""
    return "ok_" + secrets.token_hex(32)


def _validate_api_key(api_key: str | None) -> None:
    if api_key is not None and (
        not isinstance(api_key, str)
        or not api_key
        or any(not 0x21 <= ord(char) <= 0x7E for char in api_key)
    ):
        raise ValueError("api_key must be nonempty visible ASCII without whitespace")
