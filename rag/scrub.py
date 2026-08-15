"""Strip credentials out of text before it can reach a user.

Provider SDKs put the failing request's headers into the exception message, so
an unhandled 401 will happily print the key that failed. Everything that ends up
in the UI goes through scrub() first.
"""

from __future__ import annotations

import re

# Shapes, not values. Ordered longest-prefix first so sk-ant- is not eaten by sk-.
_KEY_PATTERNS = [
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"sk-proj-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"sk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"gsk_[A-Za-z0-9]{16,}"),
    re.compile(r"AIza[0-9A-Za-z_\-]{20,}"),
    re.compile(r"xai-[A-Za-z0-9]{16,}"),
    re.compile(r"hf_[A-Za-z0-9]{16,}"),
    re.compile(r"r8_[A-Za-z0-9]{16,}"),
    # Authorization: Bearer <anything>
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9_\-\.]{12,}"),
    # api_key='...' / "api-key": "..." in repr'd kwargs
    re.compile(r"(?i)(api[_\-]?key['\"]?\s*[:=]\s*['\"]?)[A-Za-z0-9_\-\.]{12,}"),
]

REDACTED = "<redacted>"


def scrub(text: object) -> str:
    """Return text with anything key-shaped replaced by <redacted>."""
    out = str(text)
    for pattern in _KEY_PATTERNS:
        if pattern.groups:
            out = pattern.sub(lambda m: m.group(1) + REDACTED, out)
        else:
            out = pattern.sub(REDACTED, out)
    return out


def scrub_exception(exc: BaseException) -> str:
    """A one-line, credential-free description of an exception."""
    return f"{type(exc).__name__}: {scrub(exc)}"
