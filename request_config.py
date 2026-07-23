"""Pure request-validation helpers for the serverless handler.

This module intentionally has no torch/RunPod/LTX imports so its behavior can
be covered by small CPU-only tests.
"""

from __future__ import annotations

import math
from numbers import Real
from typing import Any, Mapping


MAX_NEGATIVE_PROMPT_CHARS = 4096
DEFAULT_DECODE_NOISE_SCALE = 0.025
MAX_DECODE_NOISE_SCALE = 0.25


def resolve_boolean(payload: Mapping[str, Any], key: str, default: bool) -> bool:
    """Return an explicit JSON boolean without Python truthiness surprises."""
    if key not in payload:
        return default
    value = payload[key]
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")
    return value


def resolve_optional_number(
    payload: Mapping[str, Any],
    key: str,
    *,
    minimum: float,
    maximum: float,
) -> float | None:
    """Validate an optional finite JSON number in a closed interval."""
    if key not in payload:
        return None
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{key} must be a finite number")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{key} must be a finite number")
    if not minimum <= resolved <= maximum:
        raise ValueError(f"{key} must be between {minimum} and {maximum}")
    return resolved


def resolve_negative_prompt(payload: Mapping[str, Any], default: str) -> tuple[str, str]:
    """Return the effective negative prompt and whether it came from the request."""
    if "negative_prompt" not in payload:
        return default, "default"

    value = payload["negative_prompt"]
    if not isinstance(value, str):
        raise ValueError("negative_prompt must be a string")
    if len(value) > MAX_NEGATIVE_PROMPT_CHARS:
        raise ValueError(
            f"negative_prompt exceeds {MAX_NEGATIVE_PROMPT_CHARS} characters"
        )
    if "\x00" in value:
        raise ValueError("negative_prompt must not contain NUL characters")
    return value, "request"


def resolve_decode_noise(payload: Mapping[str, Any]) -> float | None:
    """Validate an explicit decoder renoise override.

    ``None`` means the request omitted the setting and the decoder's vendored
    0.025 instance default must remain untouched.
    """
    if "decode_noise" not in payload:
        return None

    return resolve_optional_number(
        payload,
        "decode_noise",
        minimum=0.0,
        maximum=MAX_DECODE_NOISE_SCALE,
    )
