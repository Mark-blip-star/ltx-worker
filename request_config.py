"""Pure request-validation helpers for the serverless handler.

This module intentionally has no torch/RunPod/LTX imports so its behavior can
be covered by small CPU-only tests.
"""

from __future__ import annotations

import hashlib
import math
from numbers import Real
from typing import Any, Mapping


MAX_NEGATIVE_PROMPT_CHARS = 4096
DEFAULT_DECODE_NOISE_SCALE = 0.025
MAX_DECODE_NOISE_SCALE = 0.25
MIN_STAGE1_STEPS = 1
MAX_STAGE1_STEPS = 64
MIN_STAGE2_SIGMA_POINTS = 2
MAX_STAGE2_SIGMA_POINTS = 16


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
    try:
        resolved = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{key} must be a finite number") from exc
    if not math.isfinite(resolved):
        raise ValueError(f"{key} must be a finite number")
    if not minimum <= resolved <= maximum:
        raise ValueError(f"{key} must be between {minimum} and {maximum}")
    return resolved


def resolve_optional_step_count(
    payload: Mapping[str, Any],
    key: str = "steps",
) -> int | None:
    """Validate an optional integer stage-1 transition count.

    ``None`` means the request omitted the lever, so the tier's existing
    default remains authoritative. JSON booleans are rejected even though
    ``bool`` subclasses ``int`` in Python.
    """
    if key not in payload:
        return None
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    if not MIN_STAGE1_STEPS <= value <= MAX_STAGE1_STEPS:
        raise ValueError(
            f"{key} must be between {MIN_STAGE1_STEPS} and {MAX_STAGE1_STEPS}"
        )
    return value


def resolve_stage2_sigmas(
    payload: Mapping[str, Any],
    key: str = "stage2_sigmas",
) -> tuple[float, ...] | None:
    """Return a bounded, finite, strictly descending stage-2 sigma grid.

    The sequence contains scheduler *points*, so its transition count is
    ``len(result) - 1``. Every point must be in ``[0, 1]``; the last point is
    exactly zero. The same validation is intentionally shared by fast and
    quality tiers.
    """
    if key not in payload:
        return None
    value = payload[key]
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list of finite numbers")
    if not MIN_STAGE2_SIGMA_POINTS <= len(value) <= MAX_STAGE2_SIGMA_POINTS:
        raise ValueError(
            f"{key} must contain between {MIN_STAGE2_SIGMA_POINTS} and "
            f"{MAX_STAGE2_SIGMA_POINTS} points"
        )

    resolved: list[float] = []
    for point in value:
        if isinstance(point, bool) or not isinstance(point, Real):
            raise ValueError(f"{key} must be a list of finite numbers")
        point_float = float(point)
        if not math.isfinite(point_float):
            raise ValueError(f"{key} must contain only finite numbers")
        if not 0.0 <= point_float <= 1.0:
            raise ValueError(f"{key} points must be between 0.0 and 1.0")
        resolved.append(point_float)

    if resolved[-1] != 0.0:
        raise ValueError(f"{key} must end at exactly 0.0")
    if not all(
        resolved[index] > resolved[index + 1]
        for index in range(len(resolved) - 1)
    ):
        raise ValueError(f"{key} must be strictly descending")
    return tuple(resolved)


def sampling_schedule_tag_suffix(
    stage1_steps: int,
    stage2_sigmas: tuple[float, ...],
) -> str:
    """Build a compact, deterministic tag suffix for an explicit schedule.

    Counts make the allocation human-readable. The digest covers every sigma
    value at a stable 17-significant-digit representation, preventing two
    different grids with the same length from sharing a runtime tag.
    """
    canonical_sigmas = ",".join(format(value, ".17g") for value in stage2_sigmas)
    digest = hashlib.sha256(canonical_sigmas.encode("ascii")).hexdigest()[:12]
    return (
        f"-s1st{stage1_steps}"
        f"-s2st{len(stage2_sigmas) - 1}"
        f"-s2h{digest}"
    )


def conditioning_strength_tag_suffix(value: float | None) -> str:
    """Return a stable runtime-tag suffix for an explicit conditioning value.

    ``repr(float)`` is Python's shortest round-trippable representation, so
    values that differ at runtime cannot silently share a friendly rounded
    tag. The result contains only tag-safe alphanumerics and underscores.
    """
    if value is None:
        return ""
    canonical = repr(float(value))
    safe = (
        canonical.replace(".", "_")
        .replace("+", "p")
        .replace("-", "m")
    )
    return f"-ic{safe}"


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
