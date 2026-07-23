"""Strict, dependency-free configuration for the regional compile canary.

This module is imported before ``torch`` so ``TORCHINDUCTOR_CACHE_DIR`` can be
set before the compiler stack initializes.  The first rollout is intentionally
limited to stage 1 with Inductor's default mode (no CUDA Graphs).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import MutableMapping


DEFAULT_CACHE_DIR = "/tmp/ltx-torchinductor-cache"


def _flag(environ: MutableMapping[str, str], name: str, default: bool) -> bool:
    raw = environ.get(name, "1" if default else "0")
    if raw not in {"0", "1"}:
        raise ValueError(f"{name} must be 0 or 1, got {raw!r}")
    return raw == "1"


@dataclass(frozen=True)
class RegionalCompileSettings:
    """Validated settings for the disabled-by-default compile canary."""

    enabled: bool
    stage1: bool
    stage2: bool
    mode: str
    backend: str = "inductor"
    fullgraph: bool = False
    dynamic: None = None
    guards_enabled: bool = True
    cache_dir: str | None = None

    @classmethod
    def from_environ(cls, environ: MutableMapping[str, str]) -> "RegionalCompileSettings":
        enabled = _flag(environ, "LTX_REGIONAL_COMPILE", False)
        stage1 = enabled and _flag(environ, "LTX_REGIONAL_COMPILE_STAGE1", True)
        stage2 = enabled and _flag(environ, "LTX_REGIONAL_COMPILE_STAGE2", False)
        mode = environ.get("LTX_REGIONAL_COMPILE_MODE", "default").strip().lower()

        if enabled and not stage1 and not stage2:
            raise ValueError("LTX_REGIONAL_COMPILE=1 requires at least one enabled stage")
        if stage2:
            raise ValueError("stage-2 regional compile is not available in the first lossless canary")
        if enabled and mode != "default":
            raise ValueError(
                "the first lossless canary only supports LTX_REGIONAL_COMPILE_MODE=default "
                "(torch.compile mode=None; CUDA Graph modes are intentionally blocked)"
            )

        cache_dir = environ.get("TORCHINDUCTOR_CACHE_DIR")
        if enabled:
            cache_dir = cache_dir or DEFAULT_CACHE_DIR

        return cls(
            enabled=enabled,
            stage1=stage1,
            stage2=stage2,
            mode=mode,
            cache_dir=cache_dir,
        )

    def prepare_environment(self, environ: MutableMapping[str, str]) -> None:
        """Apply compiler environment that must exist before importing torch."""
        if self.enabled and self.cache_dir is not None:
            environ.setdefault("TORCHINDUCTOR_CACHE_DIR", self.cache_dir)

    def telemetry(self) -> dict[str, object]:
        return asdict(self)
