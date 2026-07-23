from dataclasses import dataclass
from enum import IntEnum

import torch
from torch._prims_common import DeviceLikeType


class PerturbationType(IntEnum):
    """Types of attention perturbations for STG (Spatio-Temporal Guidance).

    The integer value is the row index into
    :attr:`BatchedPerturbationConfig._block_masks`.
    """

    SKIP_VIDEO_SELF_ATTN = 0
    SKIP_AUDIO_SELF_ATTN = 1
    SKIP_A2V_CROSS_ATTN = 2
    SKIP_V2A_CROSS_ATTN = 3


@dataclass(frozen=True)
class Perturbation:
    """A single perturbation specifying which attention type to skip and in which blocks."""

    type: PerturbationType
    blocks: list[int] | None  # None means all blocks

    def is_perturbed(self, perturbation_type: PerturbationType, block: int) -> bool:
        if self.type != perturbation_type:
            return False

        if self.blocks is None:
            return True

        return block in self.blocks


@dataclass(frozen=True)
class PerturbationConfig:
    """Configuration holding a list of perturbations for a single sample."""

    perturbations: list[Perturbation] | None

    def is_perturbed(self, perturbation_type: PerturbationType, block: int) -> bool:
        if self.perturbations is None:
            return False

        return any(perturbation.is_perturbed(perturbation_type, block) for perturbation in self.perturbations)

    @staticmethod
    def empty() -> "PerturbationConfig":
        return PerturbationConfig([])


class BatchedPerturbationConfig:
    """Tensorized per-block attention keep masks for a batch.

    The Python perturbation structure is materialized once, eagerly, into a
    ``(len(PerturbationType), num_blocks, batch)`` tensor. Compiled transformer
    blocks therefore receive runtime tensors instead of inspecting Python
    lists or branching on perturbation configuration.
    """

    _block_masks: torch.Tensor
    _block_masks_cpu: torch.Tensor | None

    def __init__(
        self,
        perturbations: list[PerturbationConfig],
        num_blocks: int,
        device: DeviceLikeType | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        keep = [
            [
                [not config.is_perturbed(PerturbationType(direction), block) for config in perturbations]
                for block in range(num_blocks)
            ]
            for direction in range(len(PerturbationType))
        ]
        self._block_masks_cpu = torch.tensor(keep, dtype=dtype, device="cpu")
        self._block_masks = self._block_masks_cpu if device is None else self._block_masks_cpu.to(device)

    @classmethod
    def from_masks(
        cls,
        block_masks: torch.Tensor,
        block_masks_cpu: torch.Tensor | None = None,
    ) -> "BatchedPerturbationConfig":
        """Construct from existing mask tensors without rebuilding Python state."""
        obj = cls.__new__(cls)
        obj._block_masks = block_masks
        obj._block_masks_cpu = block_masks_cpu
        return obj

    def batch_slice(self, start: int, end: int) -> "BatchedPerturbationConfig":
        """Return a batch-dimension view while retaining the eager host mirror."""
        cpu_mask = self._block_masks_cpu[:, :, start:end] if self._block_masks_cpu is not None else None
        return BatchedPerturbationConfig.from_masks(self._block_masks[:, :, start:end], cpu_mask)

    def mask(self, perturbation_type: PerturbationType, block: int) -> torch.Tensor:
        """Return this block's owned ``(batch, 1, 1)`` keep-mask."""
        return self._block_masks[perturbation_type, block].reshape(-1, 1, 1).clone()

    def any_in_batch(self, perturbation_type: PerturbationType, block: int) -> bool:
        assert self._block_masks_cpu is not None, "host mirror required by eager perturbation processing"
        return bool((self._block_masks_cpu[perturbation_type, block] == 0).any())

    def all_in_batch(self, perturbation_type: PerturbationType, block: int) -> bool:
        assert self._block_masks_cpu is not None, "host mirror required by eager perturbation processing"
        return bool((self._block_masks_cpu[perturbation_type, block] == 0).all())

    @staticmethod
    def empty(
        batch_size: int,
        num_blocks: int,
        device: DeviceLikeType | None = None,
        dtype: torch.dtype | None = None,
    ) -> "BatchedPerturbationConfig":
        return BatchedPerturbationConfig(
            [PerturbationConfig.empty() for _ in range(batch_size)],
            num_blocks,
            device,
            dtype,
        )
