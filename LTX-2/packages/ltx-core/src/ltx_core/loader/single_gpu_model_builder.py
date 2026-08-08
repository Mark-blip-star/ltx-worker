import hashlib
import logging
import os
from dataclasses import dataclass, field, replace
from typing import Generic

import safetensors.torch
import torch
from torch import nn

from ltx_core.loader.fuse_loras import apply_loras
from ltx_core.loader.helpers import create_meta_model, load_state_dict, read_model_config
from ltx_core.loader.module_ops import ModuleOps
from ltx_core.loader.primitives import (
    LoRAAdaptableProtocol,
    LoraPathStrengthAndSDOps,
    LoraStateDictWithStrength,
    ModelBuilderProtocol,
    StateDict,
    StateDictLoader,
)
from ltx_core.loader.registry import DummyRegistry, Registry
from ltx_core.loader.sd_ops import SDOps
from ltx_core.loader.sft_loader import SafetensorsModelStateDictLoader
from ltx_core.model.model_protocol import ModelConfigurator, ModelType

logger: logging.Logger = logging.getLogger(__name__)


def _check_uninitialized(model: nn.Module) -> list[str]:
    """Return names of any parameters/buffers still on meta device."""
    names = []
    for name, param in model.named_parameters():
        if str(param.device) == "meta":
            names.append(name)
    for name, buf in model.named_buffers():
        if str(buf.device) == "meta":
            names.append(name)
    return names


def _load_model_weights(
    meta_model: nn.Module,
    model_path: str | tuple[str, ...],
    loras: tuple[LoraPathStrengthAndSDOps, ...],
    loader: StateDictLoader,
    registry: Registry,
    device: torch.device,
    dtype: torch.dtype | None,
    model_sd_ops: SDOps | None = None,
    lora_load_device: torch.device | None = None,
) -> None:
    """Load base weights and fuse LoRAs into *meta_model* in-place."""
    if lora_load_device is None:
        lora_load_device = device

    lora_strengths = [lora.strength for lora in loras]
    fused_needed = bool(loras) and not (
        lora_strengths and min(lora_strengths) == 0 and max(lora_strengths) == 0
    )
    if fused_needed and os.environ.get("LTX_FUSED_CACHE", "1") == "1":
        # Фʼюжн детермінований по (ckpt, lora@strength, dtype) — готовий SD можна
        # читати напряму, минаючи стрімінг лор з CPU і фʼюжн-математику (25-45с холодного).
        # Джерела: $LTX_FUSED_DIR, потім <тека чекпоінта>/fused (Model Store стейджить її з репо).
        paths = model_path if isinstance(model_path, (tuple, list)) else (model_path,)
        # Ключ по basename'ах: абсолютні шляхи різняться між платформами (Modal /vol/...,
        # RunPod /runpod-volume/...), а кеш має бути спільним артефактом.
        key_src = "|".join(os.path.basename(str(p)) for p in paths) + "||" + \
                  "|".join(f"{os.path.basename(str(lr.path))}@{lr.strength}" for lr in loras) + f"||{dtype}"
        key = hashlib.sha1(key_src.encode()).hexdigest()[:16]
        candidates = []
        if os.environ.get("LTX_FUSED_DIR"):
            candidates.append(os.path.join(os.environ["LTX_FUSED_DIR"], f"{key}.safetensors"))
        candidates.append(os.path.join(os.path.dirname(str(paths[0])), "fused", f"{key}.safetensors"))
        for fpath in candidates:
            if os.path.exists(fpath):
                sd = safetensors.torch.load_file(fpath, device=str(device))
                meta_model.load_state_dict(sd, strict=False, assign=True)
                logger.info(f"fused-cache HIT {key}: {len(sd)} tensors from {fpath}")
                return
        save_dir = os.environ.get("LTX_FUSED_SAVE_DIR")
    else:
        save_dir = None

    model_sd = load_state_dict(model_path, loader, registry, device, model_sd_ops)

    if not lora_strengths or (min(lora_strengths) == 0 and max(lora_strengths) == 0):
        sd = model_sd.sd
        if dtype is not None:
            sd = {
                key: (
                    value
                    if value.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
                    or key.endswith((".weight_scale", ".input_scale"))
                    else value.to(dtype=dtype)
                )
                for key, value in model_sd.sd.items()
            }
        meta_model.load_state_dict(sd, strict=False, assign=True)
        return

    lora_state_dicts = [load_state_dict([lora.path], loader, registry, lora_load_device, lora.sd_ops) for lora in loras]
    lora_sd_and_strengths = [
        LoraStateDictWithStrength(sd, strength) for sd, strength in zip(lora_state_dicts, lora_strengths, strict=True)
    ]
    final_sd = apply_loras(
        model_sd=model_sd,
        lora_sd_and_strengths=lora_sd_and_strengths,
        dtype=dtype,
        destination_sd=model_sd if isinstance(registry, DummyRegistry) else None,
    )
    meta_model.load_state_dict(final_sd.sd, strict=False, assign=True)
    if save_dir:
        try:
            os.makedirs(save_dir, exist_ok=True)
            fpath = os.path.join(save_dir, f"{key}.safetensors")
            sd_cpu = {k: v.detach().to("cpu", copy=True).contiguous()
                      for k, v in meta_model.state_dict().items()}
            safetensors.torch.save_file(sd_cpu, fpath + ".tmp")
            os.replace(fpath + ".tmp", fpath)
            logger.info(f"fused-cache SAVED {key} -> {fpath}")
        except Exception as exc:  # noqa: BLE001 — кеш не критичний, білд уже вдався
            logger.warning(f"fused-cache save skipped: {exc!r}")


@dataclass(frozen=True)
class SingleGPUModelBuilder(Generic[ModelType], ModelBuilderProtocol[ModelType], LoRAAdaptableProtocol):
    """
    Builder for PyTorch models residing on a single GPU.
    Attributes:
        model_class_configurator: Class responsible for constructing the model from a config dict.
        model_path: Path (or tuple of shard paths) to the model's `.safetensors` checkpoint(s).
        model_sd_ops: Optional state-dict operations applied when loading the model weights.
        module_ops: Sequence of module-level mutations applied to the meta model before weight loading.
        loras: Sequence of LoRA adapters (path, strength, optional sd_ops) to fuse into the model.
        model_loader: Strategy for loading state dicts from disk. Defaults to
            :class:`SafetensorsModelStateDictLoader`.
        registry: Cache for already-loaded state dicts. Defaults to :class:`DummyRegistry` (no caching).
        lora_load_device: Device used when loading LoRA weight tensors from disk. Defaults to
            ``torch.device("cpu")``, which keeps LoRA weights in CPU memory and transfers them to
            the target GPU sequentially during fusion, reducing peak GPU memory usage compared to
            loading all LoRA weights directly onto the GPU at once.
    """

    model_class_configurator: type[ModelConfigurator[ModelType]]
    model_path: str | tuple[str, ...]
    model_sd_ops: SDOps | None = None
    module_ops: tuple[ModuleOps, ...] = field(default_factory=tuple)
    loras: tuple[LoraPathStrengthAndSDOps, ...] = field(default_factory=tuple)
    model_loader: StateDictLoader = field(default_factory=SafetensorsModelStateDictLoader)
    registry: Registry = field(default_factory=DummyRegistry)
    lora_load_device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    def lora(self, lora_path: str, strength: float, sd_ops: SDOps) -> "SingleGPUModelBuilder":
        return replace(self, loras=(*self.loras, LoraPathStrengthAndSDOps(lora_path, strength, sd_ops)))

    def with_sd_ops(self, sd_ops: SDOps | None) -> "SingleGPUModelBuilder":
        return replace(self, model_sd_ops=sd_ops)

    def with_module_ops(self, module_ops: tuple[ModuleOps, ...]) -> "SingleGPUModelBuilder":
        return replace(self, module_ops=module_ops)

    def with_loras(self, loras: tuple[LoraPathStrengthAndSDOps, ...]) -> "SingleGPUModelBuilder":
        return replace(self, loras=loras)

    def with_registry(self, registry: Registry) -> "SingleGPUModelBuilder":
        return replace(self, registry=registry)

    def with_lora_load_device(self, device: torch.device) -> "SingleGPUModelBuilder":
        return replace(self, lora_load_device=device)

    def model_config(self) -> dict:
        return read_model_config(self.model_path, self.model_loader)

    def meta_model(self, config: dict, module_ops: tuple[ModuleOps, ...]) -> ModelType:
        return create_meta_model(self.model_class_configurator, config, module_ops)

    def load_sd(
        self, paths: list[str], registry: Registry, device: torch.device | None, sd_ops: SDOps | None = None
    ) -> StateDict:
        return load_state_dict(paths, self.model_loader, registry, device, sd_ops)

    def _return_model(self, meta_model: ModelType, device: torch.device) -> ModelType:
        uninitialized = _check_uninitialized(meta_model)
        if uninitialized:
            logger.warning(f"Uninitialized parameters or buffers: {uninitialized}")
            return meta_model
        return meta_model.to(device)

    def build(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **kwargs: object,  # noqa: ARG002
    ) -> ModelType:
        device = torch.device("cuda") if device is None else device
        config = self.model_config()
        meta_model = self.meta_model(config, self.module_ops)

        _load_model_weights(
            meta_model=meta_model,
            model_path=self.model_path,
            loras=self.loras,
            loader=self.model_loader,
            registry=self.registry,
            device=device,
            dtype=dtype,
            model_sd_ops=self.model_sd_ops,
            lora_load_device=self.lora_load_device,
        )
        return self._return_model(meta_model, device)
