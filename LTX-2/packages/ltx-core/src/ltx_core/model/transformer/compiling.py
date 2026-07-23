from dataclasses import dataclass, field, replace
from typing import Any

import torch

from ltx_core.guidance.perturbations import BatchedPerturbationConfig, PerturbationType
from ltx_core.loader.module_ops import ModuleOps
from ltx_core.loader.sd_ops import SDOps
from ltx_core.model.transformer.model import LTXModel
from ltx_core.model.transformer.transformer_args import BlockPerturbationsProcessor, TransformerArgs


_DEFAULT_INDUCTOR_CONFIG: dict[str, Any] = {}
_DEFAULT_DYNAMO_CONFIG: dict[str, Any] = {
    "inline_inbuilt_nn_modules": True,
    "cache_size_limit": 256,
    "suppress_errors": False,
}
_UNSAFE_CACHE_FLAG = "unsafe_skip_cache_dynamic_shape_guards"


@dataclass(frozen=True)
class CompilationConfig:
    """Configuration for the first lossless regional compile canary.

    The first rollout intentionally accepts only Inductor's default mode.
    CUDA Graph modes remain blocked until their separate paired H200 campaign.
    """

    mode: str | None = None
    backend: str = "inductor"
    fullgraph: bool = False
    dynamic: bool | None = None
    inductor_config: dict[str, Any] = field(default_factory=lambda: dict(_DEFAULT_INDUCTOR_CONFIG))
    dynamo_config: dict[str, Any] = field(default_factory=lambda: dict(_DEFAULT_DYNAMO_CONFIG))


def _validate_lossless_config(config: CompilationConfig) -> None:
    if config.mode is not None:
        raise ValueError("lossless regional compile currently requires mode=None (CUDA Graphs disabled)")
    if config.backend != "inductor":
        raise ValueError("lossless regional compile currently requires backend='inductor'")
    if config.fullgraph:
        raise ValueError("lossless regional compile currently requires fullgraph=False")
    if config.dynamic is not None:
        raise ValueError("lossless regional compile currently requires dynamic=None")
    if config.inductor_config.get(_UNSAFE_CACHE_FLAG):
        raise ValueError(f"{_UNSAFE_CACHE_FLAG} is forbidden because it can silently corrupt dynamic shapes")
    if config.dynamo_config.get("suppress_errors") is not False:
        raise ValueError("lossless regional compile requires torch._dynamo.config.suppress_errors=False")


class CompiledBlockPerturbationsProcessor(BlockPerturbationsProcessor):
    """Mark sequence dimensions dynamic and attach stable runtime masks.

    This processor runs eagerly outside each compiled block. Both masks are
    always tensors and both Python skip flags stay false, so conditional,
    unconditional, STG, and modality passes share the same traced topology.
    """

    def __call__(
        self,
        args: TransformerArgs,
        perturbations: BatchedPerturbationConfig,
        block_idx: int,
        self_attn_type: PerturbationType,
        cross_attn_type: PerturbationType,
    ) -> TransformerArgs:
        torch._dynamo.mark_dynamic(args.x, 1)
        cos, sin = args.positional_embeddings
        torch._dynamo.mark_dynamic(cos, cos.ndim - 2)
        torch._dynamo.mark_dynamic(sin, sin.ndim - 2)

        if args.cross_positional_embeddings is not None:
            cross_cos, cross_sin = args.cross_positional_embeddings
            torch._dynamo.mark_dynamic(cross_cos, cross_cos.ndim - 2)
            torch._dynamo.mark_dynamic(cross_sin, cross_sin.ndim - 2)

        if args.self_attention_mask is not None:
            if args.self_attention_mask.shape[2] > 1:
                torch._dynamo.mark_dynamic(args.self_attention_mask, 2)
            torch._dynamo.mark_dynamic(args.self_attention_mask, 3)
        if args.context_mask is not None:
            torch._dynamo.mark_dynamic(args.context_mask, 2)
        if args.timesteps.shape[1] > 1:
            torch._dynamo.mark_dynamic(args.timesteps, 1)
        if args.embedded_timestep.shape[1] > 1:
            torch._dynamo.mark_dynamic(args.embedded_timestep, 1)
        if args.cross_scale_shift_timestep is not None and args.cross_scale_shift_timestep.shape[1] > 1:
            torch._dynamo.mark_dynamic(args.cross_scale_shift_timestep, 1)

        return replace(
            args,
            self_attn_perturbation_mask=perturbations.mask(self_attn_type, block_idx),
            self_attn_all_perturbed=False,
            cross_attn_perturbation_mask=perturbations.mask(cross_attn_type, block_idx),
            cross_attn_skip_all=False,
        )


def compile_transformer(model: LTXModel, config: CompilationConfig) -> LTXModel:
    """Compile each transformer block once with guards enabled.

    CUDA Graph step markers are deliberately absent: the accepted configuration
    is ``mode=None``. A future CUDA Graph experiment must add one marker around
    the complete model forward and remove the denoiser-level marker.
    """
    _validate_lossless_config(config)
    if hasattr(model, "forward_without_compilation"):
        raise RuntimeError("transformer is already regionally compiled")

    compiled_blocks = torch.nn.ModuleList(
        torch.compile(
            block,
            mode=config.mode,
            backend=config.backend,
            fullgraph=config.fullgraph,
            dynamic=config.dynamic,
        )
        for block in model.transformer_blocks
    )

    eager_forward = model.forward

    def patched_dynamo_forward(*args, **kwargs) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        with (
            torch._inductor.config.patch(**config.inductor_config),
            torch._dynamo.config.patch(**config.dynamo_config),  # type: ignore[attr-defined]
        ):
            return eager_forward(*args, **kwargs)

    model.transformer_blocks = compiled_blocks
    model.block_input_processor = CompiledBlockPerturbationsProcessor()
    model.forward_without_compilation = eager_forward
    model.forward = patched_dynamo_forward
    return model


def build_compile_transformer_op(config: CompilationConfig) -> ModuleOps:
    return ModuleOps(
        name="compile_transformer",
        matcher=lambda model: isinstance(model, LTXModel),
        mutator=lambda model: compile_transformer(model, config),
    )


COMPILE_TRANSFORMER = build_compile_transformer_op(CompilationConfig())


def modify_sd_ops_for_compilation(original_sd_ops: SDOps, number_of_blocks: int = 48) -> SDOps:
    for i in range(number_of_blocks):
        original_sd_ops = original_sd_ops.with_replacement(
            f"transformer_blocks.{i}.",
            f"transformer_blocks.{i}._orig_mod.",
        )
    return original_sd_ops
