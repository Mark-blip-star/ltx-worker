from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORE_SRC = ROOT / "LTX-2/packages/ltx-core/src"
sys.path.insert(0, str(CORE_SRC))

try:
    import torch

    from ltx_core.batch_split import BatchSplitAdapter
    from ltx_core.guidance.perturbations import (
        BatchedPerturbationConfig,
        Perturbation,
        PerturbationConfig,
        PerturbationType,
    )
    from ltx_core.model.transformer.modality import Modality
    from ltx_core.model.transformer.compiling import (
        CompilationConfig,
        CompiledBlockPerturbationsProcessor,
        _validate_lossless_config,
        compile_transformer,
    )
    from ltx_core.model.transformer.transformer_args import TransformerArgs
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch runtime is not installed on the host")
class TorchPerturbationRuntimeTests(unittest.TestCase):
    def test_tensorized_masks_and_batch_slices_use_real_torch_tensors(self) -> None:
        configs = [
            PerturbationConfig.empty(),
            PerturbationConfig(
                [Perturbation(PerturbationType.SKIP_VIDEO_SELF_ATTN, None)]
            ),
            PerturbationConfig(
                [
                    Perturbation(PerturbationType.SKIP_VIDEO_SELF_ATTN, [1]),
                    Perturbation(PerturbationType.SKIP_A2V_CROSS_ATTN, None),
                ]
            ),
        ]
        batch = BatchedPerturbationConfig(configs, num_blocks=3, device="cpu", dtype=torch.bfloat16)

        self.assertTrue(
            torch.equal(
                batch.mask(PerturbationType.SKIP_VIDEO_SELF_ATTN, 1).reshape(-1),
                torch.tensor([1, 0, 0], dtype=torch.bfloat16),
            )
        )
        sliced = batch.batch_slice(1, 3)
        self.assertTrue(sliced.any_in_batch(PerturbationType.SKIP_VIDEO_SELF_ATTN, 0))
        self.assertFalse(sliced.all_in_batch(PerturbationType.SKIP_VIDEO_SELF_ATTN, 0))
        first = sliced.mask(PerturbationType.SKIP_A2V_CROSS_ATTN, 0)
        second = sliced.mask(PerturbationType.SKIP_A2V_CROSS_ATTN, 0)
        self.assertNotEqual(first.data_ptr(), second.data_ptr())

    def test_compiled_processor_pins_topology_and_marks_only_sequence_dimensions(self) -> None:
        args = TransformerArgs(
            x=torch.zeros(1, 4, 8),
            context=torch.zeros(1, 6, 8),
            context_mask=torch.zeros(1, 1, 6),
            timesteps=torch.zeros(1, 4, 8),
            embedded_timestep=torch.zeros(1, 4, 8),
            positional_embeddings=(torch.zeros(1, 2, 4, 4), torch.zeros(1, 2, 4, 4)),
            cross_positional_embeddings=(torch.zeros(1, 2, 4, 4), torch.zeros(1, 2, 4, 4)),
            cross_scale_shift_timestep=torch.zeros(1, 4, 8),
            cross_gate_timestep=torch.zeros(1, 1, 8),
            enabled=True,
            self_attention_mask=torch.zeros(1, 1, 4, 4),
        )
        perturbations = BatchedPerturbationConfig(
            [PerturbationConfig.empty()],
            num_blocks=1,
            device="cpu",
            dtype=torch.float32,
        )
        marked: list[tuple[tuple[int, ...], int]] = []
        original = torch._dynamo.mark_dynamic

        def record(tensor: torch.Tensor, dim: int) -> None:
            marked.append((tuple(tensor.shape), dim))

        torch._dynamo.mark_dynamic = record
        try:
            result = CompiledBlockPerturbationsProcessor()(
                args,
                perturbations,
                0,
                PerturbationType.SKIP_VIDEO_SELF_ATTN,
                PerturbationType.SKIP_A2V_CROSS_ATTN,
            )
        finally:
            torch._dynamo.mark_dynamic = original

        self.assertFalse(result.self_attn_all_perturbed)
        self.assertFalse(result.cross_attn_skip_all)
        self.assertEqual(tuple(result.self_attn_perturbation_mask.shape), (1, 1, 1))
        self.assertEqual(tuple(result.cross_attn_perturbation_mask.shape), (1, 1, 1))
        self.assertIn(((1, 4, 8), 1), marked)
        self.assertNotIn(((1, 1, 8), 1), marked)

    def test_batch_split_slices_tensorized_masks_without_rebuilding(self) -> None:
        class EchoModel(torch.nn.Module):
            num_blocks = 2

            def __init__(self) -> None:
                super().__init__()
                self.seen_masks: list[torch.Tensor] = []

            def forward(
                self,
                video: Modality,
                audio: Modality | None,
                perturbations: BatchedPerturbationConfig,
            ) -> tuple[torch.Tensor, None]:
                self.seen_masks.append(
                    perturbations.mask(PerturbationType.SKIP_VIDEO_SELF_ATTN, 0)
                )
                return video.latent, None

        video = Modality(
            latent=torch.arange(12, dtype=torch.float32).reshape(2, 3, 2),
            sigma=torch.ones(2),
            timesteps=torch.ones(2, 3),
            positions=torch.zeros(2, 3, 3),
            context=torch.zeros(2, 4, 2),
        )
        perturbations = BatchedPerturbationConfig(
            [
                PerturbationConfig.empty(),
                PerturbationConfig(
                    [Perturbation(PerturbationType.SKIP_VIDEO_SELF_ATTN, None)]
                ),
            ],
            num_blocks=2,
            device="cpu",
            dtype=torch.float32,
        )
        inner = EchoModel()
        adapter = BatchSplitAdapter(inner, max_batch_size=1)

        output, audio = adapter(video=video, audio=None, perturbations=perturbations)

        self.assertTrue(torch.equal(output, video.latent))
        self.assertIsNone(audio)
        self.assertEqual([mask.reshape(-1).item() for mask in inner.seen_masks], [1, 0])

    def test_lossless_config_rejects_cuda_graph_and_unsafe_guard_modes(self) -> None:
        _validate_lossless_config(CompilationConfig())
        invalid = (
            CompilationConfig(mode="reduce-overhead"),
            CompilationConfig(fullgraph=True),
            CompilationConfig(dynamic=False),
            CompilationConfig(
                inductor_config={"unsafe_skip_cache_dynamic_shape_guards": True}
            ),
            CompilationConfig(dynamo_config={"suppress_errors": True}),
        )
        for config in invalid:
            with self.subTest(config=config), self.assertRaises(ValueError):
                _validate_lossless_config(config)

    def test_tiny_regional_compile_matches_eager_output(self) -> None:
        class TinyBlock(torch.nn.Module):
            def forward(self, value: torch.Tensor) -> torch.Tensor:
                return value.sin() + 1

        class TinyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.transformer_blocks = torch.nn.ModuleList([TinyBlock(), TinyBlock()])
                self.block_input_processor = None

            def forward(self, value: torch.Tensor) -> torch.Tensor:
                for block in self.transformer_blocks:
                    value = block(value)
                return value

        sample = torch.linspace(-1, 1, 32)
        eager = TinyModel()(sample)
        compiled = compile_transformer(TinyModel(), CompilationConfig())
        compiled_a = compiled(sample)
        compiled_b = compiled(sample + 0.25)
        compiled_a_again = compiled(sample)

        # Inductor may differ by one fp32 ULP while remaining numerically
        # equivalent; the real H200 gate is derived from eager-vs-eager noise.
        torch.testing.assert_close(compiled_a, eager, rtol=1e-6, atol=2e-7)
        torch.testing.assert_close(compiled_a_again, compiled_a, rtol=0, atol=0)
        self.assertFalse(torch.equal(compiled_b, compiled_a))
        self.assertEqual(len(compiled.transformer_blocks), 2)
        self.assertTrue(hasattr(compiled, "forward_without_compilation"))


if __name__ == "__main__":
    unittest.main()
