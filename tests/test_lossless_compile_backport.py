from __future__ import annotations

import ast
import importlib.util
import itertools
import math
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "LTX-2/packages/ltx-core/src/ltx_core"


class _FakeTensor:
    def __init__(self, data: object, *, dtype: object = None, device: object = "cpu") -> None:
        self._shape, self._values = self._flatten(data)
        self.dtype = dtype
        self.device = device

    @classmethod
    def _from_flat(
        cls,
        values: list[object],
        shape: tuple[int, ...],
        *,
        dtype: object,
        device: object,
    ) -> "_FakeTensor":
        tensor = cls.__new__(cls)
        tensor._shape = shape
        tensor._values = list(values)
        tensor.dtype = dtype
        tensor.device = device
        return tensor

    @classmethod
    def _flatten(cls, data: object) -> tuple[tuple[int, ...], list[object]]:
        if not isinstance(data, (list, tuple)):
            return (), [data]
        if not data:
            return (0,), []

        children = [cls._flatten(item) for item in data]
        child_shape = children[0][0]
        if any(shape != child_shape for shape, _ in children):
            raise ValueError("ragged fake tensor data is unsupported")
        return (len(data), *child_shape), [
            value for _, values in children for value in values
        ]

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def data(self) -> "_FakeTensor":
        return self

    def __getitem__(self, key: object) -> "_FakeTensor":
        keys = key if isinstance(key, tuple) else (key,)
        if len(keys) > len(self.shape):
            raise IndexError("too many fake tensor indices")
        keys = (*keys, *(slice(None) for _ in range(len(self.shape) - len(keys))))

        coordinate_axes: list[list[int]] = []
        output_shape: list[int] = []
        for size, axis_key in zip(self.shape, keys, strict=True):
            if isinstance(axis_key, int):
                index = axis_key + size if axis_key < 0 else axis_key
                if not 0 <= index < size:
                    raise IndexError("fake tensor index out of range")
                coordinate_axes.append([index])
            elif isinstance(axis_key, slice):
                indices = list(range(*axis_key.indices(size)))
                coordinate_axes.append(indices)
                output_shape.append(len(indices))
            else:
                raise TypeError(f"unsupported fake tensor index: {axis_key!r}")

        strides = [
            math.prod(self.shape[axis + 1 :])
            for axis in range(len(self.shape))
        ]
        values = [
            self._values[sum(index * stride for index, stride in zip(coords, strides, strict=True))]
            for coords in itertools.product(*coordinate_axes)
        ]
        return self._from_flat(
            values,
            tuple(output_shape),
            dtype=self.dtype,
            device=self.device,
        )

    def __eq__(self, other: object) -> "_FakeTensor":
        return self._from_flat(
            [value == other for value in self._values],
            self.shape,
            dtype=bool,
            device=self.device,
        )

    def any(self) -> bool:
        return any(self._values)

    def all(self) -> bool:
        return all(self._values)

    def reshape(self, *shape: int) -> "_FakeTensor":
        inferred = [index for index, size in enumerate(shape) if size == -1]
        if len(inferred) > 1:
            raise ValueError("only one inferred fake tensor dimension is supported")
        resolved = list(shape)
        if inferred:
            known = math.prod(size for size in resolved if size != -1)
            if known == 0 or len(self._values) % known:
                raise ValueError("invalid inferred fake tensor shape")
            resolved[inferred[0]] = len(self._values) // known
        if math.prod(resolved) != len(self._values):
            raise ValueError("invalid fake tensor shape")
        return self._from_flat(
            self._values,
            tuple(resolved),
            dtype=self.dtype,
            device=self.device,
        )

    def clone(self) -> "_FakeTensor":
        return self._from_flat(
            self._values,
            self.shape,
            dtype=self.dtype,
            device=self.device,
        )

    def to(self, device: object) -> "_FakeTensor":
        return self._from_flat(
            self._values,
            self.shape,
            dtype=self.dtype,
            device=device,
        )

    def tolist(self) -> object:
        def nest(values: list[object], shape: tuple[int, ...]) -> object:
            if not shape:
                return values[0]
            stride = math.prod(shape[1:])
            return [
                nest(values[index * stride : (index + 1) * stride], shape[1:])
                for index in range(shape[0])
            ]

        return nest(self._values, self.shape)


def _load_perturbations_with_fake_torch() -> types.ModuleType:
    fake_torch = types.ModuleType("torch")
    fake_torch.__path__ = []
    fake_torch.Tensor = _FakeTensor
    fake_torch.dtype = object
    fake_torch.tensor = lambda data, dtype=None, device=None: _FakeTensor(
        data,
        dtype=dtype,
        device=device,
    )
    fake_prims = types.ModuleType("torch._prims_common")
    fake_prims.DeviceLikeType = object

    path = CORE / "guidance/perturbations.py"
    spec = importlib.util.spec_from_file_location("_test_ltx_perturbations", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with patch.dict(
        sys.modules,
        {
            "torch": fake_torch,
            "torch._prims_common": fake_prims,
            spec.name: module,
        },
    ):
        spec.loader.exec_module(module)
    return module


class TensorizedPerturbationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_perturbations_with_fake_torch()

    def test_masks_are_precomputed_per_type_block_and_batch(self) -> None:
        p = self.module
        configs = [
            p.PerturbationConfig.empty(),
            p.PerturbationConfig(
                [p.Perturbation(p.PerturbationType.SKIP_VIDEO_SELF_ATTN, None)]
            ),
            p.PerturbationConfig(
                [
                    p.Perturbation(p.PerturbationType.SKIP_VIDEO_SELF_ATTN, [1]),
                    p.Perturbation(p.PerturbationType.SKIP_A2V_CROSS_ATTN, None),
                ]
            ),
        ]

        batch = p.BatchedPerturbationConfig(configs, num_blocks=3, device="cuda", dtype="bf16")

        self.assertEqual(
            batch.mask(p.PerturbationType.SKIP_VIDEO_SELF_ATTN, 0).data.reshape(-1).tolist(),
            [1, 0, 1],
        )
        self.assertEqual(
            batch.mask(p.PerturbationType.SKIP_VIDEO_SELF_ATTN, 1).data.reshape(-1).tolist(),
            [1, 0, 0],
        )
        self.assertEqual(
            batch.mask(p.PerturbationType.SKIP_A2V_CROSS_ATTN, 2).data.reshape(-1).tolist(),
            [1, 1, 0],
        )
        self.assertTrue(batch.any_in_batch(p.PerturbationType.SKIP_VIDEO_SELF_ATTN, 0))
        self.assertFalse(batch.all_in_batch(p.PerturbationType.SKIP_VIDEO_SELF_ATTN, 0))

    def test_batch_slice_preserves_device_mask_and_host_shortcuts(self) -> None:
        p = self.module
        configs = [
            p.PerturbationConfig.empty(),
            p.PerturbationConfig(
                [p.Perturbation(p.PerturbationType.SKIP_AUDIO_SELF_ATTN, None)]
            ),
        ]
        batch = p.BatchedPerturbationConfig(configs, num_blocks=2, device="cuda", dtype="bf16")

        sliced = batch.batch_slice(1, 2)
        mask_a = sliced.mask(p.PerturbationType.SKIP_AUDIO_SELF_ATTN, 0)
        mask_b = sliced.mask(p.PerturbationType.SKIP_AUDIO_SELF_ATTN, 0)

        self.assertEqual(mask_a.data.reshape(-1).tolist(), [0])
        self.assertEqual(mask_a.device, "cuda")
        self.assertTrue(sliced.all_in_batch(p.PerturbationType.SKIP_AUDIO_SELF_ATTN, 0))
        self.assertIsNot(mask_a, mask_b)
        self.assertIsNot(mask_a._values, mask_b._values)


class StaticCompileSafetyTests(unittest.TestCase):
    def test_compiler_is_regional_dynamic_and_fail_closed(self) -> None:
        source = (CORE / "model/transformer/compiling.py").read_text()
        self.assertIn("for block in model.transformer_blocks", source)
        self.assertIn("mode: str | None = None", source)
        self.assertIn("fullgraph: bool = False", source)
        self.assertIn("dynamic: bool | None = None", source)
        self.assertIn('"suppress_errors": False', source)
        self.assertNotIn('mode="reduce-overhead"', source)
        self.assertNotIn("unsafe_skip_cache_dynamic_shape_guards=True", source)
        self.assertNotIn("cudagraph_mark_step_begin", source)

    def test_block_has_no_python_perturbation_config_and_tgate_is_preserved(self) -> None:
        path = CORE / "model/transformer/transformer.py"
        source = path.read_text()
        tree = ast.parse(source)
        block = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "BasicAVTransformerBlock"
        )
        forward = next(
            node
            for node in block.body
            if isinstance(node, ast.FunctionDef) and node.name == "forward"
        )
        self.assertNotIn("perturbations", [arg.arg for arg in forward.args.args])
        self.assertNotIn("any_in_batch", ast.unparse(forward))
        self.assertNotIn("all_in_batch", ast.unparse(forward))
        self.assertIn("LTX_CURRENT_STEP_INDEX", source)
        self.assertIn("_tgate_video_delta", source)
        self.assertIn("_tgate_audio_delta", source)

    def test_resident_integration_is_stage1_only_and_pipeline_default_stays_eager(self) -> None:
        handler = (ROOT / "handler.py").read_text()
        self.assertIn('cache_key != "stage1"', handler)
        self.assertIn("CompilationConfig(", handler)
        self.assertIn("mode=None", handler)
        self.assertIn("backend=\"inductor\"", handler)
        self.assertIn("fullgraph=False", handler)
        self.assertIn("dynamic=None", handler)
        self.assertIn("quantization=QuantizationPolicy.fp8_scaled_mm(ckpt), torch_compile=False", handler)
        self.assertIn("cuda_graphs=False", handler)
        self.assertIn("torch_compile_transformer=False", handler)
        self.assertNotIn("_dynamo.config.suppress_errors = True", handler)
        self.assertIn("mandatory regional compile warmup failed", handler)
        self.assertLess(
            handler.index("_REGIONAL_COMPILE_SETTINGS.prepare_environment(os.environ)"),
            handler.index("import torch  # noqa: E402"),
        )
        self.assertLess(
            handler.index("out = super().get(stage, cache_key, **kw)"),
            handler.index("_compile_resident_stage(cache_key, out)"),
        )

    def test_tensorized_masks_are_built_before_batch_split_and_compile_boundary(self) -> None:
        denoisers = (
            ROOT / "LTX-2/packages/ltx-pipelines/src/ltx_pipelines/utils/denoisers.py"
        ).read_text()
        batch_split = (CORE / "batch_split.py").read_text()
        model = (CORE / "model/transformer/model.py").read_text()
        self.assertIn("num_blocks=transformer.num_blocks", denoisers)
        self.assertIn("device=reference.latent.device", denoisers)
        self.assertIn("dtype=reference.latent.dtype", denoisers)
        self.assertIn("config.batch_slice", batch_split)
        self.assertIn("self.block_input_processor(", model)

    def test_images_copy_pre_torch_config_and_slim_installs_compiler(self) -> None:
        for dockerfile in ("Dockerfile", "Dockerfile.baked", "Dockerfile.slim"):
            source = (ROOT / dockerfile).read_text()
            self.assertIn(
                "COPY regional_compile_config.py /app/regional_compile_config.py",
                source,
            )
        slim = (ROOT / "Dockerfile.slim").read_text()
        self.assertIn("curl ca-certificates g++ libgomp1", slim)


if __name__ == "__main__":
    unittest.main()
