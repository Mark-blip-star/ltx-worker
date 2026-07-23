from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class StageLoopScopingTests(unittest.TestCase):
    def test_stage_loops_are_explicit_and_stage2_is_euler(self) -> None:
        tree = ast.parse((ROOT / "stage_timing_runner.py").read_text())
        calls: dict[str, ast.Call] = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            if node.func.id != "run_stage_with_optional_cuda_graphs" or len(node.args) < 4:
                continue
            graph_key = node.args[3]
            if isinstance(graph_key, ast.Constant) and graph_key.value in {"stage1", "stage2"}:
                calls[str(graph_key.value)] = node

        self.assertEqual(set(calls), {"stage1", "stage2"})
        stage1_loop = next(kw.value for kw in calls["stage1"].keywords if kw.arg == "loop")
        stage2_loop = next(kw.value for kw in calls["stage2"].keywords if kw.arg == "loop")
        self.assertIn("gradient_estimating_euler_denoising_loop", ast.unparse(stage1_loop))
        self.assertIn("euler_denoising_loop", ast.unparse(stage1_loop))
        self.assertEqual(ast.unparse(stage2_loop), "euler_denoising_loop")

    def test_loop_is_forwarded_in_direct_and_resident_stage_paths(self) -> None:
        tree = ast.parse((ROOT / "stage_timing_runner.py").read_text())
        wrapper = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "run_stage_with_optional_cuda_graphs"
        )
        forwarded = []
        for node in ast.walk(wrapper):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name) and node.func.id == "stage":
                forwarded.append(node)
            elif (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "stage"
                and node.func.attr == "run"
            ):
                forwarded.append(node)
        self.assertEqual(len(forwarded), 2)
        for call in forwarded:
            value = next(kw.value for kw in call.keywords if kw.arg == "loop")
            self.assertEqual(ast.unparse(value), "loop")

    def test_handler_has_no_global_blocks_loop_monkeypatch(self) -> None:
        source = (ROOT / "handler.py").read_text()
        self.assertNotIn("_ltx_blocks.euler_denoising_loop", source)
        self.assertNotIn("blocks.euler_denoising_loop =", source)
        self.assertIn(
            '_STAGE1_GE_DEFAULT = os.environ.get("LTX_STAGE1_GE") == "1"',
            source,
        )
        self.assertIn(
            'resolve_boolean(inp, "stage1_ge", _STAGE1_GE_DEFAULT)',
            source,
        )
        self.assertIn('settings["stage1_ge"] = ge_on', source)

    def test_handler_imports_ltx_23_params_used_for_effective_cfg(self) -> None:
        tree = ast.parse((ROOT / "handler.py").read_text())
        imported_names = {
            alias.asname or alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and node.module == "ltx_pipelines.utils.constants"
            for alias in node.names
        }
        self.assertIn("LTX_2_3_PARAMS", imported_names)

    def test_warmup_uses_endpoint_stage1_sampler_default(self) -> None:
        source = (ROOT / "handler.py").read_text()
        self.assertIn('"stage1_ge": _STAGE1_GE_DEFAULT', source)


class PromptAndDecoderWiringTests(unittest.TestCase):
    def test_run_case_reads_case_negative_prompt(self) -> None:
        source = (ROOT / "stage_timing_runner.py").read_text()
        handler = (ROOT / "handler.py").read_text()
        self.assertIn('case.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT)', source)
        self.assertIn("[case[\"prompt\"], negative_prompt]", source)
        self.assertIn('negative_prompt_source == "default"', handler)

    def test_decode_override_targets_constructed_instance(self) -> None:
        blocks = (
            ROOT
            / "LTX-2/packages/ltx-pipelines/src/ltx_pipelines/utils/blocks.py"
        ).read_text()
        runner = (ROOT / "stage_timing_runner.py").read_text()
        handler = (ROOT / "handler.py").read_text()
        self.assertIn("decoder.decode_noise_scale = resolved", blocks)
        self.assertIn("self.last_decode_noise_scale = float(decoder.decode_noise_scale)", blocks)
        self.assertIn('decode_noise_scale=settings.get("decode_noise")', runner)
        self.assertNotIn("LTX_DECODE_NOISE_SCALE", handler)
        self.assertNotIn("LTX_DECODE_NOISE_SCALE", runner)

    def test_effective_config_is_returned(self) -> None:
        source = (ROOT / "handler.py").read_text()
        for required_key in (
            '"effective_config"',
            '"sampler"',
            '"cfg_cache"',
            '"tgate"',
            '"cas"',
            '"decode_noise"',
            '"enhance"',
            '"audio"',
            '"shape"',
            '"negative_prompt"',
        ):
            with self.subTest(required_key=required_key):
                self.assertIn(required_key, source)

    def test_latency_sensitive_booleans_are_strictly_resolved(self) -> None:
        source = (ROOT / "handler.py").read_text()
        for expression in (
            'resolve_boolean(inp, "audio", False)',
            'resolve_boolean(inp, "enhance", ENHANCE_DEFAULT)',
            'resolve_boolean(inp, "cfg_cache", _CFG_CACHE_ENV)',
        ):
            with self.subTest(expression=expression):
                self.assertIn(expression, source)
        self.assertNotIn('bool(inp.get("audio", False))', source)
        self.assertNotIn('bool(inp.get("enhance", ENHANCE_DEFAULT))', source)

    def test_request_validation_module_is_copied_into_worker_images(self) -> None:
        for dockerfile in ("Dockerfile", "Dockerfile.baked", "Dockerfile.slim"):
            with self.subTest(dockerfile=dockerfile):
                source = (ROOT / dockerfile).read_text()
                self.assertIn(
                    "COPY request_config.py /app/request_config.py",
                    source,
                )

    def test_slim_build_fails_on_application_syntax_errors(self) -> None:
        source = (ROOT / "Dockerfile.slim").read_text()
        self.assertIn("import request_config, stage_timing_runner", source)
        self.assertIn("python -m py_compile", source)
        self.assertIn("/app/request_config.py /app/stage_timing_runner.py /app/handler.py", source)
        self.assertIn("ruff check --select F821", source)


if __name__ == "__main__":
    unittest.main()
