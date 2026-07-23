from __future__ import annotations

import ast
import copy
import threading
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class QualityStepAllocationWiringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.handler = (ROOT / "handler.py").read_text()
        cls.runner = (ROOT / "stage_timing_runner.py").read_text()

    def test_quality_tier_uses_existing_16_step_default(self) -> None:
        self.assertIn(
            "else (FAST_DEFAULT_STEPS if tier == \"fast\" else 16)",
            self.handler,
        )
        self.assertIn(
            "_STAGE2_DEFAULT_VALUES = tuple(float(value) for value in "
            "STAGE_2_DISTILLED_SIGMA_VALUES)",
            self.handler,
        )

    def test_stage2_override_is_validated_before_tier_dispatch(self) -> None:
        self.assertIn(
            "requested_stage2_sigmas = resolve_stage2_sigmas(inp)",
            self.handler,
        )
        self.assertNotIn(
            'tier == "fast" and "stage2_sigmas" in inp',
            self.handler,
        )

    def test_stage2_schedule_is_request_local_not_a_mutated_module_global(self) -> None:
        self.assertNotIn(
            "base.STAGE_2_DISTILLED_SIGMAS =",
            self.handler,
        )
        self.assertIn(
            'settings["stage2_sigmas"] = stage2_effective_tensor',
            self.handler,
        )
        self.assertIn(
            'configured_stage_2_sigmas = settings.get("stage2_sigmas")',
            self.runner,
        )
        self.assertIn(
            "if configured_stage_2_sigmas is None",
            self.runner,
        )
        self.assertIn(
            "STAGE_2_DISTILLED_SIGMAS",
            self.runner,
        )

    def test_effective_config_attests_requested_and_effective_schedule(self) -> None:
        for key in (
            '"schedule"',
            '"requested_step_count"',
            '"effective_step_count"',
            '"requested_sigmas"',
            '"effective_sigmas"',
            '"effective_total_transition_count"',
        ):
            with self.subTest(key=key):
                self.assertIn(key, self.handler)
        self.assertIn(
            "sampling_schedule_tag_suffix(steps, stage2_effective_values)",
            self.handler,
        )

    def test_effective_config_attests_image_conditioning_strength(self) -> None:
        self.assertIn(
            '"conditioning_strength", minimum=0.0, maximum=1.0',
            self.handler,
        )
        self.assertIn(
            '"image_conditioning"',
            self.handler,
        )
        self.assertIn(
            '"stage1_strength"',
            self.handler,
        )
        self.assertIn(
            '"stage2_strength"',
            self.handler,
        )
        self.assertIn('"source": conditioning_strength_source', self.handler)
        self.assertIn(
            'conditioning_strength_source = "not_applicable"',
            self.handler,
        )
        self.assertIn(
            'if t2v:',
            self.handler,
        )
        self.assertIn(
            "+ conditioning_strength_tag",
            self.handler,
        )

    def test_stage_specific_image_conditioning_is_request_local(self) -> None:
        self.assertIn(
            "requested_stage_conditioning_strengths = (",
            self.handler,
        )
        self.assertIn(
            '"conditioning_strength_stage1": conditioning_strength_stage1',
            self.handler,
        )
        self.assertIn(
            '"conditioning_strength_stage2": conditioning_strength_stage2',
            self.handler,
        )
        self.assertIn(
            'settings.get(\n            "conditioning_strength_stage1"',
            self.runner,
        )
        self.assertIn(
            'settings.get(\n            "conditioning_strength_stage2"',
            self.runner,
        )
        self.assertIn("images=stage1_images", self.runner)
        self.assertIn("images=stage2_images", self.runner)

    def test_warmup_receives_fast_schedule_without_global_state(self) -> None:
        self.assertIn('"stage2_sigmas": _STAGE2_FAST', self.handler)


class RegionalCompileResponseTelemetryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = (ROOT / "handler.py").read_text()
        tree = ast.parse(cls.source)
        cls.refresh_function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_refresh_regional_compile_generation_snapshot"
        )

    def _build_namespace(self):
        class FakeCuda:
            @staticmethod
            def is_available():
                return True

            @staticmethod
            def memory_allocated():
                return 1 * 2**30

            @staticmethod
            def memory_reserved():
                return 2 * 2**30

            @staticmethod
            def max_memory_allocated():
                return 3 * 2**30

            @staticmethod
            def max_memory_reserved():
                return 4 * 2**30

        state = {
            "config": {"enabled": True},
            "runtime": {},
            "stages": {"stage1": {"status": "warmed"}},
        }
        namespace = {
            "_REGIONAL_COMPILE_REQUEST_SEQUENCE": 0,
            "_REGIONAL_COMPILE_SETTINGS": types.SimpleNamespace(enabled=True),
            "_REGIONAL_COMPILE_STATE": state,
            "_REGIONAL_COMPILE_TELEMETRY_LOCK": threading.Lock(),
            "_compiler_counter_snapshot": lambda **_: {
                "frames": {"ok": 1},
            },
            "_regional_compile_snapshot": lambda: copy.deepcopy(state),
            "torch": types.SimpleNamespace(cuda=FakeCuda()),
        }
        exec(
            compile(
                ast.fix_missing_locations(
                    ast.Module(body=[self.refresh_function], type_ignores=[]),
                ),
                str(ROOT / "handler.py"),
                "exec",
            ),
            namespace,
        )
        return namespace

    def test_latest_snapshot_is_refreshed_and_bounded(self) -> None:
        namespace = self._build_namespace()
        refresh = namespace["_refresh_regional_compile_generation_snapshot"]
        first = refresh()
        namespace["_compiler_counter_snapshot"] = lambda **_: {
            "frames": {"ok": 2},
        }
        second = refresh()

        self.assertEqual(first["post_generation"]["request_sequence"], 1)
        self.assertEqual(
            first["post_generation"]["compiler_counters"]["frames"]["ok"],
            1,
        )
        self.assertEqual(second["post_generation"]["request_sequence"], 2)
        self.assertEqual(
            second["post_generation"]["compiler_counters"]["frames"]["ok"],
            2,
        )
        self.assertEqual(
            set(namespace["_REGIONAL_COMPILE_STATE"]),
            {"config", "runtime", "stages", "post_generation"},
        )
        self.assertIsInstance(
            namespace["_REGIONAL_COMPILE_STATE"]["post_generation"],
            dict,
        )
        self.assertNotIn(
            ".append(",
            ast.unparse(self.refresh_function),
        )

    def test_telemetry_is_compile_only_and_fail_closed(self) -> None:
        namespace = self._build_namespace()
        refresh = namespace["_refresh_regional_compile_generation_snapshot"]
        namespace["_REGIONAL_COMPILE_SETTINGS"].enabled = False
        with self.assertRaises(RuntimeError):
            refresh()
        namespace["_REGIONAL_COMPILE_SETTINGS"].enabled = True
        namespace["_compiler_counter_snapshot"] = lambda **_: {}
        with self.assertRaises(RuntimeError):
            refresh()

    def test_counter_payload_and_response_refresh_are_statically_bounded(self) -> None:
        for expected in (
            "_COMPILER_COUNTER_KEYS_PER_CATEGORY = 20",
            "_COMPILER_COUNTER_KEY_MAX_CHARS = 160",
            ")[:_COMPILER_COUNTER_KEYS_PER_CATEGORY]",
            "str(key)[:_COMPILER_COUNTER_KEY_MAX_CHARS]",
            "_compiler_counter_snapshot(fail_closed=True)",
            "_refresh_regional_compile_generation_snapshot()",
            'resp["effective_config"]["regional_compile"] = '
            "regional_compile_response",
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, self.source)
        run_position = self.source.index("rec = base.run_case")
        refresh_position = self.source.index(
            "_refresh_regional_compile_generation_snapshot()",
            run_position,
        )
        response_position = self.source.index(
            'resp["effective_config"]["regional_compile"]',
            refresh_position,
        )
        self.assertLess(run_position, refresh_position)
        self.assertLess(refresh_position, response_position)


if __name__ == "__main__":
    unittest.main()
