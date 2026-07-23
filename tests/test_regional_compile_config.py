from __future__ import annotations

import unittest

from regional_compile_config import DEFAULT_CACHE_DIR, RegionalCompileSettings


class RegionalCompileSettingsTests(unittest.TestCase):
    def test_default_is_fully_disabled_and_does_not_mutate_environment(self) -> None:
        environ: dict[str, str] = {}

        settings = RegionalCompileSettings.from_environ(environ)
        settings.prepare_environment(environ)

        self.assertFalse(settings.enabled)
        self.assertFalse(settings.stage1)
        self.assertFalse(settings.stage2)
        self.assertEqual(settings.mode, "default")
        self.assertNotIn("TORCHINDUCTOR_CACHE_DIR", environ)

    def test_enabled_canary_is_stage1_default_mode_with_guards(self) -> None:
        environ = {"LTX_REGIONAL_COMPILE": "1"}

        settings = RegionalCompileSettings.from_environ(environ)
        settings.prepare_environment(environ)

        self.assertTrue(settings.enabled)
        self.assertTrue(settings.stage1)
        self.assertFalse(settings.stage2)
        self.assertIsNone(settings.dynamic)
        self.assertFalse(settings.fullgraph)
        self.assertTrue(settings.guards_enabled)
        self.assertEqual(environ["TORCHINDUCTOR_CACHE_DIR"], DEFAULT_CACHE_DIR)

    def test_explicit_cache_directory_is_preserved(self) -> None:
        environ = {
            "LTX_REGIONAL_COMPILE": "1",
            "TORCHINDUCTOR_CACHE_DIR": "/tmp/existing-compile-cache",
        }

        settings = RegionalCompileSettings.from_environ(environ)
        settings.prepare_environment(environ)

        self.assertEqual(settings.cache_dir, "/tmp/existing-compile-cache")
        self.assertEqual(environ["TORCHINDUCTOR_CACHE_DIR"], "/tmp/existing-compile-cache")

    def test_rejects_invalid_flags_empty_rollout_stage2_and_cuda_graph_mode(self) -> None:
        invalid = (
            {"LTX_REGIONAL_COMPILE": "yes"},
            {
                "LTX_REGIONAL_COMPILE": "1",
                "LTX_REGIONAL_COMPILE_STAGE1": "0",
            },
            {
                "LTX_REGIONAL_COMPILE": "1",
                "LTX_REGIONAL_COMPILE_STAGE2": "1",
            },
            {
                "LTX_REGIONAL_COMPILE": "1",
                "LTX_REGIONAL_COMPILE_MODE": "reduce-overhead",
            },
        )

        for environ in invalid:
            with self.subTest(environ=environ), self.assertRaises(ValueError):
                RegionalCompileSettings.from_environ(environ)


if __name__ == "__main__":
    unittest.main()
