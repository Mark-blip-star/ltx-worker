import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class TerminalKeyframeSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.handler = (ROOT / "handler.py").read_text()
        cls.runner = (ROOT / "stage_timing_runner.py").read_text()

    def test_files_remain_valid_python(self):
        ast.parse(self.handler)
        ast.parse(self.runner)

    def test_default_is_strictly_off(self):
        self.assertIn(
            '"terminal_keyframe_strength_stage1": (\n'
            "                requested_terminal_keyframe_strength_stage1 or 0.0",
            self.handler,
        )
        self.assertIn(
            '"terminal_keyframe_strength_stage2": (\n'
            "                requested_terminal_keyframe_strength_stage2 or 0.0",
            self.handler,
        )

    def test_only_nonzero_i2v_arms_append_terminal_guidance(self):
        self.assertIn(
            "if not t2v and terminal_keyframe_strength_stage1 > 0.0:",
            self.runner,
        )
        self.assertIn(
            "if not t2v and terminal_keyframe_strength_stage2 > 0.0:",
            self.runner,
        )
        self.assertIn("frame_idx=terminal_frame_index", self.runner)

    def test_first_frame_replacement_is_unchanged(self):
        self.assertIn("stage1_images = [] if t2v else [stage1_image]", self.runner)
        self.assertIn("stage2_images = [] if t2v else [stage2_image]", self.runner)

    def test_t2v_rejects_terminal_keyframe_request(self):
        self.assertIn(
            '"detail": "terminal keyframe strengths require image_b64"',
            self.handler,
        )


if __name__ == "__main__":
    unittest.main()
