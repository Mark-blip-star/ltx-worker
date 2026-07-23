from __future__ import annotations

import math
import unittest

from request_config import (
    DEFAULT_DECODE_NOISE_SCALE,
    MAX_DECODE_NOISE_SCALE,
    MAX_NEGATIVE_PROMPT_CHARS,
    resolve_boolean,
    resolve_decode_noise,
    resolve_negative_prompt,
    resolve_optional_number,
)


class BooleanTests(unittest.TestCase):
    def test_omitted_uses_default_and_json_booleans_are_preserved(self) -> None:
        self.assertTrue(resolve_boolean({}, "stage1_ge", True))
        self.assertFalse(resolve_boolean({}, "stage1_ge", False))
        self.assertTrue(resolve_boolean({"stage1_ge": True}, "stage1_ge", False))
        self.assertFalse(resolve_boolean({"stage1_ge": False}, "stage1_ge", True))

    def test_rejects_truthy_strings_and_numbers(self) -> None:
        for value in ("false", "true", 0, 1, None, []):
            with self.subTest(value=value), self.assertRaises(ValueError):
                resolve_boolean({"stage1_ge": value}, "stage1_ge", False)


class NegativePromptTests(unittest.TestCase):
    def test_omitted_uses_existing_default(self) -> None:
        self.assertEqual(
            resolve_negative_prompt({}, "existing default"),
            ("existing default", "default"),
        )

    def test_request_value_including_empty_string_is_preserved(self) -> None:
        self.assertEqual(resolve_negative_prompt({"negative_prompt": ""}, "default"), ("", "request"))
        self.assertEqual(
            resolve_negative_prompt({"negative_prompt": "no extra people"}, "default"),
            ("no extra people", "request"),
        )

    def test_rejects_wrong_type_nul_and_excess_length(self) -> None:
        for value in (None, 7, False, ["extra people"]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                resolve_negative_prompt({"negative_prompt": value}, "default")
        with self.assertRaises(ValueError):
            resolve_negative_prompt({"negative_prompt": "bad\x00prompt"}, "default")
        with self.assertRaises(ValueError):
            resolve_negative_prompt(
                {"negative_prompt": "x" * (MAX_NEGATIVE_PROMPT_CHARS + 1)},
                "default",
            )


class DecodeNoiseTests(unittest.TestCase):
    def test_omitted_preserves_decoder_default(self) -> None:
        self.assertEqual(DEFAULT_DECODE_NOISE_SCALE, 0.025)
        self.assertIsNone(resolve_decode_noise({}))

    def test_accepts_safe_numeric_range(self) -> None:
        for value in (0, 0.025, 0.05, MAX_DECODE_NOISE_SCALE):
            with self.subTest(value=value):
                self.assertEqual(resolve_decode_noise({"decode_noise": value}), float(value))

    def test_rejects_non_numeric_non_finite_and_out_of_range(self) -> None:
        for value in (
            True,
            "0.05",
            None,
            math.nan,
            math.inf,
            -0.001,
            MAX_DECODE_NOISE_SCALE + 0.001,
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                resolve_decode_noise({"decode_noise": value})


class OptionalNumberTests(unittest.TestCase):
    def test_omitted_returns_none_and_bounds_are_inclusive(self) -> None:
        self.assertIsNone(
            resolve_optional_number({}, "cas_amount", minimum=0.0, maximum=1.0)
        )
        self.assertEqual(
            resolve_optional_number(
                {"cas_amount": 0}, "cas_amount", minimum=0.0, maximum=1.0
            ),
            0.0,
        )
        self.assertEqual(
            resolve_optional_number(
                {"cas_amount": 1.0}, "cas_amount", minimum=0.0, maximum=1.0
            ),
            1.0,
        )

    def test_rejects_bool_string_non_finite_and_out_of_bounds(self) -> None:
        for value in (False, "0.5", None, math.nan, math.inf, -0.01, 1.01):
            with self.subTest(value=value), self.assertRaises(ValueError):
                resolve_optional_number(
                    {"cas_amount": value},
                    "cas_amount",
                    minimum=0.0,
                    maximum=1.0,
                )


if __name__ == "__main__":
    unittest.main()
