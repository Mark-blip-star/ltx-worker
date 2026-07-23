from __future__ import annotations

import math
import unittest

from request_config import (
    DEFAULT_DECODE_NOISE_SCALE,
    MAX_DECODE_NOISE_SCALE,
    MAX_NEGATIVE_PROMPT_CHARS,
    conditioning_strength_tag_suffix,
    resolve_boolean,
    resolve_decode_noise,
    resolve_negative_prompt,
    resolve_optional_number,
    resolve_optional_step_count,
    resolve_stage_conditioning_strengths,
    resolve_stage2_sigmas,
    sampling_schedule_tag_suffix,
    stage_conditioning_strength_tag_suffix,
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

    def test_conditioning_strength_accepts_only_a_finite_unit_interval(self) -> None:
        self.assertIsNone(
            resolve_optional_number(
                {},
                "conditioning_strength",
                minimum=0.0,
                maximum=1.0,
            )
        )
        for value in (0, 0.8, 0.95, 1.0):
            with self.subTest(value=value):
                self.assertEqual(
                    resolve_optional_number(
                        {"conditioning_strength": value},
                        "conditioning_strength",
                        minimum=0.0,
                        maximum=1.0,
                    ),
                    float(value),
                )
        for value in (False, "1.0", None, math.nan, math.inf, -0.01, 1.01):
            with self.subTest(value=value), self.assertRaises(ValueError):
                resolve_optional_number(
                    {"conditioning_strength": value},
                    "conditioning_strength",
                    minimum=0.0,
                    maximum=1.0,
                )
        with self.assertRaisesRegex(ValueError, "must be a finite number"):
            resolve_optional_number(
                {"conditioning_strength": 10**10000},
                "conditioning_strength",
                minimum=0.0,
                maximum=1.0,
            )

    def test_conditioning_strength_runtime_tag_is_exact_and_arm_distinct(self) -> None:
        self.assertEqual(conditioning_strength_tag_suffix(None), "")
        self.assertEqual(conditioning_strength_tag_suffix(0.8), "-ic0_8")
        self.assertEqual(conditioning_strength_tag_suffix(1.0), "-ic1_0")
        self.assertNotEqual(
            conditioning_strength_tag_suffix(0.8000001),
            conditioning_strength_tag_suffix(0.8),
        )


class StageConditioningStrengthTests(unittest.TestCase):
    def test_omitted_pair_preserves_legacy_path(self) -> None:
        self.assertIsNone(resolve_stage_conditioning_strengths({}))
        self.assertEqual(stage_conditioning_strength_tag_suffix(None), "")

    def test_accepts_exact_stage_pair_and_builds_distinct_tag(self) -> None:
        values = resolve_stage_conditioning_strengths(
            {
                "conditioning_strength_stage1": 1.0,
                "conditioning_strength_stage2": 0.8,
            }
        )
        self.assertEqual(values, (1.0, 0.8))
        self.assertEqual(
            stage_conditioning_strength_tag_suffix(values),
            "-ics1_1_0-s2_0_8",
        )
        self.assertNotEqual(
            stage_conditioning_strength_tag_suffix((1.0, 0.8)),
            stage_conditioning_strength_tag_suffix((0.8, 1.0)),
        )

    def test_requires_both_fields_and_forbids_legacy_ambiguity(self) -> None:
        for payload in (
            {"conditioning_strength_stage1": 1.0},
            {"conditioning_strength_stage2": 0.8},
            {
                "conditioning_strength": 0.8,
                "conditioning_strength_stage1": 1.0,
                "conditioning_strength_stage2": 0.8,
            },
        ):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                resolve_stage_conditioning_strengths(payload)

    def test_rejects_invalid_stage_values(self) -> None:
        for value in (False, "1.0", None, math.nan, math.inf, -0.01, 1.01):
            with self.subTest(value=value), self.assertRaises(ValueError):
                resolve_stage_conditioning_strengths(
                    {
                        "conditioning_strength_stage1": value,
                        "conditioning_strength_stage2": 0.8,
                    }
                )


class SamplingScheduleTests(unittest.TestCase):
    candidate_stage2 = (
        0.909375,
        0.77109375,
        0.5734375,
        0.31640625,
        0.0,
    )

    def test_omitted_schedule_levers_preserve_tier_defaults(self) -> None:
        self.assertIsNone(resolve_optional_step_count({}))
        self.assertIsNone(resolve_stage2_sigmas({}))

    def test_quality_15_plus_4_candidate_is_accepted_exactly(self) -> None:
        payload = {
            "tier": "quality",
            "steps": 15,
            "stage2_sigmas": list(self.candidate_stage2),
        }
        self.assertEqual(resolve_optional_step_count(payload), 15)
        self.assertEqual(
            resolve_stage2_sigmas(payload),
            self.candidate_stage2,
        )
        self.assertEqual(
            15 + len(resolve_stage2_sigmas(payload)) - 1,
            19,
        )

    def test_stage2_rejects_nan_inf_and_non_numeric_values(self) -> None:
        for invalid in (math.nan, math.inf, -math.inf, True, "0.5", None):
            values = [0.909375, 0.5, invalid, 0.0]
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                resolve_stage2_sigmas({"stage2_sigmas": values})

    def test_stage2_rejects_out_of_range_non_descending_and_nonzero_tail(self) -> None:
        invalid_grids = (
            [1.01, 0.5, 0.0],
            [0.9, -0.01, 0.0],
            [0.9, 0.5, 0.5, 0.0],
            [0.9, 0.5, 0.6, 0.0],
            [0.9, 0.5, 0.001],
            [0.0],
            [1.0, 0.9] + [0.8] * 15 + [0.0],
        )
        for values in invalid_grids:
            with self.subTest(values=values), self.assertRaises(ValueError):
                resolve_stage2_sigmas({"stage2_sigmas": values})

    def test_stage2_requires_a_json_list(self) -> None:
        for invalid in (None, "0.9,0", (0.9, 0.0), {"0": 0.9}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                resolve_stage2_sigmas({"stage2_sigmas": invalid})

    def test_stage1_steps_are_bounded_integer_not_bool(self) -> None:
        self.assertEqual(resolve_optional_step_count({"steps": 15}), 15)
        for invalid in (True, 15.0, "15", 0, 65, math.nan):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                resolve_optional_step_count({"steps": invalid})

    def test_tag_identifies_allocation_and_exact_sigma_grid(self) -> None:
        candidate = sampling_schedule_tag_suffix(15, self.candidate_stage2)
        same_count_other_grid = sampling_schedule_tag_suffix(
            15,
            (0.909375, 0.75, 0.55, 0.3, 0.0),
        )
        other_allocation = sampling_schedule_tag_suffix(16, self.candidate_stage2)
        self.assertRegex(candidate, r"^-s1st15-s2st4-s2h[0-9a-f]{12}$")
        self.assertNotEqual(candidate, same_count_other_grid)
        self.assertNotEqual(candidate, other_allocation)


if __name__ == "__main__":
    unittest.main()
