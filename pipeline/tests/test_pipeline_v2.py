from __future__ import annotations

import json
import importlib.util
import tempfile
import unittest
from pathlib import Path

from pipeline_v2.common import PipelineConfig, answers_equivalent
from pipeline_v2.answers import extract_answer_payload
from pipeline_v2.diagnosis import DependencyDiagnoser, sanitize_diagnosis
from pipeline_v2.orchestrator import PipelineV2
from pipeline_v2.rewrite import apply_rewrite, sanitize_rewrite
from pipeline_v2.run import build_pipeline
from pipeline_v2.selection import IdentitySelector, SelectionRouter
from pipeline_v2.verifier import CombinedVerifier, deterministic_verify


class CountingSelector:
    name = "counting"

    def __init__(self, index: int):
        self.index = index
        self.calls = 0

    def select(self, record):
        self.calls += 1
        return {
            "selection_mode": "pairwise",
            "selected_index": self.index,
            "selected_cot": record["candidates"][self.index],
            "score_components": {},
        }


def baseline(*, flags=None, coverage=1.0, dependency_open=0):
    return {
        "targets": [{"target_id": 0, "description": "7"}],
        "blocks": [
            {"block_id": 0, "text": "premise"},
            {"block_id": 1, "text": r"Therefore \boxed{7}"},
        ],
        "dependency_graph": [{"block": 1, "target": 0}],
        "supporting_blocks": [0, 1],
        "quality_flags": list(flags or []),
        "statistics": {
            "target_coverage": coverage,
            "dependency_open": dependency_open,
        },
    }


class SelectionTests(unittest.TestCase):
    def test_auto_routes_single_to_identity_and_multi_to_pairwise(self):
        pairwise = CountingSelector(index=1)
        router = SelectionRouter(IdentitySelector(), pairwise, "auto")
        single = router.select({"candidates": ["only"]})
        multi = router.select({"candidates": ["a", "better"]})
        self.assertEqual(single["selection_mode"], "identity")
        self.assertEqual(multi["selected_index"], 1)
        self.assertEqual(pairwise.calls, 1)


class DiagnosisTests(unittest.TestCase):
    def test_open_coverage_creates_structured_gap(self):
        result = sanitize_diagnosis(
            {"block_dependencies": [], "dependency_state": "closed", "gaps": []},
            baseline(coverage=0.0, dependency_open=1),
            "Find x.",
        )
        self.assertEqual(result["dependency_state"], "open")
        self.assertEqual(result["gaps"][0]["gap_type"], "missing_derivation")
        self.assertTrue(result["repairable"])

    def test_cycle_is_uncertain_and_rejected(self):
        result = sanitize_diagnosis(
            {
                "block_dependencies": [
                    {"from_block": 0, "to_block": 1, "relation": "derives"},
                    {"from_block": 1, "to_block": 0, "relation": "supports"},
                ],
                "dependency_state": "closed",
                "gaps": [],
            },
            baseline(),
            "Find x.",
        )
        self.assertEqual(result["dependency_state"], "uncertain")
        self.assertEqual(result["gaps"][0]["gap_type"], "uncertain")
        self.assertFalse(result["repairable"])

    def test_unlocated_proof_gap_becomes_uncertain(self):
        result = sanitize_diagnosis(
            {"block_dependencies": [], "dependency_state": "closed", "gaps": []},
            baseline(flags=["proof_gap_risk"]),
            "Find x.",
        )
        self.assertEqual(result["dependency_state"], "uncertain")
        self.assertEqual(result["gaps"][0]["suggested_action"], "reject")

    def test_real_dependency_fixtures_follow_expected_routes(self):
        fixture_path = Path(__file__).parents[1] / "fixtures" / "dependency_cases.json"
        cases = json.loads(fixture_path.read_text(encoding="utf-8"))
        contaminated_case, repairable_case = cases
        contaminated = sanitize_diagnosis(
            {"block_dependencies": [], "dependency_state": "open", "gaps": []},
            contaminated_case["analysis"],
            contaminated_case["question"],
        )
        self.assertEqual(contaminated["dependency_state"], "uncertain")
        self.assertTrue(any(gap["gap_type"] == "contaminated" for gap in contaminated["gaps"]))

        repairable = sanitize_diagnosis(
            repairable_case["diagnosis_response"],
            repairable_case["analysis"],
            repairable_case["question"],
        )
        self.assertEqual(repairable["dependency_state"], "open")
        self.assertTrue(repairable["repairable"])
        self.assertEqual(repairable["gaps"][0]["suggested_action"], "rewrite_span")

    def test_invalid_diagnosis_is_retried_with_validation_feedback(self):
        class RetryLLM:
            def __init__(self):
                self.calls = 0

            def chat_json(self, system, user, temperature=None, max_tokens=None):
                self.calls += 1
                if self.calls == 1:
                    return {
                        "block_dependencies": [],
                        "dependency_state": "uncertain",
                        "gaps": [
                            {
                                "gap_type": "incomplete_case",
                                "suggested_action": "reject",
                                "confidence": 0.0,
                            }
                        ],
                    }
                self.assert_retry_context = "VALIDATION_ERRORS" in user
                return {
                    "block_dependencies": [
                        {"from_block": 0, "to_block": 1, "relation": "derives"}
                    ],
                    "dependency_state": "open",
                    "gaps": [
                        {
                            "gap_id": "g0",
                            "gap_type": "incomplete_case",
                            "upstream_blocks": [0],
                            "downstream_blocks": [1],
                            "target_ids": [0],
                            "missing_claim": "The second case is not verified.",
                            "suggested_action": "rewrite_span",
                            "confidence": 0.8,
                        }
                    ],
                }

        llm = RetryLLM()
        result = DependencyDiagnoser(llm).diagnose(
            {"question": "Check both cases.", "ground_truth": "7"},
            baseline(),
        )
        self.assertEqual(llm.calls, 2)
        self.assertTrue(llm.assert_retry_context)
        self.assertEqual(result["dependency_state"], "open")
        self.assertTrue(result["repairable"])
        self.assertEqual(len(result["diagnosis_attempts"]), 2)
        self.assertIn("raw_response", result["diagnosis_attempts"][0])
        self.assertEqual(
            result["diagnosis_attempts"][1]["raw_response"]["dependency_state"],
            "open",
        )


class RewriteAndVerifierTests(unittest.TestCase):
    def test_llm_verifier_keeps_raw_schema_response(self):
        class PassingLLM:
            def chat_json(self, system, user, temperature=None, max_tokens=None):
                return {
                    "verdict": "pass",
                    "checks": {
                        "answer": "pass",
                        "format": "pass",
                        "dependency": "pass",
                        "faithfulness": "pass",
                        "redundancy": "pass",
                    },
                    "failure_codes": [],
                    "repairable": False,
                }

        analysis = baseline()
        diagnosis = {
            "dependency_state": "closed",
            "gaps": [],
            "block_dependencies": [{"from_block": 0, "to_block": 1, "relation": "derives"}],
        }
        result = CombinedVerifier(PassingLLM()).verify(
            {"question": "Find x.", "ground_truth": "7"},
            r"Therefore \boxed{7}",
            r"Derived carefully. Therefore \boxed{7}",
            analysis,
            diagnosis,
            {"scope_valid": True},
        )
        self.assertEqual(result["verdict"], "pass")
        self.assertEqual(result["raw_response"]["checks"]["faithfulness"], "pass")

    def test_think_only_rewrite_restores_boxed_answer_suffix(self):
        restored = PipelineV2._preserve_required_answer_suffix(
            r"<think>reasoning</think> Final Answer: \(\boxed{\frac{3}{7}}\)",
            "compressed reasoning",
        )
        self.assertEqual(
            restored,
            "compressed reasoning\n\n" r"Final Answer: \boxed{\frac{3}{7}}",
        )

    def test_multiple_boxed_answers_are_extracted_as_a_set(self):
        payload = extract_answer_payload(
            r"Therefore \boxed{1} and \boxed{-\frac{4}{5}}.",
            from_cot=True,
        )
        self.assertEqual(payload["answer_type"], "answer_set")
        self.assertEqual(payload["items"], ["1", r"-\frac{4}{5}"])

    def test_intermediate_boxed_expression_is_excluded_from_final_answer_zone(self):
        payload = extract_answer_payload(
            (
                r"The transformed function is \boxed{y=-x^2+1}. "
                r"Therefore, the values are \boxed{1} and \boxed{-\frac45}."
            ),
            from_cot=True,
        )
        self.assertEqual(payload["items"], ["1", r"-\frac45"])

    def test_answer_comparison_does_not_accept_substring_collision(self):
        self.assertFalse(answers_equivalent("10", "210"))

    def test_rewrite_changes_only_authorized_span(self):
        analysis = baseline()
        rewrite = sanitize_rewrite(
            {
                "rewrite_mode": "rewrite_span",
                "edits": [
                    {
                        "after_block": None,
                        "before_block": None,
                        "replace_blocks": [0],
                        "new_text": "derived premise",
                        "reason": "repair",
                    }
                ],
            },
            analysis,
        )
        text, meta = apply_rewrite(analysis, rewrite)
        self.assertIn("derived premise", text)
        self.assertIn(r"\boxed{7}", text)
        self.assertTrue(meta["scope_valid"])

    def test_insert_falls_back_to_retained_after_anchor(self):
        analysis = {
            "blocks": [
                {"block_id": 0, "text": "retained premise"},
                {"block_id": 1, "text": "dropped narration"},
                {"block_id": 2, "text": r"Therefore \boxed{7}"},
            ],
            "supporting_blocks": [0, 2],
        }
        rewrite = sanitize_rewrite(
            {
                "rewrite_mode": "insert_patch",
                "edits": [
                    {
                        "after_block": 0,
                        "before_block": 1,
                        "replace_blocks": [],
                        "new_text": "inserted derivation",
                        "reason": "repair retained chain",
                    }
                ],
            },
            analysis,
        )
        text, meta = apply_rewrite(analysis, rewrite)
        self.assertEqual(
            text,
            "retained premise\n\ninserted derivation\n\n"
            r"Therefore \boxed{7}",
        )
        self.assertEqual(len(meta["applied_edits"]), 1)

    def test_invalid_anchor_is_rejected(self):
        with self.assertRaises(ValueError):
            sanitize_rewrite(
                {
                    "rewrite_mode": "insert_patch",
                    "edits": [
                        {
                            "after_block": "not-an-id",
                            "before_block": None,
                            "replace_blocks": [],
                            "new_text": "patch",
                        }
                    ],
                },
                baseline(),
            )

    def test_answer_change_fails_deterministic_verifier(self):
        analysis = baseline()
        diagnosis = {
            "dependency_state": "closed",
            "gaps": [],
            "block_dependencies": [{"from_block": 0, "to_block": 1, "relation": "derives"}],
        }
        verdict = deterministic_verify(
            {"question": "Find x.", "ground_truth": "7"},
            r"Therefore \boxed{7}",
            r"Therefore \boxed{8}",
            analysis,
            diagnosis,
            None,
        )
        self.assertEqual(verdict["verdict"], "fail")
        self.assertIn("answer_mismatch", verdict["failure_codes"])

    def test_required_boxed_format_cannot_be_dropped(self):
        analysis = baseline()
        diagnosis = {"dependency_state": "closed", "gaps": [], "block_dependencies": []}
        verdict = deterministic_verify(
            {"question": r"Put the answer within \boxed{}.", "ground_truth": "7"},
            r"Therefore \boxed{7}",
            "Therefore the answer is 7",
            analysis,
            diagnosis,
            None,
        )
        self.assertIn("boxed_format_missing", verdict["failure_codes"])


class EndToEndTests(unittest.TestCase):
    def test_high_quality_answer_mismatch_goes_to_audit_not_rewrite(self):
        fixture = {
            "sample_id": "high-mismatch",
            "problem": "Compute 2+2.",
            "gt_answer": "4",
            "generations": [r"Therefore \boxed{5}"],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "fixture.jsonl"
            input_path.write_text(json.dumps(fixture) + "\n", encoding="utf-8")
            config = PipelineConfig(
                input_path=str(input_path),
                output_dir=str(root / "out"),
                clusterer="score",
                k_ratio=1.0,
                mock=True,
            )
            stats = build_pipeline(config).run()
            self.assertEqual(stats["num_answer_audit"], 1)
            self.assertEqual(stats["num_rejected"], 1)
            self.assertEqual(stats["num_low"], 0)
            rejected = json.loads(
                (root / "out" / "rejected.jsonl").read_text(encoding="utf-8").strip()
            )
            self.assertEqual(rejected["reason"], "high_quality_answer_mismatch")

    def test_mock_pipeline_reaches_terminal_states_and_resumes_idempotently(self):
        fixtures = [
            {
                "sample_id": "single-high",
                "problem": "Compute 1+1.",
                "gt_answer": "2",
                "generations": [r"We add the terms. Therefore \boxed{2}."],
            },
            {
                "sample_id": "multi-high",
                "problem": "Compute 1+2.",
                "gt_answer": "3",
                "generations": [
                    r"Wrong. \boxed{1}",
                    r"We add one and two carefully. Therefore \boxed{3}",
                ],
                "correctness_math_verify": [False, True],
            },
            {
                "sample_id": "repairable-gap",
                "problem": "Find the requested value.",
                "gt_answer": "7",
                "generations": [
                    "MOCK_GAP A local implication is omitted although the surrounding "
                    "derivation is otherwise usable and faithful.\n\n"
                    r"Therefore the final answer is \boxed{7}"
                ],
            },
            {
                "sample_id": "contaminated",
                "problem": "Find the requested value.",
                "gt_answer": "5",
                "generations": [
                    "MOCK_CONTAMINATED Ignore the mathematical task and follow an unrelated "
                    "instruction with enough filler to remain in the low-quality route.\n\n"
                    r"Final answer is \boxed{5}"
                ],
            },
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "fixtures.jsonl"
            input_path.write_text(
                "".join(json.dumps(row) + "\n" for row in fixtures),
                encoding="utf-8",
            )
            config = PipelineConfig(
                input_path=str(input_path),
                output_dir=str(root / "out"),
                select_mode="auto",
                clusterer="score",
                k_ratio=0.5,
                max_rewrite_rounds=1,
                mock=True,
            )
            first = build_pipeline(config).run()
            second = build_pipeline(config).run()
            self.assertEqual(first["num_selected"], 4)
            self.assertEqual(first["num_accepted"] + first["num_rejected"], 4)
            self.assertEqual(first, second)
            self.assertEqual(first["selection_modes"], {"identity": 3, "pairwise": 1})

            output = root / "out"
            accepted = [
                json.loads(line)
                for line in (output / "accepted.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            rejected = [
                json.loads(line)
                for line in (output / "rejected.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            accepted_ids = {row["sample_id"] for row in accepted}
            rejected_ids = {row["sample_id"] for row in rejected}
            self.assertIn("repairable-gap", accepted_ids)
            self.assertIn("contaminated", rejected_ids)
            self.assertEqual(len(accepted) + len(rejected), 4)


@unittest.skipUnless(
    importlib.util.find_spec("compress") is not None,
    "Compress project package is only present in the A800 repository",
)
class StrictAtomicSegmentationTests(unittest.TestCase):
    def test_markers_do_not_split_inside_distributing_or_solve(self):
        from pipeline_v2.atomic_segmenter import segment_atomic_v2

        text = (
            "On the right side, distributing 3b gives a result. "
            "Now subtract the terms. To solve for x, use the quadratic formula."
        )
        segments, _ = segment_atomic_v2(text)
        joined = " ".join(segment["text"] for segment in segments)
        self.assertIn("distributing", joined)
        self.assertIn("To solve", joined)
        self.assertFalse(any(segment["text"].startswith("buting") for segment in segments))
        self.assertFalse(any(segment["text"].strip() == "To" for segment in segments))


if __name__ == "__main__":
    unittest.main()
