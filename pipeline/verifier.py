from __future__ import annotations

from typing import Any, Dict, List, Optional

from .common import JsonLLM, answers_equivalent, final_answer, is_contaminated


CHECK_NAMES = ("answer", "format", "dependency", "faithfulness", "redundancy")


def deterministic_verify(
    record: Dict[str, Any],
    original_cot: str,
    candidate_cot: str,
    analysis: Dict[str, Any],
    diagnosis: Dict[str, Any],
    rewrite_meta: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    failures: List[str] = []
    reference = (
        record.get("ground_truth")
        or record.get("original_final_answer")
        or final_answer(original_cot)
    )
    predicted = final_answer(candidate_cot)
    if not candidate_cot.strip():
        failures.append("empty_output")
    if not reference or not predicted:
        failures.append("answer_unavailable")
    elif not answers_equivalent(reference, predicted):
        failures.append("answer_mismatch")
    requires_boxed = (
        r"\boxed{" in original_cot
        or r"\boxed{" in record.get("question", "")
        or "within boxed" in record.get("question", "").lower()
    )
    if requires_boxed and r"\boxed{" not in candidate_cot:
        failures.append("boxed_format_missing")
    if rewrite_meta is not None and not bool(rewrite_meta.get("scope_valid")):
        failures.append("rewrite_scope_violation")
    stats = analysis.get("statistics", {})
    if diagnosis.get("dependency_state") != "closed":
        failures.append("dependency_not_closed")
    if float(stats.get("target_coverage", 0.0) or 0.0) < 1.0:
        failures.append("target_uncovered")
    if int(stats.get("dependency_open", 0) or 0) > 0:
        failures.append("dependency_open")
    if diagnosis.get("gaps"):
        failures.append("unresolved_gap")
    if "proof_gap_risk" in analysis.get("quality_flags", []):
        failures.append("baseline_proof_gap")
    if is_contaminated(record.get("question", ""), candidate_cot):
        failures.append("contaminated")

    return {
        "verdict": "pass" if not failures else "fail",
        "checks": {
            "answer": "fail" if any(code in failures for code in ("answer_unavailable", "answer_mismatch")) else "pass",
            "format": "fail" if "boxed_format_missing" in failures else "pass",
            "dependency": "fail" if any(
                code in failures
                for code in ("dependency_not_closed", "target_uncovered", "dependency_open", "unresolved_gap", "baseline_proof_gap")
            ) else "pass",
            "faithfulness": "fail" if any(code in failures for code in ("rewrite_scope_violation", "contaminated")) else "pass",
            "redundancy": "pass",
        },
        "failure_codes": failures,
        "repairable": bool(failures) and all(
            code in {"dependency_not_closed", "unresolved_gap", "baseline_proof_gap", "boxed_format_missing"}
            for code in failures
        ),
        "source": "deterministic",
    }


class CombinedVerifier:
    def __init__(self, llm: JsonLLM):
        self.llm = llm

    def verify(
        self,
        record: Dict[str, Any],
        original_cot: str,
        candidate_cot: str,
        analysis: Dict[str, Any],
        diagnosis: Dict[str, Any],
        rewrite_meta: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        deterministic = deterministic_verify(
            record,
            original_cot,
            candidate_cot,
            analysis,
            diagnosis,
            rewrite_meta,
        )
        if deterministic["verdict"] != "pass":
            return deterministic
        response = self.llm.chat_json(
            system=(
                "Verify a refined mathematical Chain-of-Thought. Judge logical closure and "
                "faithfulness only; deterministic answer, format, and dependency checks already passed."
            ),
            user=(
                f"QUESTION:\n{record['question']}\n\n"
                f"GROUND_TRUTH:\n{record.get('ground_truth') or '(unknown)'}\n\n"
                f"ORIGINAL_COT:\n{original_cot}\n\n"
                f"CANDIDATE_COT:\n{candidate_cot}\n\n"
                f"BLOCK_DEPENDENCIES:\n{diagnosis.get('block_dependencies', [])}\n\n"
                "Return exactly this JSON shape:\n"
                "{\n"
                '  "verdict": "pass|fail|uncertain",\n'
                '  "checks": {\n'
                '    "answer": "pass|fail",\n'
                '    "format": "pass|fail",\n'
                '    "dependency": "pass|fail",\n'
                '    "faithfulness": "pass|fail",\n'
                '    "redundancy": "pass|fail"\n'
                "  },\n"
                '  "failure_codes": [],\n'
                '  "repairable": false\n'
                "}\n"
                "All five check keys are mandatory and every value must be the literal string "
                "pass or fail. Deterministic answer, format, and structural dependency checks "
                "have already passed; mark them fail only if semantic inspection finds a real "
                "problem. If verdict is pass, all five checks must be pass and failure_codes "
                "must be empty."
            ),
            temperature=0.0,
            max_tokens=1024,
        )
        verdict = str(response.get("verdict", "uncertain"))
        if verdict not in {"pass", "fail", "uncertain"}:
            verdict = "uncertain"
        checks = {
            name: "pass" if str(response.get("checks", {}).get(name, "fail")) == "pass" else "fail"
            for name in CHECK_NAMES
        }
        if verdict == "pass" and any(value != "pass" for value in checks.values()):
            verdict = "fail"
        return {
            "verdict": verdict,
            "checks": checks,
            "failure_codes": [str(code) for code in response.get("failure_codes", [])],
            "repairable": bool(response.get("repairable", False)),
            "source": "deterministic+llm",
            "raw_response": response,
        }
