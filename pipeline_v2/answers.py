from __future__ import annotations

import re
from typing import Any, Dict, List, Protocol

from .common import JsonLLM, canonical_answer, extract_boxed_values, final_answer


ANSWER_TYPES = {
    "single_value",
    "single_choice",
    "answer_set",
    "ordered_tuple",
    "interval",
    "expression",
    "equation",
    "multi_target",
    "text_answer",
    "unknown",
}


class AnswerJudge(Protocol):
    def compare(self, record: Dict[str, Any], cot: str) -> Dict[str, Any]: ...


def _unique(values: List[str]) -> List[str]:
    out: List[str] = []
    for value in values:
        cleaned = value.strip()
        if cleaned and cleaned not in out:
            out.append(cleaned)
    return out


def classify_answer(raw: str, items: List[str]) -> str:
    text = raw.strip()
    if len(items) > 1:
        return "answer_set"
    if re.fullmatch(r"[\(\[]\s*[^,]+,\s*[^,]+[\)\]]", text):
        return "ordered_tuple"
    if re.search(r"(?:\\cup|∪|[\(\[][^,]+,[^,]+[\)\]])", text):
        return "interval"
    if re.fullmatch(r"\(?[A-H]\)?", text.strip(), flags=re.IGNORECASE):
        return "single_choice"
    if "=" in text:
        alternatives = re.split(r"\bor\b|或|或者", text, flags=re.IGNORECASE)
        return "answer_set" if len([part for part in alternatives if part.strip()]) > 1 else "equation"
    if any(token in text for token in (r"\frac", r"\sqrt", "^", "+", "-", "*", "/")):
        return "expression"
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", canonical_answer(text)):
        return "single_value"
    return "text_answer" if text else "unknown"


def extract_answer_payload(raw: str, *, from_cot: bool) -> Dict[str, Any]:
    scoped_raw = raw
    if from_cot:
        cue_pattern = re.compile(
            r"(?:final\s+answer|therefore|thus|hence|"
            r"the\s+values?\s+of.+?\s+are|答案|所以|故)",
            flags=re.IGNORECASE,
        )
        cues = list(cue_pattern.finditer(raw))
        if cues:
            candidate_scope = raw[cues[-1].start() :]
            if extract_boxed_values(candidate_scope):
                scoped_raw = candidate_scope
    else:
        scoped_raw = re.sub(
            r"\\text\s*\{\s*(or|and)\s*\}",
            r" \1 ",
            raw,
            flags=re.IGNORECASE,
        )

    boxes = extract_boxed_values(scoped_raw)
    if boxes:
        items = _unique(boxes)
        extracted = items[-1] if len(items) == 1 else "{" + ", ".join(items) + "}"
    else:
        extracted = final_answer(scoped_raw) if from_cot else scoped_raw.strip()
        alternatives = re.split(
            r"\s+(?:or|and)\s+|(?:或|或者)|\s*[,;]\s*",
            extracted,
            flags=re.IGNORECASE,
        )
        items = _unique([part for part in alternatives if part.strip()])
        if len(items) <= 1:
            items = [extracted] if extracted else []
    answer_type = classify_answer(extracted, items)
    return {
        "answer_type": answer_type,
        "raw": extracted if from_cot else raw[:2000],
        "extracted": extracted,
        "items": items,
        "normalized_items": [canonical_answer(item) for item in items],
    }


def reference_answer(record: Dict[str, Any], selected_index: int) -> str:
    if record.get("ground_truth"):
        return str(record["ground_truth"])
    originals = record.get("original_final_answers", [])
    if isinstance(originals, list) and selected_index < len(originals):
        return str(originals[selected_index])
    return str(record.get("original_final_answer", ""))


def _upstream_label(record: Dict[str, Any], selected_index: int) -> str:
    source = record.get("source_record", {})
    for field in ("is_reasoning_complete", "correctness_math_verify"):
        values = source.get(field)
        if isinstance(values, list) and selected_index < len(values):
            return "pass" if bool(values[selected_index]) else "fail"
    return "unknown"


def typed_exact_equivalent(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
    left_items = {item for item in left.get("normalized_items", []) if item}
    right_items = {item for item in right.get("normalized_items", []) if item}
    return bool(left_items) and left_items == right_items


class LLMAnswerJudge:
    def __init__(self, llm: JsonLLM):
        self.llm = llm

    def compare(self, record: Dict[str, Any], cot: str) -> Dict[str, Any]:
        selected_index = int(record.get("selected_index", 0))
        candidate = extract_answer_payload(cot, from_cot=True)
        reference_raw = reference_answer(record, selected_index)
        reference = extract_answer_payload(reference_raw, from_cot=False)
        upstream = _upstream_label(record, selected_index)
        if upstream == "fail":
            return {
                "verdict": "not_equivalent",
                "answer_type": candidate["answer_type"],
                "reference": reference,
                "candidate": candidate,
                "reason": "Upstream correctness label is false.",
                "confidence": 1.0,
                "source": "upstream_veto",
            }
        if not reference["items"] or not candidate["items"]:
            return {
                "verdict": "uncertain",
                "answer_type": candidate["answer_type"],
                "reference": reference,
                "candidate": candidate,
                "reason": "Reference or candidate answer could not be extracted.",
                "confidence": 0.0,
                "source": "extraction_guard",
            }

        response = self.llm.chat_json(
            system=(
                "Compare a dataset ground-truth answer with an extracted candidate answer. "
                "Judge semantic mathematical equivalence, including choices, unordered answer "
                "sets, equations, expressions, intervals, and multiple targets. Do not solve or "
                "repair the Chain-of-Thought. Return only the requested JSON."
            ),
            user=(
                f"QUESTION:\n{record['question']}\n\n"
                f"GROUND_TRUTH:\n{reference}\n\n"
                f"CANDIDATE_ANSWER:\n{candidate}\n\n"
                "Return verdict (equivalent, not_equivalent, or uncertain), answer_type, "
                "normalized_gt, normalized_candidate, reason, and confidence in [0,1]."
            ),
            temperature=0.0,
            max_tokens=768,
        )
        verdict = str(response.get("verdict", "uncertain"))
        if verdict not in {"equivalent", "not_equivalent", "uncertain"}:
            verdict = "uncertain"
        try:
            confidence = max(0.0, min(1.0, float(response.get("confidence", 0.0))))
        except (TypeError, ValueError):
            confidence = 0.0
        answer_type = str(response.get("answer_type", candidate["answer_type"]))
        if answer_type not in ANSWER_TYPES:
            answer_type = candidate["answer_type"]
        return {
            "verdict": verdict,
            "answer_type": answer_type,
            "reference": reference,
            "candidate": candidate,
            "normalized_gt": response.get("normalized_gt"),
            "normalized_candidate": response.get("normalized_candidate"),
            "reason": str(response.get("reason", ""))[:1000],
            "confidence": confidence,
            "source": "llm_semantic_judge",
        }


class MockAnswerJudge:
    def compare(self, record: Dict[str, Any], cot: str) -> Dict[str, Any]:
        selected_index = int(record.get("selected_index", 0))
        reference = extract_answer_payload(
            reference_answer(record, selected_index),
            from_cot=False,
        )
        candidate = extract_answer_payload(cot, from_cot=True)
        verdict = (
            "equivalent"
            if typed_exact_equivalent(reference, candidate)
            else "not_equivalent"
        )
        return {
            "verdict": verdict,
            "answer_type": candidate["answer_type"],
            "reference": reference,
            "candidate": candidate,
            "reason": "mock typed comparison",
            "confidence": 1.0,
            "source": "mock",
        }


__all__ = [
    "AnswerJudge",
    "LLMAnswerJudge",
    "MockAnswerJudge",
    "extract_answer_payload",
    "typed_exact_equivalent",
]
