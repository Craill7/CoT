#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V4 joint two-turn compression pipeline.

Turn 1 asks the LLM for dependency-aware segmentation. Turn 2 continues the
same conversation and labels those blocks with fine-grained dependency labels.
V3 fast_label.py is intentionally left untouched as the baseline.
"""

from __future__ import annotations

import argparse
import json
import logging
from difflib import SequenceMatcher
import re
import sys
import time
import traceback
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from .client import LLMClient
    from .utils import (
        canonical_target,
        clean_target_text,
        equivalent_targets,
        extract_cot_text,
        find_matching_brace,
        flatten_text,
        iter_jsonl,
        load_json,
        sample_id_from_record,
        split_sentences,
        write_json,
    )
except ImportError:  # pragma: no cover - allows direct script execution.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from client import LLMClient
    from utils import (
        canonical_target,
        clean_target_text,
        equivalent_targets,
        extract_cot_text,
        find_matching_brace,
        flatten_text,
        iter_jsonl,
        load_json,
        sample_id_from_record,
        split_sentences,
        write_json,
    )


PIPELINE_VERSION = "v4_joint_two_turn"

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"

VALID_BLOCK_TYPES = {
    "setup",
    "derivation",
    "verification",
    "correction",
    "final",
    "narration",
}

VALID_LABEL_SUBTYPES = {
    "DERIVATION_SUPPORT",
    "VALIDATION_SUPPORT",
    "FINAL_ANSWER",
    "REDUNDANT_VERIFICATION",
    "SELF_CORRECTION",
    "DEAD_END",
    "NARRATION",
    "REPEATED_FINAL",
    "NON_SUPPORTING",
}

KEEP_BY_DEFAULT = {
    "DERIVATION_SUPPORT": True,
    "VALIDATION_SUPPORT": True,
    "FINAL_ANSWER": True,
    "REDUNDANT_VERIFICATION": False,
    "SELF_CORRECTION": False,
    "DEAD_END": False,
    "NARRATION": False,
    "REPEATED_FINAL": False,
    "NON_SUPPORTING": False,
}

VALID_QUALITY_FLAGS = {
    "summary_only_risk",
    "proof_gap_risk",
    "repeated_final_risk",
    "dirty_trace_risk",
    "target_over_split_risk",
    "target_uncertain_risk",
    "json_repair_risk",
}

FALLBACK_TRIGGER_FLAGS = {
    "no_boxed",
    "malformed_boxed",
    "multiple_distinct_boxed",
    "boxed_not_near_final",
    "boxed_final_conflict",
    "answer_set_candidate",
    "subquestion_ambiguous",
    "low_confidence_final_answer",
}

FINAL_MARKERS = (
    "final answer",
    "the answer is",
    "therefore, the answer is",
    "therefore the answer is",
    "thus",
    "hence",
    "answer:",
)

FINAL_LINE_PATTERNS = (
    re.compile(r"(?:\*\*)?\s*final\s+answer\s*(?:\*\*)?\s*[:：]?\s*(.*)", re.IGNORECASE),
    re.compile(r"(?:therefore,\s*)?the\s+answer\s+is\s*(.*)", re.IGNORECASE),
    re.compile(r"(?:therefore|thus|hence)\s*,?\s*(?:the\s+answer\s+is\s*)?(.*)", re.IGNORECASE),
    re.compile(r"(?:答案|最终答案|所以|因此)\s*(?:是|为|:|：)?\s*(.*)"),
)


def clean_answer_text(text: Any) -> str:
    text = re.sub(r"</?think>", "", str(text), flags=re.IGNORECASE).strip()
    text = re.sub(r"^[:：=\s]+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" \t\r\n.。")


def canonical_answer(text: Any) -> str:
    normalized = clean_answer_text(text).lower()
    normalized = re.sub(r"\\(?:boxed|text|mathrm)\s*\{([^{}]*)\}", r"\1", normalized)
    normalized = normalized.replace("{", "").replace("}", "")
    normalized = re.sub(r"\\[a-zA-Z]+", "", normalized)
    normalized = re.sub(r"[\s$`'\".,;:，。；：]+", "", normalized)
    return normalized


def equivalent_answer(left: Any, right: Any) -> bool:
    left_key = canonical_answer(left)
    right_key = canonical_answer(right)
    if not left_key or not right_key:
        return False
    if left_key == right_key:
        return True
    if len(left_key) <= 2 or len(right_key) <= 2:
        return False
    return SequenceMatcher(None, left_key, right_key).ratio() >= 0.92


@dataclass
class JsonParseResult:
    data: Dict[str, Any]
    repaired: bool


class StageError(Exception):
    """Error with stage and raw-response context for failure_cases.jsonl."""

    def __init__(
        self,
        stage: str,
        error_type: str,
        message: str,
        raw_response: Optional[str] = None,
    ):
        super().__init__(message)
        self.stage = stage
        self.error_type = error_type
        self.raw_response = raw_response


def read_prompt(name: str) -> str:
    path = PROMPT_DIR / name
    return path.read_text(encoding="utf-8").strip()


def setup_logging(output_dir: Path, truncate: bool = False) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("joint_label")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    log_path = output_dir / f"{output_dir.name}.log"
    file_handler = logging.FileHandler(log_path, mode="w" if truncate else "a", encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    return logger


def json_dumps(obj: Any, pretty: bool = False) -> str:
    if pretty:
        return json.dumps(obj, ensure_ascii=False, indent=2)
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def record_preview(record: Dict[str, Any], max_chars: int = 1200) -> str:
    try:
        preview = json.dumps(record, ensure_ascii=False)
    except Exception:
        preview = str(record)
    return preview[:max_chars]


def load_records(path: Path) -> List[Tuple[int, Dict[str, Any]]]:
    if path.suffix == ".jsonl":
        return list(iter_jsonl(path))
    payload = load_json(path)
    rows = payload if isinstance(payload, list) else [payload]
    return [(idx + 1, row) for idx, row in enumerate(rows) if isinstance(row, dict)]


def extract_question(record: Dict[str, Any]) -> str:
    for key in ("question", "problem", "prompt", "input", "instruction"):
        if key in record:
            text = flatten_text(record[key]).strip()
            if text:
                return text
    messages = record.get("messages")
    if isinstance(messages, list):
        user_parts = []
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "user":
                text = flatten_text(msg.get("content")).strip()
                if text:
                    user_parts.append(text)
        if user_parts:
            return "\n".join(user_parts)
    return ""


def extract_answer_field(record: Dict[str, Any]) -> str:
    for key in ("answer", "gt_answer", "ground_truth", "final_answer", "label"):
        if key in record:
            text = flatten_text(record[key]).strip()
            if text:
                return text
    return ""


def make_failure(
    sample_id: str,
    stage: str,
    error_type: str,
    error_message: str,
    record: Dict[str, Any],
    raw_response: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "sample_id": sample_id,
        "stage": stage,
        "error_type": error_type,
        "error_message": error_message,
        "raw_response": raw_response,
        "record_preview": record_preview(record),
    }


def classify_exception(exc: Exception) -> str:
    text = str(exc).lower()
    if "timeout" in text:
        return "timeout"
    if "request failed" in text or "connection" in text or "http" in text or "api" in text:
        return "api_error"
    if "json" in text or isinstance(exc, json.JSONDecodeError):
        return "json_parse_error"
    return "unknown"


def find_balanced_json(text: str, opener: str = "{") -> Optional[str]:
    closer = "}" if opener == "{" else "]"
    start = text.find(opener)
    while start != -1:
        depth = 0
        in_string = False
        escape = False
        for idx in range(start, len(text)):
            ch = text[idx]
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    return text[start : idx + 1]
        start = text.find(opener, start + 1)
    return None


def escape_invalid_json_backslashes(text: str) -> str:
    # LLMs often copy LaTeX commands like \( or \frac inside JSON strings
    # without JSON-escaping the backslash. Add one slash when a run would form
    # an illegal escape, and also protect alphabetic LaTeX commands such as \frac.
    def repl(match: re.Match[str]) -> str:
        slashes = match.group(1)
        next_char = match.group(2)
        valid_json_escape = next_char in '"\\/bfnrtu'
        if len(slashes) % 2 == 1 and (not valid_json_escape or next_char.isalpha()):
            return slashes + '\\' + next_char
        return slashes + next_char

    return re.sub(r'(\\+)(.)', repl, text)


def simple_json_repair(candidate: str) -> str:
    repaired = candidate.strip()
    repaired = repaired.replace("\ufeff", "")
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    repaired = re.sub(r"```json\s*", "", repaired, flags=re.IGNORECASE)
    repaired = repaired.replace("```", "")
    repaired = escape_invalid_json_backslashes(repaired)
    return repaired.strip()


def parse_json_response(response: str, expected_key: Optional[str] = None) -> JsonParseResult:
    if response is None:
        raise ValueError("Empty LLM response.")
    text = response.strip()
    candidates: List[Tuple[str, bool]] = []
    candidates.append((text, False))

    for match in re.finditer(r"```(?:json)?\s*\n?(.*?)\n?```", text, flags=re.DOTALL | re.IGNORECASE):
        candidates.append((match.group(1).strip(), False))

    balanced = find_balanced_json(text, "{")
    if balanced:
        candidates.append((balanced, False))

    seen = set()
    last_error: Optional[Exception] = None
    for candidate, repaired in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        for payload, is_repaired in ((candidate, repaired), (simple_json_repair(candidate), True)):
            try:
                data = json.loads(payload)
                if not isinstance(data, dict):
                    raise ValueError("Parsed JSON is not an object.")
                if expected_key and expected_key not in data:
                    raise ValueError(f"Parsed JSON missing expected key: {expected_key}")
                return JsonParseResult(data=data, repaired=is_repaired)
            except Exception as exc:
                last_error = exc

    raise ValueError(f"Failed to parse JSON response: {last_error}")


def request_json_stage(
    client: LLMClient,
    messages: List[Dict[str, str]],
    expected_key: str,
    stage: str,
    temperature: float,
    max_tokens: int,
    parse_retries: int,
) -> Tuple[Dict[str, Any], str, bool]:
    raw_response = None
    last_error: Optional[Exception] = None
    attempts = max(1, parse_retries + 1)
    for attempt in range(attempts):
        try:
            raw_response = client._request(  # noqa: SLF001 - V4 needs raw multi-turn control.
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            parsed = parse_json_response(raw_response, expected_key=expected_key)
            return parsed.data, raw_response, parsed.repaired
        except Exception as exc:
            last_error = exc
            if attempt + 1 < attempts:
                continue
    raise StageError(stage, classify_exception(last_error or ValueError("unknown")), str(last_error), raw_response)


def extract_boxed_spans(text: str) -> Tuple[List[Dict[str, Any]], int]:
    spans: List[Dict[str, Any]] = []
    malformed = 0
    for idx, match in enumerate(re.finditer(r"\\boxed\s*\{", text)):
        open_idx = match.end() - 1
        close_idx = find_matching_brace(text, open_idx)
        if close_idx is None:
            malformed += 1
            continue
        value = text[open_idx + 1 : close_idx].strip()
        if value:
            spans.append(
                {
                    "boxed_index": idx,
                    "value": value,
                    "start": match.start(),
                    "end": close_idx + 1,
                    "evidence": text[match.start() : close_idx + 1],
                }
            )
    return spans, malformed


def ordered_unique(values: Iterable[str]) -> List[str]:
    result: List[str] = []
    keys = set()
    for value in values:
        cleaned = clean_answer_text(value)
        key = canonical_answer(cleaned)
        if not cleaned or not key or key in keys:
            continue
        keys.add(key)
        result.append(cleaned)
    return result


def last_final_marker_pos(text: str) -> int:
    lowered = text.lower()
    positions = [lowered.rfind(marker) for marker in FINAL_MARKERS]
    return max(positions)


def final_tail_start(text: str) -> int:
    marker_pos = last_final_marker_pos(text)
    if marker_pos >= 0:
        return marker_pos
    return max(0, int(len(text) * 0.65))


def extract_final_candidates(text: str) -> List[Dict[str, str]]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    candidates: List[Dict[str, str]] = []
    for idx, line in enumerate(lines[-18:]):
        for pattern in FINAL_LINE_PATTERNS:
            match = pattern.search(line)
            if not match:
                continue
            tail = match.group(1).strip()
            evidence = line
            if not tail and idx + 1 < len(lines[-18:]):
                tail = lines[-18:][idx + 1].strip()
                evidence = f"{line}\n{tail}"
            cleaned = clean_answer_text(tail)
            if cleaned:
                candidates.append({"description": cleaned, "evidence": evidence})
            break

    if not candidates:
        sentences = split_sentences(text)
        for sentence in reversed(sentences[-8:]):
            lowered = sentence.lower()
            if any(marker in lowered for marker in ("answer", "therefore", "thus", "hence", "so")):
                cleaned = clean_answer_text(sentence)
                if cleaned:
                    candidates.append({"description": cleaned, "evidence": sentence})
                    break
    return candidates


def is_answer_set_candidate(text: str, boxed_spans: Sequence[Dict[str, Any]]) -> bool:
    if len(boxed_spans) < 2:
        return False
    tail_start = final_tail_start(text)
    tail_spans = [span for span in boxed_spans if int(span["start"]) >= tail_start]
    if len(ordered_unique(span["value"] for span in tail_spans)) >= 2:
        return True
    last_spans = list(boxed_spans[-6:])
    if len(ordered_unique(span["value"] for span in last_spans)) >= 2:
        first = int(last_spans[0]["start"])
        last = int(last_spans[-1]["end"])
        between = text[first:last].lower()
        return last - first <= 900 and bool(re.search(r",| and | or |;|，|、", between))
    return False


def detect_subquestion(context: str) -> Optional[str]:
    patterns = (
        r"\(([a-zA-Z])\)",
        r"\bpart\s*([a-zA-Z])\b",
        r"\bsubquestion\s*([a-zA-Z0-9]+)\b",
        r"\b(first|second|third|fourth)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, context, flags=re.IGNORECASE)
        if match:
            return match.group(1).lower()
    return None


def make_target(
    target_id: int,
    target_type: str,
    description: str,
    source: str,
    evidence: str,
    confidence: str,
    items: Optional[List[str]] = None,
    subquestion: Optional[str] = None,
    boxed_indices: Optional[List[int]] = None,
    flags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    return {
        "target_id": target_id,
        "type": target_type if target_type in {"single_answer", "answer_set", "subquestion_answer", "unknown"} else "unknown",
        "description": clean_answer_text(description),
        "items": items or [],
        "subquestion": subquestion,
        "source": source,
        "boxed_indices": boxed_indices or [],
        "evidence": evidence,
        "confidence": confidence if confidence in {"high", "medium", "low"} else "low",
        "extraction_flags": flags or [],
    }


def target_matches_text(target: Dict[str, Any], text: str) -> bool:
    if not text:
        return False
    if target.get("type") == "answer_set":
        return all(canonical_answer(item) in canonical_answer(text) for item in target.get("items", []) if item)
    desc = str(target.get("description", ""))
    desc_key = canonical_answer(desc)
    text_key = canonical_answer(text)
    return equivalent_answer(desc, text) or bool(desc_key and text_key and desc_key in text_key)


def normalize_targets_from_rule(cot_text: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    boxed_spans, malformed_count = extract_boxed_spans(cot_text)
    flags: List[str] = []
    targets: List[Dict[str, Any]] = []
    if malformed_count:
        flags.append("malformed_boxed")

    final_candidates = extract_final_candidates(cot_text)
    last_box_end = max((int(span["end"]) for span in boxed_spans), default=-1)
    if boxed_spans and last_box_end < int(len(cot_text) * 0.65) and last_final_marker_pos(cot_text) < 0:
        flags.append("boxed_not_near_final")

    if boxed_spans:
        unique_values = ordered_unique(span["value"] for span in boxed_spans)
        if is_answer_set_candidate(cot_text, boxed_spans):
            flags.append("answer_set_candidate")
            tail_start = final_tail_start(cot_text)
            sequence_spans = [span for span in boxed_spans if int(span["start"]) >= tail_start] or list(boxed_spans[-6:])
            items = ordered_unique(span["value"] for span in sequence_spans)
            boxed_indices = [int(span["boxed_index"]) for span in sequence_spans]
            evidence = cot_text[int(sequence_spans[0]["start"]) : int(sequence_spans[-1]["end"])]
            targets = [
                make_target(
                    0,
                    "answer_set",
                    "{" + ", ".join(items) + "}",
                    "boxed_sequence",
                    evidence,
                    "medium",
                    items=items,
                    boxed_indices=boxed_indices,
                    flags=["answer_set"],
                )
            ]
        elif len(unique_values) == 1:
            all_indices = [int(span["boxed_index"]) for span in boxed_spans if equivalent_answer(span["value"], unique_values[0])]
            target_flags = ["repeated_boxed_dedup"] if len(all_indices) > 1 else []
            targets = [
                make_target(
                    0,
                    "single_answer",
                    unique_values[0],
                    "boxed",
                    boxed_spans[-1]["evidence"],
                    "high" if last_box_end >= int(len(cot_text) * 0.65) else "medium",
                    boxed_indices=all_indices,
                    flags=target_flags,
                )
            ]
        else:
            subq_targets: List[Dict[str, Any]] = []
            ambiguous_subq = False
            for span in boxed_spans:
                start = max(0, int(span["start"]) - 220)
                context = cot_text[start : int(span["start"])]
                subq = detect_subquestion(context)
                if subq:
                    subq_targets.append(
                        make_target(
                            len(subq_targets),
                            "subquestion_answer",
                            span["value"],
                            "boxed",
                            span["evidence"],
                            "medium",
                            subquestion=subq,
                            boxed_indices=[int(span["boxed_index"])],
                            flags=["subquestion"],
                        )
                    )
                else:
                    ambiguous_subq = True
            if subq_targets and not ambiguous_subq:
                targets = subq_targets
            else:
                flags.append("multiple_distinct_boxed")
                if subq_targets:
                    flags.append("subquestion_ambiguous")
                tail_start = final_tail_start(cot_text)
                candidate_spans = [span for span in boxed_spans if int(span["start"]) >= tail_start] or boxed_spans[-min(5, len(boxed_spans)) :]
                dedup_values = ordered_unique(span["value"] for span in candidate_spans)
                for value in dedup_values:
                    spans_for_value = [span for span in candidate_spans if equivalent_answer(span["value"], value)]
                    targets.append(
                        make_target(
                            len(targets),
                            "single_answer",
                            value,
                            "boxed",
                            spans_for_value[-1]["evidence"],
                            "low",
                            boxed_indices=[int(span["boxed_index"]) for span in spans_for_value],
                            flags=["multiple_distinct_boxed"],
                        )
                    )
    else:
        flags.append("no_boxed")
        if final_candidates:
            candidate = final_candidates[-1]
            targets = [
                make_target(
                    0,
                    "single_answer",
                    candidate["description"],
                    "final_sentence",
                    candidate["evidence"],
                    "medium",
                )
            ]
        else:
            flags.append("low_confidence_final_answer")

    if not boxed_spans and targets and targets[0].get("source") == "final_sentence":
        if len(str(targets[0].get("description", ""))) > 240:
            flags.append("low_confidence_final_answer")
            targets[0]["confidence"] = "low"
            targets[0]["extraction_flags"].append("low_confidence_final_answer")

    if boxed_spans and final_candidates and targets:
        final_text = final_candidates[-1]["description"]
        if final_text and not any(target_matches_text(target, final_text) for target in targets):
            flags.append("boxed_final_conflict")

    for target in targets:
        merged_flags = ordered_flag_union(target.get("extraction_flags", []), flags)
        target["extraction_flags"] = merged_flags
    return targets, ordered_flag_union(flags, [])


def ordered_flag_union(*flag_lists: Iterable[str]) -> List[str]:
    result: List[str] = []
    seen = set()
    for flags in flag_lists:
        for flag in flags:
            if flag and flag not in seen:
                seen.add(flag)
                result.append(flag)
    return result


def normalize_llm_targets(payload: Dict[str, Any], inherited_flags: Sequence[str]) -> List[Dict[str, Any]]:
    raw_targets = payload.get("targets", [])
    if not isinstance(raw_targets, list):
        return []
    global_flags = payload.get("extraction_flags", [])
    if not isinstance(global_flags, list):
        global_flags = []

    targets: List[Dict[str, Any]] = []
    for raw in raw_targets:
        if not isinstance(raw, dict):
            continue
        desc = clean_answer_text(str(raw.get("description", "")))
        items = raw.get("items", [])
        if not isinstance(items, list):
            items = []
        target_type = str(raw.get("type", "unknown"))
        if target_type == "answer_set" and not items:
            items = ordered_unique(re.split(r"[,;，、]|\band\b", desc.strip("{}")))
        evidence = str(raw.get("evidence", ""))
        confidence = str(raw.get("confidence", "low"))
        targets.append(
            make_target(
                len(targets),
                target_type,
                desc,
                "llm_fallback",
                evidence,
                confidence,
                items=[clean_answer_text(str(item)) for item in items if clean_answer_text(str(item))],
                subquestion=raw.get("subquestion"),
                boxed_indices=[],
                flags=ordered_flag_union(inherited_flags, global_flags),
            )
        )
    return targets


def run_target_fallback(
    client: LLMClient,
    prompt: str,
    question: str,
    cot_text: str,
    trigger_flags: Sequence[str],
    temperature: float,
    max_tokens: int,
    parse_retries: int,
) -> Tuple[List[Dict[str, Any]], str, bool]:
    user = (
        "QUESTION:\n{question}\n\n"
        "TRIGGER_FLAGS:\n{flags}\n\n"
        "COT:\n{cot}\n\n"
        "Extract only final answer targets explicitly stated in the CoT."
    ).format(question=question or "(unknown)", flags=json_dumps(list(trigger_flags)), cot=cot_text)
    messages = [
        {"role": "system", "content": prompt + "\n\nRespond ONLY with valid JSON. No markdown fences, no extra text."},
        {"role": "user", "content": user},
    ]
    data, raw, repaired = request_json_stage(
        client,
        messages,
        expected_key="targets",
        stage="target_fallback",
        temperature=temperature,
        max_tokens=max_tokens,
        parse_retries=parse_retries,
    )
    targets = normalize_llm_targets(data, trigger_flags)
    return targets, raw, repaired


def extract_targets_v4(
    client: LLMClient,
    prompt: str,
    question: str,
    cot_text: str,
    temperature: float,
    max_tokens: int,
    parse_retries: int,
    logger: logging.Logger,
) -> Tuple[List[Dict[str, Any]], Optional[str], bool]:
    rule_targets, rule_flags = normalize_targets_from_rule(cot_text)
    fallback_raw: Optional[str] = None
    repaired = False
    should_fallback = bool(FALLBACK_TRIGGER_FLAGS & set(rule_flags))
    if should_fallback:
        try:
            fallback_targets, fallback_raw, repaired = run_target_fallback(
                client,
                prompt,
                question,
                cot_text,
                rule_flags,
                temperature,
                max_tokens,
                parse_retries,
            )
            if fallback_targets:
                return fallback_targets, fallback_raw, repaired
        except StageError:
            if not rule_targets:
                raise
            logger.warning("target fallback failed; continuing with rule targets")
        except Exception as exc:
            if not rule_targets:
                raise StageError("target_fallback", classify_exception(exc), str(exc), fallback_raw) from exc
            logger.warning("target fallback failed; continuing with rule targets: %s", exc)

    if not rule_targets:
        raise StageError("target_extraction", "empty_targets", "No targets extracted from CoT.", None)
    return rule_targets, fallback_raw, repaired


def target_answer_alignment(targets: Sequence[Dict[str, Any]], answer_text: str) -> str:
    if not answer_text or not targets:
        return "unknown"
    answer_key = canonical_answer(answer_text)
    if not answer_key:
        return "unknown"
    for target in targets:
        if target.get("type") == "answer_set":
            item_keys = [canonical_answer(str(item)) for item in target.get("items", []) if canonical_answer(str(item))]
            if item_keys and all(key in answer_key for key in item_keys):
                return "match"
        desc = str(target.get("description", ""))
        if equivalent_answer(desc, answer_text):
            return "match"
        desc_key = canonical_answer(desc)
        if desc_key and (desc_key in answer_key or answer_key in desc_key):
            return "match"
    return "mismatch"


def make_segmentation_messages(prompt: str, question: str, cot_text: str) -> List[Dict[str, str]]:
    user = (
        "QUESTION:\n{question}\n\n"
        "ORIGINAL_COT:\n{cot}\n\n"
        "Return dependency-aware blocks as JSON."
    ).format(question=question or "(unknown)", cot=cot_text)
    return [
        {"role": "system", "content": prompt + "\n\nRespond ONLY with valid JSON. No markdown fences, no extra text."},
        {"role": "user", "content": user},
    ]


def regex_for_whitespace_flexible(candidate: str) -> Optional[re.Pattern[str]]:
    pieces = [re.escape(part) for part in re.split(r"\s+", candidate.strip()) if part]
    if not pieces:
        return None
    return re.compile(r"\s+".join(pieces), flags=re.DOTALL)


def restore_original_span(cot_text: str, candidate: str, cursor: int) -> Tuple[str, int, bool]:
    if not candidate:
        return "", cursor, False
    exact_idx = cot_text.find(candidate, cursor)
    if exact_idx >= 0:
        return cot_text[exact_idx : exact_idx + len(candidate)], exact_idx + len(candidate), True
    exact_idx = cot_text.find(candidate)
    if exact_idx >= 0:
        return cot_text[exact_idx : exact_idx + len(candidate)], exact_idx + len(candidate), True

    pattern = regex_for_whitespace_flexible(candidate)
    if pattern is not None:
        match = pattern.search(cot_text, cursor) or pattern.search(cot_text)
        if match:
            return cot_text[match.start() : match.end()], match.end(), True
    return candidate, cursor, False


def sanitize_blocks(data: Dict[str, Any], cot_text: str) -> Tuple[List[Dict[str, Any]], bool]:
    raw_blocks = data.get("blocks", [])
    if not isinstance(raw_blocks, list) or not raw_blocks:
        raise StageError("segmentation", "empty_blocks", "Segmentation response contains no blocks.")

    blocks: List[Dict[str, Any]] = []
    cursor = 0
    had_span_repair = False
    for raw in raw_blocks:
        if not isinstance(raw, dict):
            continue
        text = str(raw.get("text", "")).strip()
        if not text:
            continue
        original, cursor, ok = restore_original_span(cot_text, text, cursor)
        if not ok:
            raise StageError(
                "segmentation",
                "schema_error",
                f"Block {len(blocks)} text is not a continuous original CoT span.",
            )
        had_span_repair = had_span_repair or original != text
        block_type = str(raw.get("type", "derivation")).strip()
        if block_type not in VALID_BLOCK_TYPES:
            block_type = "derivation"
            had_span_repair = True
        blocks.append({"block_id": len(blocks), "type": block_type, "text": original})

    if not blocks:
        raise StageError("segmentation", "empty_blocks", "No valid blocks after sanitizing segmentation.")
    return blocks, had_span_repair


def make_dependency_user(prompt: str, question: str, targets: Sequence[Dict[str, Any]], blocks: Sequence[Dict[str, Any]]) -> str:
    return (
        "{prompt}\n\n"
        "QUESTION:\n{question}\n\n"
        "TARGETS:\n{targets}\n\n"
        "TURN_1_BLOCKS:\n{blocks}\n\n"
        "Label every block_id exactly once."
    ).format(
        prompt=prompt,
        question=question or "(unknown)",
        targets=json_dumps(list(targets), pretty=True),
        blocks=json_dumps(list(blocks), pretty=True),
    )


def bool_from_any(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    return default


def sanitize_dependency(
    data: Dict[str, Any],
    blocks: Sequence[Dict[str, Any]],
    targets: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    target_ids = {int(target["target_id"]) for target in targets}
    all_target_ids = sorted(target_ids)
    raw_labels = data.get("labels", [])
    if not isinstance(raw_labels, list):
        raise StageError("dependency_labeling", "schema_error", "Dependency response labels is not a list.")

    by_block: Dict[int, Dict[str, Any]] = {}
    local_flags: List[str] = []
    for raw in raw_labels:
        if not isinstance(raw, dict):
            continue
        try:
            block_id = int(raw.get("block_id"))
        except Exception:
            local_flags.append("json_repair_risk")
            continue
        if block_id in by_block:
            local_flags.append("json_repair_risk")
            continue
        by_block[block_id] = raw

    labels: List[Dict[str, Any]] = []
    for block in blocks:
        block_id = int(block["block_id"])
        raw = by_block.get(block_id)
        if raw is None:
            labels.append(
                {
                    "block_id": block_id,
                    "type": "NON_SUPPORTING",
                    "label_subtype": "NON_SUPPORTING",
                    "keep": False,
                    "supports": [],
                    "reason": "Missing from dependency response.",
                }
            )
            local_flags.append("json_repair_risk")
            continue

        subtype = str(raw.get("label", raw.get("label_subtype", "NON_SUPPORTING"))).strip()
        if subtype not in VALID_LABEL_SUBTYPES:
            subtype = "NON_SUPPORTING"
            local_flags.append("json_repair_risk")

        keep = bool_from_any(raw.get("keep"), KEEP_BY_DEFAULT[subtype])
        if subtype in {"DERIVATION_SUPPORT", "VALIDATION_SUPPORT", "FINAL_ANSWER"}:
            keep = True
        if subtype in {"REDUNDANT_VERIFICATION", "DEAD_END", "NARRATION", "REPEATED_FINAL", "NON_SUPPORTING"}:
            keep = False

        supports_raw = raw.get("supports", [])
        if not isinstance(supports_raw, list):
            supports_raw = []
            local_flags.append("json_repair_risk")
        supports: List[int] = []
        for item in supports_raw:
            try:
                tid = int(item)
            except Exception:
                local_flags.append("json_repair_risk")
                continue
            if tid in target_ids and tid not in supports:
                supports.append(tid)

        if keep and not supports and all_target_ids:
            supports = all_target_ids
            local_flags.append("target_uncertain_risk")
        if not keep:
            supports = []

        labels.append(
            {
                "block_id": block_id,
                "type": "SUPPORTING" if keep else "NON_SUPPORTING",
                "label_subtype": subtype,
                "keep": keep,
                "supports": supports,
                "reason": str(raw.get("reason", ""))[:500],
            }
        )

    raw_quality = data.get("quality_flags", [])
    if not isinstance(raw_quality, list):
        raw_quality = []
        local_flags.append("json_repair_risk")
    quality_flags = [str(flag) for flag in raw_quality if str(flag) in VALID_QUALITY_FLAGS]
    quality_flags = ordered_flag_union(quality_flags, local_flags)
    return labels, quality_flags


def token_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def compute_v4_statistics(
    blocks: Sequence[Dict[str, Any]],
    labels: Sequence[Dict[str, Any]],
    targets: Sequence[Dict[str, Any]],
    quality_flags: Sequence[str],
) -> Dict[str, Any]:
    block_by_id = {int(block["block_id"]): block for block in blocks}
    kept_labels = [label for label in labels if bool(label.get("keep"))]
    kept_ids = {int(label["block_id"]) for label in kept_labels}
    supported_targets = {
        int(target_id)
        for label in kept_labels
        for target_id in label.get("supports", [])
        if any(int(target["target_id"]) == int(target_id) for target in targets)
    }

    num_blocks = len(blocks)
    num_targets = len(targets)
    num_supporting = len(kept_labels)
    num_redundant = num_blocks - num_supporting
    total_tokens = sum(token_count(str(block.get("text", ""))) for block in blocks)
    supporting_tokens = sum(token_count(str(block_by_id[bid].get("text", ""))) for bid in kept_ids if bid in block_by_id)
    dependency_open = max(num_targets - len(supported_targets), 0)
    return {
        "num_blocks": num_blocks,
        "num_targets": num_targets,
        "num_supporting": num_supporting,
        "num_redundant": num_redundant,
        "compression_ratio_blocks": round(num_supporting / num_blocks, 4) if num_blocks else 0.0,
        "compression_ratio": round(num_supporting / num_blocks, 4) if num_blocks else 0.0,
        "compression_ratio_tokens": round(supporting_tokens / total_tokens, 4) if total_tokens else 0.0,
        "num_tokens_total": total_tokens,
        "num_tokens_supporting": supporting_tokens,
        "target_coverage": round(len(supported_targets) / num_targets, 4) if num_targets else 0.0,
        "dependency_open": dependency_open,
        "dependency_open_rate": round(dependency_open / num_targets, 4) if num_targets else 0.0,
        "label_subtype_distribution": dict(Counter(str(label.get("label_subtype", label.get("type", "UNKNOWN"))) for label in labels)),
        "block_type_distribution": dict(Counter(str(block.get("type", "unknown")) for block in blocks)),
        "num_quality_flags": len(quality_flags),
    }


def add_heuristic_quality_flags(
    labels: Sequence[Dict[str, Any]],
    targets: Sequence[Dict[str, Any]],
    quality_flags: Sequence[str],
    statistics: Dict[str, Any],
) -> List[str]:
    flags = list(quality_flags)
    subtypes = [str(label.get("label_subtype", "")) for label in labels]
    kept_subtypes = [str(label.get("label_subtype", "")) for label in labels if label.get("keep")]
    if "REPEATED_FINAL" in subtypes:
        flags.append("repeated_final_risk")
    if statistics.get("dependency_open", 0) > 0:
        flags.append("proof_gap_risk")
    if "FINAL_ANSWER" in kept_subtypes and not any(s in kept_subtypes for s in ("DERIVATION_SUPPORT", "VALIDATION_SUPPORT")):
        flags.append("summary_only_risk")
    if any(str(target.get("confidence", "")) == "low" or str(target.get("type", "")) == "unknown" for target in targets):
        flags.append("target_uncertain_risk")
    return [flag for flag in ordered_flag_union(flags) if flag in VALID_QUALITY_FLAGS]


def assemble_output(
    sample_id: str,
    source_field: Optional[str],
    targets: Sequence[Dict[str, Any]],
    alignment: str,
    blocks: Sequence[Dict[str, Any]],
    labels: Sequence[Dict[str, Any]],
    quality_flags: Sequence[str],
    raw_responses: Dict[str, Optional[str]],
) -> Dict[str, Any]:
    dependency_graph = [
        {"block": int(label["block_id"]), "target": int(target_id)}
        for label in labels
        if label.get("keep")
        for target_id in label.get("supports", [])
    ]
    supporting_blocks = [int(label["block_id"]) for label in labels if label.get("keep")]
    redundant_blocks = [int(label["block_id"]) for label in labels if not label.get("keep")]
    statistics = compute_v4_statistics(blocks, labels, targets, quality_flags)
    quality_flags = add_heuristic_quality_flags(labels, targets, quality_flags, statistics)
    statistics = compute_v4_statistics(blocks, labels, targets, quality_flags)
    return {
        "sample_id": sample_id,
        "source_field": source_field,
        "targets": list(targets),
        "target_answer_alignment": alignment,
        "blocks": list(blocks),
        "block_labels": list(labels),
        "dependency_graph": dependency_graph,
        "supporting_blocks": supporting_blocks,
        "redundant_blocks": redundant_blocks,
        "quality_flags": quality_flags,
        "statistics": statistics,
        "raw_responses": raw_responses,
        "pipeline_version": PIPELINE_VERSION,
    }


def validate_v4_output(payload: Dict[str, Any]) -> None:
    required = {
        "sample_id",
        "targets",
        "blocks",
        "block_labels",
        "dependency_graph",
        "supporting_blocks",
        "redundant_blocks",
        "statistics",
        "raw_responses",
        "pipeline_version",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Missing output fields: {missing}")
    block_ids = {int(block["block_id"]) for block in payload["blocks"]}
    target_ids = {int(target["target_id"]) for target in payload["targets"]}
    if not block_ids:
        raise ValueError("Output has no blocks.")
    if not target_ids:
        raise ValueError("Output has no targets.")
    for label in payload["block_labels"]:
        block_id = int(label["block_id"])
        if block_id not in block_ids:
            raise ValueError(f"Label references unknown block_id={block_id}")
        if label.get("type") not in {"SUPPORTING", "NON_SUPPORTING"}:
            raise ValueError(f"Invalid V3-compatible label type: {label.get('type')}")
        for target_id in label.get("supports", []):
            if int(target_id) not in target_ids:
                raise ValueError(f"Label references unknown target_id={target_id}")


def process_one(
    client: LLMClient,
    prompts: Dict[str, str],
    record: Dict[str, Any],
    fallback_id: str,
    temperature: float,
    max_tokens: int,
    parse_retries: int,
    logger: logging.Logger,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    sample_id = sample_id_from_record(record, fallback_id)
    try:
        question = extract_question(record)
        cot_text, source_field = extract_cot_text(record)
        if not cot_text:
            raise StageError("target_extraction", "empty_cot", "No CoT text found in record.")

        targets, target_fallback_raw, target_repaired = extract_targets_v4(
            client,
            prompts["target_fallback"],
            question,
            cot_text,
            temperature,
            max_tokens,
            parse_retries,
            logger,
        )
        if not targets:
            raise StageError("target_extraction", "empty_targets", "No targets extracted from CoT.")

        answer_text = extract_answer_field(record)
        alignment = target_answer_alignment(targets, answer_text)

        seg_messages = make_segmentation_messages(prompts["segmentation"], question, cot_text)
        seg_data, seg_raw, seg_repaired = request_json_stage(
            client,
            seg_messages,
            expected_key="blocks",
            stage="segmentation",
            temperature=temperature,
            max_tokens=max_tokens,
            parse_retries=parse_retries,
        )
        blocks, span_repaired = sanitize_blocks(seg_data, cot_text)
        if not blocks:
            raise StageError("segmentation", "empty_blocks", "No blocks returned by segmentation.")

        dep_user = make_dependency_user(prompts["dependency"], question, targets, blocks)
        dep_messages = seg_messages + [
            {"role": "assistant", "content": seg_raw},
            {"role": "user", "content": dep_user},
        ]
        dep_data, dep_raw, dep_repaired = request_json_stage(
            client,
            dep_messages,
            expected_key="labels",
            stage="dependency_labeling",
            temperature=temperature,
            max_tokens=max_tokens,
            parse_retries=parse_retries,
        )
        labels, quality_flags = sanitize_dependency(dep_data, blocks, targets)
        if target_repaired or seg_repaired or dep_repaired or span_repaired:
            quality_flags = ordered_flag_union(quality_flags, ["json_repair_risk"])

        payload = assemble_output(
            sample_id=sample_id,
            source_field=source_field,
            targets=targets,
            alignment=alignment,
            blocks=blocks,
            labels=labels,
            quality_flags=quality_flags,
            raw_responses={
                "target_fallback_raw_response": target_fallback_raw,
                "segmentation_raw_response": seg_raw,
                "dependency_raw_response": dep_raw,
            },
        )
        validate_v4_output(payload)
        return payload, None
    except StageError as exc:
        return None, make_failure(sample_id, exc.stage, exc.error_type, str(exc), record, exc.raw_response)
    except Exception as exc:
        return None, make_failure(
            sample_id,
            "output_assembly",
            classify_exception(exc),
            f"{exc}\n{traceback.format_exc()}",
            record,
            None,
        )


def load_success_checkpoint(ckpt_path: Path) -> set[str]:
    success_ids: set[str] = set()
    if not ckpt_path.exists():
        return success_ids
    for _, row in iter_jsonl(ckpt_path):
        if row.get("status") == "success":
            success_ids.add(str(row.get("sample_id", "")))
    return success_ids


def aggregate_stats(comp_path: Path, fail_path: Path, elapsed: float) -> Dict[str, Any]:
    results = list(iter_jsonl(comp_path)) if comp_path.exists() else []
    failures = list(iter_jsonl(fail_path)) if fail_path.exists() else []
    rows = [row for _, row in results]
    fail_rows = [row for _, row in failures]

    if not rows:
        return {
            "timestamp": datetime.now().isoformat(),
            "pipeline_version": PIPELINE_VERSION,
            "num_success": 0,
            "num_failed": len(fail_rows),
            "elapsed_seconds": round(elapsed, 1),
            "failure_breakdown": dict(Counter(row.get("error_type", "unknown") for row in fail_rows)),
        }

    def stat_values(key: str) -> List[float]:
        return [float(row.get("statistics", {}).get(key, 0.0)) for row in rows]

    def mean(values: Sequence[float]) -> float:
        return round(sum(values) / len(values), 4) if values else 0.0

    def median(values: Sequence[float]) -> float:
        if not values:
            return 0.0
        vals = sorted(values)
        mid = len(vals) // 2
        if len(vals) % 2:
            return round(vals[mid], 4)
        return round((vals[mid - 1] + vals[mid]) / 2, 4)

    label_subtypes: Counter[str] = Counter()
    block_types: Counter[str] = Counter()
    quality_flags: Counter[str] = Counter()
    for row in rows:
        label_subtypes.update(
            str(label.get("label_subtype", label.get("type", "UNKNOWN"))) for label in row.get("block_labels", [])
        )
        block_types.update(str(block.get("type", "unknown")) for block in row.get("blocks", []))
        quality_flags.update(str(flag) for flag in row.get("quality_flags", []))

    failure_breakdown = Counter(row.get("error_type", row.get("type", "unknown")) for row in fail_rows)
    total_seen = len(rows) + len(fail_rows)
    return {
        "timestamp": datetime.now().isoformat(),
        "pipeline_version": PIPELINE_VERSION,
        "num_success": len(rows),
        "num_failed": len(fail_rows),
        "elapsed_seconds": round(elapsed, 1),
        "mean_cr_blocks": mean(stat_values("compression_ratio_blocks")),
        "median_cr_blocks": median(stat_values("compression_ratio_blocks")),
        "mean_cr_tokens": mean(stat_values("compression_ratio_tokens")),
        "median_cr_tokens": median(stat_values("compression_ratio_tokens")),
        "mean_target_coverage": mean(stat_values("target_coverage")),
        "median_target_coverage": median(stat_values("target_coverage")),
        "mean_dependency_open_rate": mean(stat_values("dependency_open_rate")),
        "label_subtype_distribution": dict(label_subtypes),
        "block_type_distribution": dict(block_types),
        "quality_flag_distribution": dict(quality_flags),
        "quality_flag_rates": {flag: round(count / len(rows), 4) for flag, count in quality_flags.items()},
        "risk_rates": {
            "repeated_final_risk": round(quality_flags.get("repeated_final_risk", 0) / len(rows), 4),
            "summary_only_risk": round(quality_flags.get("summary_only_risk", 0) / len(rows), 4),
            "proof_gap_risk": round(quality_flags.get("proof_gap_risk", 0) / len(rows), 4),
            "dirty_trace_risk": round(quality_flags.get("dirty_trace_risk", 0) / len(rows), 4),
        },
        "parse_failure_rate": round(failure_breakdown.get("json_parse_error", 0) / total_seen, 4) if total_seen else 0.0,
        "failure_breakdown": dict(failure_breakdown),
    }


def write_checkpoint(ckpt_path: Path, sample_id: str, status: str, output_file: str, extra: Optional[Dict[str, Any]] = None) -> None:
    row = {
        "sample_id": sample_id,
        "status": status,
        "output_file": output_file,
        "timestamp": datetime.now().isoformat(),
    }
    if extra:
        row.update(extra)
    append_jsonl(ckpt_path, row)


def main() -> None:
    parser = argparse.ArgumentParser(description="V4 joint two-turn CoT compression pipeline.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("output_joint_3k"))
    parser.add_argument("--base-url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--model", type=str, default="Qwen2.5-32B-Instruct")
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--max-retries", type=int, default=2)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir, truncate=args.no_resume)

    comp_path = output_dir / "compressed.jsonl"
    fail_path = output_dir / "failure_cases.jsonl"
    stats_path = output_dir / "stats.json"
    ckpt_path = output_dir / "checkpoint.jsonl"

    if args.no_resume:
        for path in (comp_path, fail_path, stats_path, ckpt_path):
            if path.exists():
                path.unlink()

    prompts = {
        "target_fallback": read_prompt("fallback.txt"),
        "segmentation": read_prompt("segmentation.txt"),
        "dependency": read_prompt("dependency.txt"),
    }

    client = LLMClient(
        base_url=args.base_url,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )

    done_ids = set() if args.no_resume else load_success_checkpoint(ckpt_path)
    if done_ids:
        logger.info("[resume] skipping %d successful samples from checkpoint", len(done_ids))

    records = load_records(args.input)
    if args.limit is not None:
        records = records[: args.limit]

    total_to_visit = len(records)
    succ = 0
    fail = 0
    skipped = 0
    start = time.time()
    logger.info(
        "start V4 joint pipeline | records=%d | output=%s | model=%s | base_url=%s",
        total_to_visit,
        output_dir,
        args.model,
        args.base_url,
    )

    for ordinal, (line_no, record) in enumerate(records, start=1):
        sample_id = sample_id_from_record(record, str(line_no))
        if sample_id in done_ids:
            skipped += 1
            continue

        item_start = time.time()
        result, failure = process_one(
            client=client,
            prompts=prompts,
            record=record,
            fallback_id=str(line_no),
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            parse_retries=args.max_retries,
            logger=logger,
        )
        elapsed = time.time() - start
        processed = succ + fail + 1
        avg = elapsed / processed if processed else 0.0

        if result is not None:
            succ += 1
            append_jsonl(comp_path, result)
            write_checkpoint(ckpt_path, sample_id, "success", str(comp_path))
            stats = result["statistics"]
            logger.info(
                "[%d/%d] OK %s | blocks=%d keep=%d cr=%.4f cov=%.4f flags=%d | %.1fs item | %.1fs avg",
                ordinal,
                total_to_visit,
                sample_id,
                stats["num_blocks"],
                stats["num_supporting"],
                stats["compression_ratio_blocks"],
                stats["target_coverage"],
                stats["num_quality_flags"],
                time.time() - item_start,
                avg,
            )
        else:
            fail += 1
            assert failure is not None
            append_jsonl(fail_path, failure)
            write_checkpoint(
                ckpt_path,
                sample_id,
                "failed",
                str(fail_path),
                {"stage": failure.get("stage"), "error_type": failure.get("error_type")},
            )
            logger.info(
                "[%d/%d] FAIL %s | stage=%s type=%s | %.1fs item | %.1fs avg",
                ordinal,
                total_to_visit,
                sample_id,
                failure.get("stage"),
                failure.get("error_type"),
                time.time() - item_start,
                avg,
            )

    aggregate = aggregate_stats(comp_path, fail_path, time.time() - start)
    write_json(stats_path, aggregate, pretty=True)
    logger.info(
        "DONE | success=%d failed=%d skipped=%d elapsed=%.1fs mean_cr=%s mean_cov=%s",
        succ,
        fail,
        skipped,
        time.time() - start,
        aggregate.get("mean_cr_blocks"),
        aggregate.get("mean_target_coverage"),
    )


if __name__ == "__main__":
    main()
