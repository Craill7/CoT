#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V4.2 marker-atomic joint compression pipeline.

V4.2 keeps V3/V4 baselines intact and adds:
Step 0: local marker-based atomic segmentation.
Turn 1: LLM block refinement over atomic segments.
Turn 2: LLM block labeling.
Turn 3: LLM dependency graph and global quality check.

Final block text is assembled locally from exact original spans. LLM-generated
text is trusted only for validated internal_split substrings.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from . import labeler as v4
    from .segmenter import segment_atomic
    from .utils import extract_cot_text, iter_jsonl, sample_id_from_record, write_json
except ImportError:  # pragma: no cover - allows direct script execution.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import labeler as v4  # type: ignore
    from segmenter import segment_atomic  # type: ignore
    from utils import extract_cot_text, iter_jsonl, sample_id_from_record, write_json  # type: ignore


PIPELINE_VERSION = "v4.2_marker_atomic_joint"
PROJECT_DIR = Path(__file__).resolve().parents[1]
# Qwen2.5-32B may be served with a larger vLLM max-model-len, but this
# checkpoint's effective position budget is 32k. Keep requests under that
# safer envelope so context-window retries do not destabilize the server.
CONTEXT_WINDOW = 32768
CONTEXT_MARGIN = 4096
COMPACT_MIN_OUTPUT_TOKENS = 4096
COMPACT_SEGMENT_TEXT_CHARS = 96
COMPACT_BLOCK_TEXT_CHARS = 1200
TARGET_FALLBACK_TAIL_CHARS = 12000

VALID_LABEL_SUBTYPES = {
    "DERIVATION_SUPPORT",
    "VALIDATION_SUPPORT",
    "FINAL_ANSWER",
    "REDUNDANT_DERIVATION",
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
    "REDUNDANT_DERIVATION": False,
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
    "undercompressed_risk",
    "overcompressed_risk",
}

VALID_ENGINEERING_FLAGS = {
    "json_repair_risk",
    "schema_repair_risk",
    "internal_split_used",
    "segment_limit_relaxed_markers",
    "segment_limit_applied",
    "overlap_risk",
    "context_window_retry",
    "context_compact_mode",
    "target_fallback_used",
    "exact_substring_repair",
}

DIRTY_TRACE_PATTERNS = (
    r"\bokay\b",
    r"\bhmm\b",
    r"\blet me think\b",
    r"\blet me check\b",
    r"\bmaybe\b",
    r"\bi think\b",
    r"\bi'm not sure\b",
    r"\bthis is confusing\b",
    r"\btime constraints\b",
    r"\bhold on\b",
    r"\bwait\b",
)


def setup_logging(output_dir: Path, truncate: bool = False) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("joint_label_v42")
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


def ordered_flags(*flag_lists: Iterable[str]) -> List[str]:
    return v4.ordered_flag_union(*flag_lists)


def filter_quality_flags(flags: Iterable[Any]) -> List[str]:
    return [flag for flag in ordered_flags(str(flag) for flag in flags) if flag in VALID_QUALITY_FLAGS]


def filter_engineering_flags(flags: Iterable[Any]) -> List[str]:
    return [flag for flag in ordered_flags(str(flag) for flag in flags) if flag in VALID_ENGINEERING_FLAGS]


def bool_from_any(value: Any, default: bool) -> bool:
    return v4.bool_from_any(value, default)


def strip_json_control_chars(text: Any) -> str:
    cleaned = str(text or "")
    cleaned = cleaned.replace("\x08oxed", r"\boxed")
    cleaned = cleaned.replace("\x0boxed", r"\boxed")
    cleaned = "".join(ch for ch in cleaned if ch in "\n\t" or ord(ch) >= 32)
    return cleaned


def flatten_generation_candidates(value: Any) -> List[str]:
    if isinstance(value, list):
        return [v4.flatten_text(item) for item in value if v4.flatten_text(item).strip()]
    text_value = v4.flatten_text(value).strip()
    return [text_value] if text_value else []


def extract_think_spans(text_value: str) -> List[str]:
    spans = []
    for match in re.finditer(r"<think\b[^>]*>(.*?)</think>", text_value, flags=re.IGNORECASE | re.DOTALL):
        span = match.group(1).strip()
        if span:
            spans.append(span)
    return spans


def extract_scoped_cot_text(record: Dict[str, Any]) -> Tuple[str, Optional[str], str]:
    generations = record.get("generations")
    if generations is not None:
        spans: List[str] = []
        for candidate in flatten_generation_candidates(generations):
            spans.extend(extract_think_spans(candidate))
        if spans:
            return "\n\n[THINK_SPAN_BREAK]\n\n".join(spans).strip(), "generations", "think_only"
    cot_text, source_field = extract_cot_text(record)
    return cot_text, source_field, "fallback_full_generation"


def unwrap_boxed_once(text_value: str) -> Optional[str]:
    stripped = text_value.strip()
    match = re.search(r"\\boxed\s*\{", stripped)
    if not match:
        return None
    open_idx = match.end() - 1
    close_idx = v4.find_matching_brace(stripped, open_idx)
    if close_idx is None:
        return None
    before = stripped[:match.start()].strip(" $`:.：，,;；")
    after = stripped[close_idx + 1 :].strip(" $`:.：，,;；")
    if before or after:
        return None
    return stripped[open_idx + 1 : close_idx].strip()


def boxed_values_from_text(text_value: str) -> List[str]:
    values = []
    for match in re.finditer(r"\\boxed\s*\{", text_value):
        open_idx = match.end() - 1
        close_idx = v4.find_matching_brace(text_value, open_idx)
        if close_idx is None:
            continue
        value = text_value[open_idx + 1 : close_idx].strip()
        if value:
            values.append(value)
    return v4.ordered_unique(values)


def strip_answer_wrappers(text_value: str) -> str:
    text_value = strip_json_control_chars(text_value).strip()
    boxed = unwrap_boxed_once(text_value)
    if boxed is not None:
        text_value = boxed
    wrapper_patterns = [
        r"^(?:final\s+answer|answer)\s*[:：=\-]*\s*",
        r"^(?:therefore|thus|hence|so)\s*,?\s*(?:the\s+)?(?:final\s+)?answer\s*(?:is|:|=)?\s*",
        r"^(?:the\s+)?answer\s+(?:is|are)\s*",
        r"^(?:the\s+)?function\s+(?:is|equals)\s*",
        r"^(?:the\s+)?values?\s+(?:of\s+[^:=]+\s+)?(?:is|are)\s*",
        r"^(?:values?\s+of\s+[^:=]+)\s*[:：=\-]+\s*",
        r"^(?:function\s+expression\s+of\s+[^:=]+)\s*[:：=\-]+\s*",
    ]
    changed = True
    while changed:
        changed = False
        for pattern in wrapper_patterns:
            new_value = re.sub(pattern, "", text_value, flags=re.IGNORECASE).strip()
            if new_value != text_value:
                text_value = new_value
                changed = True
    text_value = text_value.strip(" \t\r\n.。;；,，")
    boxed = unwrap_boxed_once(text_value)
    return boxed if boxed is not None else text_value


def has_answer_signal(text_value: str) -> bool:
    return bool(re.search(r"\\boxed|\d|=|\\frac|\\sqrt|\$|[A-Za-z]_[A-Za-z0-9]", text_value))


def normalize_target_items(raw_items: Any, description: str, boxes: Sequence[str], target_type: str) -> List[str]:
    items: List[str] = []
    if isinstance(raw_items, list):
        items = [strip_answer_wrappers(item) for item in raw_items if strip_answer_wrappers(item)]
    if not items and boxes and (len(boxes) >= 2 or target_type == "answer_set"):
        items = [strip_answer_wrappers(item) for item in boxes if strip_answer_wrappers(item)]
    if not items and target_type == "answer_set":
        pieces = re.split(r"[,;，、]|\band\b|\bor\b", description.strip("{}"), flags=re.IGNORECASE)
        items = [strip_answer_wrappers(piece) for piece in pieces if strip_answer_wrappers(piece)]
    return v4.ordered_unique(items)


def description_from_items(items: Sequence[str]) -> str:
    return "{" + ", ".join(str(item) for item in items) + "}"


def extract_value_variable(evidence: str) -> Optional[str]:
    patterns = [
        r"values?\s+of\s+\\\(\s*([A-Za-z][A-Za-z0-9_]*)\s*\\\)\s+(?:is|are)",
        r"values?\s+of\s+[$`]?(?:\\)?([A-Za-z][A-Za-z0-9_]*)[$`]?\s+(?:is|are)",
    ]
    for pattern in patterns:
        match = re.search(pattern, evidence, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def subquestion_description_from_items(items: Sequence[str], evidence: str) -> str:
    variable = extract_value_variable(evidence)
    if variable:
        return " or ".join(f"{variable} = {item}" for item in items)
    return description_from_items(items)


def canonicalize_targets_v42(targets: Sequence[Dict[str, Any]], cot_text: str) -> List[Dict[str, Any]]:
    cleaned_targets: List[Dict[str, Any]] = []
    for raw_target in targets:
        target = dict(raw_target)
        evidence = strip_json_control_chars(target.get("evidence", ""))
        description = strip_answer_wrappers(target.get("description", ""))
        evidence_boxes = boxed_values_from_text(evidence)
        desc_boxes = boxed_values_from_text(description)
        boxes = evidence_boxes or desc_boxes

        target_type = str(target.get("type", "single_answer"))
        items = normalize_target_items(target.get("items", []), description, boxes, target_type)
        if target_type == "answer_set" or len(items) >= 2:
            if items:
                description = (
                    subquestion_description_from_items(items, evidence)
                    if target_type == "subquestion_answer"
                    else description_from_items(items)
                )
        else:
            if boxes:
                description = strip_answer_wrappers(boxes[-1])
            elif items:
                description = items[0]
            elif (not has_answer_signal(description)) and evidence:
                evidence_clean = strip_answer_wrappers(evidence)
                if has_answer_signal(evidence_clean) and len(evidence_clean) <= 220:
                    description = evidence_clean

        target["items"] = items
        target["description"] = strip_answer_wrappers(description)
        target["evidence"] = evidence
        target["target_id"] = len(cleaned_targets)
        target["extraction_flags"] = v4.ordered_flag_union(target.get("extraction_flags", []), [])
        cleaned_targets.append(target)
    return cleaned_targets


def resolve_marker_dict(path: Path) -> Path:
    if path.is_absolute() or path.exists():
        return path
    candidate = PROJECT_DIR / path
    return candidate if candidate.exists() else path


def atomic_category_to_block_type(category: str) -> str:
    mapping = {
        "setup": "setup",
        "derivation": "derivation",
        "verification": "verification",
        "correction": "correction",
        "final": "final",
        "narration": "narration",
        "conclusion": "derivation",
        "unknown": "derivation",
    }
    return mapping.get(str(category), "derivation")


def normalize_block_type(raw: Any, fallback: str = "derivation") -> Tuple[str, bool]:
    block_type = str(raw or fallback).strip()
    if block_type in v4.VALID_BLOCK_TYPES:
        return block_type, False
    return fallback if fallback in v4.VALID_BLOCK_TYPES else "derivation", True


def parse_int_list(value: Any) -> Tuple[List[int], bool]:
    repaired = False
    if not isinstance(value, list):
        return [], True
    result: List[int] = []
    for item in value:
        try:
            parsed = int(item)
        except Exception:
            repaired = True
            continue
        if parsed not in result:
            result.append(parsed)
        else:
            repaired = True
    return result, repaired


def make_refinement_messages(
    prompt: str,
    question: str,
    targets: Sequence[Dict[str, Any]],
    atomic_segments: Sequence[Dict[str, Any]],
) -> List[Dict[str, str]]:
    user = (
        "QUESTION:\n{question}\n\n"
        "TARGETS:\n{targets}\n\n"
        "ATOMIC_SEGMENTS:\n{segments}\n\n"
        "Construct reasoning blocks over these atomic segments. Cover every segment exactly once unless using "
        "validated internal_split for a mixed segment."
    ).format(
        question=question or "(unknown)",
        targets=v4.json_dumps(list(targets), pretty=True),
        segments=v4.json_dumps(list(atomic_segments), pretty=True),
    )
    return [
        {"role": "system", "content": prompt + "\n\nRespond ONLY with valid JSON. No markdown fences, no extra text."},
        {"role": "user", "content": user},
    ]




def compact_text_for_prompt(text: Any, max_chars: int = COMPACT_SEGMENT_TEXT_CHARS) -> Tuple[str, bool]:
    value = str(text or "")
    if len(value) <= max_chars:
        return value, False
    head = max_chars // 2
    tail = max_chars - head
    return value[:head] + "\n...[omitted middle in compact mode]...\n" + value[-tail:], True


def make_compact_refinement_messages(
    prompt: str,
    question: str,
    targets: Sequence[Dict[str, Any]],
    atomic_segments: Sequence[Dict[str, Any]],
) -> List[Dict[str, str]]:
    compact_segments = []
    for seg in atomic_segments:
        text, truncated = compact_text_for_prompt(seg.get("text", ""))
        compact_segments.append(
            {
                "seg_id": int(seg["seg_id"]),
                "char_start": int(seg["char_start"]),
                "char_end": int(seg["char_end"]),
                "marker": seg.get("marker"),
                "marker_category": seg.get("marker_category", "unknown"),
                "priority": seg.get("priority", "medium"),
                "text": text,
                "text_truncated": truncated,
            }
        )
    user = (
        "QUESTION:\n{question}\n\n"
        "TARGETS:\n{targets}\n\n"
        "ATOMIC_SEGMENTS_COMPACT:\n{segments}\n\n"
        "Compact mode: long segment text may be shortened, but segment ids and character ranges are exact. "
        "Prefer segment_group. Use internal_split only when text_truncated=false for that parent segment and "
        "the split text is visible exactly in the compact input. Cover every segment exactly once."
    ).format(
        question=question or "(unknown)",
        targets=v4.json_dumps(list(targets), pretty=True),
        segments=v4.json_dumps(compact_segments, pretty=True),
    )
    return [
        {"role": "system", "content": prompt + "\n\nRespond ONLY with valid JSON. No markdown fences, no extra text."},
        {"role": "user", "content": user},
    ]


def make_target_fallback_messages(
    prompt: str,
    question: str,
    cot_text: str,
    trigger_flags: Sequence[str],
    compact: bool = False,
) -> List[Dict[str, str]]:
    cot_payload = cot_text
    cot_header = "COT"
    compact_note = ""
    if compact and len(cot_text) > TARGET_FALLBACK_TAIL_CHARS:
        cot_payload = cot_text[-TARGET_FALLBACK_TAIL_CHARS:]
        cot_header = "COT_TAIL_COMPACT"
        compact_note = (
            "\nCompact mode: only the tail of the CoT is shown because target fallback only needs "
            "the author-stated final answer. Extract only answers explicitly present in this provided text; "
            "if the final answer is not visible, return an unknown low-confidence target."
        )
    user = (
        "QUESTION:\n{question}\n\n"
        "TRIGGER_FLAGS:\n{flags}\n\n"
        "{cot_header}:\n{cot}\n\n"
        "Extract only final answer targets explicitly stated in the CoT."
        "{compact_note}"
    ).format(
        question=question or "(unknown)",
        flags=v4.json_dumps(list(trigger_flags)),
        cot_header=cot_header,
        cot=cot_payload,
        compact_note=compact_note,
    )
    return [
        {"role": "system", "content": prompt + "\n\nRespond ONLY with valid JSON. No markdown fences, no extra text."},
        {"role": "user", "content": user},
    ]


def make_labeling_user(
    prompt: str,
    question: str,
    targets: Sequence[Dict[str, Any]],
    blocks: Sequence[Dict[str, Any]],
) -> str:
    return (
        "{prompt}\n\n"
        "QUESTION:\n{question}\n\n"
        "TARGETS:\n{targets}\n\n"
        "BLOCKS:\n{blocks}\n\n"
        "Label every block_id exactly once."
    ).format(
        prompt=prompt,
        question=question or "(unknown)",
        targets=v4.json_dumps(list(targets), pretty=True),
        blocks=v4.json_dumps(list(blocks), pretty=True),
    )


def make_graph_user(
    prompt: str,
    question: str,
    targets: Sequence[Dict[str, Any]],
    blocks: Sequence[Dict[str, Any]],
    labels: Sequence[Dict[str, Any]],
) -> str:
    del blocks, labels
    return (
        "{prompt}\n\n"
        "QUESTION:\n{question}\n\n"
        "TARGETS:\n{targets}\n\n"
        "Use the BLOCKS from the previous user turn and the sanitized TURN_2_LABELS from the previous assistant turn. "
        "Do not require the blocks to be repeated here. Return final dependency_graph, final_keep_blocks, "
        "final_drop_blocks, quality_flags, and any overrides."
    ).format(
        prompt=prompt,
        question=question or "(unknown)",
        targets=v4.json_dumps(list(targets), pretty=True),
    )




def compact_block_text_limit(num_blocks: int) -> int:
    if num_blocks <= 0:
        return COMPACT_BLOCK_TEXT_CHARS
    return max(160, min(COMPACT_BLOCK_TEXT_CHARS, 24000 // num_blocks))

def make_compact_labeling_messages(
    prompt: str,
    question: str,
    targets: Sequence[Dict[str, Any]],
    blocks: Sequence[Dict[str, Any]],
) -> List[Dict[str, str]]:
    compact_blocks = []
    text_limit = compact_block_text_limit(len(blocks))
    for block in blocks:
        text, truncated = compact_text_for_prompt(block.get("text", ""), text_limit)
        compact_blocks.append(
            {
                "block_id": int(block["block_id"]),
                "type": block.get("type", "derivation"),
                "source": block.get("source", "segment_group"),
                "segment_ids": block.get("segment_ids", []),
                "parent_segment_id": block.get("parent_segment_id"),
                "text": text,
                "text_truncated": truncated,
            }
        )
    user = (
        "QUESTION:\n{question}\n\n"
        "TARGETS:\n{targets}\n\n"
        "BLOCKS:\n{blocks}\n\n"
        "Compact mode: atomic segments and previous raw responses are omitted. Label every block_id exactly once."
    ).format(
        question=question or "(unknown)",
        targets=v4.json_dumps(list(targets), pretty=True),
        blocks=v4.json_dumps(compact_blocks, pretty=True),
    )
    return [
        {"role": "system", "content": prompt + "\n\nRespond ONLY with valid JSON. No markdown fences, no extra text."},
        {"role": "user", "content": user},
    ]


def make_compact_graph_messages(
    prompt: str,
    question: str,
    targets: Sequence[Dict[str, Any]],
    blocks: Sequence[Dict[str, Any]],
    labels: Sequence[Dict[str, Any]],
) -> List[Dict[str, str]]:
    block_by_id = {int(block["block_id"]): block for block in blocks}
    kept_blocks = []
    dropped_labels = []
    for label in labels:
        row = {
            "block_id": int(label["block_id"]),
            "label_subtype": label.get("label_subtype", label.get("type", "NON_SUPPORTING")),
            "keep": bool(label.get("keep")),
            "supports": label.get("supports", []),
            "reason": label.get("reason", ""),
        }
        if label.get("keep"):
            block = block_by_id.get(int(label["block_id"]), {})
            text, truncated = compact_text_for_prompt(block.get("text", ""), compact_block_text_limit(len(kept_blocks) + 1))
            row["block_type"] = block.get("type", "derivation")
            row["text"] = text
            row["text_truncated"] = truncated
            kept_blocks.append(row)
        else:
            dropped_labels.append(row)
    user = (
        "QUESTION:\n{question}\n\n"
        "TARGETS:\n{targets}\n\n"
        "KEPT_CANDIDATE_BLOCKS_WITH_TEXT:\n{kept}\n\n"
        "DROPPED_BLOCK_LABELS_NO_TEXT:\n{dropped}\n\n"
        "Compact mode: dropped block text is omitted. Perform final dependency graph and quality check over final kept blocks."
    ).format(
        question=question or "(unknown)",
        targets=v4.json_dumps(list(targets), pretty=True),
        kept=v4.json_dumps(kept_blocks, pretty=True),
        dropped=v4.json_dumps(dropped_labels, pretty=True),
    )
    return [
        {"role": "system", "content": prompt + "\n\nRespond ONLY with valid JSON. No markdown fences, no extra text."},
        {"role": "user", "content": user},
    ]


def estimate_message_tokens(messages: Sequence[Dict[str, str]]) -> int:
    total = 32 + 4 * len(messages)
    for message in messages:
        content = str(message.get("content", ""))
        # vLLM tokenization for math-heavy JSON/LaTeX prompts is much denser
        # than the simple local counter. Use a conservative char-based floor.
        total += max(v4.token_count(content), len(content) // 2)
    return total


def budget_max_tokens(messages: Sequence[Dict[str, str]], requested_max_tokens: int) -> Tuple[int, int, bool]:
    prompt_tokens = estimate_message_tokens(messages)
    available = CONTEXT_WINDOW - prompt_tokens - CONTEXT_MARGIN
    actual = max(0, min(requested_max_tokens, available))
    return actual, prompt_tokens, available < COMPACT_MIN_OUTPUT_TOKENS


def is_context_window_error(exc: v4.StageError) -> bool:
    message = str(exc).lower()
    return exc.error_type in {"api_error", "context_window_error"} and (
        "maximum context length" in message
        or "context window" in message
        or "max model len" in message
        or "input_tokens" in message
        or "400" in message
    )


def request_json_stage_budgeted(
    client: v4.LLMClient,
    messages: List[Dict[str, str]],
    expected_key: str,
    stage: str,
    temperature: float,
    max_tokens: int,
    parse_retries: int,
    compact_messages: Optional[List[Dict[str, str]]] = None,
    fallback_max_tokens: int = 8192,
) -> Tuple[Dict[str, Any], str, bool, List[str]]:
    active_messages = messages
    flags: List[str] = []
    actual_max_tokens, prompt_tokens, should_compact = budget_max_tokens(active_messages, max_tokens)
    if should_compact and compact_messages is not None:
        active_messages = compact_messages
        flags.append("context_compact_mode")
        actual_max_tokens, prompt_tokens, _ = budget_max_tokens(active_messages, max_tokens)

    if actual_max_tokens < 1024:
        raise v4.StageError(
            stage,
            "context_window_error",
            f"Estimated context window exhausted before request: prompt_tokens~{prompt_tokens}, requested_max_tokens={max_tokens}.",
        )

    try:
        data, raw, repaired = v4.request_json_stage(
            client,
            active_messages,
            expected_key=expected_key,
            stage=stage,
            temperature=temperature,
            max_tokens=actual_max_tokens,
            parse_retries=parse_retries,
        )
        return data, raw, repaired, filter_engineering_flags(flags)
    except v4.StageError as exc:
        if not is_context_window_error(exc):
            raise

        retry_messages = active_messages
        if compact_messages is not None and active_messages is not compact_messages:
            retry_messages = compact_messages
            flags.append("context_compact_mode")
        retry_max_tokens = min(max_tokens, fallback_max_tokens)
        retry_actual, retry_prompt_tokens, _ = budget_max_tokens(retry_messages, retry_max_tokens)
        if retry_actual < 1024:
            raise v4.StageError(
                stage,
                "context_window_error",
                f"Context window retry impossible: prompt_tokens~{retry_prompt_tokens}, requested_max_tokens={retry_max_tokens}. Original error: {exc}",
                exc.raw_response,
            ) from exc
        flags.append("context_window_retry")
        try:
            data, raw, repaired = v4.request_json_stage(
                client,
                retry_messages,
                expected_key=expected_key,
                stage=stage,
                temperature=temperature,
                max_tokens=retry_actual,
                parse_retries=parse_retries,
            )
            return data, raw, repaired, filter_engineering_flags(flags)
        except v4.StageError as retry_exc:
            if is_context_window_error(retry_exc):
                raise v4.StageError(stage, "context_window_error", str(retry_exc), retry_exc.raw_response) from retry_exc
            raise


def extract_targets_v42(
    client: v4.LLMClient,
    prompt: str,
    question: str,
    cot_text: str,
    temperature: float,
    max_tokens: int,
    parse_retries: int,
    logger: logging.Logger,
) -> Tuple[List[Dict[str, Any]], Optional[str], bool, List[str]]:
    rule_targets, rule_flags = v4.normalize_targets_from_rule(cot_text)
    fallback_raw: Optional[str] = None
    repaired = False
    engineering_flags: List[str] = []
    should_fallback = bool(v4.FALLBACK_TRIGGER_FLAGS & set(rule_flags))
    if should_fallback:
        engineering_flags.append("target_fallback_used")
        try:
            messages = make_target_fallback_messages(prompt, question, cot_text, rule_flags, compact=False)
            compact_messages = make_target_fallback_messages(prompt, question, cot_text, rule_flags, compact=True)
            data, fallback_raw, repaired, budget_flags = request_json_stage_budgeted(
                client,
                messages,
                expected_key="targets",
                stage="target_fallback",
                temperature=temperature,
                max_tokens=max_tokens,
                parse_retries=parse_retries,
                compact_messages=compact_messages,
                fallback_max_tokens=4096,
            )
            engineering_flags.extend(budget_flags)
            fallback_targets = v4.normalize_llm_targets(data, rule_flags)
            if fallback_targets:
                return fallback_targets, fallback_raw, repaired, filter_engineering_flags(engineering_flags)
        except v4.StageError:
            if not rule_targets:
                raise
            logger.warning("target fallback failed; continuing with rule targets")
        except Exception as exc:
            if not rule_targets:
                raise v4.StageError("target_fallback", v4.classify_exception(exc), str(exc), fallback_raw) from exc
            logger.warning("target fallback failed; continuing with rule targets: %s", exc)

    if not rule_targets:
        raise v4.StageError("target_extraction", "empty_targets", "No targets extracted from CoT.", None)
    return rule_targets, fallback_raw, repaired, filter_engineering_flags(engineering_flags)


def compact_refinement_context(blocks: Sequence[Dict[str, Any]], flags: Sequence[str]) -> str:
    compact_blocks = []
    for block in blocks:
        compact_blocks.append(
            {
                "block_id": int(block["block_id"]),
                "source": block.get("source", "segment_group"),
                "segment_ids": block.get("segment_ids", []),
                "parent_segment_id": block.get("parent_segment_id"),
                "block_type": block.get("type", "derivation"),
                "reason": block.get("reason", ""),
            }
        )
    return v4.json_dumps({"blocks": compact_blocks, "segmentation_flags": list(flags)}, pretty=False)


def compact_labeling_context(labels: Sequence[Dict[str, Any]], flags: Sequence[str]) -> str:
    compact_labels = []
    for label in labels:
        compact_labels.append(
            {
                "block_id": int(label["block_id"]),
                "label": label.get("label_subtype", label.get("type", "NON_SUPPORTING")),
                "keep": bool(label.get("keep")),
                "supports": label.get("supports", []),
                "reason": label.get("reason", ""),
            }
        )
    return v4.json_dumps({"labels": compact_labels, "labeling_flags": list(flags)}, pretty=False)


def _ranges_overlap(left: Tuple[int, int], right: Tuple[int, int]) -> bool:
    return max(left[0], right[0]) < min(left[1], right[1])


def sanitize_refined_blocks(
    data: Dict[str, Any],
    atomic_segments: Sequence[Dict[str, Any]],
    cot_text: str,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    raw_blocks = data.get("blocks", [])
    if not isinstance(raw_blocks, list) or not raw_blocks:
        raise v4.StageError("block_refinement", "empty_blocks", "Block refinement response contains no blocks.")

    atomic_by_id = {int(seg["seg_id"]): seg for seg in atomic_segments}
    all_seg_ids = sorted(atomic_by_id)
    used_group_segments: set[int] = set()
    split_parent_ids: set[int] = set()
    split_ranges_by_parent: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    blocks: List[Dict[str, Any]] = []
    engineering_flags: List[str] = []

    raw_seg_flags = data.get("segmentation_flags", [])
    if isinstance(raw_seg_flags, list):
        engineering_flags.extend(str(flag) for flag in raw_seg_flags if str(flag) in VALID_ENGINEERING_FLAGS)

    def append_group(raw: Dict[str, Any], segment_ids: List[int], reason: str) -> None:
        nonlocal engineering_flags
        if not segment_ids:
            raise v4.StageError("block_refinement", "schema_error", "segment_group has no segment_ids.")
        unknown = [sid for sid in segment_ids if sid not in atomic_by_id]
        if unknown:
            raise v4.StageError("block_refinement", "schema_error", f"Unknown segment_ids: {unknown[:5]}")

        sorted_ids = sorted(segment_ids)
        if sorted_ids != segment_ids:
            engineering_flags.append("schema_repair_risk")
            segment_ids = sorted_ids

        start_sid = segment_ids[0]
        end_sid = segment_ids[-1]
        expanded_ids = [sid for sid in all_seg_ids if start_sid <= sid <= end_sid]
        if expanded_ids != segment_ids:
            segment_ids = expanded_ids
            engineering_flags.append("schema_repair_risk")

        overlap = [sid for sid in segment_ids if sid in used_group_segments or sid in split_parent_ids]
        if overlap:
            raise v4.StageError(
                "block_refinement",
                "schema_error",
                f"Atomic segment(s) reused across blocks: {overlap[:5]}",
            )

        for sid in segment_ids:
            used_group_segments.add(sid)

        start = int(atomic_by_id[segment_ids[0]]["char_start"])
        end = int(atomic_by_id[segment_ids[-1]]["char_end"])
        text = cot_text[start:end]
        fallback_type = atomic_category_to_block_type(str(atomic_by_id[segment_ids[0]].get("marker_category", "unknown")))
        block_type, repaired = normalize_block_type(raw.get("block_type", raw.get("type")), fallback_type)
        if repaired:
            engineering_flags.append("schema_repair_risk")
        blocks.append(
            {
                "block_id": len(blocks),
                "source": "segment_group",
                "segment_ids": segment_ids,
                "parent_segment_id": None,
                "type": block_type,
                "text": text,
                "reason": reason[:500],
            }
        )

    def append_internal_split(raw: Dict[str, Any], reason: str) -> None:
        nonlocal engineering_flags
        try:
            parent_id = int(raw.get("parent_segment_id"))
        except Exception as exc:
            raise v4.StageError("block_refinement", "schema_error", "internal_split missing parent_segment_id.") from exc
        if parent_id not in atomic_by_id:
            raise v4.StageError("block_refinement", "schema_error", f"Unknown parent_segment_id={parent_id}")
        if parent_id in used_group_segments:
            raise v4.StageError(
                "block_refinement",
                "schema_error",
                f"Parent segment {parent_id} already used by segment_group.",
            )

        parent = atomic_by_id[parent_id]
        text = str(raw.get("text", ""))
        if not text:
            raise v4.StageError("block_refinement", "exact_substring_error", "internal_split has empty text.")
        parent_text = str(parent.get("text", ""))
        rel_start = parent_text.find(text)
        if rel_start < 0:
            raise v4.StageError(
                "block_refinement",
                "exact_substring_error",
                f"internal_split text is not an exact substring of parent segment {parent_id}.",
            )
        start = int(parent["char_start"]) + rel_start
        end = start + len(text)
        split_range = (start, end)
        for old_range in split_ranges_by_parent[parent_id]:
            if _ranges_overlap(split_range, old_range):
                raise v4.StageError(
                    "block_refinement",
                    "schema_error",
                    f"Overlapping internal_split ranges for parent segment {parent_id}.",
                )
        split_ranges_by_parent[parent_id].append(split_range)
        split_parent_ids.add(parent_id)
        engineering_flags.append("internal_split_used")

        fallback_type = atomic_category_to_block_type(str(parent.get("marker_category", "unknown")))
        block_type, repaired = normalize_block_type(raw.get("block_type", raw.get("type")), fallback_type)
        if repaired:
            engineering_flags.append("schema_repair_risk")
        blocks.append(
            {
                "block_id": len(blocks),
                "source": "internal_split",
                "segment_ids": [],
                "parent_segment_id": parent_id,
                "type": block_type,
                "text": cot_text[start:end],
                "reason": reason[:500],
            }
        )

    for raw in raw_blocks:
        if not isinstance(raw, dict):
            engineering_flags.append("schema_repair_risk")
            continue
        reason = str(raw.get("reason", ""))
        source = str(raw.get("source", "")).strip()
        if source not in {"segment_group", "internal_split"}:
            source = "segment_group" if "segment_ids" in raw else "internal_split"
            engineering_flags.append("schema_repair_risk")

        if source == "segment_group":
            segment_ids, repaired = parse_int_list(raw.get("segment_ids", []))
            if repaired:
                engineering_flags.append("schema_repair_risk")
            append_group(raw, segment_ids, reason)
        else:
            append_internal_split(raw, reason)

    if not blocks:
        raise v4.StageError("block_refinement", "empty_blocks", "No valid blocks after local refinement.")

    unused = [sid for sid in all_seg_ids if sid not in used_group_segments and sid not in split_parent_ids]
    if unused:
        engineering_flags.append("schema_repair_risk")
        for sid in unused:
            seg = atomic_by_id[sid]
            fallback_type = atomic_category_to_block_type(str(seg.get("marker_category", "unknown")))
            blocks.append(
                {
                    "block_id": len(blocks),
                    "source": "segment_group",
                    "segment_ids": [sid],
                    "parent_segment_id": None,
                    "type": fallback_type,
                    "text": cot_text[int(seg["char_start"]) : int(seg["char_end"])],
                    "reason": "Added by local repair for omitted atomic segment.",
                }
            )

    for block in blocks:
        text = str(block.get("text", ""))
        if not text:
            raise v4.StageError("block_refinement", "empty_blocks", f"Block {block['block_id']} has empty text.")
        if text not in cot_text:
            raise v4.StageError(
                "block_refinement",
                "exact_substring_error",
                f"Block {block['block_id']} text is not present in the original CoT.",
            )

    return blocks, filter_engineering_flags(engineering_flags)


def sanitize_block_labels(
    data: Dict[str, Any],
    blocks: Sequence[Dict[str, Any]],
    targets: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
    raw_labels = data.get("labels", [])
    if not isinstance(raw_labels, list):
        raise v4.StageError("block_labeling", "schema_error", "Block labeling response labels is not a list.")

    target_ids = {int(target["target_id"]) for target in targets}
    by_block: Dict[int, Dict[str, Any]] = {}
    engineering_flags: List[str] = []
    for raw in raw_labels:
        if not isinstance(raw, dict):
            engineering_flags.append("schema_repair_risk")
            continue
        try:
            block_id = int(raw.get("block_id"))
        except Exception:
            engineering_flags.append("schema_repair_risk")
            continue
        if block_id in by_block:
            engineering_flags.append("schema_repair_risk")
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
                    "reason": "Missing from block labeling response.",
                }
            )
            engineering_flags.append("schema_repair_risk")
            continue

        subtype = str(raw.get("label", raw.get("label_subtype", "NON_SUPPORTING"))).strip()
        if subtype not in VALID_LABEL_SUBTYPES:
            subtype = "NON_SUPPORTING"
            engineering_flags.append("schema_repair_risk")

        keep = bool_from_any(raw.get("keep"), KEEP_BY_DEFAULT[subtype])
        if subtype in {"DERIVATION_SUPPORT", "VALIDATION_SUPPORT", "FINAL_ANSWER"}:
            keep = True
        if subtype in {"REDUNDANT_DERIVATION", "REDUNDANT_VERIFICATION", "DEAD_END", "NARRATION", "REPEATED_FINAL", "NON_SUPPORTING"}:
            keep = False

        supports_raw = raw.get("supports", [])
        if not isinstance(supports_raw, list):
            supports_raw = []
            engineering_flags.append("schema_repair_risk")
        supports: List[int] = []
        for item in supports_raw:
            try:
                target_id = int(item)
            except Exception:
                engineering_flags.append("schema_repair_risk")
                continue
            if target_id in target_ids and target_id not in supports:
                supports.append(target_id)

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

    quality_flags = []
    raw_flags = data.get("labeling_flags", [])
    if isinstance(raw_flags, list):
        quality_flags.extend(str(flag) for flag in raw_flags)
    else:
        engineering_flags.append("schema_repair_risk")
    return labels, filter_quality_flags(quality_flags), filter_engineering_flags(engineering_flags)


def parse_graph_edges(
    raw_edges: Any,
    block_ids: set[int],
    target_ids: set[int],
) -> Tuple[List[Dict[str, int]], List[str]]:
    edges: List[Dict[str, int]] = []
    engineering_flags: List[str] = []
    if not isinstance(raw_edges, list):
        return edges, ["schema_repair_risk"]
    seen: set[Tuple[int, int]] = set()
    for raw in raw_edges:
        if not isinstance(raw, dict):
            engineering_flags.append("schema_repair_risk")
            continue
        try:
            block_id = int(raw.get("block", raw.get("block_id")))
            target_id = int(raw.get("target", raw.get("target_id")))
        except Exception:
            engineering_flags.append("schema_repair_risk")
            continue
        if block_id not in block_ids or target_id not in target_ids:
            engineering_flags.append("schema_repair_risk")
            continue
        key = (block_id, target_id)
        if key not in seen:
            seen.add(key)
            edges.append({"block": block_id, "target": target_id})
    return edges, filter_engineering_flags(engineering_flags)


def sanitize_dependency_graph(
    data: Dict[str, Any],
    blocks: Sequence[Dict[str, Any]],
    targets: Sequence[Dict[str, Any]],
    labels: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, int]], List[str], List[str]]:
    block_ids = {int(block["block_id"]) for block in blocks}
    target_ids = {int(target["target_id"]) for target in targets}
    labels_by_id = {int(label["block_id"]): dict(label) for label in labels}
    engineering_flags: List[str] = []

    edges, edge_flags = parse_graph_edges(data.get("dependency_graph", []), block_ids, target_ids)
    engineering_flags.extend(edge_flags)

    has_keep_key = "final_keep_blocks" in data
    keep_raw, keep_repaired = parse_int_list(data.get("final_keep_blocks", []))
    drop_raw, drop_repaired = parse_int_list(data.get("final_drop_blocks", []))
    if keep_repaired or drop_repaired:
        engineering_flags.append("schema_repair_risk")
    final_keep = {bid for bid in keep_raw if bid in block_ids}
    final_drop = {bid for bid in drop_raw if bid in block_ids}

    base_keep = {int(label["block_id"]) for label in labels if bool(label.get("keep"))}
    keep_set = final_keep if has_keep_key else set(base_keep)
    keep_set -= final_drop

    raw_overrides = data.get("override_labels", [])
    if raw_overrides is None:
        raw_overrides = []
    if not isinstance(raw_overrides, list):
        raw_overrides = []
        engineering_flags.append("schema_repair_risk")

    override_reason_by_id: Dict[int, str] = {}
    for raw in raw_overrides:
        if not isinstance(raw, dict):
            engineering_flags.append("schema_repair_risk")
            continue
        try:
            block_id = int(raw.get("block_id"))
        except Exception:
            engineering_flags.append("schema_repair_risk")
            continue
        if block_id not in block_ids:
            engineering_flags.append("schema_repair_risk")
            continue
        if "new_keep" in raw:
            if bool_from_any(raw.get("new_keep"), block_id in keep_set):
                keep_set.add(block_id)
            else:
                keep_set.discard(block_id)
        new_label = str(raw.get("new_label", "")).strip()
        if new_label in VALID_LABEL_SUBTYPES:
            labels_by_id[block_id]["label_subtype"] = new_label
        elif new_label:
            engineering_flags.append("schema_repair_risk")
        reason = str(raw.get("reason", "")).strip()
        if reason:
            override_reason_by_id[block_id] = reason[:500]

    edges_by_block: Dict[int, List[int]] = defaultdict(list)
    for edge in edges:
        block_id = int(edge["block"])
        target_id = int(edge["target"])
        if block_id in keep_set and target_id not in edges_by_block[block_id]:
            edges_by_block[block_id].append(target_id)

    graph_is_empty = not edges
    final_labels: List[Dict[str, Any]] = []
    for block in blocks:
        block_id = int(block["block_id"])
        label = labels_by_id.get(
            block_id,
            {
                "block_id": block_id,
                "label_subtype": "NON_SUPPORTING",
                "reason": "Missing label repaired locally.",
            },
        )
        if block_id not in labels_by_id:
            engineering_flags.append("schema_repair_risk")

        keep = block_id in keep_set
        subtype = str(label.get("label_subtype", "NON_SUPPORTING"))
        if subtype not in VALID_LABEL_SUBTYPES:
            subtype = "NON_SUPPORTING"
            engineering_flags.append("schema_repair_risk")

        if subtype in {
            "REDUNDANT_DERIVATION",
            "REDUNDANT_VERIFICATION",
            "SELF_CORRECTION",
            "DEAD_END",
            "NARRATION",
            "REPEATED_FINAL",
            "NON_SUPPORTING",
        }:
            keep = False

        supports = edges_by_block.get(block_id, [])
        if keep and not supports and graph_is_empty:
            supports = [int(tid) for tid in label.get("supports", []) if int(tid) in target_ids]
        if not keep:
            supports = []

        reason = override_reason_by_id.get(block_id, str(label.get("reason", ""))[:500])
        final_labels.append(
            {
                "block_id": block_id,
                "type": "SUPPORTING" if keep else "NON_SUPPORTING",
                "label_subtype": subtype,
                "keep": keep,
                "supports": supports,
                "reason": reason,
            }
        )

    dependency_graph = [
        {"block": int(label["block_id"]), "target": int(target_id)}
        for label in final_labels
        if label.get("keep")
        for target_id in label.get("supports", [])
    ]

    raw_quality = data.get("quality_flags", [])
    if not isinstance(raw_quality, list):
        raw_quality = []
        engineering_flags.append("schema_repair_risk")
    quality_flags = filter_quality_flags(str(flag) for flag in raw_quality)
    return final_labels, dependency_graph, quality_flags, filter_engineering_flags(engineering_flags)


def block_has_dirty_trace(text: str) -> bool:
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in DIRTY_TRACE_PATTERNS)


def compute_v42_statistics(
    atomic_segments: Sequence[Dict[str, Any]],
    blocks: Sequence[Dict[str, Any]],
    labels: Sequence[Dict[str, Any]],
    targets: Sequence[Dict[str, Any]],
    quality_flags: Sequence[str],
    engineering_flags: Sequence[str],
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

    num_atomic_segments = len(atomic_segments)
    num_blocks = len(blocks)
    num_targets = len(targets)
    num_supporting = len(kept_labels)
    num_redundant = num_blocks - num_supporting
    total_tokens = sum(v4.token_count(str(block.get("text", ""))) for block in blocks)
    supporting_tokens = sum(v4.token_count(str(block_by_id[bid].get("text", ""))) for bid in kept_ids if bid in block_by_id)
    dependency_open = max(num_targets - len(supported_targets), 0)
    redundant_derivation_count = sum(1 for label in labels if label.get("label_subtype") == "REDUNDANT_DERIVATION")

    return {
        "num_atomic_segments": num_atomic_segments,
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
        "redundant_derivation_count": redundant_derivation_count,
        "redundant_derivation_rate": round(redundant_derivation_count / num_blocks, 4) if num_blocks else 0.0,
        "atomic_to_block_ratio": round(num_blocks / num_atomic_segments, 4) if num_atomic_segments else 0.0,
        "label_subtype_distribution": dict(
            Counter(str(label.get("label_subtype", label.get("type", "UNKNOWN"))) for label in labels)
        ),
        "block_type_distribution": dict(Counter(str(block.get("type", "unknown")) for block in blocks)),
        "atomic_marker_distribution": dict(
            Counter(str(seg.get("marker_category", "unknown")) for seg in atomic_segments)
        ),
        "num_quality_flags": len(quality_flags),
        "num_engineering_flags": len(engineering_flags),
    }


def add_heuristic_quality_flags(
    blocks: Sequence[Dict[str, Any]],
    labels: Sequence[Dict[str, Any]],
    targets: Sequence[Dict[str, Any]],
    quality_flags: Sequence[str],
    statistics: Dict[str, Any],
) -> List[str]:
    raw_flags = set(filter_quality_flags(quality_flags))
    # Final output quality flags should describe risks remaining in kept blocks.
    # Keep the LLM's proof-gap judgment, but recompute other observable risks
    # locally after final_keep/final_drop and override labels have been applied.
    flags = ["proof_gap_risk"] if "proof_gap_risk" in raw_flags else []
    block_by_id = {int(block["block_id"]): block for block in blocks}
    kept_labels = [label for label in labels if label.get("keep")]
    kept_subtypes = [str(label.get("label_subtype", "")) for label in kept_labels]

    if statistics.get("dependency_open", 0) > 0:
        flags.append("proof_gap_risk")
    if "FINAL_ANSWER" in kept_subtypes and not any(s in kept_subtypes for s in ("DERIVATION_SUPPORT", "VALIDATION_SUPPORT")):
        flags.append("summary_only_risk")
    if any(block_has_dirty_trace(str(block_by_id.get(int(label["block_id"]), {}).get("text", ""))) for label in kept_labels):
        flags.append("dirty_trace_risk")
    if any(str(label.get("label_subtype", "")) == "REDUNDANT_DERIVATION" for label in kept_labels):
        flags.append("undercompressed_risk")

    kept_final_by_target: Counter[int] = Counter()
    for label in kept_labels:
        if label.get("label_subtype") in {"FINAL_ANSWER", "REPEATED_FINAL"}:
            for target_id in label.get("supports", []):
                kept_final_by_target[int(target_id)] += 1
    if any(count > 1 for count in kept_final_by_target.values()):
        flags.append("repeated_final_risk")
    if any(str(label.get("label_subtype", "")) == "REPEATED_FINAL" for label in kept_labels):
        flags.append("repeated_final_risk")
    if statistics.get("num_supporting", 0) <= 1 and statistics.get("num_blocks", 0) >= 4:
        flags.append("overcompressed_risk")

    return filter_quality_flags(flags)


def assemble_output(
    sample_id: str,
    source_field: Optional[str],
    source_scope: str,
    targets: Sequence[Dict[str, Any]],
    alignment: str,
    atomic_segments: Sequence[Dict[str, Any]],
    blocks: Sequence[Dict[str, Any]],
    labels: Sequence[Dict[str, Any]],
    dependency_graph: Sequence[Dict[str, int]],
    quality_flags: Sequence[str],
    engineering_flags: Sequence[str],
    raw_responses: Dict[str, Optional[str]],
) -> Dict[str, Any]:
    engineering_flags = filter_engineering_flags(engineering_flags)
    statistics = compute_v42_statistics(atomic_segments, blocks, labels, targets, quality_flags, engineering_flags)
    quality_flags = add_heuristic_quality_flags(blocks, labels, targets, quality_flags, statistics)
    statistics = compute_v42_statistics(atomic_segments, blocks, labels, targets, quality_flags, engineering_flags)
    supporting_blocks = [int(label["block_id"]) for label in labels if label.get("keep")]
    redundant_blocks = [int(label["block_id"]) for label in labels if not label.get("keep")]
    return {
        "sample_id": sample_id,
        "source_field": source_field,
        "source_scope": source_scope,
        "targets": list(targets),
        "target_answer_alignment": alignment,
        "atomic_segments": list(atomic_segments),
        "blocks": list(blocks),
        "block_labels": list(labels),
        "dependency_graph": list(dependency_graph),
        "supporting_blocks": supporting_blocks,
        "redundant_blocks": redundant_blocks,
        "quality_flags": quality_flags,
        "engineering_flags": engineering_flags,
        "statistics": statistics,
        "raw_responses": raw_responses,
        "pipeline_version": PIPELINE_VERSION,
    }


def validate_v42_output(payload: Dict[str, Any], cot_text: str) -> None:
    required = {
        "sample_id",
        "targets",
        "atomic_segments",
        "blocks",
        "block_labels",
        "dependency_graph",
        "supporting_blocks",
        "redundant_blocks",
        "quality_flags",
        "engineering_flags",
        "statistics",
        "raw_responses",
        "pipeline_version",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Missing output fields: {missing}")

    target_ids = {int(target["target_id"]) for target in payload["targets"]}
    block_ids = {int(block["block_id"]) for block in payload["blocks"]}
    if not target_ids:
        raise ValueError("Output has no targets.")
    if not block_ids:
        raise ValueError("Output has no blocks.")

    for seg in payload["atomic_segments"]:
        start = int(seg["char_start"])
        end = int(seg["char_end"])
        if cot_text[start:end] != seg.get("text"):
            raise ValueError(f"Atomic segment {seg.get('seg_id')} is not an exact original span.")

    for block in payload["blocks"]:
        text = str(block.get("text", ""))
        if not text or text not in cot_text:
            raise ValueError(f"Block {block.get('block_id')} text is not an exact original substring.")
        if block.get("source") not in {"segment_group", "internal_split"}:
            raise ValueError(f"Invalid block source: {block.get('source')}")
        if block.get("type") not in v4.VALID_BLOCK_TYPES:
            raise ValueError(f"Invalid block type: {block.get('type')}")

    for label in payload["block_labels"]:
        block_id = int(label["block_id"])
        if block_id not in block_ids:
            raise ValueError(f"Label references unknown block_id={block_id}")
        if label.get("type") not in {"SUPPORTING", "NON_SUPPORTING"}:
            raise ValueError(f"Invalid V3-compatible label type: {label.get('type')}")
        if label.get("label_subtype") not in VALID_LABEL_SUBTYPES:
            raise ValueError(f"Invalid label_subtype: {label.get('label_subtype')}")
        for target_id in label.get("supports", []):
            if int(target_id) not in target_ids:
                raise ValueError(f"Label references unknown target_id={target_id}")

    for edge in payload["dependency_graph"]:
        if int(edge.get("block")) not in block_ids:
            raise ValueError(f"Dependency graph references unknown block: {edge}")
        if int(edge.get("target")) not in target_ids:
            raise ValueError(f"Dependency graph references unknown target: {edge}")


def process_one(
    client: v4.LLMClient,
    prompts: Dict[str, str],
    record: Dict[str, Any],
    fallback_id: str,
    marker_dict: Path,
    max_atomic_segments: int,
    temperature: float,
    max_tokens: int,
    parse_retries: int,
    logger: logging.Logger,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    sample_id = sample_id_from_record(record, fallback_id)
    raw_context: Dict[str, Optional[str]] = {
        "target_fallback_raw_response": None,
        "block_refinement_raw_response": None,
        "block_labeling_raw_response": None,
        "dependency_graph_raw_response": None,
    }
    try:
        question = v4.extract_question(record)
        cot_text, source_field, source_scope = extract_scoped_cot_text(record)
        if not cot_text:
            raise v4.StageError("atomic_segmentation", "empty_cot", "No CoT text found in record.")

        try:
            atomic_segments, atomic_engineering_flags = segment_atomic(
                cot_text,
                marker_dict=marker_dict,
                max_atomic_segments=max_atomic_segments,
            )
        except Exception as exc:
            raise v4.StageError("atomic_segmentation", "exact_substring_error", str(exc)) from exc
        if not atomic_segments:
            raise v4.StageError("atomic_segmentation", "empty_segments", "Atomic segmentation produced no segments.")

        engineering_flags: List[str] = filter_engineering_flags(atomic_engineering_flags)

        targets, target_fallback_raw, target_repaired, target_engineering = extract_targets_v42(
            client,
            prompts["target_fallback"],
            question,
            cot_text,
            temperature,
            max_tokens,
            parse_retries,
            logger,
        )
        raw_context["target_fallback_raw_response"] = target_fallback_raw
        if not targets:
            raise v4.StageError("target_extraction", "empty_targets", "No targets extracted from CoT.")
        targets = canonicalize_targets_v42(targets, cot_text)
        engineering_flags.extend(target_engineering)
        if target_repaired:
            engineering_flags.append("json_repair_risk")

        answer_text = v4.extract_answer_field(record)
        alignment = v4.target_answer_alignment(targets, answer_text)

        refine_messages = make_refinement_messages(
            prompts["refine_blocks"],
            question,
            targets,
            atomic_segments,
        )
        compact_refine_messages = make_compact_refinement_messages(
            prompts["refine_blocks"],
            question,
            targets,
            atomic_segments,
        )
        refine_data, refine_raw, refine_repaired, refine_budget_flags = request_json_stage_budgeted(
            client,
            refine_messages,
            expected_key="blocks",
            stage="block_refinement",
            temperature=temperature,
            max_tokens=max_tokens,
            parse_retries=parse_retries,
            compact_messages=compact_refine_messages,
            fallback_max_tokens=8192,
        )
        raw_context["block_refinement_raw_response"] = refine_raw
        blocks, refine_engineering = sanitize_refined_blocks(refine_data, atomic_segments, cot_text)
        engineering_flags.extend(refine_budget_flags)
        engineering_flags.extend(refine_engineering)
        if refine_repaired:
            engineering_flags.append("json_repair_risk")

        refine_context = compact_refinement_context(blocks, refine_engineering)
        labeling_user = make_labeling_user(prompts["block_labeling"], question, targets, blocks)
        labeling_messages = refine_messages + [
            {"role": "assistant", "content": refine_context},
            {"role": "user", "content": labeling_user},
        ]
        compact_labeling_messages = make_compact_labeling_messages(prompts["block_labeling"], question, targets, blocks)
        label_data, label_raw, label_repaired, label_retry_flags = request_json_stage_budgeted(
            client,
            labeling_messages,
            expected_key="labels",
            stage="block_labeling",
            temperature=temperature,
            max_tokens=max_tokens,
            parse_retries=parse_retries,
            compact_messages=compact_labeling_messages,
            fallback_max_tokens=8192,
        )
        raw_context["block_labeling_raw_response"] = label_raw
        labels, label_quality, label_engineering = sanitize_block_labels(label_data, blocks, targets)
        del label_quality
        quality_flags: List[str] = []
        engineering_flags.extend(label_retry_flags)
        engineering_flags.extend(label_engineering)
        if label_repaired:
            engineering_flags.append("json_repair_risk")

        label_context = compact_labeling_context(labels, quality_flags)
        graph_user = make_graph_user(prompts["dependency_graph"], question, targets, blocks, labels)
        graph_messages = labeling_messages + [
            {"role": "assistant", "content": label_context},
            {"role": "user", "content": graph_user},
        ]
        compact_graph_messages = make_compact_graph_messages(prompts["dependency_graph"], question, targets, blocks, labels)
        graph_data, graph_raw, graph_repaired, graph_budget_flags = request_json_stage_budgeted(
            client,
            graph_messages,
            expected_key="dependency_graph",
            stage="dependency_graph",
            temperature=temperature,
            max_tokens=min(max_tokens, 8192),
            parse_retries=parse_retries,
            compact_messages=compact_graph_messages,
            fallback_max_tokens=4096,
        )
        raw_context["dependency_graph_raw_response"] = graph_raw
        labels, dependency_graph, graph_quality, graph_engineering = sanitize_dependency_graph(
            graph_data,
            blocks,
            targets,
            labels,
        )
        quality_flags = filter_quality_flags(ordered_flags(quality_flags, graph_quality))
        engineering_flags.extend(graph_budget_flags)
        engineering_flags.extend(graph_engineering)
        if graph_repaired:
            engineering_flags.append("json_repair_risk")

        payload = assemble_output(
            sample_id=sample_id,
            source_field=source_field,
            source_scope=source_scope,
            targets=targets,
            alignment=alignment,
            atomic_segments=atomic_segments,
            blocks=blocks,
            labels=labels,
            dependency_graph=dependency_graph,
            quality_flags=quality_flags,
            engineering_flags=engineering_flags,
            raw_responses=raw_context,
        )
        validate_v42_output(payload, cot_text)
        return payload, None
    except v4.StageError as exc:
        raw_response = exc.raw_response
        if raw_response is None:
            raw_response = raw_context.get(f"{exc.stage}_raw_response")
        return None, v4.make_failure(sample_id, exc.stage, exc.error_type, str(exc), record, raw_response)
    except Exception as exc:
        return None, v4.make_failure(
            sample_id,
            "output_assembly",
            v4.classify_exception(exc),
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


def write_checkpoint(ckpt_path: Path, sample_id: str, status: str, stage: str, output_file: str) -> None:
    v4.append_jsonl(
        ckpt_path,
        {
            "sample_id": sample_id,
            "status": status,
            "stage": stage,
            "output_file": output_file,
            "timestamp": datetime.now().isoformat(),
        },
    )


def _mean(values: Sequence[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    mid = len(vals) // 2
    if len(vals) % 2:
        return round(vals[mid], 4)
    return round((vals[mid - 1] + vals[mid]) / 2, 4)


def aggregate_stats(comp_path: Path, fail_path: Path, elapsed: float) -> Dict[str, Any]:
    results = [row for _, row in iter_jsonl(comp_path)] if comp_path.exists() else []
    failures = [row for _, row in iter_jsonl(fail_path)] if fail_path.exists() else []
    failure_breakdown = Counter(row.get("error_type", row.get("type", "unknown")) for row in failures)
    failure_stages = Counter(row.get("stage", "unknown") for row in failures)
    total_seen = len(results) + len(failures)

    if not results:
        return {
            "timestamp": datetime.now().isoformat(),
            "pipeline_version": PIPELINE_VERSION,
            "num_success": 0,
            "num_failed": len(failures),
            "elapsed_seconds": round(elapsed, 1),
            "failure_breakdown": dict(failure_breakdown),
            "failure_stage_distribution": dict(failure_stages),
        }

    def stat_values(key: str) -> List[float]:
        return [float(row.get("statistics", {}).get(key, 0.0)) for row in results]

    label_subtypes: Counter[str] = Counter()
    block_types: Counter[str] = Counter()
    atomic_markers: Counter[str] = Counter()
    quality_flags: Counter[str] = Counter()
    engineering_flags: Counter[str] = Counter()
    dirty_kept = 0
    kept_total = 0

    for row in results:
        label_subtypes.update(
            str(label.get("label_subtype", label.get("type", "UNKNOWN"))) for label in row.get("block_labels", [])
        )
        block_types.update(str(block.get("type", "unknown")) for block in row.get("blocks", []))
        atomic_markers.update(str(seg.get("marker_category", "unknown")) for seg in row.get("atomic_segments", []))
        quality_flags.update(str(flag) for flag in row.get("quality_flags", []))
        engineering_flags.update(str(flag) for flag in row.get("engineering_flags", []))
        block_by_id = {int(block.get("block_id", -1)): block for block in row.get("blocks", [])}
        for label in row.get("block_labels", []):
            if label.get("keep"):
                kept_total += 1
                block = block_by_id.get(int(label.get("block_id", -1)), {})
                if block_has_dirty_trace(str(block.get("text", ""))):
                    dirty_kept += 1

    return {
        "timestamp": datetime.now().isoformat(),
        "pipeline_version": PIPELINE_VERSION,
        "num_success": len(results),
        "num_failed": len(failures),
        "elapsed_seconds": round(elapsed, 1),
        "mean_cr_blocks": _mean(stat_values("compression_ratio_blocks")),
        "median_cr_blocks": _median(stat_values("compression_ratio_blocks")),
        "mean_cr_tokens": _mean(stat_values("compression_ratio_tokens")),
        "median_cr_tokens": _median(stat_values("compression_ratio_tokens")),
        "mean_target_coverage": _mean(stat_values("target_coverage")),
        "median_target_coverage": _median(stat_values("target_coverage")),
        "mean_dependency_open_rate": _mean(stat_values("dependency_open_rate")),
        "mean_redundant_derivation_count": _mean(stat_values("redundant_derivation_count")),
        "mean_redundant_derivation_rate": _mean(stat_values("redundant_derivation_rate")),
        "mean_num_atomic_segments": _mean(stat_values("num_atomic_segments")),
        "median_num_atomic_segments": _median(stat_values("num_atomic_segments")),
        "mean_num_blocks": _mean(stat_values("num_blocks")),
        "median_num_blocks": _median(stat_values("num_blocks")),
        "mean_atomic_to_block_ratio": _mean(stat_values("atomic_to_block_ratio")),
        "label_subtype_distribution": dict(label_subtypes),
        "block_type_distribution": dict(block_types),
        "atomic_marker_distribution": dict(atomic_markers),
        "quality_flag_distribution": dict(quality_flags),
        "engineering_flag_distribution": dict(engineering_flags),
        "quality_flag_rates": {flag: round(count / len(results), 4) for flag, count in quality_flags.items()},
        "engineering_flag_rates": {flag: round(count / len(results), 4) for flag, count in engineering_flags.items()},
        "risk_rates": {
            "repeated_final_risk": round(quality_flags.get("repeated_final_risk", 0) / len(results), 4),
            "summary_only_risk": round(quality_flags.get("summary_only_risk", 0) / len(results), 4),
            "proof_gap_risk": round(quality_flags.get("proof_gap_risk", 0) / len(results), 4),
            "dirty_trace_risk": round(quality_flags.get("dirty_trace_risk", 0) / len(results), 4),
            "undercompressed_risk": round(quality_flags.get("undercompressed_risk", 0) / len(results), 4),
            "overcompressed_risk": round(quality_flags.get("overcompressed_risk", 0) / len(results), 4),
        },
        "redundant_derivation_count": label_subtypes.get("REDUNDANT_DERIVATION", 0),
        "redundant_derivation_rate": round(label_subtypes.get("REDUNDANT_DERIVATION", 0) / max(sum(label_subtypes.values()), 1), 4),
        "context_window_retry_count": engineering_flags.get("context_window_retry", 0),
        "context_window_retry_rate": round(engineering_flags.get("context_window_retry", 0) / len(results), 4),
        "context_compact_mode_count": engineering_flags.get("context_compact_mode", 0),
        "context_compact_mode_rate": round(engineering_flags.get("context_compact_mode", 0) / len(results), 4),
        "undercompressed_risk_count": quality_flags.get("undercompressed_risk", 0),
        "undercompressed_risk_rate": round(quality_flags.get("undercompressed_risk", 0) / len(results), 4),
        "dirty_trace_in_kept_rate": round(dirty_kept / kept_total, 4) if kept_total else 0.0,
        "parse_failure_rate": round(failure_breakdown.get("json_parse_error", 0) / total_seen, 4) if total_seen else 0.0,
        "schema_error_rate": round(failure_breakdown.get("schema_error", 0) / total_seen, 4) if total_seen else 0.0,
        "failure_breakdown": dict(failure_breakdown),
        "failure_stage_distribution": dict(failure_stages),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="V4.2 marker-atomic joint CoT compression pipeline.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("output_joint_v42_3k"))
    parser.add_argument("--base-url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--model", type=str, default="Qwen2.5-32B-Instruct")
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--max-atomic-segments", type=int, default=160)
    parser.add_argument("--marker-dict", type=Path, default=Path("legacy_v1/logic_markers_dict.json"))
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
        "target_fallback": v4.read_prompt("fallback.txt"),
        "refine_blocks": v4.read_prompt("refine_blocks.txt"),
        "block_labeling": v4.read_prompt("block_labeling.txt"),
        "dependency_graph": v4.read_prompt("dependency_graph.txt"),
    }

    client = v4.LLMClient(
        base_url=args.base_url,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )

    marker_dict = resolve_marker_dict(args.marker_dict)
    done_ids = set() if args.no_resume else load_success_checkpoint(ckpt_path)
    if done_ids:
        logger.info("[resume] skipping %d successful samples from checkpoint", len(done_ids))

    records = v4.load_records(args.input)
    if args.limit is not None:
        records = records[: args.limit]

    total_to_visit = len(records)
    succ = 0
    fail = 0
    skipped = 0
    start = time.time()
    logger.info(
        "start V4.2 marker-atomic joint pipeline | records=%d | output=%s | model=%s | base_url=%s | marker_dict=%s",
        total_to_visit,
        output_dir,
        args.model,
        args.base_url,
        marker_dict,
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
            marker_dict=marker_dict,
            max_atomic_segments=args.max_atomic_segments,
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
            v4.append_jsonl(comp_path, result)
            write_checkpoint(ckpt_path, sample_id, "success", "done", str(comp_path))
            stats = result["statistics"]
            logger.info(
                "[%d/%d] OK %s | atomic=%d blocks=%d keep=%d cr=%.4f cov=%.4f qflags=%d eflags=%d | %.1fs item | %.1fs avg",
                ordinal,
                total_to_visit,
                sample_id,
                stats["num_atomic_segments"],
                stats["num_blocks"],
                stats["num_supporting"],
                stats["compression_ratio_blocks"],
                stats["target_coverage"],
                stats["num_quality_flags"],
                stats["num_engineering_flags"],
                time.time() - item_start,
                avg,
            )
        else:
            fail += 1
            assert failure is not None
            v4.append_jsonl(fail_path, failure)
            write_checkpoint(ckpt_path, sample_id, "failed", str(failure.get("stage", "unknown")), str(fail_path))
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
        "DONE | success=%d failed=%d skipped=%d elapsed=%.1fs mean_cr=%s mean_cov=%s dirty_kept=%s",
        succ,
        fail,
        skipped,
        time.time() - start,
        aggregate.get("mean_cr_blocks"),
        aggregate.get("mean_target_coverage"),
        aggregate.get("dirty_trace_in_kept_rate"),
    )


if __name__ == "__main__":
    main()
