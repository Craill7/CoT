#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared helpers for Long-CoT dependency-oriented compression.

The functions in this module are intentionally conservative. They preserve
original text, extract only explicitly stated targets, and mark uncertain
dependencies as NON_SUPPORTING.
"""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple


TEXT_FIELD_PRIORITY: Tuple[str, ...] = (
    "original_cot",
    "cot",
    "chain_of_thought",
    "reasoning",
    "rationale",
    "response",
    "output",
    "generation",
    "generations",
    "solution",
    "messages",
)

FINAL_ANSWER_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(
        r"(?:final\s+answer|answer|therefore|thus|so)\s*(?:is|:|=)?\s*(.+)$",
        flags=re.IGNORECASE,
    ),
    re.compile(r"(?:答案|最终答案|所以|因此)\s*(?:是|为|:|：)?\s*(.+)$"),
)

REASONING_MARKERS: Tuple[str, ...] = (
    "because",
    "since",
    "therefore",
    "thus",
    "hence",
    "so",
    "then",
    "we get",
    "we have",
    "implies",
    "derive",
    "calculate",
    "solve",
    "substitute",
    "simplify",
    "verify",
    "check",
    "assume",
    "let",
    "given",
    "as a result",
    "因为",
    "由于",
    "所以",
    "因此",
    "可得",
    "得到",
    "计算",
    "化简",
    "代入",
    "验证",
    "设",
)

NON_SUPPORTING_MARKERS: Tuple[str, ...] = (
    "wait",
    "oops",
    "mistake",
    "wrong",
    "irrelevant",
    "not needed",
    "ignore",
    "instead",
    "let me think",
    "I should",
    "actually",
    "sorry",
    "等等",
    "错了",
    "不对",
    "无关",
    "忽略",
    "重新",
)

STOPWORDS: Set[str] = {
    "the",
    "and",
    "that",
    "this",
    "then",
    "with",
    "from",
    "have",
    "will",
    "into",
    "case",
    "therefore",
    "thus",
    "hence",
    "because",
    "since",
    "answer",
    "final",
    "boxed",
    "math",
    "text",
    "frac",
    "sqrt",
    "left",
    "right",
    "cdot",
    "times",
    "so",
    "we",
    "is",
    "are",
    "to",
    "of",
    "in",
    "for",
    "a",
    "an",
    "it",
    "as",
    "on",
    "by",
    "be",
    "or",
    "if",
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig", errors="replace") as f:
        return json.load(f)


def write_json(path: Path, payload: Any, pretty: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        if pretty:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        else:
            json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))


def iter_jsonl(path: Path) -> Iterator[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8-sig", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"JSONL record at {path}:{line_no} is not an object.")
            yield line_no, obj


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def flatten_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(flatten_text(item) for item in value if item is not None)
    if isinstance(value, dict):
        preferred: List[str] = []
        for key in ("content", "text", "response", "output", "reasoning"):
            if key in value:
                preferred.append(flatten_text(value[key]))
        if preferred:
            return "\n".join(part for part in preferred if part.strip())
        return "\n".join(flatten_text(v) for v in value.values())
    return str(value)


def extract_cot_text(record: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    for field in TEXT_FIELD_PRIORITY:
        if field not in record:
            continue
        value = record[field]
        if field == "generations" and isinstance(value, list):
            candidates = [flatten_text(item).strip() for item in value]
            candidates = [item for item in candidates if item]
            if candidates:
                return max(candidates, key=len), field
            continue
        text = flatten_text(value).strip()
        if text:
            return text, field
    return "", None


def sample_id_from_record(record: Dict[str, Any], fallback: str) -> str:
    for key in ("sample_id", "id", "uid", "qid", "question_id", "index"):
        if key in record and record[key] is not None:
            return str(record[key])
    return fallback


def find_matching_brace(text: str, open_brace_idx: int) -> Optional[int]:
    if open_brace_idx < 0 or open_brace_idx >= len(text) or text[open_brace_idx] != "{":
        return None
    depth = 0
    escaped = False
    for idx in range(open_brace_idx, len(text)):
        ch = text[idx]
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return idx
    return None


def extract_boxed_values(text: str) -> List[str]:
    values: List[str] = []
    for match in re.finditer(r"\\boxed\s*\{", text):
        open_idx = match.end() - 1
        close_idx = find_matching_brace(text, open_idx)
        if close_idx is None:
            continue
        value = text[open_idx + 1 : close_idx].strip()
        if value:
            values.append(value)
    return values


def strip_think_tags(text: str) -> str:
    return re.sub(r"</?think>", "", text, flags=re.IGNORECASE).strip()


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?\u3002\uff01\uff1f])\s+|\n+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def clean_target_text(text: str) -> str:
    text = strip_think_tags(text).strip()
    text = re.sub(r"^[:：=\-\s]+", "", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" \t\r\n.。")
    return text


def canonical_target(text: str) -> str:
    text = clean_target_text(text).lower()
    text = re.sub(r"\\(?:boxed|text|mathrm)\s*\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\[a-zA-Z]+", "", text)
    text = re.sub(r"[\s$`'\".,;:，。；：]+", "", text)
    return text


def equivalent_targets(left: str, right: str) -> bool:
    left_key = canonical_target(left)
    right_key = canonical_target(right)
    if not left_key or not right_key:
        return False
    if left_key == right_key:
        return True
    return SequenceMatcher(None, left_key, right_key).ratio() >= 0.92


def merge_targets(raw_targets: Sequence[str]) -> List[Dict[str, Any]]:
    merged: List[str] = []
    for raw in raw_targets:
        target = clean_target_text(raw)
        if not target:
            continue
        if any(equivalent_targets(target, existing) for existing in merged):
            continue
        merged.append(target)
    return [{"target_id": idx, "description": target} for idx, target in enumerate(merged)]


def fallback_final_answer(text: str) -> List[str]:
    sentences = split_sentences(text)
    candidates: List[str] = []
    for sentence in reversed(sentences[-8:]):
        for pattern in FINAL_ANSWER_PATTERNS:
            match = pattern.search(sentence)
            if match:
                candidate = clean_target_text(match.group(1))
                if candidate:
                    candidates.append(candidate)
                    break
        if candidates:
            break
    if not candidates and sentences:
        candidates.append(clean_target_text(sentences[-1]))
    return candidates


def extract_targets(cot_text: str) -> List[Dict[str, Any]]:
    boxed = extract_boxed_values(cot_text)
    raw_targets = boxed if boxed else fallback_final_answer(cot_text)
    return merge_targets(raw_targets)


def normalize_newlines(text: str) -> str:
    return re.sub(r"\r\n?", "\n", text).strip()


def should_start_new_block(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if re.match(r"^(?:step|case|part)\s*\d+[\).:\-]", stripped, flags=re.IGNORECASE):
        return True
    if re.match(r"^\(?\d+[\).]\s+", stripped):
        return True
    if re.match(r"^[A-Za-z]\)\s+", stripped):
        return True
    if re.match(r"^(?:First|Second|Third|Next|Finally|Now|Then|Thus|Therefore|Hence)\b", stripped):
        return True
    if re.match(r"^(?:首先|其次|然后|接着|最后|因此|所以)", stripped):
        return True
    return False


def merge_short_blocks(blocks: Sequence[str], min_chars: int = 28) -> List[str]:
    merged: List[str] = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        if merged and len(block) < min_chars:
            merged[-1] = f"{merged[-1]}\n{block}".strip()
        else:
            merged.append(block)
    return merged


def split_long_paragraph(paragraph: str, max_chars: int = 900) -> List[str]:
    paragraph = paragraph.strip()
    if len(paragraph) <= max_chars:
        return [paragraph]
    sentences = split_sentences(paragraph)
    if len(sentences) <= 1:
        return [paragraph]

    blocks: List[str] = []
    current: List[str] = []
    current_len = 0
    for sentence in sentences:
        sentence_len = len(sentence)
        if current and current_len + sentence_len > max_chars and should_start_new_block(sentence):
            blocks.append(" ".join(current).strip())
            current = [sentence]
            current_len = sentence_len
        else:
            current.append(sentence)
            current_len += sentence_len + 1
    if current:
        blocks.append(" ".join(current).strip())
    return blocks or [paragraph]


def segment_blocks(cot_text: str) -> List[Dict[str, Any]]:
    text = normalize_newlines(strip_think_tags(cot_text))
    if not text:
        return []

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", text) if part.strip()]
    coarse_blocks: List[str] = []
    for paragraph in paragraphs:
        lines = [line.rstrip() for line in paragraph.split("\n") if line.strip()]
        if len(lines) <= 1:
            coarse_blocks.extend(split_long_paragraph(paragraph))
            continue

        current: List[str] = []
        for line in lines:
            if current and should_start_new_block(line):
                coarse_blocks.append("\n".join(current).strip())
                current = [line.strip()]
            else:
                current.append(line.strip())
        if current:
            coarse_blocks.append("\n".join(current).strip())

    if len(coarse_blocks) <= 1:
        coarse_blocks = split_long_paragraph(text)

    merged = merge_short_blocks(coarse_blocks)
    return [{"block_id": idx, "text": block} for idx, block in enumerate(merged)]


def tokenize_for_dependency(text: str) -> Set[str]:
    normalized = text.lower()
    normalized = re.sub(r"\\[a-zA-Z]+", " ", normalized)
    tokens = re.findall(r"[a-zA-Z_][a-zA-Z_0-9]*|[-+]?\d+(?:\.\d+)?|[\u4e00-\u9fff]+", normalized)
    return {token for token in tokens if token not in STOPWORDS and len(token) > 1}


def has_reasoning_signal(text: str) -> bool:
    lowered = text.lower()
    return (
        any(marker in lowered for marker in REASONING_MARKERS)
        or bool(re.search(r"[=<>≤≥]|\\frac|\\sqrt|\\sum|\\int|\d\s*[-+*/]\s*\d", text))
    )


def is_likely_non_supporting(text: str) -> bool:
    lowered = text.lower()
    if any(marker in lowered for marker in NON_SUPPORTING_MARKERS):
        return True
    if len(tokenize_for_dependency(text)) <= 1 and not re.search(r"\\boxed|\d|[=<>]", text):
        return True
    return False


def block_supports_target(block_text: str, target_text: str) -> bool:
    block_key = canonical_target(block_text)
    target_key = canonical_target(target_text)
    target_tokens = tokenize_for_dependency(target_text)
    block_tokens = tokenize_for_dependency(block_text)

    if target_key and target_key in block_key:
        return True

    if is_likely_non_supporting(block_text):
        return False

    if target_tokens:
        overlap = target_tokens & block_tokens
        if len(overlap) == len(target_tokens) and has_reasoning_signal(block_text):
            return True
        if len(overlap) >= 2 and has_reasoning_signal(block_text):
            return True

    target_numbers = set(re.findall(r"[-+]?\d+(?:\.\d+)?", target_text))
    block_numbers = set(re.findall(r"[-+]?\d+(?:\.\d+)?", block_text))
    if target_numbers and target_numbers <= block_numbers and has_reasoning_signal(block_text):
        return True

    return False


def label_dependencies(
    blocks: Sequence[Dict[str, Any]], targets: Sequence[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    labels: List[Dict[str, Any]] = []
    graph: List[Dict[str, Any]] = []

    for block in blocks:
        block_id = int(block["block_id"])
        text = str(block.get("text", ""))
        supports: List[int] = []
        for target in targets:
            target_id = int(target["target_id"])
            target_text = str(target.get("description", ""))
            if block_supports_target(text, target_text):
                supports.append(target_id)
                graph.append(
                    {
                        "block_id": block_id,
                        "target_id": target_id,
                        "block": block_id,
                        "target": target_id,
                    }
                )

        labels.append(
            {
                "block_id": block_id,
                "type": "SUPPORTING" if supports else "NON_SUPPORTING",
                "supports": supports,
            }
        )

    return labels, graph


def compute_statistics(
    blocks: Sequence[Dict[str, Any]],
    labels: Sequence[Dict[str, Any]],
    targets: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    num_blocks = len(blocks)
    supporting = [label for label in labels if label["type"] == "SUPPORTING"]
    supported_targets = {target_id for label in labels for target_id in label.get("supports", [])}
    num_supporting = len(supporting)
    num_redundant = num_blocks - num_supporting
    compression_ratio_blocks = num_supporting / num_blocks if num_blocks else 0.0
    target_coverage = len(supported_targets) / len(targets) if targets else 0.0
    return {
        "num_blocks": num_blocks,
        "num_supporting": num_supporting,
        "num_redundant": num_redundant,
        "compression_ratio_blocks": compression_ratio_blocks,
        "compression_ratio": compression_ratio_blocks,
        "target_coverage": target_coverage,
    }


def validate_output(payload: Dict[str, Any]) -> None:
    required = {
        "sample_id",
        "targets",
        "blocks",
        "block_labels",
        "dependency_graph",
        "redundant_blocks",
        "supporting_blocks",
        "statistics",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Missing output fields: {missing}")

    block_ids = {block["block_id"] for block in payload["blocks"]}
    target_ids = {target["target_id"] for target in payload["targets"]}
    for label in payload["block_labels"]:
        if label["block_id"] not in block_ids:
            raise ValueError(f"Label references unknown block_id={label['block_id']}")
        if label["type"] not in {"SUPPORTING", "NON_SUPPORTING"}:
            raise ValueError(f"Invalid label type: {label['type']}")
        for target_id in label["supports"]:
            if target_id not in target_ids:
                raise ValueError(f"Label references unknown target_id={target_id}")
        if label["type"] == "NON_SUPPORTING" and label["supports"]:
            raise ValueError("NON_SUPPORTING label cannot have supports.")
        if label["type"] == "SUPPORTING" and not label["supports"]:
            raise ValueError("SUPPORTING label must support at least one target.")


def compress_sample(record: Dict[str, Any], fallback_id: str = "0") -> Dict[str, Any]:
    cot_text, _ = extract_cot_text(record)
    sample_id = sample_id_from_record(record, fallback_id)
    targets = extract_targets(cot_text)
    blocks = segment_blocks(cot_text)
    labels, graph = label_dependencies(blocks, targets)
    supporting_blocks = [
        label["block_id"] for label in labels if label["type"] == "SUPPORTING"
    ]
    redundant_blocks = [
        label["block_id"] for label in labels if label["type"] == "NON_SUPPORTING"
    ]
    payload = {
        "sample_id": sample_id,
        "targets": targets,
        "blocks": blocks,
        "block_labels": labels,
        "dependency_graph": graph,
        "redundant_blocks": redundant_blocks,
        "supporting_blocks": supporting_blocks,
        "statistics": compute_statistics(blocks, labels, targets),
    }
    validate_output(payload)
    return payload
