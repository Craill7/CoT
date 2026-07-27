from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Protocol, Tuple


CONTAMINATION_PATTERNS = (
    r"\u5c06\u4e0a\u9762\u7684\u6587\u672c\u7ffb\u8bd1",
    r"将上面的文本翻译",
    r"translate the (?:above|following) text",
    r"ignore (?:the )?previous",
    r"system prompt",
)


class JsonLLM(Protocol):
    def chat_json(
        self,
        system: str,
        user: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]: ...


def first_text(record: Dict[str, Any], fields: Iterable[str]) -> str:
    for field in fields:
        value = record.get(field)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def flatten_candidates(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        out: List[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
            elif isinstance(item, dict):
                text = first_text(item, ("text", "output", "response", "content"))
                if text:
                    out.append(text)
        return out
    return []


def stable_sample_id(record: Dict[str, Any], fallback: str) -> str:
    for field in ("sample_id", "uuid", "id", "problem_hash"):
        value = record.get(field)
        if value is not None and str(value).strip():
            return str(value).strip()
    question = first_text(record, ("problem", "instruction", "question", "prompt"))
    if question:
        return hashlib.sha1(question.encode("utf-8")).hexdigest()[:20]
    return fallback


def normalize_record(record: Dict[str, Any], fallback: str) -> Dict[str, Any]:
    question = first_text(record, ("problem", "instruction", "question", "prompt"))
    candidates = flatten_candidates(record.get("generations"))
    if not candidates:
        for field in ("output", "cot", "response", "solution_text"):
            candidates = flatten_candidates(record.get(field))
            if candidates:
                break
    ground_truth = first_text(record, ("gt_answer", "answer", "ground_truth"))
    raw_original = record.get("original_final_answer")
    original_finals = (
        [str(value).strip() for value in raw_original if str(value).strip()]
        if isinstance(raw_original, list)
        else []
    )
    original_final = (
        str(raw_original).strip()
        if raw_original is not None and not isinstance(raw_original, list)
        else first_text(record, ("final_answer",))
    )
    return {
        "sample_id": stable_sample_id(record, fallback),
        "question": question,
        "candidates": candidates,
        "candidate_count": len(candidates),
        "ground_truth": ground_truth,
        "original_final_answer": original_final,
        "original_final_answers": original_finals,
        "solution": first_text(record, ("solution", "reference_solution")),
        "source_record": record,
    }


def iter_jsonl(path: Path) -> Iterator[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if line.strip():
                yield line_no, json.loads(line)


def json_fingerprint(payload: Dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def is_contaminated(*texts: str) -> bool:
    joined = "\n".join(texts).lower()
    return any(re.search(pattern, joined, flags=re.IGNORECASE) for pattern in CONTAMINATION_PATTERNS)


def extract_boxed_values(text: str) -> List[str]:
    values: List[str] = []
    marker = r"\boxed{"
    cursor = 0
    while True:
        start = text.find(marker, cursor)
        if start < 0:
            break
        pos = start + len(marker)
        depth = 0
        for idx in range(pos, len(text)):
            char = text[idx]
            if char == "{":
                depth += 1
            elif char == "}":
                if depth == 0:
                    value = text[pos:idx].strip()
                    if value:
                        values.append(value)
                    cursor = idx + 1
                    break
                depth -= 1
        else:
            break
    return values


def canonical_answer(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\\(?:boxed|fbox)\s*\{(.*)\}", r"\1", text)
    text = text.replace("$", "").replace("\\left", "").replace("\\right", "")
    text = re.sub(r"\s+", "", text)
    return text.strip(".,;:")


def answers_equivalent(left: Any, right: Any) -> bool:
    left_key = canonical_answer(left)
    right_key = canonical_answer(right)
    if not left_key or not right_key:
        return False
    return left_key == right_key


def final_answer(text: str) -> str:
    boxes = extract_boxed_values(text)
    if boxes:
        return boxes[-1]
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    patterns = (
        r"(?:final answer|answer is|therefore|thus|hence)\s*[:：]?\s*(.+)$",
        r"(?:\u7b54\u6848|\u6240\u4ee5)\s*[:：]?\s*(.+)$",
        r"(?:final answer|answer is|therefore|thus|hence)\s*[:：]?\s*(.+)$",
        r"(?:答案|故|所以)\s*[:：]?\s*(.+)$",
    )
    for line in reversed(lines[-16:]):
        for pattern in patterns:
            match = re.search(pattern, line, flags=re.IGNORECASE)
            if match and match.group(1).strip():
                return match.group(1).strip()
    return ""


def answer_gate(record: Dict[str, Any], selected_index: int, cot: str) -> str:
    source = record.get("source_record", {})
    predicted = final_answer(cot)
    if not predicted:
        return "fail"
    for field in ("is_reasoning_complete", "correctness_math_verify"):
        values = source.get(field)
        if isinstance(values, list) and selected_index < len(values):
            if not bool(values[selected_index]):
                return "fail"
            if not record.get("ground_truth"):
                return "pass"
            break
    original_finals = record.get("original_final_answers", [])
    per_candidate_original = (
        original_finals[selected_index]
        if isinstance(original_finals, list) and selected_index < len(original_finals)
        else ""
    )
    reference = (
        record.get("ground_truth")
        or per_candidate_original
        or record.get("original_final_answer")
        or predicted
    )
    if reference and predicted:
        return "pass" if answers_equivalent(reference, predicted) else "fail"
    return "uncertain"


@dataclass(frozen=True)
class PipelineConfig:
    input_path: str
    output_dir: str
    pipeline_version: str = "v2.1.5"
    base_url: str = "http://localhost:8000/v1"
    model: str = "/ky200t/models/Qwen/Qwen2.5-32B-Instruct"
    select_mode: str = "auto"
    clusterer: str = "ifd_kmeans"
    model_path: str = "/ky200t/models/Qwen/Qwen2.5-32B-Instruct"
    num_clusters: int = 40
    k_ratio: float = 0.8
    max_length: int = 2048
    gpu_id: int = 0
    llm_max_tokens: int = 4096
    max_rewrite_rounds: int = 1
    limit: Optional[int] = None
    resume: bool = False
    mock: bool = False

    def fingerprint(self) -> str:
        payload = asdict(self)
        payload.pop("resume", None)
        payload.pop("output_dir", None)
        return json_fingerprint(payload)


class JsonlStage:
    """Append-only JSONL stage with restart-safe key de-duplication."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records: Dict[Tuple[str, str], Dict[str, Any]] = {}
        if path.exists():
            for _, row in iter_jsonl(path):
                key = (str(row.get("sample_id", "")), str(row.get("config_fingerprint", "")))
                if all(key):
                    self.records[key] = row

    def get(self, sample_id: str, fingerprint: str) -> Optional[Dict[str, Any]]:
        return self.records.get((str(sample_id), str(fingerprint)))

    def append(self, row: Dict[str, Any]) -> bool:
        key = (str(row.get("sample_id", "")), str(row.get("config_fingerprint", "")))
        if not all(key):
            raise ValueError("stage row requires sample_id and config_fingerprint")
        if key in self.records:
            return False
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.records[key] = row
        return True


def stage_row(record: Dict[str, Any], fingerprint: str, **extra: Any) -> Dict[str, Any]:
    return {
        "sample_id": str(record["sample_id"]),
        "config_fingerprint": fingerprint,
        **extra,
    }
