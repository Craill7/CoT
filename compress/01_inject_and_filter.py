#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_inject_and_filter.py

功能：
1. 读取 CoT/3k_test_data/data_sample_3k.jsonl。
2. 将每条样本原生字段 answer 注入到 generations 的 \\boxed{...} 中。
3. 提取 <think>...</think> 内部思考过程并统计长度。
4. 按 70 分位数阈值进行物理分流：
   - 长度 > 阈值：CoT/compress/data_group_A_long.jsonl
   - 长度 <= 阈值：CoT/compress/data_group_C_keep.jsonl

说明：
- 本脚本不依赖第三方库，只使用 Python 标准库。
- 为了避免正则在嵌套花括号时失效，\\boxed{...} 替换使用轻量级括号扫描器实现。
"""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


ROOT_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT_DIR / "3k_test_data" / "data_sample_3k.jsonl"
OUTPUT_LONG_PATH = ROOT_DIR / "compress" / "data_group_A_long.jsonl"
OUTPUT_KEEP_PATH = ROOT_DIR / "compress" / "data_group_C_keep.jsonl"


THINK_RE = re.compile(r"<think>(.*?)</think>", flags=re.IGNORECASE | re.DOTALL)
BOXED_TOKEN_RE = re.compile(r"\\boxed\s*\{")


def iter_jsonl(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    """逐行读取 JSONL，并在 JSON 解析失败时给出明确行号。"""
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSON decode failed at line {line_no}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Line {line_no} is not a JSON object.")
            yield line_no, obj


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    """以 UTF-8 JSONL 写出，保留中文和数学符号的可读性。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def find_matching_brace(text: str, open_brace_idx: int) -> Optional[int]:
    """
    给定 text 中某个左花括号的位置，返回与之匹配的右花括号位置。

    该函数用于处理类似 \\boxed{\\frac{1}{2}} 的嵌套结构。普通正则
    r"\\\\boxed\\{.*?\\}" 会在第一个右花括号处提前停止，导致替换错误。
    """
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


def replace_first_boxed_answer(text: str, answer: str) -> Tuple[str, bool]:
    """
    将第一个合法的 \\boxed{...} 内部内容替换为 answer。

    返回：
        (new_text, replaced)

    边缘情况：
    - 找不到 \\boxed{：返回原文和 False。
    - 找到 \\boxed{ 但花括号不闭合：继续寻找下一个候选；若全部失败则返回 False。
    - answer 中如含反斜杠或花括号，按普通文本原样放入 boxed。
    """
    for match in BOXED_TOKEN_RE.finditer(text):
        open_brace_idx = match.end() - 1
        close_brace_idx = find_matching_brace(text, open_brace_idx)
        if close_brace_idx is None:
            continue
        new_text = text[: open_brace_idx + 1] + answer + text[close_brace_idx:]
        return new_text, True
    return text, False


def inject_ground_truth(generation: Any, answer: Any) -> str:
    """
    将 Ground Truth answer 注入到 generations 文本中。

    generations/answer 在脏数据中可能不是字符串，因此统一转 str；
    None 会转为空字符串，避免写出 "None" 作为答案。
    """
    generation_text = "" if generation is None else str(generation)
    answer_text = "" if answer is None else str(answer).strip()

    injected, replaced = replace_first_boxed_answer(generation_text, answer_text)
    if replaced:
        return injected

    suffix = f"\nFinal Answer\n\\boxed{{{answer_text}}}"
    if generation_text.endswith("\n"):
        return generation_text.rstrip("\n") + suffix
    return generation_text + suffix


def extract_think_text(generation: str) -> str:
    """
    提取 <think>...</think> 内部文本。

    若存在多个 think 块，拼接所有块用于长度统计；若标签缺失或不闭合，
    退化为使用全文长度，避免把异常格式误判为 0 长度。
    """
    matches = THINK_RE.findall(generation)
    if matches:
        return "\n".join(m.strip() for m in matches)

    lower_text = generation.lower()
    start = lower_text.find("<think>")
    if start >= 0:
        return generation[start + len("<think>") :].strip()

    return generation.strip()


def percentile_nearest_rank(values: List[int], percentile: float) -> int:
    """
    使用 nearest-rank 风格计算分位数阈值。

    对于 70 分位数，返回排序后约 70% 位置的值。随后使用 length > threshold
    分到 A 组，因此 A 组约为最长的 30%，与需求一致。
    """
    if not values:
        raise ValueError("Cannot compute percentile on an empty list.")
    if not 0 <= percentile <= 100:
        raise ValueError("percentile must be in [0, 100].")

    sorted_values = sorted(values)
    # 使用线性位置四舍五入，比简单 ceil 更接近常见数据科学工具的结果。
    pos = round((percentile / 100.0) * (len(sorted_values) - 1))
    pos = max(0, min(pos, len(sorted_values) - 1))
    return sorted_values[pos]


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    processed_rows: List[Dict[str, Any]] = []
    lengths: List[int] = []

    for line_no, row in iter_jsonl(INPUT_PATH):
        if "answer" not in row:
            raise KeyError(f"Line {line_no} has no required field 'answer'.")

        injected_generation = inject_ground_truth(row.get("generations", ""), row.get("answer"))
        think_text = extract_think_text(injected_generation)
        think_char_len = len(think_text)

        row["generations"] = injected_generation
        row["think_char_len"] = think_char_len
        processed_rows.append(row)
        lengths.append(think_char_len)

    if not processed_rows:
        raise ValueError(f"No valid samples found in {INPUT_PATH}")

    threshold = percentile_nearest_rank(lengths, 70)
    group_a_long = [row for row in processed_rows if row["think_char_len"] > threshold]
    group_c_keep = [row for row in processed_rows if row["think_char_len"] <= threshold]

    write_jsonl(OUTPUT_LONG_PATH, group_a_long)
    write_jsonl(OUTPUT_KEEP_PATH, group_c_keep)

    print("Length statistics over <think>...</think> text")
    print(f"  count       : {len(lengths)}")
    print(f"  min         : {min(lengths)}")
    print(f"  max         : {max(lengths)}")
    print(f"  mean        : {statistics.mean(lengths):.2f}")
    print(f"  p70 threshold: {threshold}")
    print("Routing result")
    print(f"  Group A long (>  p70): {len(group_a_long)} -> {OUTPUT_LONG_PATH}")
    print(f"  Group C keep (<= p70): {len(group_c_keep)} -> {OUTPUT_KEEP_PATH}")


if __name__ == "__main__":
    main()
