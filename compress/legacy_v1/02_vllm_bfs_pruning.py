#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_vllm_bfs_pruning.py

功能：
对 01_inject_and_filter.py 生成的 Group A 超长样本执行“逆向逻辑 BFS 裁剪”。

核心思想：
1. 将 <think>...</think> 内部思考过程切分为若干逻辑块。
2. 从答案附近的尾部逻辑块开始，逐步向前搜索/扩展候选骨架。
3. 每次调用本地 vLLM 判断当前候选：
   - useful: 当前新增逻辑块是否对最终解题有帮助。
   - sufficient: 当前候选骨架是否足以推出最终答案。
4. 一旦 sufficient=True，即认为找到黄金骨架；否则继续扩展。
5. 若搜索结束仍无法 sufficient=True，则写入 failed 文件。

依赖：
    pip install openai tqdm

vLLM OpenAI-compatible endpoint:
    base_url = "http://127.0.0.1:8001/v1"
    api_key  = "EMPTY"
    model    = "Qwen2.5-32B-Instruct"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm


ROOT_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT_DIR / "compress" / "data_group_A_long.jsonl"
OUTPUT_PRUNED_PATH = ROOT_DIR / "compress" / "data_group_A_pruned.jsonl"
OUTPUT_FAILED_PATH = ROOT_DIR / "compress" / "data_group_A_failed.jsonl"

DEFAULT_BASE_URL = "http://127.0.0.1:8001/v1"
DEFAULT_API_KEY = "EMPTY"
DEFAULT_MODEL = "Qwen2.5-32B-Instruct"

THINK_RE = re.compile(r"<think>(.*?)</think>", flags=re.IGNORECASE | re.DOTALL)
BOXED_RE = re.compile(r"\\boxed\s*\{", flags=re.DOTALL)


class PruningError(RuntimeError):
    """单条样本裁剪失败时抛出的可捕获异常。"""


@dataclass(frozen=True)
class JudgeResult:
    useful: bool
    sufficient: bool
    raw: Dict[str, Any]


@dataclass(frozen=True)
class CandidateState:
    """
    搜索状态。

    block_indices 使用升序保存，便于重建原始推理顺序；搜索扩展时从后向前添加。
    """

    block_indices: Tuple[int, ...]

    @property
    def first_index(self) -> int:
        return self.block_indices[0]


def iter_jsonl(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
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


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    """同步追加一行 JSONL；调用方通过 asyncio.Lock 保证并发安全。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_think_span(generation: str) -> Tuple[str, Optional[Tuple[int, int]]]:
    """
    返回思考文本及其在 generation 中的 span。

    若找不到完整 <think>...</think>，返回全文和 None；后续重建时会采用
    “全文替换为裁剪后内容 + 原答案”的保守策略。
    """
    match = THINK_RE.search(generation)
    if match:
        return match.group(1).strip(), match.span(1)
    return generation.strip(), None


def extract_final_answer_text(row: Dict[str, Any], generation: str) -> str:
    """
    获取最终答案。优先使用原生 answer 字段；缺失时尝试从 \\boxed{...} 提取。
    """
    answer = row.get("answer")
    if answer is not None and str(answer).strip():
        return str(answer).strip()

    boxed = extract_first_boxed_content(generation)
    if boxed:
        return boxed.strip()
    return ""


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


def extract_first_boxed_content(text: str) -> Optional[str]:
    for match in BOXED_RE.finditer(text):
        open_idx = match.end() - 1
        close_idx = find_matching_brace(text, open_idx)
        if close_idx is not None:
            return text[open_idx + 1 : close_idx]
    return None


def split_logic_blocks(think_text: str) -> List[str]:
    """
    将长思考过程切分为逻辑块。

    设计原则：
    - 优先按空行、Step/步骤、编号列表等显式边界切分。
    - 若某块仍过长，再按句号/分号/换行等弱边界细分。
    - 避免生成过碎的块，否则 BFS 调用次数会膨胀。
    """
    text = think_text.strip()
    if not text:
        return []

    # 显式边界：空行、Step N、步骤 N、编号列表等。
    explicit_boundary = re.compile(
        r"(?:\n\s*\n+)|(?=\n?\s*(?:Step|步骤)\s*\d+[:：.\)]\s*)|(?=\n?\s*\d+\s*[\.、\)]\s+)",
        flags=re.IGNORECASE,
    )
    rough_blocks = [b.strip() for b in explicit_boundary.split(text) if b and b.strip()]

    blocks: List[str] = []
    max_chars_per_block = 900
    min_chars_to_flush = 180
    weak_sentence_re = re.compile(r"(?<=[。！？.!?；;])\s+|\n+")

    for rough in rough_blocks:
        if len(rough) <= max_chars_per_block:
            blocks.append(rough)
            continue

        buf: List[str] = []
        buf_len = 0
        parts = [p.strip() for p in weak_sentence_re.split(rough) if p and p.strip()]
        for part in parts:
            if buf and buf_len + len(part) > max_chars_per_block and buf_len >= min_chars_to_flush:
                blocks.append(" ".join(buf).strip())
                buf = []
                buf_len = 0
            buf.append(part)
            buf_len += len(part)
        if buf:
            blocks.append(" ".join(buf).strip())

    return [b for b in blocks if b]


def build_candidate_text(blocks: Sequence[str], indices: Sequence[int]) -> str:
    return "\n\n".join(blocks[i].strip() for i in indices if 0 <= i < len(blocks)).strip()


def rebuild_generation_with_pruned_think(
    original_generation: str,
    think_span: Optional[Tuple[int, int]],
    pruned_think: str,
    answer: str,
) -> str:
    """将裁剪后的思考骨架放回 generations，尽量保留原始答案区格式。"""
    if think_span is not None:
        start, end = think_span
        return original_generation[:start] + pruned_think + original_generation[end:]

    return f"<think>\n{pruned_think}\n</think>\n\nFinal Answer\n\\boxed{{{answer}}}"


def coerce_bool(value: Any) -> bool:
    """兼容模型返回 true/false、'true'/'false'、1/0、yes/no 等形式。"""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value != 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "y", "1", "是", "有用", "充分"}:
            return True
        if normalized in {"false", "no", "n", "0", "否", "无用", "不充分"}:
            return False
    return False


def parse_json_object(text: str) -> Dict[str, Any]:
    """
    解析模型输出 JSON。

    即便 response_format=json_object，部分 OpenAI-compatible 服务仍可能返回
    Markdown 代码块或前后冗余文本；这里做一次容错提取。
    """
    text = text.strip()
    if not text:
        raise ValueError("Empty model response.")

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        obj = json.loads(text[start : end + 1])
        if isinstance(obj, dict):
            return obj

    raise ValueError(f"Cannot parse JSON object from model response: {text[:300]}")


class VLLMJudgeClient:
    """基于 AsyncOpenAI 的本地 vLLM 判别客户端。"""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 256,
        max_retries: int = 3,
    ) -> None:
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries

    async def judge(
        self,
        question: str,
        answer: str,
        candidate_reasoning: str,
        new_block: str,
    ) -> JudgeResult:
        prompt = build_judge_prompt(question, answer, candidate_reasoning, new_block)
        last_exc: Optional[BaseException] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a strict verifier for mathematical and logical reasoning. "
                                "Return only a JSON object."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    response_format={"type": "json_object"},
                )
                content = resp.choices[0].message.content or ""
                obj = parse_json_object(content)
                return JudgeResult(
                    useful=coerce_bool(obj.get("useful", False)),
                    sufficient=coerce_bool(obj.get("sufficient", False)),
                    raw=obj,
                )
            except Exception as exc:  # noqa: BLE001 - 需要重试并最终落 failed
                last_exc = exc
                await asyncio.sleep(min(2.0 * attempt, 8.0))

        raise PruningError(f"Judge request failed after {self.max_retries} retries: {last_exc}")


def build_judge_prompt(question: str, answer: str, candidate_reasoning: str, new_block: str) -> str:
    """
    构造裁剪判别提示词。

    useful 判断“新加入的块是否有助于推导答案”；sufficient 判断“候选骨架整体
    是否已经足以推出最终答案”。要求模型严格返回 JSON。
    """
    return f"""
Given a problem, the ground-truth final answer, and a candidate reasoning skeleton, judge whether the skeleton is useful and sufficient.

Definitions:
- useful: The newly added reasoning block contains information that is relevant and helpful for deriving the final answer. Redundant restatements or dead-end calculations are not useful.
- sufficient: The full candidate reasoning skeleton contains enough logically connected steps to derive the final answer without relying on hidden missing steps.

Return a JSON object only:
{{
  "useful": true or false,
  "sufficient": true or false
}}

Problem:
{question}

Ground-truth answer:
{answer}

Newly added reasoning block:
{new_block}

Full candidate reasoning skeleton:
{candidate_reasoning}
""".strip()


def get_question_text(row: Dict[str, Any]) -> str:
    """兼容不同数据字段命名。"""
    for key in ("question", "problem", "prompt", "input", "query"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    # 最后兜底：不要把 generations 带入问题，避免判别器偷看完整推理。
    return ""


async def prune_one_sample(
    row: Dict[str, Any],
    line_no: int,
    judge_client: VLLMJudgeClient,
    request_sem: asyncio.Semaphore,
    max_states: int,
    max_blocks: int,
) -> Dict[str, Any]:
    """
    对单条样本执行后向 BFS 裁剪。

    搜索策略：
    - 初始状态：只包含最后一个逻辑块。
    - 每个状态向前扩展一个块，形成更完整的候选骨架。
    - 若新增块 useful=False 且当前骨架 sufficient=False，则不再保留该分支。
    - 若 sufficient=True，立即返回当前候选。

    由于后向连续扩展通常已覆盖 CoT 骨架抽取需求，这里的状态空间保持为线性
    BFS，可避免组合爆炸和死循环。max_states/max_blocks 作为保险阈值。
    """
    generation = str(row.get("generations", "") or "")
    think_text, think_span = extract_think_span(generation)
    blocks = split_logic_blocks(think_text)
    if not blocks:
        raise PruningError("No reasoning blocks extracted from generations.")

    if len(blocks) > max_blocks:
        # 保留尾部 max_blocks 个块；逆向裁剪通常最依赖靠近结论的推理。
        offset = len(blocks) - max_blocks
        blocks = blocks[offset:]
    else:
        offset = 0

    question = get_question_text(row)
    answer = extract_final_answer_text(row, generation)
    if not answer:
        raise PruningError("Missing answer and cannot extract boxed final answer.")

    visited = set()
    queue: List[CandidateState] = [CandidateState(block_indices=(len(blocks) - 1,))]
    states_checked = 0
    best_useful_state: Optional[CandidateState] = None
    judge_trace: List[Dict[str, Any]] = []

    while queue and states_checked < max_states:
        state = queue.pop(0)
        if state.block_indices in visited:
            continue
        visited.add(state.block_indices)
        states_checked += 1

        candidate_text = build_candidate_text(blocks, state.block_indices)
        new_block = blocks[state.first_index]

        async with request_sem:
            result = await judge_client.judge(
                question=question,
                answer=answer,
                candidate_reasoning=candidate_text,
                new_block=new_block,
            )

        judge_trace.append(
            {
                "indices": [i + offset for i in state.block_indices],
                "useful": result.useful,
                "sufficient": result.sufficient,
                "raw": result.raw,
            }
        )

        if result.useful:
            best_useful_state = state

        if result.sufficient:
            pruned_generation = rebuild_generation_with_pruned_think(
                original_generation=generation,
                think_span=think_span,
                pruned_think=candidate_text,
                answer=answer,
            )
            out = dict(row)
            out["generations"] = pruned_generation
            out["pruned_think"] = candidate_text
            out["pruning_meta"] = {
                "status": "success",
                "line_no": line_no,
                "original_block_count": len(split_logic_blocks(think_text)),
                "searched_block_count": len(blocks),
                "kept_block_indices": [i + offset for i in state.block_indices],
                "kept_block_count": len(state.block_indices),
                "states_checked": states_checked,
                "judge_trace": judge_trace,
            }
            return out

        # 后向扩展：继续把更早的一个逻辑块加入骨架。
        prev_idx = state.first_index - 1
        if prev_idx >= 0:
            # 如果当前块被判定 useful=False，仍允许从最后一个状态向前扩展一次，
            # 因为尾部结论句常常只是答案复述，本身不 useful 但需要前序推理支撑。
            if result.useful or len(state.block_indices) == 1:
                next_indices = tuple(range(prev_idx, state.block_indices[-1] + 1))
                if next_indices not in visited:
                    queue.append(CandidateState(block_indices=next_indices))

    if best_useful_state is not None:
        best_text = build_candidate_text(blocks, best_useful_state.block_indices)
        reason = (
            "BFS ended without sufficient=True. "
            f"Best useful state kept {len(best_useful_state.block_indices)} blocks."
        )
    else:
        best_text = ""
        reason = "BFS ended without sufficient=True and no useful state was found."

    raise PruningError(
        f"{reason} states_checked={states_checked}, blocks={len(blocks)}, "
        f"best_preview={best_text[:200]!r}"
    )


async def process_one_with_outputs(
    row: Dict[str, Any],
    line_no: int,
    judge_client: VLLMJudgeClient,
    request_sem: asyncio.Semaphore,
    write_lock: asyncio.Lock,
    args: argparse.Namespace,
) -> Tuple[str, int]:
    """
    包装单条处理：成功写 pruned，失败写 failed。

    返回 (status, line_no)，供 tqdm 汇总。
    """
    try:
        pruned = await prune_one_sample(
            row=row,
            line_no=line_no,
            judge_client=judge_client,
            request_sem=request_sem,
            max_states=args.max_states,
            max_blocks=args.max_blocks,
        )
        async with write_lock:
            append_jsonl(args.output_pruned, pruned)
        return "success", line_no
    except Exception as exc:  # noqa: BLE001 - 单样本失败不能中断全局任务
        failed = dict(row)
        failed["pruning_meta"] = {
            "status": "failed",
            "line_no": line_no,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        async with write_lock:
            append_jsonl(args.output_failed, failed)
        return "failed", line_no


async def run(args: argparse.Namespace) -> None:
    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    rows = list(iter_jsonl(args.input))
    if not rows:
        print(f"No samples found in {args.input}")
        return

    # 清空旧输出，避免多次运行时 JSONL 混杂。
    args.output_pruned.parent.mkdir(parents=True, exist_ok=True)
    args.output_failed.parent.mkdir(parents=True, exist_ok=True)
    args.output_pruned.write_text("", encoding="utf-8")
    args.output_failed.write_text("", encoding="utf-8")

    judge_client = VLLMJudgeClient(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_retries=args.max_retries,
    )
    request_sem = asyncio.Semaphore(args.concurrency)
    write_lock = asyncio.Lock()

    tasks = [
        process_one_with_outputs(
            row=row,
            line_no=line_no,
            judge_client=judge_client,
            request_sem=request_sem,
            write_lock=write_lock,
            args=args,
        )
        for line_no, row in rows
    ]

    success = 0
    failed = 0
    started = time.time()

    for coro in tqdm.as_completed(tasks, total=len(tasks), desc="BFS pruning"):
        status, _line_no = await coro
        if status == "success":
            success += 1
        else:
            failed += 1

    elapsed = time.time() - started
    print("Pruning finished")
    print(f"  total   : {len(rows)}")
    print(f"  success : {success} -> {args.output_pruned}")
    print(f"  failed  : {failed} -> {args.output_failed}")
    print(f"  elapsed : {elapsed:.2f}s")
    if elapsed > 0:
        print(f"  speed   : {len(rows) / elapsed:.2f} samples/s")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backward BFS pruning with local vLLM.")
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--output-pruned", type=Path, default=OUTPUT_PRUNED_PATH)
    parser.add_argument("--output-failed", type=Path, default=OUTPUT_FAILED_PATH)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-states", type=int, default=128)
    parser.add_argument("--max-blocks", type=int, default=80)
    args = parser.parse_args(argv)

    if args.concurrency <= 0:
        parser.error("--concurrency must be positive.")
    if args.max_states <= 0:
        parser.error("--max-states must be positive.")
    if args.max_blocks <= 0:
        parser.error("--max-blocks must be positive.")
    if not math.isfinite(args.temperature) or args.temperature < 0:
        parser.error("--temperature must be a non-negative finite number.")

    return args


def main() -> None:
    args = parse_args()
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
